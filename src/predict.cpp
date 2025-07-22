#include "predict.h"
#include <iostream>
#include <filesystem>
#include <numeric>
#include "post_process.cpp"

namespace fs = std::filesystem;
namespace Predict {

static std::unique_ptr<nvinfer1::IRuntime> g_runtime;

// TensorRT 错误记录器
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << "TensorRT Logger: " << msg << std::endl;
        }
    }
} gLogger;

// 创建TensorRT引擎
nvinfer1::ICudaEngine* createTensorRTEngine(
    const std::string& onnx_path,
    const Predict::Config::Tensorrt& config,
    std::ofstream& log_file
) {
    log_file << "Creating TensorRT engine from: " << onnx_path << std::endl;
    
    // 创建构建器和网络
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    
    // 解析ONNX模型
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR))) {
        log_file << "Failed to parse ONNX file" << std::endl;
        return nullptr;
    }
    
    // 配置构建器
    nvinfer1::IBuilderConfig* configBuilder = builder->createBuilderConfig();
    configBuilder->setMaxWorkspaceSize(config.max_workspace_size);
    if (config.fp16_mode && builder->platformHasFastFp16()) {
        configBuilder->setFlag(nvinfer1::BuilderFlag::kFP16);
        log_file << "Using FP16 precision for TensorRT engine" << std::endl;
    } else {
        log_file << "Using FP32 precision for TensorRT engine" << std::endl;
    }

    // 创建优化配置文件
    nvinfer1::Dims4 fixed(1, 1, 512, 512);
    const char* inName = network->getInput(0)->getName();
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(inName, nvinfer1::OptProfileSelector::kMIN, fixed);
    profile->setDimensions(inName, nvinfer1::OptProfileSelector::kOPT, fixed);
    profile->setDimensions(inName, nvinfer1::OptProfileSelector::kMAX, fixed);

    // 将配置文件添加到构建器
    configBuilder->addOptimizationProfile(profile);
    log_file << "Optimization profile created with dimensions: [1, 1, 512, 512]" << std::endl;

    // 构建引擎
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *configBuilder);
    
    // 释放资源
    parser->destroy();
    network->destroy();
    configBuilder->destroy();
    builder->destroy();
    
    if (!engine) {
        log_file << "Failed to create TensorRT engine (check dimension ranges)" << std::endl;
    }
    return engine;
}

// 加载或创建TensorRT引擎（支持缓存）
nvinfer1::ICudaEngine* loadTensorRTEngine(
    const std::string& onnx_path,
    const Predict::Config::Tensorrt& config,
    std::ofstream& log_file
) {
    nvinfer1::ICudaEngine* engine = nullptr;
    
    // 检查是否有缓存的引擎
    if (!config.engine_cache_path.empty()) {
        std::ifstream file(config.engine_cache_path, std::ios::binary);
        if (file.good()) {
            log_file << "Loading cached TensorRT engine from: " << config.engine_cache_path << std::endl;
            
            // 读取缓存的引擎
            std::vector<char> trtModelStream;
            file.seekg(0, file.end);
            size_t size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
            
            // 创建运行时并反序列化引擎
            nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
            engine = runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
            // runtime->destroy();
            
            if (engine) {
                log_file << "TensorRT engine loaded from cache" << std::endl;
                return engine;
            }
        }
    }
    
    // 没有缓存或加载失败，从头创建引擎
    engine = createTensorRTEngine(onnx_path, config, log_file);
    if (!engine) {
        log_file << "Failed to create TensorRT engine" << std::endl;
        return nullptr;
    }
    
    // 保存引擎到缓存
    if (!config.engine_cache_path.empty()) {
        log_file << "Serializing TensorRT engine to cache: " << config.engine_cache_path << std::endl;
        
        // 序列化引擎
        nvinfer1::IHostMemory* serializedEngine = engine->serialize();
        
        // 写入文件
        std::ofstream file(config.engine_cache_path, std::ios::binary);
        if (file) {
            file.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
            file.close();
        }
        
        serializedEngine->destroy();
    }
    
    return engine;
}

// 释放资源
void freeResources(void* session, EngineType engine_type) {
    if (!session) return;
    
    if (engine_type == EngineType::ONNX_RUNTIME) {
        delete static_cast<Ort::Session*>(session);
    } else if (engine_type == EngineType::TENSORRT) {
        static_cast<nvinfer1::ICudaEngine*>(session)->destroy();
    }
    g_runtime.reset();
}

// 加载模型
void* loadModel(
    const std::string& model_path,
    const Config& config,
    std::ofstream& log_file
) {
    try {
        if (config.engine_type == EngineType::ONNX_RUNTIME) {
            log_file << "Loading ONNX model with ONNX Runtime: " << model_path << std::endl;
            
            // 创建环境
            static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SegmentationModel");
            
            // 配置会话选项
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(config.onnx.intra_op_num_threads);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // 启用GPU（若配置且支持）
            if (config.onnx.use_gpu) {
                try {
                    #ifdef USE_CUDA
                    session_options.AppendExecutionProvider_CUDA({});
                    #endif
                    log_file << "Using CUDA execution provider" << std::endl;
                } catch (...) {
                    log_file << "CUDA not available, falling back to CPU" << std::endl;
                }
            } else {
                log_file << "Using CPU execution provider" << std::endl;
            }
            
            // 将std::string转换为ORTCHAR_T*
            std::wstring w_model_path = std::wstring(model_path.begin(), model_path.end());
            const ORTCHAR_T* ort_model_path = w_model_path.c_str();
            
            // 创建会话
            Ort::Session* session = new Ort::Session(env, ort_model_path, session_options);
            log_file << "ONNX model loaded successfully" << std::endl;
            
            return session;
        }
        else if (config.engine_type == EngineType::TENSORRT) {
            log_file << "Loading ONNX model with TensorRT: " << model_path << std::endl;
            
            // 加载或创建TensorRT引擎
            nvinfer1::ICudaEngine* engine = loadTensorRTEngine(model_path, config.tensorrt, log_file);
            if (!engine) {
                throw std::runtime_error("Failed to load TensorRT engine");
            }
            
            log_file << "TensorRT engine loaded successfully" << std::endl;
            return engine;
        }
        else {
            throw std::runtime_error("Unsupported inference engine type");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model: " + std::string(e.what()));
    }
}

// 图像预处理（保持不变）
std::vector<float> preprocess_image(const cv::Mat& gray_img) {
    // 仅检查图像类型（8位灰度图），不限制尺寸
    assert(gray_img.type() == CV_8UC1 && "Input must be 8-bit grayscale image");

    // 获取图像总像素数（由实际尺寸决定）
    const int total_pixels = gray_img.rows * gray_img.cols;
    
    // 创建输出向量
    std::vector<float> input_data(total_pixels);

    // 直接访问原始数据进行归一化（[0,255] → [0,1]）
    const uchar* src_data = gray_img.data;
    float* dst_data = input_data.data();
    for (int i = 0; i < total_pixels; ++i) {
        dst_data[i] = static_cast<float>(src_data[i]) / 255.0f;
    }

    return input_data;
}

// TensorRT推理实现
cv::Mat predict_mask_tensorrt(
    nvinfer1::ICudaEngine* engine,
    const cv::Mat& gray_img,
    const Predict::Config::Tensorrt& config
) {
    try {
        // 创建执行上下文
        nvinfer1::IExecutionContext* context = engine->createExecutionContext();
        
        // 获取输入输出绑定索引
        int input_idx = engine->getBindingIndex(config.input_name.c_str());
        int output_idx = engine->getBindingIndex(config.output_name.c_str());
        
        // 检查输入尺寸是否在优化配置的范围内（H和W）
        int input_h = gray_img.rows;
        int input_w = gray_img.cols;
        if (input_h < 128 || input_h > 4267 || input_w < 128 || input_w > 4267) {  // 对应步骤1的min/max
            throw std::runtime_error("Input size out of range! Expected H/W between 128-4267, got " + std::to_string(input_h) + "x" + std::to_string(input_w));
        }

        // 预处理输入数据
        std::vector<float> input_data = preprocess_image(gray_img);
        std::vector<int64_t> input_shape = {1, 1, gray_img.rows, gray_img.cols};
        
        // 设置动态输入尺寸（如果模型支持）
        context->setInputShape(config.input_name.c_str(), nvinfer1::Dims4(1, 1, gray_img.rows, gray_img.cols));
        
        // 获取输出尺寸
        nvinfer1::Dims output_dims = context->getBindingDimensions(output_idx);
        int num_classes = output_dims.d[1];
        int output_height = output_dims.d[2];
        int output_width = output_dims.d[3];
        
        // 分配GPU和CPU内存
        void* buffers[2];
        size_t input_size = 1 * 1 * input_h * input_w;
        size_t output_size = num_classes * output_height * output_width;
        
        cudaMalloc(&buffers[input_idx], input_size * sizeof(float));
        cudaMalloc(&buffers[output_idx], output_size * sizeof(float));
        
        // 创建CUDA流
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // 将输入数据复制到GPU
        cudaMemcpyAsync(buffers[input_idx], input_data.data(), input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
        
        // 执行推理
        context->enqueueV2(buffers, stream, nullptr);
        
        // 分配CPU输出缓冲区
        std::vector<float> output_data(output_size);
        
        // 将输出数据复制回CPU
        cudaMemcpyAsync(output_data.data(), buffers[output_idx], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
        
        // 等待流完成
        cudaStreamSynchronize(stream);
        
        // 释放GPU内存和流
        cudaFree(buffers[input_idx]);
        cudaFree(buffers[output_idx]);
        cudaStreamDestroy(stream);
        
        // 后处理：argmax获取类别掩码
        cv::Mat pred_mask(output_height, output_width, CV_8UC1);
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                int base_idx = i * output_width + j;  // H*W维度偏移
                int max_idx = 0;
                float max_val = output_data[base_idx];  // 类别0的概率

                for (int c = 1; c < num_classes; ++c) {
                    int idx = c * output_height * output_width + base_idx;  // 类别c的偏移
                    if (output_data[idx] > max_val) {
                        max_val = output_data[idx];
                        max_idx = c;
                    }
                }

                pred_mask.at<uchar>(i, j) = static_cast<uchar>(max_idx);
            }
        }
        // 释放上下文
        context->destroy();
        
        return pred_mask;
    } catch (const std::exception& e) {
        throw std::runtime_error("TensorRT inference failed: " + std::string(e.what()));
    }
}

// ONNX Runtime推理实现（保持原有逻辑）
cv::Mat predict_mask_onnx(
    Ort::Session* session,
    const cv::Mat& gray_img,
    const Predict::Config::Onnx& config
) {
    try {
        // 输入输出节点名
        const char* input_names[] = {config.input_name.c_str()};
        const char* output_names[] = {config.output_name.c_str()};

        // 预处理输入数据（使用实际图像尺寸）
        std::vector<float> input_data = preprocess_image(gray_img);
        // 输入形状：[N=1, C=1, H=图像高度, W=图像宽度]（完全由输入图像决定）
        std::vector<int64_t> input_shape = {1, 1, gray_img.rows, gray_img.cols};

        // 创建输入张量
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(),
            input_data.size(),
            input_shape.data(),
            input_shape.size()
        );

        // 执行推理
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_tensor));
        
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names,
            input_tensors.data(),
            1,
            output_names,
            1
        );

        // 解析输出张量
        Ort::Value& output_tensor = output_tensors[0];
        auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_dims = output_info.GetShape();

        // 提取输出参数（输出尺寸与输入尺寸一致，因模型通常保持空间尺寸）
        int num_classes = output_dims[1];
        int output_height = output_dims[2];
        int output_width = output_dims[3];
        float* output_data = output_tensor.GetTensorMutableData<float>();

        // 后处理：argmax获取类别掩码
        cv::Mat pred_mask(output_height, output_width, CV_8UC1);
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                int base_idx = i * output_width + j;  // H*W维度偏移
                int max_idx = 0;
                float max_val = output_data[base_idx];  // 类别0的概率

                for (int c = 1; c < num_classes; ++c) {
                    int idx = c * output_height * output_width + base_idx;  // 类别c的偏移
                    if (output_data[idx] > max_val) {
                        max_val = output_data[idx];
                        max_idx = c;
                    }
                }

                pred_mask.at<uchar>(i, j) = static_cast<uchar>(max_idx);
            }
        }

        return pred_mask;
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("ONNX Runtime inference failed: " + std::string(e.what()));
    }
}

// 模型推理（根据引擎类型选择实现）
cv::Mat predict_mask(
    void* session,
    const cv::Mat& gray_img,
    const Config& config
) {
    if (config.engine_type == EngineType::ONNX_RUNTIME) {
        return predict_mask_onnx(static_cast<Ort::Session*>(session), gray_img, config.onnx);
    }
    else if (config.engine_type == EngineType::TENSORRT) {
        return predict_mask_tensorrt(static_cast<nvinfer1::ICudaEngine*>(session), gray_img, config.tensorrt);
    }
    else {
        throw std::runtime_error("Unsupported inference engine type");
    }
}

// 掩码可视化（保持不变）
cv::Mat mask_to_image(const cv::Mat& mask) {
    cv::Mat vis_mask(mask.size(), CV_8UC1);
    uchar lut[256] = {0};
    lut[1] = 128;
    lut[2] = 255;
    cv::LUT(mask, cv::Mat(1, 256, CV_8UC1, lut), vis_mask);
    return vis_mask;
}

// 单张图像预测主函数
void predict_single_image(
    void* session,
    const std::string& input_img_path,
    const std::string& output_mask_path,
    const Config& config,
    std::ofstream& log_file
) {
    try {
        std::string msg = "Start predicting: " + fs::path(input_img_path).filename().string();
        log_file << msg << std::endl;
        std::cout << msg << std::endl;

        // 读取图像
        cv::Mat gray_img = cv::imread(input_img_path, cv::IMREAD_GRAYSCALE);
        if (gray_img.empty()) {
            throw std::runtime_error("Failed to read image: " + input_img_path);
        }
        // 打印输入图像尺寸
        log_file << "Input image size: " << gray_img.cols << "x" << gray_img.rows << std::endl;

        // 记录开始时间
        auto start_time = std::chrono::high_resolution_clock::now();

        // 推理（自动适应输入尺寸）
        cv::Mat pred_mask = predict_mask(session, gray_img, config);
        pred_mask = postprocess_mask(pred_mask);
        log_file << "Postprocess applied" << std::endl;
        std::cout << "Postprocess applied" << std::endl;

        // 记录结束时间并计算耗时
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        log_file << "Inference time: " << duration << " ms" << std::endl;
        std::cout << "Inference time: " << duration << " ms" << std::endl;

        // 保存结果（输出尺寸与输入一致）
        cv::Mat vis_mask = mask_to_image(pred_mask);
        fs::create_directories(fs::path(output_mask_path).parent_path());
        std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
        if (!cv::imwrite(output_mask_path, vis_mask, params)) {
            throw std::runtime_error("Failed to save mask: " + output_mask_path);
        }

        msg = "Mask saved to: " + output_mask_path + " (size: " + std::to_string(vis_mask.cols) + "x" + std::to_string(vis_mask.rows) + ")";
        log_file << msg << std::endl;
        std::cout << msg << std::endl;

    } catch (const std::exception& e) {
        std::string error = "Prediction error: " + std::string(e.what());
        log_file << error << std::endl;
        std::cerr << error << std::endl;
    }
}

}  // namespace Predict