#include "predict.h"
#include <iostream>
#include <filesystem>
#include <numeric>
#include <thread>
#include "post_process.cpp"

namespace fs = std::filesystem;
namespace Predict {

// TensorRT 错误记录器
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << "TensorRT Logger: " << msg << std::endl;
        }
    }
} gLogger;

// 全局运行时（用于反序列化引擎）
static std::unique_ptr<nvinfer1::IRuntime> g_runtime;

// 为每个引擎缓存一个推理上下文（线程局部存储，避免锁竞争）
static thread_local std::unordered_map<nvinfer1::ICudaEngine*, TensorRTContext> g_context_cache;

// 加载TensorRT引擎
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
            if (!g_runtime) {
                g_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
            }
            engine = g_runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
            
            if (engine) {
                log_file << "TensorRT engine loaded from cache" << std::endl;
                return engine;
            }
        }
    }
    
    log_file << "No valid TensorRT engine cache found. Please ensure the cache path is correct." << std::endl;
    return nullptr;
}

// 释放资源
void freeResources(void* session, EngineType engine_type) {
    if (!session) return;
    
    if (engine_type == EngineType::TENSORRT) {
        auto* engine = static_cast<nvinfer1::ICudaEngine*>(session);
        // 释放线程局部的上下文和设备内存
        if (g_context_cache.count(engine)) {
            auto& ctx = g_context_cache[engine];
            if (ctx.input_buffer) cudaFree(ctx.input_buffer);
            if (ctx.output_buffer) cudaFree(ctx.output_buffer);
            if (ctx.stream) cudaStreamDestroy(ctx.stream);
            g_context_cache.erase(engine);
        }
        engine->destroy();
    } else if (engine_type == EngineType::ONNX_RUNTIME) {
        delete static_cast<Ort::Session*>(session);
    }
    g_runtime.reset();
}

// 加载模型（保持原有逻辑）
void* loadModel(
    const std::string& model_path,
    const Config& config,
    std::ofstream& log_file
) {
    try {
        if (config.engine_type == EngineType::ONNX_RUNTIME) {
            log_file << "Loading ONNX model with ONNX Runtime: " << model_path << std::endl;
            
            static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SegmentationModel");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(config.onnx.intra_op_num_threads);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
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
            
            std::wstring w_model_path = std::wstring(model_path.begin(), model_path.end());
            return new Ort::Session(env, w_model_path.c_str(), session_options);
        }
        else if (config.engine_type == EngineType::TENSORRT) {
            log_file << "Loading TensorRT engine: " << model_path << std::endl;
            auto* engine = loadTensorRTEngine(model_path, config.tensorrt, log_file);
            if (!engine) {
                throw std::runtime_error("Failed to load TensorRT engine");
            }
            return engine;
        }
        else {
            throw std::runtime_error("Unsupported inference engine type");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model: " + std::string(e.what()));
    }
}

// 图像预处理（适配16位输入，优化内存访问）
std::vector<float> preprocess_image(const cv::Mat& gray_img) {
    // 支持16位和8位输入（自动适配）
    const int type = gray_img.type();
    assert(type == CV_8UC1 || type == CV_16UC1 && "Input must be 8-bit or 16-bit grayscale");

    const int total_pixels = gray_img.rows * gray_img.cols;
    std::vector<float> input_data(total_pixels);
    float* dst = input_data.data();

    // 16位转float（0-65535 → 0-1）
    if (type == CV_16UC1) {
        const uint16_t* src = gray_img.ptr<uint16_t>();
        for (int i = 0; i < total_pixels; ++i) {
            dst[i] = static_cast<float>(src[i]) / 65535.0f;
        }
    } 
    // 8位转float（0-255 → 0-1）
    else {
        const uchar* src = gray_img.data;
        for (int i = 0; i < total_pixels; ++i) {
            dst[i] = static_cast<float>(src[i]) / 255.0f;
        }
    }

    return input_data;
}

// TensorRT推理实现（核心优化）
cv::Mat predict_mask_tensorrt(
    nvinfer1::ICudaEngine* engine,
    const cv::Mat& gray_img,
    const Predict::Config::Tensorrt& config
) {
    try {
        const int input_h = gray_img.rows;
        const int input_w = gray_img.cols;
        
        // 1. 初始化并复用执行上下文、流和设备内存
        TensorRTContext* ctx;
        if (!g_context_cache.count(engine)) {
            g_context_cache[engine] = TensorRTContext();
            ctx = &g_context_cache[engine];
            
            // 创建执行上下文和CUDA流
            ctx->context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
            cudaStreamCreate(&ctx->stream);
            
            // 获取输入绑定索引（仅输入可用getProfileDimensions）
            const int input_idx = engine->getBindingIndex(config.input_name.c_str());
            const int output_idx = engine->getBindingIndex(config.output_name.c_str());
            
            // 仅对输入绑定获取优化配置的尺寸范围（输出维度不通过此API获取）
            auto min_input_dims = engine->getProfileDimensions(input_idx, 0, nvinfer1::OptProfileSelector::kMIN);
            auto max_input_dims = engine->getProfileDimensions(input_idx, 0, nvinfer1::OptProfileSelector::kMAX);
            
            // 预分配输入设备内存（基于输入最大尺寸）
            ctx->max_input_size = 1 * 1 * max_input_dims.d[2] * max_input_dims.d[3] * sizeof(float);
            cudaMalloc(&ctx->input_buffer, ctx->max_input_size);
            
            // 输出内存不预分配固定大小（因输出尺寸由输入尺寸动态决定）
            ctx->output_buffer = nullptr;  // 改为动态分配
            
        } else {
            ctx = &g_context_cache[engine];
        }

        // 2. 检查输入尺寸是否在引擎支持的范围内（仅检查输入）
        const int input_idx = engine->getBindingIndex(config.input_name.c_str());
        auto min_dims = engine->getProfileDimensions(input_idx, 0, nvinfer1::OptProfileSelector::kMIN);
        auto max_dims = engine->getProfileDimensions(input_idx, 0, nvinfer1::OptProfileSelector::kMAX);
        
        if (input_h < min_dims.d[2] || input_h > max_dims.d[2] || 
            input_w < min_dims.d[3] || input_w > max_dims.d[3]) {
            throw std::runtime_error(
                "Input size out of range! Expected H: " + std::to_string(min_dims.d[2]) + "-" + std::to_string(max_dims.d[2]) +
                ", W: " + std::to_string(min_dims.d[3]) + "-" + std::to_string(max_dims.d[3]) +
                ", got " + std::to_string(input_h) + "x" + std::to_string(input_w)
            );
        }

        // 3. 设置输入形状并获取输出维度
        ctx->context->setInputShape(config.input_name.c_str(), nvinfer1::Dims4(1, 1, 512, 512));  // 输入形状：[N=1, C=1, H=512, W=512]
        const int output_idx = engine->getBindingIndex(config.output_name.c_str());
        nvinfer1::Dims output_dims = ctx->context->getBindingDimensions(output_idx);  // 从执行上下文获取输出维度（正确方式）
        
        const int num_classes = 3;
        const int output_h = 512;
        const int output_w = 512;

        // 4. 动态分配输出设备内存（根据实际输出尺寸）
        const size_t input_size = 1 * 1 * input_h * input_w * sizeof(float);
        const size_t output_size = num_classes * output_h * output_w * sizeof(float);
        
        // 若预分配内存不足，重新分配（仅在必要时）
        if (ctx->output_buffer == nullptr || output_size > ctx->max_output_size) {
            if (ctx->output_buffer) cudaFree(ctx->output_buffer);
            cudaMalloc(&ctx->output_buffer, output_size);
            ctx->max_output_size = output_size;
        }

        // 5. 预处理输入数据
        std::vector<float> input_data = preprocess_image(gray_img);

        

        // auto start_time = std::chrono::high_resolution_clock::now();
        
        // 6. 异步执行推理
        void* buffers[] = {ctx->input_buffer, ctx->output_buffer};
        cudaMemcpyAsync(buffers[input_idx], input_data.data(), input_size, cudaMemcpyHostToDevice, ctx->stream);
        ctx->context->enqueueV2(buffers, ctx->stream, nullptr);

        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        //     std::chrono::high_resolution_clock::now() - start_time).count();

        // std::cout<<"Pure prediction time: "<< duration <<std::endl;

        // 7. 读取输出数据
        std::vector<float> output_data(num_classes * output_h * output_w);
        cudaMemcpyAsync(output_data.data(), buffers[output_idx], output_size, cudaMemcpyDeviceToHost, ctx->stream);
        cudaStreamSynchronize(ctx->stream);

        // 8. 后处理：生成正确尺寸的掩码
        cv::Mat pred_mask(output_h, output_w, CV_8UC1);
        
        if (num_classes == 1) {
            // 二分类：阈值化
            cv::Mat output_mat(output_h, output_w, CV_32FC1, output_data.data());
            output_mat.convertTo(pred_mask, CV_8UC1, 255.0);
            cv::threshold(pred_mask, pred_mask, 127, 255, cv::THRESH_BINARY);
        } else {
            // 多分类：argmax（修复版本）
            
            // 创建单通道的概率最大值矩阵和类别索引矩阵
            cv::Mat max_prob(output_h, output_w, CV_32FC1, cv::Scalar(-FLT_MAX));  // 初始化为负无穷
            cv::Mat class_idx(output_h, output_w, CV_8UC1, cv::Scalar(0));         // 初始化为类别0
            
            // 遍历每个类别，更新最大值和对应类别索引
            for (int c = 0; c < num_classes; ++c) {
                // 获取当前类别的概率图
                cv::Mat class_channel(output_h, output_w, CV_32FC1, &output_data[c*output_h*output_w]);
                
                // 找出当前类别概率大于max_prob的位置
                cv::Mat update_mask;
                cv::compare(class_channel, max_prob, update_mask, cv::CMP_GT);
                
                // 更新最大值矩阵
                class_channel.copyTo(max_prob, update_mask);
                
                // 更新类别索引矩阵
                cv::Mat class_mask = cv::Mat::ones(output_h, output_w, CV_8UC1) * c;
                class_mask.copyTo(class_idx, update_mask);
            }
            // 直接使用class_idx作为预测掩码
            class_idx.copyTo(pred_mask);
        }
        return pred_mask;

    } catch (const std::exception& e) {
        throw std::runtime_error("TensorRT inference failed: " + std::string(e.what()));
    }
}

// ONNX Runtime推理实现
cv::Mat predict_mask_onnx(
    Ort::Session* session,
    const cv::Mat& gray_img,
    const Predict::Config::Onnx& config
) {
    try {
        const char* input_names[] = {config.input_name.c_str()};
        const char* output_names[] = {config.output_name.c_str()};

        std::vector<float> input_data = preprocess_image(gray_img);
        std::vector<int64_t> input_shape = {1, 1, gray_img.rows, gray_img.cols};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size()
        );

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_tensor));
        
        auto output_tensors = session->Run(Ort::RunOptions{nullptr},
            input_names, input_tensors.data(), 1,
            output_names, 1
        );

        Ort::Value& output_tensor = output_tensors[0];
        auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_dims = output_info.GetShape();

        int num_classes = output_dims[1];
        int output_h = output_dims[2];
        int output_w = output_dims[3];
        float* output_data = output_tensor.GetTensorMutableData<float>();

        // 优化后处理（同TensorRT）
        cv::Mat pred_mask(output_h, output_w, CV_8UC1);
        if (num_classes == 1) {
            cv::Mat output_mat(output_h, output_w, CV_32FC1, output_data);
            output_mat.convertTo(pred_mask, CV_8UC1, 255.0);
            cv::threshold(pred_mask, pred_mask, 127, 255, cv::THRESH_BINARY);
        } else {
            cv::Mat output_mat(output_h, output_w, CV_32FC(num_classes), output_data);
            cv::reduce(output_mat, pred_mask, 2, cv::REDUCE_MAX);
            pred_mask.convertTo(pred_mask, CV_8UC1);
        }

        return pred_mask;
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("ONNX Runtime inference failed: " + std::string(e.what()));
    }
}

// 模型推理分发（保持不变）
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

// 单张图像预测主函数（支持直接传入cv::Mat）
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

        // 读取图像（支持8/16位灰度图）
        cv::Mat gray_img = cv::imread(input_img_path, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
        if (gray_img.empty()) {
            throw std::runtime_error("Failed to read image: " + input_img_path);
        }
        log_file << "Input image size: " << gray_img.cols << "x" << gray_img.rows 
                 << " (type: " << (gray_img.type() == CV_16UC1 ? "16-bit" : "8-bit") << ")" << std::endl;

        // 推理计时
        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat pred_mask = predict_mask(session, gray_img, config);
        pred_mask = postprocess_mask(pred_mask);

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();

        log_file << "Postprocess applied" << std::endl;
        std::cout << "Postprocess applied" << std::endl;
        log_file << "Inference time: " << duration << " ms" << std::endl;
        std::cout << "Inference time: " << duration << " ms" << std::endl;

        // 保存结果
        cv::Mat vis_mask = mask_to_image(pred_mask);
        fs::create_directories(fs::path(output_mask_path).parent_path());
        std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};  // 快速保存
        if (!cv::imwrite(output_mask_path, vis_mask, params)) {
            throw std::runtime_error("Failed to save mask: " + output_mask_path);
        }

        msg = "Mask saved to: " + output_mask_path + " (size: " + 
              std::to_string(vis_mask.cols) + "x" + std::to_string(vis_mask.rows) + ")";
        log_file << msg << std::endl;
        std::cout << msg << std::endl;

    } catch (const std::exception& e) {
        std::string error = "Prediction error: " + std::string(e.what());
        log_file << error << std::endl;
        std::cerr << error << std::endl;
    }
}

// 新增：直接传入cv::Mat的预测接口（用于无文件路径场景）
void predict_single_image(
    void* session,
    const cv::Mat& gray_img,
    const std::string& output_mask_path,
    const Config& config,
    std::ofstream& log_file
) {
    try {
        log_file << "Start predicting from in-memory image" << std::endl;
        std::cout << "Start predicting from in-memory image" << std::endl;

        if (gray_img.empty()) {
            throw std::runtime_error("Input image is empty");
        }
        log_file << "Input image size: " << gray_img.cols << "x" << gray_img.rows 
                 << " (type: " << (gray_img.type() == CV_16UC1 ? "16-bit" : "8-bit") << ")" << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat pred_mask = predict_mask(session, gray_img, config);
        pred_mask = postprocess_mask(pred_mask);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();

        log_file << "Postprocess applied" << std::endl;
        std::cout << "Postprocess applied" << std::endl;
        log_file << "Inference time: " << duration << " ms" << std::endl;
        std::cout << "Inference time: " << duration << " ms" << std::endl;

        // 保存结果
        cv::Mat vis_mask = mask_to_image(pred_mask);
        fs::create_directories(fs::path(output_mask_path).parent_path());
        std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
        if (!cv::imwrite(output_mask_path, vis_mask, params)) {
            throw std::runtime_error("Failed to save mask: " + output_mask_path);
        }

        std::string msg = "Mask saved to: " + output_mask_path + " (size: " + 
              std::to_string(vis_mask.cols) + "x" + std::to_string(vis_mask.rows) + ")";
        log_file << msg << std::endl;
        std::cout << msg << std::endl;

    } catch (const std::exception& e) {
        std::string error = "Prediction error: " + std::string(e.what());
        log_file << error << std::endl;
        std::cerr << error << std::endl;
    }
}

}  // namespace Predict