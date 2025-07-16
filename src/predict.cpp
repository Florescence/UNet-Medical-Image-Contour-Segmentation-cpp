#include "predict.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include "post_process.cpp"

namespace fs = std::filesystem;
namespace Predict {

// 释放 TensorRT 资源的工具函数
void freeTrtResources(nvinfer1::IExecutionContext* context, nvinfer1::ICudaEngine* engine, nvinfer1::IRuntime* runtime) {
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
}

TRTLogger::TRTLogger(std::ofstream& log_file) : log_file_(log_file) {}

void TRTLogger::log(Severity severity, const char* msg) noexcept {
    std::string severity_str;
    switch (severity) {
        case Severity::kINTERNAL_ERROR: severity_str = "[INTERNAL ERROR]"; break;
        case Severity::kERROR:          severity_str = "[ERROR]"; break;
        case Severity::kWARNING:        severity_str = "[WARNING]"; break;
        case Severity::kINFO:           severity_str = "[INFO]"; break;
        default:                        severity_str = "[UNKNOWN]";
    }
    std::string log_msg = "TRT " + severity_str + ": " + std::string(msg);
    //std::cout << log_msg << std::endl;  // 同时输出到控制台
    log_file_ << log_msg << std::endl;  // 输出到日志文件
}

/**
 * 用 TensorRT 解析 ONNX 模型并构建引擎（支持动态形状和序列化缓存）
 */
nvinfer1::ICudaEngine* buildTrtEngine(
    const std::string& onnx_path, 
    const std::string& engine_cache_path, 
    TRTLogger& logger, 
    const TRTConfig& trt_config) {
    // 日志输出通过 logger 写入文件
    logger.log(nvinfer1::ILogger::Severity::kINFO, "Attempting to load or build TensorRT engine...");

    // 若存在缓存的引擎文件，直接加载
    if (fs::exists(engine_cache_path)) {
        std::ifstream engine_file(engine_cache_path, std::ios::binary);
        if (engine_file) {
            engine_file.seekg(0, std::ios::end);
            size_t size = engine_file.tellg();
            engine_file.seekg(0, std::ios::beg);
            std::vector<char> engine_data(size);
            engine_file.read(engine_data.data(), size);
            
            nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
            nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), size);
            runtime->destroy();
            
            std::string msg = "Loaded TensorRT engine from cache: " + engine_cache_path;
            logger.log(nvinfer1::ILogger::Severity::kINFO, msg.c_str());
            return engine;
        }
    }

    // 无缓存时，从 ONNX 构建引擎
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    nvinfer1::IBuilderConfig* builder_config = builder->createBuilderConfig();
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

    // 解析 ONNX 模型
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::string error_msg = "Failed to parse ONNX model! Errors: ";
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            error_msg += parser->getError(i)->desc();
        }
        throw std::runtime_error(error_msg);
    }

    // 关键修改：检查并修正模型输入维度（处理动态维度）
    nvinfer1::ITensor* input_tensor = network->getInput(0);
    nvinfer1::Dims input_dims = input_tensor->getDimensions();
    logger.log(nvinfer1::ILogger::Severity::kINFO, "Original model input dimensions:");

    // 修正负数维度（动态维度）
    for (int i = 0; i < input_dims.nbDims; ++i) {
        if (input_dims.d[i] < 0) {
            input_dims.d[i] = (i == 0) ? 1 : input_dims.d[i];  // 批次维度设为1
            logger.log(nvinfer1::ILogger::Severity::kINFO, ("Replaced negative dimension at index " + std::to_string(i) + " with " + std::to_string(input_dims.d[i])).c_str());
        }
    }

    // 验证所有维度非负
    for (int i = 0; i < input_dims.nbDims; ++i) {
        if (input_dims.d[i] < 0) {
            throw std::runtime_error("Input dimension " + std::to_string(i) + " is still negative after correction!");
        }
    }

    // 配置引擎参数（使用 TRTConfig）
    builder_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, trt_config.workspace_size);
    if (trt_config.use_fp16 && builder->platformHasFastFp16()) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        logger.log(nvinfer1::ILogger::Severity::kINFO, "Enabled FP16 precision for acceleration");
    }

    // 为动态输入添加优化配置文件
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    const char* input_name = "input";

    // 检查输入维度是否为 [N, C, H, W]
    if (input_dims.nbDims != 4) {
        throw std::runtime_error("Expected 4D input (N, C, H, W), but got " + std::to_string(input_dims.nbDims) + "D");
    }

    // 设置动态尺寸范围
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, 
        nvinfer1::Dims4(input_dims.d[0], input_dims.d[1], trt_config.min_height, trt_config.min_width));
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, 
        nvinfer1::Dims4(input_dims.d[0], input_dims.d[1], trt_config.opt_height, trt_config.opt_width));
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, 
        nvinfer1::Dims4(input_dims.d[0], input_dims.d[1], trt_config.max_height, trt_config.max_width));
    builder_config->addOptimizationProfile(profile);

    // 构建引擎并序列化
    nvinfer1::IHostMemory* serialized_engine = builder->buildSerializedNetwork(*network, *builder_config);
    if (!serialized_engine) {
        throw std::runtime_error("Failed to build serialized engine (check input shape or TRTConfig)");
    }
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *builder_config);
    if (!engine) {
        throw std::runtime_error("Failed to build TensorRT engine from network");
    }

    // 保存引擎到缓存
    std::ofstream engine_file(engine_cache_path, std::ios::binary);
    engine_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    logger.log(nvinfer1::ILogger::Severity::kINFO, ("Built and saved TensorRT engine to: " + engine_cache_path).c_str());

    // 释放中间资源
    parser->destroy();
    builder_config->destroy();
    network->destroy();
    builder->destroy();
    serialized_engine->destroy();

    return engine;
}

/**
 * 图像预处理（保持不变）
 */
std::vector<float> preprocess_image(const cv::Mat& gray_img) {
    std::vector<float> input_data(gray_img.rows * gray_img.cols);
    for (int i = 0; i < gray_img.rows; ++i) {
        for (int j = 0; j < gray_img.cols; ++j) {
            input_data[i * gray_img.cols + j] = static_cast<float>(gray_img.at<uchar>(i, j)) / 255.0f;
        }
    }
    return input_data;
}

/**
 * 模型预测：适配动态形状输入
 */
cv::Mat predict_mask(nvinfer1::ICudaEngine* engine, const cv::Mat& gray_img, std::ofstream& log_file) {
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // 获取输入输出绑定信息
    const int input_idx = engine->getBindingIndex("input");
    const int output_idx = engine->getBindingIndex("output");
    nvinfer1::Dims input_dims = engine->getBindingDimensions(input_idx);

    // 动态设置当前输入尺寸
    input_dims.d[2] = gray_img.rows;  // H
    input_dims.d[3] = gray_img.cols;  // W
    if (!context->setBindingDimensions(input_idx, input_dims)) {
        throw std::runtime_error("Input size out of range! Check TRTConfig min/max dimensions");
    }

    // 获取调整后的输出尺寸
    nvinfer1::Dims output_dims = context->getBindingDimensions(output_idx);

    // 计算内存大小
    size_t input_size = 1;
    for (int i = 0; i < input_dims.nbDims; ++i) input_size *= input_dims.d[i];
    input_size *= sizeof(float);

    size_t output_size = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) output_size *= output_dims.d[i];
    output_size *= sizeof(float);

    // 分配 GPU 内存
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    // 预处理并拷贝数据
    std::vector<float> input_data = preprocess_image(gray_img);
    cudaMemcpy(d_input, input_data.data(), input_size, cudaMemcpyHostToDevice);

    // 执行推理
    void* bindings[] = {d_input, d_output};
    context->executeV2(bindings);

    // 拷贝输出数据到 CPU
    std::vector<float> output_data(output_size / sizeof(float));
    cudaMemcpy(output_data.data(), d_output, output_size, cudaMemcpyDeviceToHost);

    // 释放 GPU 内存和上下文
    cudaFree(d_input);
    cudaFree(d_output);
    context->destroy();

    // 后处理生成掩码
    cv::Mat pred_mask(gray_img.rows, gray_img.cols, CV_8UC1);
    int H = gray_img.rows, W = gray_img.cols;
    int num_classes = output_dims.d[1];  // 从输出维度获取类别数
    log_file << "Prediction: Output classes detected - " << num_classes << std::endl;

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            int max_idx = 0;
            float max_val = output_data[i * W * num_classes + j * num_classes + 0];
            for (int c = 1; c < num_classes; ++c) {
                float val = output_data[i * W * num_classes + j * num_classes + c];
                if (val > max_val) {
                    max_val = val;
                    max_idx = c;
                }
            }
            pred_mask.at<uchar>(i, j) = static_cast<uchar>(max_idx);
        }
    }

    return pred_mask;
}

/**
 * 掩码可视化（保持不变）
 */
cv::Mat mask_to_image(const cv::Mat& mask) {
    cv::Mat vis_mask(mask.size(), CV_8UC1);
    uchar lut[256] = {0};
    lut[1] = 128;
    lut[2] = 255;
    cv::LUT(mask, cv::Mat(1, 256, CV_8UC1, lut), vis_mask);
    return vis_mask;
}

/**
 * 单张图像预测主函数（支持日志文件输出）
 */
void predict_single_image(
    nvinfer1::ICudaEngine* engine,
    const std::string& input_img_path,
    const std::string& output_mask_path,
    std::ofstream& log_file
) {
    try {
        std::string msg = "Start Predicting: " + fs::path(input_img_path).filename().string();
        log_file << msg << std::endl;
        std::cout << msg << std::endl;

        cv::Mat gray_img = cv::imread(input_img_path, cv::IMREAD_GRAYSCALE);
        if (gray_img.empty()) {
            throw std::runtime_error("Failed to read image: " + input_img_path);
        }

        cv::Mat pred_mask = predict_mask(engine, gray_img, log_file);
        pred_mask = postprocess_mask(pred_mask);
        log_file << "Postprocess applied" << std::endl;
        std::cout << "Postprocess applied" << std::endl;

        cv::Mat vis_mask = mask_to_image(pred_mask);
        fs::create_directories(fs::path(output_mask_path).parent_path());
        std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
        if (!cv::imwrite(output_mask_path, vis_mask, params)) {
            throw std::runtime_error("Failed to save mask: " + output_mask_path);
        }

        msg = "Mask saved to: " + output_mask_path;
        log_file << msg << std::endl;
        std::cout << msg << std::endl;

    } catch (const std::exception& e) {
        std::string error = "Prediction error: " + std::string(e.what());
        log_file << error << std::endl;
        std::cerr << error << std::endl;
    }
}

}  // namespace Predict