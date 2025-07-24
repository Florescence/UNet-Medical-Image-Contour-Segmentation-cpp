#pragma once
#include <string>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// TensorRT 相关头文件
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>

namespace Predict {
    // 推理引擎类型
    enum class EngineType {
        ONNX_RUNTIME,
        TENSORRT
    };

    // 前向声明嵌套结构体，避免类型名解析错误
    struct Config;

    // 释放资源
    void freeResources(void* session, EngineType engine_type);

    // 加载模型
    void* loadModel(
        const std::string& model_path,
        const Config& config,
        std::ofstream& log_file
    );

    // 图像预处理
    std::vector<float> preprocess_image(const cv::Mat& gray_img);

    // 模型推理
    cv::Mat predict_mask(
        void* session,
        const cv::Mat& gray_img,
        const Config& config
    );

    // 掩码可视化
    cv::Mat mask_to_image(const cv::Mat& mask);

    // 单张图像预测主函数
    void predict_single_image(
        void* session,
        const std::string& input_img_path,
        const std::string& output_mask_path,
        const Config& config,
        std::ofstream& log_file
    );

    // 配置结构体（放在函数声明后，避免嵌套类型提前引用）
    struct Config {
        EngineType engine_type = EngineType::TENSORRT;

        // ONNX Runtime配置
        struct Onnx {
            std::string input_name = "input";
            std::string output_name = "output";
            bool use_gpu = true;
            size_t inter_op_num_threads = 1;
            size_t intra_op_num_threads = 4;
        } onnx;  // 实例名，避免与外层结构体名冲突

        // TensorRT配置
        struct Tensorrt {
            std::string input_name = "input";
            std::string output_name = "output";
            int max_batch_size = 1;
            int max_workspace_size = 8ULL << 30;
            int fp16_mode = true;
            std::string engine_cache_path = "";
        } tensorrt;  // 实例名，避免冲突
    };
    // TensorRT 推理上下文缓存（线程安全）
    struct TensorRTContext {
        std::unique_ptr<nvinfer1::IExecutionContext> context;
        cudaStream_t stream;
        void* input_buffer = nullptr;   // 预分配的输入设备内存
        void* output_buffer = nullptr;  // 预分配的输出设备内存
        size_t max_input_size = 0;      // 最大输入内存（字节）
        size_t max_output_size = 0;     // 最大输出内存（字节）
    };
}