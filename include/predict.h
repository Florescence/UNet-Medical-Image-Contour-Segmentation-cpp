#pragma once
#include <string>
#include <fstream>  // 添加日志文件流支持
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>  // 引入 ONNX Runtime C++ API
#include <NvInfer.h>       // TensorRT 核心头文件
#include <NvOnnxParser.h>  // ONNX 解析器
#include <cuda_runtime.h>  // CUDA 运行时

namespace Predict {
    struct TRTConfig {
        int min_width = 512;   // 固定为512
        int min_height = 512;  // 固定为512
        int opt_width = 512;   // 固定为512
        int opt_height = 512;  // 固定为512
        int max_width = 512;   // 固定为512
        int max_height = 512;  // 固定为512
        size_t workspace_size = 64 * 1024 * 1024;  // 128MB
        bool use_fp16 = true;
    };

    /**
     * TensorRT 日志类，支持输出到文件
     */
    class TRTLogger : public nvinfer1::ILogger {
    private:
        std::ofstream& log_file_;  // 日志文件流引用
    public:
        TRTLogger(std::ofstream& log_file);  // 构造函数
        void log(Severity severity, const char* msg) noexcept override;
    };

    // 函数声明（更新函数签名以支持日志文件）
    void freeTrtResources(nvinfer1::IExecutionContext* context, nvinfer1::ICudaEngine* engine, nvinfer1::IRuntime* runtime);
    nvinfer1::ICudaEngine* buildTrtEngine(
        const std::string& onnx_path,
        const std::string& engine_cache_path,
        TRTLogger& logger,
        const TRTConfig& trt_config
    );

    std::vector<float> preprocess_image(const cv::Mat& gray_img);
    cv::Mat predict_mask(nvinfer1::ICudaEngine* engine, const cv::Mat& gray_img, std::ofstream& log_file);  // 添加日志参数
    cv::Mat mask_to_image(const cv::Mat& mask);
    void predict_single_image(
        nvinfer1::ICudaEngine* engine,
        const std::string& input_img_path,
        const std::string& output_mask_path,
        std::ofstream& log_file  // 添加日志参数
    );
}