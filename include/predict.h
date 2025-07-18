#pragma once
#include <string>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace Predict {
    // ONNX Runtime配置参数（移除固定尺寸限制）
    struct ONNXConfig {
        std::string input_name = "input";  // ONNX模型输入节点名
        std::string output_name = "output";// ONNX模型输出节点名
        bool use_gpu = true;               // 是否使用GPU推理
        size_t inter_op_num_threads = 1;   // 线程数配置
        size_t intra_op_num_threads = 4;
    };

    // 释放ONNX Runtime资源
    void freeOnnxResources(Ort::Session* session = nullptr, Ort::Env* env = nullptr);

    // 加载ONNX模型
    Ort::Session* loadOnnxModel(
        const std::string& onnx_path,
        Ort::Env& env,
        const ONNXConfig& config,
        std::ofstream& log_file
    );

    // 图像预处理（支持任意尺寸）
    std::vector<float> preprocess_image(const cv::Mat& gray_img);  // 移除config依赖（尺寸由图像本身决定）

    // 模型推理（支持任意输入尺寸）
    cv::Mat predict_mask(Ort::Session* session, const cv::Mat& gray_img, const ONNXConfig& config);

    // 掩码可视化
    cv::Mat mask_to_image(const cv::Mat& mask);

    // 单张图像预测主函数
    void predict_single_image(
        Ort::Session* session,
        const std::string& input_img_path,
        const std::string& output_mask_path,
        const ONNXConfig& config,
        std::ofstream& log_file
    );
}