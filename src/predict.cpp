#include "predict.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <numeric>
#include "post_process.cpp"

namespace fs = std::filesystem;
namespace Predict {

// 释放ONNX Runtime资源
void freeOnnxResources(Ort::Session* session, Ort::Env* env) {
    if (session) delete session;
    if (env) delete env;
}

// 加载ONNX模型
Ort::Session* loadOnnxModel(
    const std::string& onnx_path,
    Ort::Env& env,
    const ONNXConfig& config,
    std::ofstream& log_file
) {
    try {
        log_file << "Loading ONNX model: " << onnx_path << std::endl;
        
        // 配置会话选项
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(config.intra_op_num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 启用GPU（若配置且支持）
        if (config.use_gpu) {
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
        std::wstring w_onnx_path = std::wstring(onnx_path.begin(), onnx_path.end());
        const ORTCHAR_T* ort_model_path = w_onnx_path.c_str();

        // 创建会话
        Ort::Session* session = new Ort::Session(env, ort_model_path, session_options);
        log_file << "ONNX model loaded successfully" << std::endl;

        return session;
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("Failed to load ONNX model: " + std::string(e.what()));
    }
}

// 图像预处理（支持任意尺寸，仅保留归一化）
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

// 模型推理（支持任意输入尺寸）
cv::Mat predict_mask(Ort::Session* session, const cv::Mat& gray_img, const ONNXConfig& config) {
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
        throw std::runtime_error("Inference failed: " + std::string(e.what()));
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
    Ort::Session* session,
    const std::string& input_img_path,
    const std::string& output_mask_path,
    const ONNXConfig& config,
    std::ofstream& log_file
) {
    try {
        std::string msg = "Start predicting: " + fs::path(input_img_path).filename().string();
        log_file << msg << std::endl;
        std::cout << msg << std::endl;

        // 读取图像（支持任意尺寸）
        cv::Mat gray_img = cv::imread(input_img_path, cv::IMREAD_GRAYSCALE);
        if (gray_img.empty()) {
            throw std::runtime_error("Failed to read image: " + input_img_path);
        }
        // 打印输入图像尺寸
        log_file << "Input image size: " << gray_img.cols << "x" << gray_img.rows << std::endl;

        // 推理（自动适应输入尺寸）
        cv::Mat pred_mask = predict_mask(session, gray_img, config);
        pred_mask = postprocess_mask(pred_mask);
        log_file << "Postprocess applied" << std::endl;
        std::cout << "Postprocess applied" << std::endl;

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