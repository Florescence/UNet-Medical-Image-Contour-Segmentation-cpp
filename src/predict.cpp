#include "predict.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <c10/util/Exception.h>
#include "post_process.cpp"  // 包含后处理函数

namespace fs = std::filesystem;
namespace Predict{

// 常量定义（与Python版本保持一致）
const int INPUT_CHANNELS = 1;    // 输入图像通道数（灰度图）
const int NUM_CLASSES = 3;       // 多分类类别数（0/1/2）

/**
 * 图像预处理：转为Tensor并标准化（优化版本）
 * 假设输入已为8位灰度图（由第二步归一化保证）
 */
torch::Tensor preprocess_image(const cv::Mat& gray_img) {
    // 直接创建Tensor，避免额外内存拷贝（利用OpenCV和PyTorch内存连续性）
    torch::Tensor tensor = torch::from_blob(
        gray_img.data,
        {1, INPUT_CHANNELS, gray_img.rows, gray_img.cols},
        torch::kUInt8
    ).clone();  // 克隆数据以分离生命周期（确保Tensor不依赖原始cv::Mat）

    // 直接在Tensor上执行归一化（替代OpenCV的convertTo）
    tensor = tensor.to(torch::kFloat32).div(255.0);
    
    return tensor;
}

/**
 * 模型预测：使用预加载的模型执行推理（模型由外部传入）
 */
cv::Mat predict_mask(torch::jit::script::Module& model, const cv::Mat& gray_img, torch::Device device) {
    // 预处理图像对齐模型输入
    torch::Tensor input_tensor = preprocess_image(gray_img).to(device);

    // 模型推理（移除冗余的张量验证，假设输入有效）
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    torch::NoGradGuard no_grad;
    torch::Tensor output = model.forward(inputs).toTensor();

    // 获取预测结果并转换为掩码
    torch::Tensor pred_indices = output.argmax(1).squeeze(0).to(torch::kCPU).to(torch::kUInt8);
    
    // 直接创建cv::Mat，避免内存拷贝（利用连续内存布局）
    cv::Mat pred_mask(gray_img.rows, gray_img.cols, CV_8UC1);
    std::memcpy(pred_mask.data, pred_indices.data_ptr(), pred_indices.numel() * sizeof(uint8_t));

    return pred_mask;  // 移除冗余的clone
}

/**
 * 掩码可视化：将前景转为白色，其他保持黑色（优化版本）
 */
cv::Mat mask_to_image(const cv::Mat& mask) {
    // 预分配结果矩阵（与输入相同尺寸和类型）
    cv::Mat vis_mask(mask.size(), CV_8UC1);
    
    // 使用LUT（查找表）加速颜色映射（比setTo更高效）
    uchar lut[256] = {0};  // 初始化所有值为0
    lut[1] = 128;          // 类别1映射为128
    lut[2] = 255;          // 类别2映射为255
    
    cv::LUT(mask, cv::Mat(1, 256, CV_8UC1, lut), vis_mask);
    
    return vis_mask;
}

/**
 * 单张图像预测主函数（接收预加载的模型）
 */
void predict_single_image(
    torch::jit::script::Module& model,  // 传入预加载的模型
    torch::Device& device,              // 传入设备（与模型一致）
    const std::string& input_img_path,
    const std::string& output_mask_path
) {
    try {
        std::cout << "Start Predicting: " << fs::path(input_img_path).filename() << std::endl;

        // 读取输入图像（假设为8位灰度图，跳过颜色转换）
        cv::Mat gray_img = cv::imread(input_img_path, cv::IMREAD_GRAYSCALE);
        if (gray_img.empty()) {
            throw std::runtime_error("Failed to Read Image: " + input_img_path);
        }
        std::cout << "Input Image Scale: " << gray_img.cols << "x" << gray_img.rows << std::endl;

        // 模型预测（使用传入的模型和设备）
        std::cout << "Prediction In Progress, Generating Raw Mask . . ." << std::endl;
        cv::Mat pred_mask = predict_mask(model, gray_img, device);
        
        // 应用后处理
        pred_mask = postprocess_mask(pred_mask);
        std::cout << "Postprocess Applied" << std::endl;

        // 转换为可视化掩码
        cv::Mat vis_mask = mask_to_image(pred_mask);

        // 保存输出掩码（优化：仅在必要时创建目录）
        const fs::path output_path(output_mask_path);
        if (!fs::exists(output_path.parent_path())) {
            fs::create_directories(output_path.parent_path());
        }
        
        // 设置PNG压缩参数（平衡速度与文件大小）
        std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
        if (!cv::imwrite(output_mask_path, vis_mask, params)) {
            throw std::runtime_error("Failed to Save Mask: " + output_mask_path);
        }
        
        std::cout << "Mask Saved to: " << output_mask_path << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "PyTorch Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Error: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown Error: An unexpected error occurred." << std::endl;
    }
}
}