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
 * 图像预处理：转为Tensor并标准化（对应Python的BasicDataset.preprocess）
 */
torch::Tensor preprocess_image(const cv::Mat& gray_img) {
    // 转换为float类型
    cv::Mat float_img;
    gray_img.convertTo(float_img, CV_32F, 1.0 / 255.0, 0.0);
    float_img = float_img.clone();

    // std::cout << "\n===== float_img Convert Validation =====" << std::endl;
    // double float_min, float_max;
    // cv::minMaxLoc(float_img, &float_min, &float_max);
    // std::cout << "float_img Value Range: [" << float_min << ", " << float_max << "] (Should be same with gray_img)" << std::endl;

    // 转换为Tensor：[H, W] -> [1, C, H, W]（添加批次和通道维度）
    torch::Tensor tensor = torch::from_blob(
        float_img.data,
        {1, INPUT_CHANNELS, gray_img.rows, gray_img.cols},
        torch::kFloat32
    ).clone();

    return tensor;
}

/**
 * 模型预测：加载TorchScript模型并执行推理
 */
cv::Mat predict_mask(torch::jit::script::Module& model, const cv::Mat& gray_img, torch::Device device) {
    // 预处理图像对齐模型输入
    torch::Tensor input_tensor = preprocess_image(gray_img).to(device);

    // 验证张量有效性
    if (!input_tensor.defined()) {
        throw std::runtime_error("Input tensor is undefined (invalid data)");
    }
    if (input_tensor.numel() == 0) {
        throw std::runtime_error("Input tensor is empty (invalid size)");
    }
    if (torch::isnan(input_tensor).any().item<bool>()) {
        throw std::runtime_error("Input tensor contains NaN values");
    }

    // 模型推理
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    torch::NoGradGuard no_grad;
    torch::Tensor output = model.forward(inputs).toTensor();

    torch::Tensor pred_indices = output.argmax(1).squeeze(0).to(torch::kCPU).to(torch::kInt64);

    // 截断到[0,2]并转为uint8
    pred_indices = pred_indices.clamp(0, 2).to(torch::kUInt8);
    
    // 转换为cv::Mat
    cv::Mat pred_mask(gray_img.rows, gray_img.cols, CV_8UC1);
    std::memcpy(pred_mask.data, pred_indices.data_ptr(), pred_indices.numel() * sizeof(uint8_t));

    return pred_mask.clone();
}

/**
 * 掩码可视化：将前景转为白色，其他保持黑色
 */
cv::Mat mask_to_image(const cv::Mat& mask) {
    cv::Mat vis_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    vis_mask.setTo(0, mask == 0);
    vis_mask.setTo(128, mask == 1);
    vis_mask.setTo(255, mask == 2);  // 类别2是前景
    return vis_mask;
}

/**
 * 单张图像预测主函数
 */
void predict_single_image(
    const std::string& model_path,
    const std::string& input_img_path,
    const std::string& output_mask_path
) {
    try {
        std::cout << "Start Processing: " << input_img_path << std::endl;

        // 1. 初始化设备
        torch::Device device = torch::kCPU;
        std::cout << "Device: CPU" << std::endl;

        // 2. 加载模型
        torch::jit::script::Module model = torch::jit::load(model_path);
        model.to(device);
        model.eval();  // 切换到推理模式
        std::cout << "Model Loaded: " << model_path << std::endl;

        // 3. 读取输入图像并转为灰度图
        cv::Mat img = cv::imread(input_img_path);
        if (img.empty()) {
            throw std::runtime_error("Failed to Read Image: " + input_img_path);
        }
        cv::Mat gray_img;
        cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);  // 转为8位灰度图
        std::cout << "Input Image Scale: " << gray_img.cols << "x" << gray_img.rows << std::endl;

        // 4. 模型预测
        std::cout << "Prediction In Progress, Generating Raw Mask . . ." << std::endl;
        cv::Mat pred_mask = predict_mask(model, gray_img, device);
        
        // 5. 应用后处理
        pred_mask = postprocess_mask(pred_mask);
        std::cout << "Postprocess Applied" << std::endl;

        // 6. 转换为可视化掩码（0->0, 1->128, 2->255）
        cv::Mat vis_mask = mask_to_image(pred_mask);

        // 7. 保存输出掩码
        fs::create_directories(fs::path(output_mask_path).parent_path());
        cv::imwrite(output_mask_path, vis_mask);
        std::cout << "Mask Saved to: " << output_mask_path << std::endl;

    } catch (const c10::Error& e) {  // 捕获PyTorch的c10::Error异常
    std::cerr << "PyTorch Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {  // 保留原有std异常捕获
        std::cerr << "Standard Error: " << e.what() << std::endl;
    }
    catch (...) {  // 捕获所有其他未定义异常
        std::cerr << "Unknown Error: An unexpected error occurred." << std::endl;
    }
}
}
// // 测试主函数（硬编码参数）
// int main() {
//     // 硬编码测试参数（根据实际需求修改）
//     const std::string model_path = "../unet_model.pt";       // TorchScript模型路径
//     const std::string input_img_path = "output/test.png";       // 输入图像路径（需为PNG/JPG）
//     const std::string output_mask_path = "output/test_mask.png";  // 输出掩码路径

//     // 执行单张图像预测
//     predict_single_image(model_path, input_img_path, output_mask_path);

//     return 0;
// }