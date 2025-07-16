#include "png_denormalize.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace PngDenormalize{

/**
 * 从JSON文件加载原始尺寸和缩放尺寸信息
 */
json load_original_sizes(const std::string& json_path) {
    std::ifstream f(json_path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Fail to Open JSON File: " + json_path);
    }
    json j;
    f >> j;
    return j;
}

/**
 * 单张图像反归一化处理
 * 步骤：直接将缩放后的图像（长边=512）缩放回原始尺寸
 */
cv::Mat denormalize_single_image(const cv::Mat& scaled_img, 
                                int orig_width, int orig_height,
                                int scaled_width, int scaled_height) {
    // 计算缩放比例（原始尺寸 / 缩放后尺寸）
    const double scale_x = static_cast<double>(orig_width) / scaled_width;
    const double scale_y = static_cast<double>(orig_height) / scaled_height;

    cv::Mat denormalized_img;
    // 根据缩放方向选择插值方式（缩小用AREA，放大用LINEAR）
    const int interpolation = (scaled_width < orig_width || scaled_height < orig_height)
        ? cv::INTER_AREA
        : cv::INTER_LINEAR;
    
    cv::resize(
        scaled_img, 
        denormalized_img, 
        cv::Size(orig_width, orig_height), 
        scale_x, scale_y,  // 显式指定缩放因子，提升精度
        interpolation
    );

    return denormalized_img;
}

/**
 * 保存反归一化后的图像
 */
void save_denormalized_image(const cv::Mat& img, const std::string& output_path) {
    fs::path output_fs(output_path);
    if (!fs::exists(output_fs.parent_path())) {
        fs::create_directories(output_fs.parent_path());
    }
    
    // 最低压缩等级，最快保存速度
    std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
    if (!cv::imwrite(output_path, img, params)) {
        throw std::runtime_error("Fail to Save Denormalized Image: " + output_path);
    }
}

/**
 * 单张图像反归一化主函数（适配无黑边逻辑）
 */
void denormalize_single_png(const std::string& input_img_path,
                           const std::string& output_img_path,
                           const std::string& json_path) {
    try {
        // 1. 读取输入图像（缩放后的图像，长边=512）
        cv::Mat scaled_img = cv::imread(input_img_path, cv::IMREAD_GRAYSCALE);
        if (scaled_img.empty()) {
            throw std::runtime_error("Fail to Read Input Image: " + input_img_path);
        }

        // 2. 加载原始尺寸和缩放尺寸信息
        const std::string filename = fs::path(input_img_path).filename().string();
        json sizes_json = load_original_sizes(json_path);
        
        // 检查JSON中是否包含必要信息
        if (!sizes_json.contains(filename)) {
            throw std::runtime_error("JSON file missing size info for: " + filename);
        }
        if (!sizes_json[filename].contains("original_width") || 
            !sizes_json[filename].contains("original_height") ||
            !sizes_json[filename].contains("scaled_width") ||
            !sizes_json[filename].contains("scaled_height")) {
            throw std::runtime_error("JSON file missing scaled/original size info for: " + filename);
        }

        // 解析尺寸参数
        int orig_width = sizes_json[filename]["original_width"];
        int orig_height = sizes_json[filename]["original_height"];
        int scaled_width = sizes_json[filename]["scaled_width"];
        int scaled_height = sizes_json[filename]["scaled_height"];

        // 验证输入图像尺寸与JSON记录一致
        if (scaled_img.cols != scaled_width || scaled_img.rows != scaled_height) {
            throw std::runtime_error(
                "Scaled image size mismatch: " + 
                std::to_string(scaled_img.cols) + "x" + std::to_string(scaled_img.rows) + 
                " (actual) vs " + 
                std::to_string(scaled_width) + "x" + std::to_string(scaled_height) + " (JSON)"
            );
        }

        std::cout << "Original Size: " << orig_width << "x" << orig_height << std::endl;
        std::cout << "Scaled Size: " << scaled_width << "x" << scaled_height << std::endl;

        // 3. 执行反归一化（直接缩放回原始尺寸，无黑边裁剪）
        cv::Mat denormalized_img = denormalize_single_image(
            scaled_img, orig_width, orig_height,
            scaled_width, scaled_height
        );

        // 4. 保存结果
        save_denormalized_image(denormalized_img, output_img_path);
        std::cout << "Denormalize Complete, Saved to: " << output_img_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Processing Failure: " << e.what() << std::endl;
    }
}
}