#include "png_normalize.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace PngNormalize{

// 常量定义（长边目标尺寸）
const int TARGET_LONG_EDGE = 512;  // 长边缩放到512

/**
 * 读取PNG图片并转换为8位灰度图
 */
cv::Mat read_and_convert_to_gray(const std::string& img_path) {
    cv::Mat img = cv::imread(img_path, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("Failed to Read Image: " + img_path);
    }
    // 16位图转8位（保持原有优化）
    if (img.depth() == CV_16U) {
        img.convertTo(img, CV_8U, 1.0 / 256.0);
    }
    return img;
}

/**
 * 等比例缩放图片（长边=512，短边按比例缩放）
 */
cv::Mat normalize_image(const cv::Mat& gray_img, int& original_width, int& original_height,
                       int& scaled_width, int& scaled_height) {  // 输出缩放后的尺寸
    // 获取原始尺寸
    original_width = gray_img.cols;
    original_height = gray_img.rows;

    // 计算缩放比例（长边缩放到TARGET_LONG_EDGE）
    const double scale = (original_width >= original_height) 
        ? static_cast<double>(TARGET_LONG_EDGE) / original_width  // 宽为长边
        : static_cast<double>(TARGET_LONG_EDGE) / original_height; // 高为长边

    // 计算缩放后的尺寸（短边按比例缩放，必然小于等于512）
    scaled_width = static_cast<int>(original_width * scale);
    scaled_height = static_cast<int>(original_height * scale);

    // 优化缩放（缩小用INTER_AREA，放大用INTER_LINEAR）
    cv::Mat resized;
    const int interpolation = (scaled_width < original_width || scaled_height < original_height)
        ? cv::INTER_AREA
        : cv::INTER_LINEAR;
    cv::resize(gray_img, resized, cv::Size(scaled_width, scaled_height), 0, 0, interpolation);

    return resized;  // 直接返回缩放后的图像
}

/**
 * 保存归一化后的图片
 */
void save_normalized_image(const cv::Mat& normalized_img, const std::string& output_path) {
    fs::path output_fs(output_path);
    if (!fs::exists(output_fs.parent_path())) {
        fs::create_directories(output_fs.parent_path());
    }
    
    // 最低压缩等级，最快保存速度
    std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
    if (!cv::imwrite(output_path, normalized_img, params)) {
        throw std::runtime_error("Fail to Save Normalized Image: " + output_path);
    }
}

/**
 * 保存原始尺寸和缩放尺寸到JSON（新增缩放后尺寸记录）
 */
void save_original_size(const std::string& json_path, const std::string& filename, 
                       int original_width, int original_height,
                       int scaled_width, int scaled_height) {  // 新增缩放后尺寸
    json j;
    j[filename] = {
        {"original_width", original_width},
        {"original_height", original_height},
        {"scaled_width", scaled_width},    // 缩放后宽度（<=512）
        {"scaled_height", scaled_height}   // 缩放后高度（<=512）
    };

    std::ofstream f(json_path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Fail to Open JSON File: " + json_path);
    }
    f << j << std::endl;
}

/**
 * 单张图片归一化主函数
 */
void normalize_single_png(const std::string& input_path, 
                         const std::string& output_img_path,
                         const std::string& output_json_path) {
    try {
        std::cout << "Start Processing: " << fs::path(input_path).filename() << std::endl;

        // 步骤1: 读取并转换为灰度图
        cv::Mat gray_img = read_and_convert_to_gray(input_path);

        // 步骤2: 等比例缩放（长边=512）
        int original_width, original_height;
        int scaled_width, scaled_height;  // 缩放后的尺寸
        cv::Mat normalized_img = normalize_image(
            gray_img, original_width, original_height,
            scaled_width, scaled_height  // 输出缩放后尺寸
        );

        // 步骤3: 保存归一化图片
        save_normalized_image(normalized_img, output_img_path);

        // 步骤4: 保存原始尺寸和缩放尺寸到JSON
        save_original_size(
            output_json_path, 
            fs::path(input_path).filename().string(),
            original_width, original_height,
            scaled_width, scaled_height  // 记录缩放后尺寸
        );

        std::cout << "Processing Complete: " << std::endl;
        std::cout << "  Normalized Image: " << output_img_path << std::endl;
        std::cout << "  Scale Record: " << output_json_path << std::endl;
        std::cout << "  Original Size: " << original_width << "x" << original_height << std::endl;
        std::cout << "  Scaled Size: " << scaled_width << "x" << scaled_height << " (long edge=512)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Processing Failure: " << e.what() << std::endl;
    }
}
}