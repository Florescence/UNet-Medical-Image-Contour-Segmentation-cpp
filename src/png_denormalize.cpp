#include "png_denormalize.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace PngDenormalize{

// 常量定义
const int TARGET_SIZE = 512;  // 归一化时的目标尺寸（固定512x512）

/**
 * 从JSON文件加载原始尺寸信息（优化版）
 */
json load_original_sizes(const std::string& json_path) {
    // 以二进制模式打开，减少文本转换开销
    std::ifstream f(json_path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Fail to Open JSON File: " + json_path);
    }
    json j;
    f >> j;
    return j;
}

/**
 * 单张图像反归一化处理（优化版）
 * 步骤：裁剪黑边 -> 缩放回原始尺寸
 */
cv::Mat denormalize_single_image(const cv::Mat& normalized_img, 
                                int orig_width, int orig_height) {
    // 计算缩放比例和填充黑边位置（合并条件判断，减少分支）
    const bool width_is_longer = (orig_width >= orig_height);
    const double scale = width_is_longer 
        ? static_cast<double>(TARGET_SIZE) / orig_width 
        : static_cast<double>(TARGET_SIZE) / orig_height;

    const int new_width = width_is_longer ? TARGET_SIZE : static_cast<int>(orig_width * scale);
    const int new_height = width_is_longer ? static_cast<int>(orig_height * scale) : TARGET_SIZE;
    const int padding_x = width_is_longer ? 0 : (TARGET_SIZE - new_width) / 2;
    const int padding_y = width_is_longer ? (TARGET_SIZE - new_height) / 2 : 0;

    // 裁剪黑边（使用矩形直接定位，避免临时变量）
    cv::Mat cropped_img = normalized_img(cv::Rect(padding_x, padding_y, new_width, new_height));

    // 缩放回原始尺寸（根据缩放方向选择最优插值）
    cv::Mat denormalized_img;
    const int interpolation = (new_width < orig_width || new_height < orig_height)
        ? cv::INTER_AREA  // 缩小用AREA，保留细节
        : cv::INTER_LINEAR; // 放大用LINEAR，平衡速度和质量
    
    cv::resize(
        cropped_img, 
        denormalized_img, 
        cv::Size(orig_width, orig_height), 
        0, 0, 
        interpolation
    );

    return denormalized_img;
}

/**
 * 保存反归一化后的图像（优化版）
 */
void save_denormalized_image(const cv::Mat& img, const std::string& output_path) {
    // 仅在目录不存在时创建（减少系统调用）
    fs::path output_fs(output_path);
    if (!fs::exists(output_fs.parent_path())) {
        fs::create_directories(output_fs.parent_path());
    }
    
    // 降低PNG压缩等级（9→3→0级平衡速度和文件大小）
    std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
    if (!cv::imwrite(output_path, img, params)) {
        throw std::runtime_error("Fail to Save Denormalized Image: " + output_path);
    }
}

/**
 * 单张图像反归一化主函数（优化版）
 */
void denormalize_single_png(const std::string& input_img_path,
                           const std::string& output_img_path,
                           const std::string& json_path) {
    try {
        // 1. 读取输入图像（归一化后的512x512图像）
        cv::Mat normalized_img = cv::imread(input_img_path, cv::IMREAD_GRAYSCALE);
        if (normalized_img.empty()) {
            throw std::runtime_error("Fail to Read Input Image: " + input_img_path);
        }

        // 验证输入图像尺寸（快速检查，避免后续无效计算）
        if (normalized_img.rows != TARGET_SIZE || normalized_img.cols != TARGET_SIZE) {
            throw std::runtime_error("Input Image Scale is not 512x512 (current: " 
                + std::to_string(normalized_img.cols) + "x" + std::to_string(normalized_img.rows) + ")");
        }

        // 2. 加载原始尺寸信息（提前获取文件名）
        const std::string filename = fs::path(input_img_path).filename().string();
        json sizes_json = load_original_sizes(json_path);
        
        // 检查JSON中是否包含目标文件的尺寸信息
        if (!sizes_json.contains(filename)) {
            throw std::runtime_error("JSON file missing size info for: " + filename);
        }

        int orig_width = sizes_json[filename]["width"];
        int orig_height = sizes_json[filename]["height"];
        std::cout << "Original Size: " << orig_width << "x" << orig_height << std::endl;

        // 3. 执行反归一化（裁剪黑边+缩放）
        cv::Mat denormalized_img = denormalize_single_image(
            normalized_img, orig_width, orig_height
        );

        // 4. 保存结果（复用目录创建逻辑）
        save_denormalized_image(denormalized_img, output_img_path);
        std::cout << "Denormalize Complete, Saved to: " << output_img_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Processing Failure: " << e.what() << std::endl;
    }
}
}