#include "png_denormalize.h"
#include <iostream>
#include <fstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

// 常量定义（与Python版本保持一致）
const int TARGET_SIZE = 512;  // 归一化时的目标尺寸（固定512x512）

/**
 * 从JSON文件加载原始尺寸信息
 */
json load_original_sizes(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error("Fail to Open JSON File: " + json_path);
    }
    json j;
    f >> j;
    return j;
}

/**
 * 单张图像反归一化处理
 * 步骤：裁剪黑边 -> 缩放回原始尺寸
 */
cv::Mat denormalize_single_image(const cv::Mat& normalized_img, 
                                int orig_width, int orig_height) {
    // 计算归一化时的缩放比例和填充黑边的位置
    int padding_x = 0, padding_y = 0;
    int new_width = 0, new_height = 0;

    if (orig_width >= orig_height) {
        // 宽为长边时的填充计算
        double scale = static_cast<double>(TARGET_SIZE) / orig_width;
        new_width = TARGET_SIZE;
        new_height = static_cast<int>(orig_height * scale);
        padding_y = (TARGET_SIZE - new_height) / 2;
    } else {
        // 高为长边时的填充计算
        double scale = static_cast<double>(TARGET_SIZE) / orig_height;
        new_height = TARGET_SIZE;
        new_width = static_cast<int>(orig_width * scale);
        padding_x = (TARGET_SIZE - new_width) / 2;
    }

    // 裁剪黑边（去除归一化时添加的填充）
    cv::Rect crop_region(
        padding_x, 
        padding_y, 
        new_width, 
        new_height
    );
    cv::Mat cropped_img = normalized_img(crop_region);

    // 缩放回原始尺寸（使用LANCZOS插值，对应Python的Image.LANCZOS）
    cv::Mat denormalized_img;
    cv::resize(
        cropped_img, 
        denormalized_img, 
        cv::Size(orig_width, orig_height), 
        0, 0, 
        cv::INTER_LANCZOS4  // 高质量插值
    );

    return denormalized_img;
}

/**
 * 保存反归一化后的图像
 */
void save_denormalized_image(const cv::Mat& img, const std::string& output_path) {
    // 创建输出目录（如果不存在）
    fs::create_directories(fs::path(output_path).parent_path());
    
    // 保存为PNG（最高质量）
    std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 9};
    if (!cv::imwrite(output_path, img, params)) {
        throw std::runtime_error("Fail to Save Denormalized Image: " + output_path);
    }
}

/**
 * 单张图像反归一化主函数
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
        // 验证输入图像尺寸是否为512x512
        if (normalized_img.rows != TARGET_SIZE || normalized_img.cols != TARGET_SIZE) {
            throw std::runtime_error("The Input Image Scale is not 512x512, Fail to Match Normalize Format");
        }

        // 2. 加载原始尺寸信息
        json sizes_json = load_original_sizes(json_path);
        std::string filename = fs::path(input_img_path).filename().string();
        if (!sizes_json.contains(filename)) {
            throw std::runtime_error("Cannot Find Initial Scale Info from JSON File " + filename);
        }
        int orig_width = sizes_json[filename]["width"];
        int orig_height = sizes_json[filename]["height"];
        std::cout << "Initial Scale: " << orig_width << "x" << orig_height << std::endl;

        // 3. 执行反归一化（裁剪黑边+缩放）
        cv::Mat denormalized_img = denormalize_single_image(
            normalized_img, orig_width, orig_height
        );

        // 4. 保存结果
        save_denormalized_image(denormalized_img, output_img_path);
        std::cout << "Denormalize Complete, Image Saved to: " << output_img_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Processing Failure: " << e.what() << std::endl;
    }
}

// 测试主函数（硬编码参数）
// int main() {
//     // 硬编码测试参数（根据实际需求修改）
//     const std::string input_img_path = "output/test.png";    // 输入归一化图像（512x512）
//     const std::string output_img_path = "output/test.png"; // 输出反归一化图像
//     const std::string json_path = "output/original_sizes.json";    // 原始尺寸JSON路径

//     // 执行单张图像反归一化
//     denormalize_single_png(input_img_path, output_img_path, json_path);

//     return 0;
// }