#include "png_normalize.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace PngNormalize{

// 常量定义（目标尺寸固定为512x512）
const int TARGET_SIZE = 512;

/**
 * 读取PNG图片并转换为8位灰度图
 */
cv::Mat read_and_convert_to_gray(const std::string& img_path) {
    // 优化：使用IMREAD_ANYDEPTH | IMREAD_GRAYSCALE确保兼容16位图，减少后续转换
    cv::Mat img = cv::imread(img_path, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("Failed to Read Image: " + img_path);
    }
    // 若为16位图直接缩放到8位（避免后续归一化重复操作）
    if (img.depth() == CV_16U) {
        img.convertTo(img, CV_8U, 1.0 / 256.0); // 16->8位快速转换（适用于0-65535范围）
    }
    return img;
}

/**
 * 等比例缩放图片并居中放置在512x512黑色画布上
 */
cv::Mat normalize_image(const cv::Mat& gray_img, int& original_width, int& original_height) {
    // 获取原始尺寸（减少重复调用cols/rows）
    original_width = gray_img.cols;
    original_height = gray_img.rows;

    // 计算缩放比例（合并条件判断，减少分支跳转）
    const double scale = (original_width >= original_height) 
        ? static_cast<double>(TARGET_SIZE) / original_width 
        : static_cast<double>(TARGET_SIZE) / original_height;
    const int new_width = static_cast<int>(original_width * scale);
    const int new_height = static_cast<int>(original_height * scale);

    // 优化：预分配内存并使用resize的输出参数形式，减少临时变量
    cv::Mat resized;
    resized.create(new_height, new_width, CV_8UC1);
    // 优化：缩小图用INTER_AREA（速度更快），放大图用INTER_LINEAR（平衡速度和质量）
    const int interpolation = (new_width < original_width || new_height < original_height)
        ? cv::INTER_AREA
        : cv::INTER_LINEAR;
    cv::resize(gray_img, resized, resized.size(), 0, 0, interpolation);

    // 优化：直接创建画布并计算ROI，避免冗余操作
    cv::Mat normalized(TARGET_SIZE, TARGET_SIZE, CV_8UC1, cv::Scalar(0));
    const cv::Rect roi(
        (TARGET_SIZE - new_width) / 2,
        (TARGET_SIZE - new_height) / 2,
        new_width,
        new_height
    );
    // 优化：使用copyTo的快速路径（确保类型和通道匹配）
    resized.copyTo(normalized(roi));

    return normalized;
}

/**
 * 保存归一化后的图片
 */
void save_normalized_image(const cv::Mat& normalized_img, const std::string& output_path) {
    // 优化：仅在目录不存在时创建（减少系统调用）
    fs::path output_fs(output_path);
    if (!fs::exists(output_fs.parent_path())) {
        fs::create_directories(output_fs.parent_path());
    }
    
    // 优化：降低PNG压缩等级（9→3→0），大幅提升保存速度（视觉质量差异可忽略）
    std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
    if (!cv::imwrite(output_path, normalized_img, params)) {
        throw std::runtime_error("Fail to Save Normalized Image: " + output_path);
    }
}

/**
 * 保存原始尺寸信息到JSON文件
 */
void save_original_size(const std::string& json_path, const std::string& filename, 
                       int original_width, int original_height) {
    // 优化：直接构造键值对，减少中间json对象操作
    json j;
    j[filename] = {{"width", original_width}, {"height", original_height}};

    // 优化：使用二进制模式打开文件，减少文本转换开销
    std::ofstream f(json_path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Fail to Open JSON File: " + json_path);
    }
    // 优化：取消格式化输出（去掉std::setw），提升JSON写入速度
    f << j << std::endl;
}

/**
 * 单张图片归一化主函数
 * @param input_path 输入PNG图片路径
 * @param output_img_path 输出归一化图片路径
 * @param output_json_path 输出尺寸JSON路径
 */
void normalize_single_png(const std::string& input_path, 
                         const std::string& output_img_path,
                         const std::string& output_json_path) {
    try {
        std::cout << "Start Processing: " << fs::path(input_path).filename() << std::endl;

        // 步骤1: 读取并转换为灰度图（优化了16位转8位流程）
        cv::Mat gray_img = read_and_convert_to_gray(input_path);

        // 步骤2: 归一化处理（优化了插值方式和内存操作）
        int original_width, original_height;
        cv::Mat normalized_img = normalize_image(gray_img, original_width, original_height);

        // 步骤3: 保存归一化图片（优化了压缩等级）
        save_normalized_image(normalized_img, output_img_path);

        // 步骤4: 保存原始尺寸到JSON（优化了JSON构造和写入）
        save_original_size(output_json_path, fs::path(input_path).filename().string(), 
                          original_width, original_height);

        std::cout << "Processing Complete: " << std::endl;
        std::cout << "  Normalized Image: " << output_img_path << std::endl;
        std::cout << "  Scale Record: " << output_json_path << std::endl;
        std::cout << "  Initial Scale: " << original_width << "x" << original_height << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Processing Failure: " << e.what() << std::endl;
    }
}
}
// 主函数（硬编码测试参数）
// int main() {
//     // 测试参数（根据实际需求修改）
//     const std::string input_path = "output/test.png"; // 输入单张PNG路径
//     const std::string output_img_path = "output/test.png"; // 输出归一化图片路径
//     const std::string output_json_path = "output/original_sizes.json"; // 输出尺寸JSON路径

//     // 执行归一化
//     normalize_single_png(input_path, output_img_path, output_json_path);

//     return 0;
// }