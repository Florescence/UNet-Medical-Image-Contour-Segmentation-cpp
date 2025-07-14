#include "raw2png.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;
namespace Raw2Png {

// 读取16位RAW图像数据
cv::Mat read_raw_image(const std::string& raw_path, int width, int height) {
    // 打开RAW文件
    std::ifstream file(raw_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Fail to Open .raw File: " + raw_path);
    }

    // 计算文件大小并验证
    const size_t expected_size = static_cast<size_t>(width * height * 2); // 16位/像素
    file.seekg(0, std::ios::end);
    const size_t actual_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (actual_size != expected_size) {
        std::cerr << "Warning: File Size Unmatch - Actual Size: " << actual_size 
                  << " Bytes, Expected: " << expected_size << " Bytes" << std::endl;
    }

    // 读取16位数据
    std::vector<uint16_t> raw_data(width * height);
    file.read(reinterpret_cast<char*>(raw_data.data()), expected_size);

    // 转换为OpenCV矩阵 (height x width, 16位灰度)
    return cv::Mat(height, width, CV_16UC1, raw_data.data()).clone();
}

// 直接将16位RAW转为8位灰度图（无窗宽窗位处理）
cv::Mat convert_to_8bit(const cv::Mat& raw_image) {
    cv::Mat img_8bit;
    // 直接归一化到0-255范围
    cv::normalize(raw_image, img_8bit, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    return img_8bit;
}

// 保存为PNG图片
bool save_png(const cv::Mat& image, const std::string& output_path) {
    // 创建输出目录（如果不存在）
    fs::create_directories(fs::path(output_path).parent_path());
    
    // 保存图片
    if (!cv::imwrite(output_path, image)) {
        throw std::runtime_error("Fail to Save PNG File: " + output_path);
    }
    return true;
}

// 单张RAW转PNG的核心函数
bool raw_to_png(const std::string& raw_path, const std::string& output_path, int width, int height) {
    try {
        std::cout << "Start Processing: " << fs::path(raw_path).filename() << std::endl;
        
        // 步骤1: 读取RAW图像
        cv::Mat raw_image = read_raw_image(raw_path, width, height);
        
        // 步骤2: 转为8位灰度
        cv::Mat img_8bit = convert_to_8bit(raw_image);
        
        // 步骤3: 保存为PNG
        save_png(img_8bit, output_path);
        
        std::cout << "Processing Complete: " << output_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Processing Failure: " << e.what() << std::endl;
        return false;
    }
}
}
// // 主函数（硬编码参数测试）
// int main() {
//     // 硬编码测试参数（根据实际需求修改）
//     const std::string raw_path = "test.raw";       // 输入RAW文件路径
//     const std::string output_path = "output/test.png"; // 输出PNG路径
//     const int width = 4267;                        // 图像宽度
//     const int height = 4267;                       // 图像高度

//     // 调用处理函数
//     raw_to_png(raw_path, output_path, width, height);
    
//     return 0;
// }