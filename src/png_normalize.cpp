#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

// 常量定义（目标尺寸固定为512x512）
const int TARGET_SIZE = 512;

/**
 * 读取PNG图片并转换为8位灰度图
 */
cv::Mat read_and_convert_to_gray(const std::string& img_path) {
    // 读取图片（支持任意通道，自动转换为灰度）
    cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("Failed to Read Image: " + img_path);
    }
    return img;
}

/**
 * 等比例缩放图片并居中放置在512x512黑色画布上
 */
cv::Mat normalize_image(const cv::Mat& gray_img, int& original_width, int& original_height) {
    // 获取原始尺寸
    original_width = gray_img.cols;
    original_height = gray_img.rows;

    // 计算缩放比例
    double scale;
    int new_width, new_height;

    if (original_width >= original_height) {
        // 宽为长边
        scale = static_cast<double>(TARGET_SIZE) / original_width;
        new_width = TARGET_SIZE;
        new_height = static_cast<int>(original_height * scale);
    } else {
        // 高为长边
        scale = static_cast<double>(TARGET_SIZE) / original_height;
        new_height = TARGET_SIZE;
        new_width = static_cast<int>(original_width * scale);
    }

    // 高质量缩放（LANCZOS插值对应OpenCV的INTER_LANCZOS4）
    cv::Mat resized;
    cv::resize(gray_img, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LANCZOS4);

    // 创建黑色背景画布
    cv::Mat normalized(TARGET_SIZE, TARGET_SIZE, CV_8UC1, cv::Scalar(0));

    // 计算居中粘贴位置
    int paste_x = (TARGET_SIZE - new_width) / 2;
    int paste_y = (TARGET_SIZE - new_height) / 2;

    // 粘贴缩放后的图像
    cv::Rect roi(paste_x, paste_y, new_width, new_height);
    resized.copyTo(normalized(roi));

    return normalized;
}

/**
 * 保存归一化后的图片
 */
void save_normalized_image(const cv::Mat& normalized_img, const std::string& output_path) {
    // 创建输出目录
    fs::create_directories(fs::path(output_path).parent_path());

    // 保存图片（PNG格式，最高质量）
    std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 9};  // 最高压缩比
    if (!cv::imwrite(output_path, normalized_img, params)) {
        throw std::runtime_error("Fail to Save Normalized Image: " + output_path);
    }
}

/**
 * 保存原始尺寸信息到JSON文件
 */
void save_original_size(const std::string& json_path, const std::string& filename, 
                       int original_width, int original_height) {
    // 构建JSON数据
    json j;
    j[filename]["width"] = original_width;
    j[filename]["height"] = original_height;

    // 保存到文件
    std::ofstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error("Fail to Open JSON File: " + json_path);
    }
    f << std::setw(4) << j << std::endl;  // 格式化输出，便于阅读
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

        // 步骤1: 读取并转换为灰度图
        cv::Mat gray_img = read_and_convert_to_gray(input_path);

        // 步骤2: 归一化处理（获取原始尺寸）
        int original_width, original_height;
        cv::Mat normalized_img = normalize_image(gray_img, original_width, original_height);

        // 步骤3: 保存归一化图片
        save_normalized_image(normalized_img, output_img_path);

        // 步骤4: 保存原始尺寸到JSON
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

// 主函数（硬编码测试参数）
int main() {
    // 测试参数（根据实际需求修改）
    const std::string input_path = "output/test.png"; // 输入单张PNG路径
    const std::string output_img_path = "output/test.png"; // 输出归一化图片路径
    const std::string output_json_path = "output/original_sizes.json"; // 输出尺寸JSON路径

    // 执行归一化
    normalize_single_png(input_path, output_img_path, output_json_path);

    return 0;
}