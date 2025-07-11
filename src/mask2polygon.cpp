#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

// 常量定义
const std::string JSON_VERSION = "1.0.2.799"; // 与Falcon版本一致
const cv::Scalar CONTOUR_COLOR = cv::Scalar(0, 0, 255);  // 红色轮廓
const int CONTOUR_THICKNESS = 4;  // 轮廓线宽

/**
 * 从JSON文件加载原始尺寸信息
 */
json load_size_json(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error("Fail to Open JSON File: " + json_path);
    }
    json j;
    f >> j;
    return j;
}

/**
 * 查找原始PNG图像（模仿Python的_find_original_png逻辑）
 */
std::string find_original_png(const std::string& base_name, const std::string& output_dir) {
    // 候选路径列表（优先级从高到低）
    std::vector<std::string> candidates = {
        output_dir + "/" + base_name + ".png",
        fs::path(output_dir).parent_path().string() + "/" + base_name + ".png"
    };

    for (const auto& path : candidates) {
        if (fs::exists(path) && fs::path(path).extension() == ".png") {
            return path;
        }
    }
    return "";  // 未找到
}

/**
 * 提取掩码轮廓（对应Python的cv2.findContours）
 */
std::vector<std::vector<cv::Point>> extract_contours(const cv::Mat& mask) {
    cv::Mat binary_mask;
    cv::threshold(mask, binary_mask, 127, 255, cv::THRESH_BINARY);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

/**
 * 生成轮廓JSON文件（对应Python的JSON结构）
 */
void generate_json(const std::vector<std::vector<cv::Point>>& contours,
                  const std::string& json_path,
                  const std::string& base_name,
                  int original_width,
                  int original_height) {
    json j;
    j["version"] = JSON_VERSION;
    j["imagePath"] = base_name;
    j["imageWidth"] = original_width;
    j["imageHeight"] = original_height;
    j["imageData"] = nullptr;
    j["flags"] = json::object();
    j["shapes"] = json::array();

    // 添加每个轮廓的点集
    for (const auto& contour : contours) {
        json shape;
        shape["label"] = 1;
        shape["labelIndex"] = 0;
        shape["shape_type"] = "polygon";
        shape["description"] = "";
        shape["mask"] = nullptr;
        shape["group_id"] = nullptr;
        shape["flags"] = json::object();

        // 转换点坐标格式
        json points;
        for (const auto& pt : contour) {
            points.push_back({pt.x, pt.y});
        }
        shape["points"] = points;
        j["shapes"].push_back(shape);
    }

    // 保存JSON文件
    std::ofstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error("Fail to Create JSON File: " + json_path);
    }
    f << std::setw(4) << j << std::endl;
}

/**
 * 创建轮廓叠加的覆盖图
 */
void create_overlay_image(const std::vector<std::vector<cv::Point>>& contours,
                         const std::string& original_png_path,
                         const std::string& overlay_path) {
    cv::Mat original_img = cv::imread(original_png_path);
    if (original_img.empty()) {
        throw std::runtime_error("Fail to Read Initial Mask: " + original_png_path);
    }

    // 绘制红色轮廓
    cv::drawContours(original_img, contours, -1, CONTOUR_COLOR, CONTOUR_THICKNESS);

    // 保存覆盖图
    if (!cv::imwrite(overlay_path, original_img)) {
        throw std::runtime_error("Fail to Save Overlay PNG: " + overlay_path);
    }
}

/**
 * 单张掩码处理主函数
 */
void process_single_mask(const std::string& mask_path,
                        const std::string& output_dir,
                        const std::string& json_path) {
    try {
        // 1. 提取文件名（不含扩展名）
        std::string base_name = fs::path(mask_path).stem().string();
        std::cout << "Processing Mask: " << fs::path(mask_path).filename() << std::endl;

        // 2. 加载原始尺寸信息
        json sizes_json = load_size_json(json_path);
        std::string mask_filename = fs::path(mask_path).filename().string();
        if (!sizes_json.contains(mask_filename)) {
            throw std::runtime_error("Cannot Find Scale Info from JSON File " + mask_filename);
        }
        int original_width = sizes_json[mask_filename]["width"];
        int original_height = sizes_json[mask_filename]["height"];
        std::cout << "Initial Scale: " << original_width << "x" << original_height << std::endl;

        // 3. 读取掩码图像
        cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            throw std::runtime_error("Fail to Read Mask File: " + mask_path);
        }

        // 4. 提取轮廓
        std::vector<std::vector<cv::Point>> contours = extract_contours(mask);
        if (contours.empty()) {
            std::cout << "Warning: No Contours Detected" << std::endl;
            return;
        }
        std::cout << "Extract " << contours.size() << " Contours" << std::endl;

        // 5. 生成JSON文件
        std::string output_json_path = output_dir + "/" + base_name + ".json";
        generate_json(contours, output_json_path, base_name, original_width, original_height);
        std::cout << "JSON Saved to: " << output_json_path << std::endl;

        // 6. 创建覆盖图
        std::string original_png = find_original_png(base_name, output_dir);
        if (original_png.empty()) {
            std::cout << "Warning: Cannot Find Initial PNG, Skip Producing Overlay Image" << std::endl;
            return;
        }
        std::string overlay_path = output_dir + "/" + base_name + "_contour_overlay.png";
        create_overlay_image(contours, original_png, overlay_path);
        std::cout << "Overlay Image Saved to: " << overlay_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Processing Failure: " << e.what() << std::endl;
    }
}

// 测试主函数（硬编码参数）
int main() {
    // 硬编码测试参数（根据实际需求修改）
    const std::string mask_path = "output/test.png";    // 输入掩码路径
    const std::string output_dir = "output/test.png";            // 输出目录
    const std::string size_json_path = "output/original_sizes.json";  // 尺寸JSON路径

    // 创建输出目录
    fs::create_directories(output_dir);

    // 处理单张掩码
    process_single_mask(mask_path, output_dir, size_json_path);

    return 0;
}