#include "mask2polygon.h"
#include <iostream>
#include <fstream>

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace Mask2Polygon{
// 常量定义
const std::string JSON_VERSION = "1.0.2.812";
const cv::Scalar CONTOUR_COLOR = cv::Scalar(0, 0, 255);  // 红色轮廓
const int CONTOUR_THICKNESS = 1;  // 轮廓线宽

/**
 * 从JSON文件加载原始尺寸和缩放尺寸信息
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
 * 提取掩码轮廓
 */
std::vector<std::vector<cv::Point>> extract_contours(const cv::Mat& mask) {
    cv::Mat binary_mask;
    cv::threshold(mask, binary_mask, 127, 255, cv::THRESH_BINARY);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

/**
 * 将轮廓点坐标从缩放图映射到原图
 */
std::vector<std::vector<cv::Point>> map_contour_points(
    const std::vector<std::vector<cv::Point>>& contours,
    double scale_x, double scale_y) {
    
    std::vector<std::vector<cv::Point>> mapped_contours;
    mapped_contours.reserve(contours.size());
    
    for (const auto& contour : contours) {
        std::vector<cv::Point> mapped_contour;
        mapped_contour.reserve(contour.size());
        
        // 对每个点进行坐标映射（乘以缩放因子）
        for (const auto& pt : contour) {
            int x = static_cast<int>(pt.x * scale_x);
            int y = static_cast<int>(pt.y * scale_y);
            mapped_contour.push_back(cv::Point(x, y));
        }
        
        mapped_contours.push_back(mapped_contour);
    }
    
    return mapped_contours;
}

/**
 * 生成轮廓JSON文件（对应Falcon的JSON结构）
 */
void generate_json(const std::vector<std::vector<cv::Point>>& contours,
                  const std::string& json_path,
                  const std::string& base_name,
                  int original_width,
                  int original_height) {
    // 保持原有逻辑不变
    json j;
    j["version"] = "1.0.2.812";
    j["imagePath"] = base_name + ".raw";
    j["imageData"] = nullptr;
    j["flags"] = json::object();
    j["shapes"] = json::array();

    for (const auto& contour : contours) {
        json shape;
        shape["label"] = 1;
        shape["labelIndex"] = 0;

        json points;
        for (const auto& pt : contour) {
            points.push_back({pt.x, pt.y});
        }
        shape["points"] = points;

        shape["shape_type"] = "polygon";
        shape["description"] = "";
        shape["mask"] = nullptr;
        shape["group_id"] = nullptr;
        shape["flags"] = json::object();
        
        j["shapes"].push_back(shape);
    }
    
    j["imageWidth"] = original_width;
    j["imageHeight"] = original_height;

    std::ofstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error("Fail to Create JSON File: " + json_path);
    }
    f << std::setw(4) << j << std::endl;
}

/**
 * 创建轮廓叠加的覆盖图（直接使用映射后的坐标）
 */
void create_overlay_image(const std::vector<std::vector<cv::Point>>& contours,
                         const std::string& original_png_path,
                         const std::string& overlay_path) {
    cv::Mat original_img = cv::imread(original_png_path);
    if (original_img.empty()) {
        throw std::runtime_error("Fail to Read Original Image: " + original_png_path);
    }

    // 直接在原图上绘制映射后的轮廓
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
                        const std::string& json_path,
                        const std::string& original_png,
                        const std::string& base_name) {
    try {
        // 1. 提取文件名
        std::cout << "Processing Mask: " << base_name + ".png" << std::endl;

        // 2. 加载尺寸信息（新增读取scaled_width/scaled_height）
        json sizes_json = load_size_json(json_path);
        std::string mask_filename;
        if (sizes_json.contains(base_name + ".raw")) {
            mask_filename = base_name + ".raw";
        }
        else if (sizes_json.contains(base_name + ".tif")){
            mask_filename = base_name + ".tif";
        } 
        else {
            throw std::runtime_error("Cannot Find Size Info in JSON: " + base_name + ".raw/.tif");
        }
        

        int original_width = sizes_json[mask_filename]["original_width"];
        int original_height = sizes_json[mask_filename]["original_height"];
        int scaled_width = sizes_json[mask_filename]["scaled_width"];
        int scaled_height = sizes_json[mask_filename]["scaled_height"];
        
        std::cout << "Original Size: " << original_width << "x" << original_height << std::endl;
        std::cout << "Scaled Size: " << scaled_width << "x" << scaled_height << std::endl;

        // 3. 读取掩码图像
        cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            throw std::runtime_error("Fail to Read Mask File: " + mask_path);
        }

        // 4. 验证掩码尺寸与JSON记录一致
        if (mask.cols != scaled_width || mask.rows != scaled_height) {
            throw std::runtime_error(
                "Mask size mismatch: " + 
                std::to_string(mask.cols) + "x" + std::to_string(mask.rows) + 
                " (actual) vs " + 
                std::to_string(scaled_width) + "x" + std::to_string(scaled_height) + " (JSON)"
            );
        }

        // 5. 提取轮廓
        std::vector<std::vector<cv::Point>> contours = extract_contours(mask);
        if (contours.empty()) {
            std::cout << "Warning: No Contours Detected" << std::endl;
            return;
        }
        std::cout << "Extracted " << contours.size() << " Contours" << std::endl;

        // 6. 创建覆盖图（使用映射前的坐标）
        if (!original_png.empty()) {
            std::string overlay_path = output_dir + "/" + base_name + "_contour_overlay.png";
            create_overlay_image(contours, original_png, overlay_path);
            std::cout << "Overlay Image Saved to: " << overlay_path << std::endl;
        } else {
            std::cout << "Warning: Original PNG not provided, skipping overlay generation" << std::endl;
        }

        // 7. 计算坐标缩放因子并映射轮廓点
        double scale_x = static_cast<double>(original_width) / scaled_width;
        double scale_y = static_cast<double>(original_height) / scaled_height;
        
        std::vector<std::vector<cv::Point>> mapped_contours = 
            map_contour_points(contours, scale_x, scale_y);

        // 8. 生成JSON文件（使用映射后的坐标）
        std::string output_json_path = output_dir + "/" + base_name + ".json";
        generate_json(mapped_contours, output_json_path, base_name, original_width, original_height);
        std::cout << "JSON Saved to: " << output_json_path << std::endl;

        // // 8. 创建覆盖图（直接使用映射后的坐标）
        // if (!original_png.empty()) {
        //     std::string overlay_path = output_dir + "/" + base_name + "_contour_overlay.png";
        //     create_overlay_image(mapped_contours, original_png, overlay_path);
        //     std::cout << "Overlay Image Saved to: " << overlay_path << std::endl;
        // } else {
        //     std::cout << "Warning: Original PNG not provided, skipping overlay generation" << std::endl;
        // }

    } catch (const std::exception& e) {
        std::cerr << "Processing Failure: " << e.what() << std::endl;
    }
}
}