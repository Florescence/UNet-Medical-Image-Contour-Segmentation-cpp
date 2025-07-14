#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"

namespace Mask2Polygon {
    nlohmann::json load_size_json(const std::string& json_path);
    std::vector<std::vector<cv::Point>> extract_contours(const cv::Mat& mask);
    void generate_json(const std::vector<std::vector<cv::Point>>& contours,
                     const std::string& json_path,
                     const std::string& base_name,
                     int original_width,
                     int original_height);
    void create_overlay_image(const std::vector<std::vector<cv::Point>>& contours,
                            const std::string& original_png_path,
                            const std::string& overlay_path);
    void process_single_mask(const std::string& mask_path,
                            const std::string& output_dir,
                            const std::string& json_path,
                            const std::string& original_png);
}