#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"

namespace PngDenormalize {
    nlohmann::json load_original_sizes(const std::string& json_path);
    cv::Mat denormalize_single_image(const cv::Mat& normalized_img, int orig_width, int orig_height);
    void save_denormalized_image(const cv::Mat& img, const std::string& output_path);
    void denormalize_single_png(const std::string& input_img_path,
                               const std::string& output_img_path,
                               const std::string& json_path);
}