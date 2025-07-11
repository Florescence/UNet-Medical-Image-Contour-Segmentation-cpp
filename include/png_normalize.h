#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"

namespace PngNormalize {
    cv::Mat read_and_convert_to_gray(const std::string& img_path);
    cv::Mat normalize_image(const cv::Mat& gray_img, int& original_width, int& original_height);
    void save_normalized_image(const cv::Mat& normalized_img, const std::string& output_path);
    void save_original_size(const std::string& json_path, const std::string& filename, 
                          int original_width, int original_height);
    void normalize_single_png(const std::string& input_path, 
                             const std::string& output_img_path,
                             const std::string& output_json_path);
}