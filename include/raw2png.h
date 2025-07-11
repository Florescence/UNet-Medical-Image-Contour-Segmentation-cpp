#pragma once
#include <string>
#include <opencv2/opencv.hpp>

namespace Raw2Png {
    cv::Mat read_raw_image(const std::string& raw_path, int width, int height);
    cv::Mat convert_to_8bit(const cv::Mat& raw_image);
    bool save_png(const cv::Mat& image, const std::string& output_path);
    bool raw_to_png(const std::string& raw_path, const std::string& output_path, int width, int height);
}