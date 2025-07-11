#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

namespace Predict {
    torch::Tensor preprocess_image(const cv::Mat& gray_img);
    cv::Mat predict_mask(torch::jit::script::Module& model, const cv::Mat& gray_img, torch::Device device);
    cv::Mat mask_to_image(const cv::Mat& mask);
    void predict_single_image(const std::string& model_path,
                             const std::string& input_img_path,
                             const std::string& output_mask_path);
}