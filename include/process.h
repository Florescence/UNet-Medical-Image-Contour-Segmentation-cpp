#ifndef ONNX_PROCESS_H
#define ONNX_PROCESS_H

#include <string>
#include <opencv2/opencv.hpp>

namespace MedicalSeg {
    const std::vector<int64_t> FIXED_INPUT_SHAPE = {1, 1, 512, 512};   // (batch, channel, height, width)
    const std::vector<int64_t> FIXED_OUTPUT_SHAPE = {1, 3, 512, 512};  // (batch, classes, height, width)
    // 处理单张图像
    bool process_single_image(const std::string& raw_path, int width, int height, 
                            const std::string& output_dir);

    void cleanup_ort_context();
}  // namespace MedicalSeg

#endif  // PROCESS_H