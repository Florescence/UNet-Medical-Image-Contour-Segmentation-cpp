#include "process.h"
#include "preprocess.h"
#include "mask2polygon.h"
#include "initialize.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <filesystem>
#include "postprocess.cpp"

namespace fs = std::filesystem;
namespace MedicalSeg {

// 耗时统计结构体，与GPU版本保持一致
struct TimeStats {
    int preprocess = 0;    // 预处理耗时(ms)
    int inference = 0;     // 推理耗时(ms)
    int postprocess = 0;   // 后处理耗时(ms)
    int io = 0;            // IO操作耗时(ms)
    int total = 0;         // 总耗时(ms)
};

// ONNX Runtime推理上下文
struct OrtContext {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> input_shape = FIXED_INPUT_SHAPE;
    std::vector<int64_t> output_shape = FIXED_OUTPUT_SHAPE;
    TimeStats stats;       // 当前上下文的耗时统计
};

// 线程局部ONNX上下文
static thread_local OrtContext g_ort_context;

// 初始化ONNX推理上下文
void initialize_ort_context() {
    // 无需从模型获取形状，直接使用预定义的固定形状
    g_ort_context.input_shape = FIXED_INPUT_SHAPE;
    g_ort_context.output_shape = FIXED_OUTPUT_SHAPE;

    // 验证固定形状的合法性
    for (auto dim : g_ort_context.input_shape) {
        if (dim <= 0) {
            throw std::runtime_error("Invalid fixed input shape (contains non-positive value)");
        }
    }
    for (auto dim : g_ort_context.output_shape) {
        if (dim <= 0) {
            throw std::runtime_error("Invalid fixed output shape (contains non-positive value)");
        }
    }
}

// 图像预处理（复用Preprocess::preprocess_raw的resize结果，仅做归一化）
std::vector<float> preprocess_image_onnx(const cv::Mat& gray_img, TimeStats& stats) {
    auto start = std::chrono::high_resolution_clock::now();

    const int type = gray_img.type();
    assert(type == CV_8UC1 && "Input must be 8-bit grayscale (from preprocess_raw)");

    // 注意：此处不再手动resize，因为Preprocess::preprocess_raw已将图像缩放到512x512
    // 仅做归一化（与预处理的minmax缩放配合，转为[0,1]范围）
    std::vector<float> input_data(512 * 512);  // 1*1*512*512 = 262144元素
    for (int i = 0; i < 512; ++i) {
        for (int j = 0; j < 512; ++j) {
            input_data[i * 512 + j] = static_cast<float>(gray_img.at<uchar>(i, j)) / 255.0f;
        }
    }

    // 记录预处理耗时（仅包含归一化，不包含resize）
    auto end = std::chrono::high_resolution_clock::now();
    stats.preprocess = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return input_data;
}

// 执行ONNX推理（使用固定形状，依赖preprocess_raw的resize结果）
cv::Mat execute_onnx_inference(const cv::Mat& gray_img, TimeStats& stats) {
    initialize_ort_context();  // 确保使用固定形状
    Ort::Session* session = get_onnx_session();
    const std::vector<std::string>& input_names = get_input_names();
    const std::vector<std::string>& output_names = get_output_names();

    // 1. 预处理输入（仅归一化，resize已由Preprocess::preprocess_raw完成）
    std::vector<float> input_data = preprocess_image_onnx(gray_img, stats);

    // 2. 创建输入张量（添加计时）
    auto tensor_start = std::chrono::high_resolution_clock::now();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault
    );
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),  // 262144元素（1*1*512*512）
        g_ort_context.input_shape.data(),
        g_ort_context.input_shape.size()
    );
    if (!input_tensor.IsTensor()) {
        throw std::runtime_error("Failed to create input tensor with fixed shape");
    }
    auto tensor_end = std::chrono::high_resolution_clock::now();
    int tensor_time = std::chrono::duration_cast<std::chrono::milliseconds>(tensor_end - tensor_start).count();

    // 3. 转换输入输出名称为C字符串
    std::vector<const char*> input_names_cstr;
    std::vector<const char*> output_names_cstr;
    for (const auto& name : input_names) {
        input_names_cstr.push_back(name.c_str());
    }
    for (const auto& name : output_names) {
        output_names_cstr.push_back(name.c_str());
    }

    // 4. 执行推理（核心计时）
    auto inference_start = std::chrono::high_resolution_clock::now();
    std::vector<Ort::Value> output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        input_names_cstr.data(),
        &input_tensor,
        1,
        output_names_cstr.data(),
        1
    );
    auto inference_end = std::chrono::high_resolution_clock::now();
    stats.inference = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start).count();

    // 5. 处理输出张量（添加计时）
    auto output_start = std::chrono::high_resolution_clock::now();
    if (output_tensors.empty()) {
        throw std::runtime_error("No output tensor generated");
    }
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_element_count = 1;
    for (auto dim : g_ort_context.output_shape) {
        output_element_count *= dim;  // 1*3*512*512 = 786432元素
    }
    auto output_end = std::chrono::high_resolution_clock::now();
    int output_time = std::chrono::duration_cast<std::chrono::milliseconds>(output_end - output_start).count();

    // 6. 后处理：argmax获取类别（3类，添加计时）
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    cv::Mat pred_mask(512, 512, CV_8UC1);
    for (int i = 0; i < 512; ++i) {
        for (int j = 0; j < 512; ++j) {
            int max_idx = 0;
            float max_val = -1e9;
            // 遍历3个类别
            for (int c = 0; c < 3; ++c) {
                float val = output_data[c * 512 * 512 + i * 512 + j];
                if (val > max_val) {
                    max_val = val;
                    max_idx = c;
                }
            }
            pred_mask.at<uchar>(i, j) = static_cast<uchar>(max_idx);
        }
    }
    auto postprocess_end = std::chrono::high_resolution_clock::now();
    stats.postprocess += std::chrono::duration_cast<std::chrono::milliseconds>(postprocess_end - postprocess_start).count();

    // 记录详细耗时
    auto& log_file = get_log_file();
    log_file << "  Tensor creation time: " << tensor_time << " ms" << std::endl;
    log_file << "  Output processing time: " << output_time << " ms" << std::endl;

    return pred_mask;
}

// 掩码可视化（保持不变）
cv::Mat mask_to_image(const cv::Mat& mask) {
    cv::Mat vis_mask(mask.size(), CV_8UC1);
    uchar lut[256] = {0};
    lut[1] = 128;
    lut[2] = 255;
    cv::LUT(mask, cv::Mat(1, 256, CV_8UC1, lut), vis_mask);
    return vis_mask;
}

// 处理单张图像（完全复用Preprocess::preprocess_raw的resize逻辑）
bool process_single_image(const std::string& raw_path, int width, int height, 
                         const std::string& output_dir) {
    try {
        auto& log_file = get_log_file();
        auto* session = get_onnx_session();
        
        if (!session) {
            throw std::runtime_error("ONNX Runtime not initialized");
        }
        
        log_file << "\n=== Processing Image: " << fs::path(raw_path).filename().string() << " ===" << std::endl;
        
        // 创建输出目录
        const std::string base_name = fs::path(raw_path).stem().string();
        
        // 初始化耗时统计
        TimeStats stats;
        auto total_start_time = std::chrono::high_resolution_clock::now();
        
        // 预处理RAW文件（核心：复用Preprocess::preprocess_raw，包含resize到512x512）
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        const std::string preprocessed_png_path = output_dir + "/" + base_name + "_normalized.png";
        const std::string size_json_path = output_dir + "/" + base_name + "_original_sizes.json";
        const std::string pred_mask_path = output_dir + "/" + base_name + "_mask.png";
        
        bool ok = Preprocess::preprocess_raw(raw_path, preprocessed_png_path, size_json_path, width, height);
        if (!ok) {
            throw std::runtime_error("Preprocessing failed");
        }
        // 记录Preprocess::preprocess_raw的总耗时（包含resize）
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        stats.preprocess = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start).count();
        
        // 读取预处理后的图像（已由preprocess_raw缩放到512x512）
        auto io_start = std::chrono::high_resolution_clock::now();
        cv::Mat gray_img = cv::imread(preprocessed_png_path, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
        if (gray_img.empty()) {
            throw std::runtime_error("Failed to read preprocessed image");
        }
        // 验证图像尺寸是否为512x512（确保preprocess_raw生效）
        if (gray_img.rows != 512 || gray_img.cols != 512) {
            throw std::runtime_error("Preprocessed image size is not 512x512 (got " + 
                                    std::to_string(gray_img.rows) + "x" + std::to_string(gray_img.cols) + ")");
        }
        auto io_end = std::chrono::high_resolution_clock::now();
        stats.io += std::chrono::duration_cast<std::chrono::milliseconds>(io_end - io_start).count();
        
        // 执行ONNX推理（使用preprocess_raw处理后的图像）
        cv::Mat pred_mask = execute_onnx_inference(gray_img, stats);
        
        // 应用后处理（添加计时）
        auto postprocess_start = std::chrono::high_resolution_clock::now();
        pred_mask = postprocess_mask(pred_mask);
        auto postprocess_end = std::chrono::high_resolution_clock::now();
        stats.postprocess += std::chrono::duration_cast<std::chrono::milliseconds>(postprocess_end - postprocess_start).count();

        // 保存掩码（添加IO计时）
        io_start = std::chrono::high_resolution_clock::now();
        cv::Mat vis_mask = mask_to_image(pred_mask);
        std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
        if (!cv::imwrite(pred_mask_path, vis_mask, params)) {
            throw std::runtime_error("Failed to save mask");
        }
        io_end = std::chrono::high_resolution_clock::now();
        stats.io += std::chrono::duration_cast<std::chrono::milliseconds>(io_end - io_start).count();
        
        // 生成轮廓（复用原逻辑，添加IO计时）
        io_start = std::chrono::high_resolution_clock::now();
        Mask2Polygon::process_single_mask(pred_mask_path, output_dir, size_json_path, preprocessed_png_path, base_name);
        io_end = std::chrono::high_resolution_clock::now();
        stats.io += std::chrono::duration_cast<std::chrono::milliseconds>(io_end - io_start).count();
        
        // 计算总处理时间
        auto total_end_time = std::chrono::high_resolution_clock::now();
        stats.total = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time).count();
        
        // 输出详细耗时统计（包含preprocess_raw的完整耗时）
        log_file << "  Preprocessing time (including resize): " << stats.preprocess << " ms" << std::endl;
        log_file << "  Inference time: " << stats.inference << " ms" << std::endl;
        log_file << "  Postprocessing time: " << stats.postprocess << " ms" << std::endl;
        log_file << "  IO time (read/write): " << stats.io << " ms" << std::endl;
        log_file << "  Total processing time: " << stats.total << " ms" << std::endl;
        log_file << "Processing completed for: " << base_name << std::endl;
        
        // 控制台输出总耗时
        std::cout << "Total processing time: " << stats.total << " ms" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Processing error: " << e.what() << std::endl;
        auto& log_file = get_log_file();
        log_file << "Processing error: " << e.what() << std::endl;
        return false;
    }
}

void cleanup_ort_context() {
    g_ort_context.input_shape.clear();
    g_ort_context.output_shape.clear();
}

}  // namespace MedicalSeg