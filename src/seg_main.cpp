#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "raw2png.h"
#include "png_normalize.h"
#include "predict.h"
#include "mask2polygon.h"

namespace fs = std::filesystem;
using namespace std::chrono;

// 日志文件流（全局，用于所有日志输出）
std::ofstream log_file;

// 自定义日志输出函数（同时输出到控制台和日志文件）
void log(const std::string& message) {
    //std::cout << message << std::endl;
    log_file << message << std::endl;
}

// 打印使用说明
void print_usage(const std::string& program_name) {
    std::cerr << "Medical Image Segmentation Tool" << std::endl;
    std::cerr << "Usage: " << program_name << " <raw_file_path> <onnx_model_path> <output_directory> <width> <height>" << std::endl;
    std::cerr << "Example: " << program_name << " ./test.raw ./unet.onnx ./output 512 512" << std::endl;
}

// 检查文件是否存在
bool file_exists(const std::string& path, const std::string& desc) {
    if (!fs::exists(path)) {
        std::string error = "Error: " + desc + " not found - " + path;
        log(error);
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    // 验证参数
    if (argc != 6) {
        std::cerr << "Invalid number of arguments! Expected 5, got " << argc - 1 << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // 解析参数
    const std::string raw_path = argv[1];
    const std::string onnx_path = argv[2];
    const std::string output_dir = argv[3];
    const int raw_width = std::atoi(argv[4]);
    const int raw_height = std::atoi(argv[5]);

    // 创建输出目录和日志文件
    try {
        fs::create_directories(output_dir);
        std::string log_path = output_dir + "/segmentation_log.txt";
        log_file.open(log_path, std::ios::out | std::ios::trunc);  // 覆盖已有日志
        if (!log_file.is_open()) {
            throw std::runtime_error("Failed to create log file: " + log_path);
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    log("=== Medical Image Segmentation Pipeline Started ===");
    auto program_start = high_resolution_clock::now();

    // 输入验证
    if (!file_exists(raw_path, "RAW file")) return 1;
    if (!file_exists(onnx_path, "ONNX model")) return 1;

    try {
        // 创建输出子目录
        const std::string raw_png_dir = output_dir + "/1_raw_png";
        const std::string norm_png_dir = output_dir + "/2_normalized_png";
        const std::string pred_mask_dir = output_dir + "/3_pred_masks";
        const std::string polygon_dir = output_dir + "/4_polygons";
        fs::create_directories(raw_png_dir);
        fs::create_directories(norm_png_dir);
        fs::create_directories(pred_mask_dir);
        fs::create_directories(polygon_dir);

        // 打印配置信息
        log("\n[Configuration]");
        log("Input RAW: " + fs::path(raw_path).filename().string() + " (" + std::to_string(raw_width) + "x" + std::to_string(raw_height) + ")");
        log("Model: " + fs::path(onnx_path).filename().string());
        log("Output Dir: " + output_dir);
        log("Device: GPU (fixed)");

        // 加载TensorRT引擎
        auto model_load_start = high_resolution_clock::now();
        Predict::TRTLogger logger(log_file);  // 传递日志文件给TRTLogger
        const std::string engine_cache_path = output_dir + "/trt_engine_512x512.cache";
        
        // 固定TRT配置为512x512
        Predict::TRTConfig trt_config;
        trt_config.min_width = 512;
        trt_config.min_height = 512;
        trt_config.opt_width = 512;
        trt_config.opt_height = 512;
        trt_config.max_width = 512;
        trt_config.max_height = 512;
        trt_config.workspace_size = 64 * 1024 * 1024;  // 128MB工作空间
        trt_config.use_fp16 = true;

        nvinfer1::ICudaEngine* engine = Predict::buildTrtEngine(onnx_path, engine_cache_path, logger, trt_config);
        if (!engine) {
            throw std::runtime_error("Failed to create TensorRT engine");
        }
        auto model_load_end = high_resolution_clock::now();
        log("\nEngine loaded in " + std::to_string(duration_cast<milliseconds>(model_load_end - model_load_start).count()) + " ms");

        // 核心处理流程
        auto core_start = high_resolution_clock::now();

        // 步骤1: RAW转PNG
        auto step1_start = high_resolution_clock::now();
        const std::string raw_png_path = raw_png_dir + "/test.png";
        Raw2Png::raw_to_png(raw_path, raw_png_path, raw_width, raw_height);
        auto step1_end = high_resolution_clock::now();
        log("Step 1/4: RAW to PNG - " + std::to_string(duration_cast<milliseconds>(step1_end - step1_start).count()) + " ms");

        // 步骤2: PNG归一化
        auto step2_start = high_resolution_clock::now();
        const std::string norm_png_path = norm_png_dir + "/test.png";
        const std::string size_json_path = norm_png_dir + "/original_sizes.json";
        PngNormalize::normalize_single_png(raw_png_path, norm_png_path, size_json_path);
        auto step2_end = high_resolution_clock::now();
        log("Step 2/4: Image Normalization - " + std::to_string(duration_cast<milliseconds>(step2_end - step2_start).count()) + " ms");

        // 步骤3: 模型预测
        auto step3_start = high_resolution_clock::now();
        const std::string pred_mask_path = pred_mask_dir + "/test.png";
        
        // 执行推理
        Predict::predict_single_image(engine, norm_png_path, pred_mask_path, log_file);
        auto step3_end = high_resolution_clock::now();
        log("Step 3/4: Model Inference - " + std::to_string(duration_cast<milliseconds>(step3_end - step3_start).count()) + " ms");

        // 步骤4: 生成轮廓
        auto step4_start = high_resolution_clock::now();
        Mask2Polygon::process_single_mask(pred_mask_path, polygon_dir, size_json_path, raw_png_path);
        auto step4_end = high_resolution_clock::now();
        log("Step 4/4: Polygon Generation - " + std::to_string(duration_cast<milliseconds>(step4_end - step4_start).count()) + " ms");

        // 处理总结
        auto core_end = high_resolution_clock::now();
        const auto core_time = duration_cast<milliseconds>(core_end - core_start).count();
        const auto step1_time = duration_cast<milliseconds>(step1_end - step1_start).count();
        const auto step2_time = duration_cast<milliseconds>(step2_end - step2_start).count();
        const auto step3_time = duration_cast<milliseconds>(step3_end - step3_start).count();
        const auto step4_time = duration_cast<milliseconds>(step4_end - step4_start).count();

        log("\n[Processing Summary]");
        log("Total processing time: " + std::to_string(core_time) + " ms");
        log("Breakdown:");
        log("  Conversion: " + std::to_string(step1_time * 100.0 / core_time) + "% | Normalization: " + std::to_string(step2_time * 100.0 / core_time) + "%");
        log("  Inference: " + std::to_string(step3_time * 100.0 / core_time) + "% | Polygon: " + std::to_string(step4_time * 100.0 / core_time) + "%");

        // 释放资源
        Predict::freeTrtResources(nullptr, engine, nullptr);
        log("\nResources released. Processing completed.");

    } catch (const std::exception& e) {
        std::string error = "[ERROR] " + std::string(e.what());
        log(error);
        std::cerr << error << std::endl;  // 错误同时显示在控制台
        log_file.close();
        return 1;
    } catch (...) {
        std::string error = "[ERROR] Unknown exception occurred";
        log(error);
        std::cerr << error << std::endl;
        log_file.close();
        return 1;
    }

    // 总执行时间
    auto program_end = high_resolution_clock::now();
    log("\n=== Pipeline Completed ===");
    log("Total execution time: " + std::to_string(duration_cast<milliseconds>(program_end - program_start).count()) + " ms");
    log("Results saved to: " + output_dir);

    // 关闭日志文件
    log_file.close();
    return 0;
}