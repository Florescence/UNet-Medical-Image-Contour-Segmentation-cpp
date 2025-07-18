#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "raw2png.h"
#include "png_normalize.h"
#include "predict.h"
#include "mask2polygon.h"

namespace fs = std::filesystem;
using namespace std::chrono;

// 日志文件流（全局）
std::ofstream log_file;

// 自定义日志输出函数
void log(const std::string& message) {
    log_file << message << std::endl;
}

// 打印使用说明（新增归一化控制参数说明）
void print_usage(const std::string& program_name) {
    std::cerr << "Medical Image Segmentation Tool (ONNX Runtime)" << std::endl;
    std::cerr << "Usage: " << program_name << " <raw_file_path> <onnx_model_path> <output_directory> <width> <height> [--no-normalize]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --no-normalize   Skip image normalization step (use raw PNG directly for inference)" << std::endl;
    std::cerr << "Example:" << std::endl;
    std::cerr << "  With normalization: " << program_name << " ./test.raw ./unet.onnx ./output 512 512" << std::endl;
    std::cerr << "  Without normalization: " << program_name << " ./test.raw ./unet.onnx ./output 512 512 --no-normalize" << std::endl;
}

// 检查文件是否存在
bool file_exists(const std::string& path, const std::string& desc) {
    if (!fs::exists(path)) {
        std::cerr << "Error: " << desc << " not found - " << path << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Medical Image Segmentation Pipeline Started ===" << std::endl;
    auto program_start = high_resolution_clock::now();

    // 解析控制参数（是否跳过归一化）
    bool skip_normalize = false;
    int expected_argc = 6; // 基础参数数量
    if (argc == 7 && std::string(argv[6]) == "--no-normalize") {
        skip_normalize = true;
        expected_argc = 7;
    }

    // 验证参数数量
    if (argc != expected_argc) {
        std::cerr << "Invalid number of arguments! Expected " << expected_argc - 1 << ", got " << argc - 1 << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // 解析基础参数
    const std::string raw_path = argv[1];
    const std::string onnx_path = argv[2];
    const std::string output_dir = argv[3];
    const int raw_width = std::atoi(argv[4]);
    const int raw_height = std::atoi(argv[5]);

    // 创建输出目录和日志文件
    try {
        fs::create_directories(output_dir);
        std::string log_path = output_dir + "/segmentation_log.txt";
        log_file.open(log_path, std::ios::out | std::ios::trunc);
        if (!log_file.is_open()) {
            throw std::runtime_error("Failed to create log file: " + log_path);
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    // 输入验证
    if (!file_exists(raw_path, "RAW file")) {
        log_file.close();
        return 1;
    }
    if (!file_exists(onnx_path, "ONNX model")) {
        log_file.close();
        return 1;
    }

    try {
        // 创建输出子目录
        const std::string raw_png_dir = output_dir + "/1_raw_png";
        const std::string norm_png_dir = output_dir + "/2_normalized_png";
        const std::string pred_mask_dir = output_dir + "/3_pred_masks";
        const std::string polygon_dir = output_dir + "/4_polygons";
        fs::create_directories(raw_png_dir);
        if (!skip_normalize) fs::create_directories(norm_png_dir); // 仅在需要时创建归一化目录
        fs::create_directories(pred_mask_dir);
        fs::create_directories(polygon_dir);

        // 打印配置信息（包含归一化开关状态）
        log("=== Medical Image Segmentation Pipeline Started ===");
        log("\n[Configuration]");
        log("Input RAW: " + fs::path(raw_path).filename().string() + " (" + std::to_string(raw_width) + "x" + std::to_string(raw_height) + ")");
        log("ONNX Model: " + fs::path(onnx_path).filename().string());
        log("Output Dir: " + output_dir);
        log("Device: GPU (ONNX Runtime)");
        log("Normalization: " + std::string(skip_normalize ? "Disabled" : "Enabled"));

        // 初始化ONNX Runtime环境并加载模型
        auto model_load_start = high_resolution_clock::now();
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MedicalSegmentation");
        Predict::ONNXConfig onnx_config;
        onnx_config.use_gpu = true;
        onnx_config.input_name = "input";
        onnx_config.output_name = "output";

        Ort::Session* session = Predict::loadOnnxModel(onnx_path, env, onnx_config, log_file);
        if (!session) {
            throw std::runtime_error("Failed to load ONNX model");
        }
        auto model_load_end = high_resolution_clock::now();
        log("\nModel loaded in " + std::to_string(duration_cast<milliseconds>(model_load_end - model_load_start).count()) + " ms");

        // 核心处理流程（步骤1-3，不含polygon）
        auto core_start = high_resolution_clock::now();
        std::string input_to_infer; // 推理步骤的输入图像路径（可能是raw_png或norm_png）
        auto step1_time = 0ms, step2_time = 0ms, step3_time = 0ms;

        // 步骤1: RAW转PNG
        auto step1_start = high_resolution_clock::now();
        const std::string raw_png_path = raw_png_dir + "/test.png";
        Raw2Png::raw_to_png(raw_path, raw_png_path, raw_width, raw_height);
        auto step1_end = high_resolution_clock::now();
        step1_time = duration_cast<milliseconds>(step1_end - step1_start);
        log("Step 1/4: RAW to PNG - " + std::to_string(step1_time.count()) + " ms");
        input_to_infer = raw_png_path; // 默认使用raw_png作为推理输入（若不执行归一化）

        // 步骤2: PNG归一化（可选）
        if (!skip_normalize) {
            auto step2_start = high_resolution_clock::now();
            const std::string norm_png_path = norm_png_dir + "/test.png";
            const std::string size_json_path = norm_png_dir + "/original_sizes.json";
            PngNormalize::normalize_single_png(raw_png_path, norm_png_path, size_json_path);
            auto step2_end = high_resolution_clock::now();
            step2_time = duration_cast<milliseconds>(step2_end - step2_start);
            log("Step 2/4: Image Normalization - " + std::to_string(step2_time.count()) + " ms");
            input_to_infer = norm_png_path; // 若执行归一化，使用归一化后的图像作为推理输入
        } else {
            log("Step 2/4: Image Normalization - Skipped (--no-normalize specified)");
        }

        // 步骤3: 模型预测（使用步骤1或2的输出作为输入）
        auto step3_start = high_resolution_clock::now();
        const std::string pred_mask_path = pred_mask_dir + "/test.png";
        Predict::predict_single_image(session, input_to_infer, pred_mask_path, onnx_config, log_file);
        auto step3_end = high_resolution_clock::now();
        step3_time = duration_cast<milliseconds>(step3_end - step3_start);
        log("Step 3/4: Model Inference - " + std::to_string(step3_time.count()) + " ms");

        // 核心流程结束（步骤1-3）
        auto core_end = high_resolution_clock::now();
        const auto core_time = duration_cast<milliseconds>(core_end - core_start).count();

        // 步骤4: 生成轮廓（单独计时，不纳入核心流程）
        auto step4_start = high_resolution_clock::now();
        const std::string size_json_path = norm_png_dir + "/original_sizes.json"; // 即使跳过归一化，也可能需要尺寸信息（根据mask2polygon实现）
        Mask2Polygon::process_single_mask(pred_mask_path, polygon_dir, size_json_path, raw_png_path);
        auto step4_end = high_resolution_clock::now();
        auto step4_time = duration_cast<milliseconds>(step4_end - step4_start);
        log("Step 4/4: Polygon Generation - " + std::to_string(step4_time.count()) + " ms (not included in core time)");

        // 处理总结（核心流程仅包含步骤1-3）
        log("\n[Processing Summary]");
        log("Core processing time (steps 1-3): " + std::to_string(core_time) + " ms");
        log("Breakdown:");
        if (skip_normalize) {
            log("  Conversion: " + std::to_string(step1_time.count() * 100.0 / core_time) + "% | Inference: " + std::to_string(step3_time.count() * 100.0 / core_time) + "%");
        } else {
            log("  Conversion: " + std::to_string(step1_time.count() * 100.0 / core_time) + "% | Normalization: " + std::to_string(step2_time.count() * 100.0 / core_time) + "% | Inference: " + std::to_string(step3_time.count() * 100.0 / core_time) + "%");
        }
        log("Polygon generation time (step 4): " + std::to_string(step4_time.count()) + " ms");

        // 释放资源
        Predict::freeOnnxResources(session, nullptr);
        log("\nResources released. Processing completed.");

    } catch (const std::exception& e) {
        std::string error = "[ERROR] " + std::string(e.what());
        log(error);
        std::cerr << error << std::endl;
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