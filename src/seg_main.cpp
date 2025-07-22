#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cstdlib>
#include <opencv2/opencv.hpp>
// #include "raw2png.h"
// #include "png_normalize.h"
#include "preprocess.h"
#include "predict.h"  // 适配新的predict.h（含TensorRT支持）
#include "mask2polygon.h"

namespace fs = std::filesystem;
using namespace std::chrono;

// 日志文件流（全局）
std::ofstream log_file;

// 自定义日志输出函数
void log(const std::string& message) {
    log_file << message << std::endl;
    std::cout << message << std::endl;  // 同时输出到控制台
}

// 打印使用说明（新增引擎选择参数）
void print_usage(const std::string& program_name) {
    std::cerr << "Medical Image Segmentation Tool (ONNX/TensorRT)" << std::endl;
    std::cerr << "Usage: " << program_name << " <raw_file_path> <model_path> <output_directory> <width> <height> [--no-normalize] [--engine <onnx|tensorrt>]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --no-normalize   Skip image normalization step" << std::endl;
    std::cerr << "  --engine         Select inference engine (default: onnx)" << std::endl;
    std::cerr << "Example:" << std::endl;
    std::cerr << "  ONNX Runtime: " << program_name << " ./test.raw ./model.onnx ./output 512 512" << std::endl;
    std::cerr << "  TensorRT: " << program_name << " ./test.raw ./model.onnx ./output 512 512 --engine tensorrt" << std::endl;
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

    // 解析控制参数（默认值）
    bool skip_normalize = false;
    Predict::EngineType engine_type = Predict::EngineType::ONNX_RUNTIME;  // 默认ONNX Runtime
    std::string engine_str = "onnx";

    // 解析命令行参数
    int arg_idx = 1;
    if (argc < 6) {
        print_usage(argv[0]);
        return 1;
    }

    // 基础参数
    const std::string raw_path = argv[arg_idx++];
    const std::string model_path = argv[arg_idx++];
    const std::string output_dir = argv[arg_idx++];
    const int raw_width = std::atoi(argv[arg_idx++]);
    const int raw_height = std::atoi(argv[arg_idx++]);

    // 解析可选参数
    for (; arg_idx < argc; arg_idx++) {
        std::string arg = argv[arg_idx];
        if (arg == "--no-normalize") {
            skip_normalize = true;
        } else if (arg == "--engine" && arg_idx + 1 < argc) {
            engine_str = argv[++arg_idx];
            if (engine_str == "tensorrt") {
                engine_type = Predict::EngineType::TENSORRT;
            } else if (engine_str == "onnx") {
                engine_type = Predict::EngineType::ONNX_RUNTIME;
            } else {
                std::cerr << "Invalid engine type: " << engine_str << ". Use 'onnx' or 'tensorrt'." << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Invalid argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

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
    if (!file_exists(raw_path, "RAW file") || 
        !file_exists(model_path, "Model file") || 
        raw_width <= 0 || raw_height <= 0) {
        log_file.close();
        return 1;
    }

    try {
        // 打印配置信息
        log("=== Medical Image Segmentation Pipeline Configuration ===");
        log("Input RAW: " + fs::path(raw_path).filename().string() + " (" + std::to_string(raw_width) + "x" + std::to_string(raw_height) + ")");
        log("Model Path: " + fs::path(model_path).filename().string());
        log("Output Dir: " + output_dir);
        log("Inference Engine: " + engine_str);
        log("Normalization: " + std::string(skip_normalize ? "Disabled" : "Enabled"));

        // 创建输出子目录
        // const std::string raw_png_dir = output_dir + "/1_raw_png";
        // const std::string norm_png_dir = output_dir + "/2_normalized_png";
        // const std::string pred_mask_dir = output_dir + "/3_pred_masks";
        // const std::string polygon_dir = output_dir + "/4_polygons";
        const std::string preprocess_dir = output_dir + "/1_preprocessed_png";
        const std::string pred_mask_dir = output_dir + "/2_pred_masks";
        const std::string polygon_dir = output_dir + "/3_polygons";
        // fs::create_directories(raw_png_dir);
        // if (!skip_normalize) fs::create_directories(norm_png_dir);
        fs::create_directories(preprocess_dir);
        fs::create_directories(pred_mask_dir);
        fs::create_directories(polygon_dir);

        // 配置推理引擎参数
        Predict::Config config;
        config.engine_type = engine_type;

        // 配置ONNX Runtime参数（若使用）
        if (engine_type == Predict::EngineType::ONNX_RUNTIME) {
            config.onnx.input_name = "input";
            config.onnx.output_name = "output";
            config.onnx.use_gpu = true;
            config.onnx.intra_op_num_threads = 4;
        }
        // 配置TensorRT参数（若使用）
        else if (engine_type == Predict::EngineType::TENSORRT) {
            config.tensorrt.input_name = "input";
            config.tensorrt.output_name = "output";
            config.tensorrt.max_batch_size = 1;
            config.tensorrt.fp16_mode = true;  // 启用FP16加速
            config.tensorrt.engine_cache_path = output_dir + "/trt_engine_cache.trt";  // 缓存优化后的引擎
        }

        // 加载模型（适配新的loadModel函数）
        auto model_load_start = high_resolution_clock::now();
        void* session = Predict::loadModel(model_path, config, log_file);
        if (!session) {
            throw std::runtime_error("Failed to load model with " + engine_str);
        }
        auto model_load_end = high_resolution_clock::now();
        log("\nModel loaded in " + std::to_string(duration_cast<milliseconds>(model_load_end - model_load_start).count()) + " ms");

        // 核心处理流程
        auto core_start = high_resolution_clock::now();
        auto step1_time = 0ms, step2_time = 0ms, step3_time = 0ms;

        // // 步骤1: RAW转PNG
        // auto step1_start = high_resolution_clock::now();
        // const std::string raw_png_path = raw_png_dir + "/test.png";
        // Raw2Png::raw_to_png(raw_path, raw_png_path, raw_width, raw_height);
        // auto step1_end = high_resolution_clock::now();
        // step1_time = duration_cast<milliseconds>(step1_end - step1_start);
        // log("Step 1/4: RAW to PNG - " + std::to_string(step1_time.count()) + " ms");
        // input_to_infer = raw_png_path;  // 默认推理输入为raw_png（若不归一化）

        // // 步骤2: PNG归一化（可选）
        // if (!skip_normalize) {
        //     auto step2_start = high_resolution_clock::now();
        //     const std::string norm_png_path = norm_png_dir + "/test.png";
        //     const std::string size_json_path = norm_png_dir + "/original_sizes.json";
        //     PngNormalize::normalize_single_png(raw_png_path, norm_png_path, size_json_path);
        //     auto step2_end = high_resolution_clock::now();
        //     step2_time = duration_cast<milliseconds>(step2_end - step2_start);
        //     log("Step 2/4: Image Normalization - " + std::to_string(step2_time.count()) + " ms");
        //     input_to_infer = norm_png_path;  // 归一化后使用norm_png作为推理输入
        // } else {
        //     log("Step 2/4: Image Normalization - Skipped (--no-normalize specified)");
        // }

        // // 步骤3: 模型预测（适配新的predict_single_image函数）
        // auto step3_start = high_resolution_clock::now();
        // const std::string pred_mask_path = pred_mask_dir + "/test.png";
        // Predict::predict_single_image(session, input_to_infer, pred_mask_path, config, log_file);
        // auto step3_end = high_resolution_clock::now();
        // step3_time = duration_cast<milliseconds>(step3_end - step3_start);
        // log("Step 3/4: Model Inference - " + std::to_string(step3_time.count()) + " ms");

        // // 核心流程时间统计
        // auto core_end = high_resolution_clock::now();
        // const auto core_time = duration_cast<milliseconds>(core_end - core_start).count();

        // // 步骤4: 生成轮廓（保持不变）
        // auto step4_start = high_resolution_clock::now();
        // const std::string size_json_path = norm_png_dir + "/original_sizes.json";
        // Mask2Polygon::process_single_mask(pred_mask_path, polygon_dir, size_json_path, raw_png_path);
        // auto step4_end = high_resolution_clock::now();
        // auto step4_time = duration_cast<milliseconds>(step4_end - step4_start);
        // log("Step 4/4: Polygon Generation - " + std::to_string(step4_time.count()) + " ms (not included in core time)");

        // // 处理总结
        // log("\n[Processing Summary]");
        // log("Core processing time (steps 1-3): " + std::to_string(core_time) + " ms");
        // log("Breakdown:");
        // if (skip_normalize) {
        //     log("  Conversion: " + std::to_string(100.0 * step1_time.count() / core_time) + "% | Inference: " + std::to_string(100.0 * step3_time.count() / core_time) + "%");
        // } else {
        //     log("  Conversion: " + std::to_string(100.0 * step1_time.count() / core_time) + "% | Normalization: " + std::to_string(100.0 * step2_time.count() / core_time) + "% | Inference: " + std::to_string(100.0 * step3_time.count() / core_time) + "%");
        // }
        const std::string base_name = fs::path(raw_path).stem().string();

        // 合并步骤1：预处理
        auto step1_start = high_resolution_clock::now();
        const std::string preprocessed_png_path  = preprocess_dir + "/" + base_name + ".png";
        const std::string size_json_path = preprocess_dir + "/original_sizes.json";
        bool ok = Preprocess::preprocess_raw(raw_path, preprocessed_png_path, size_json_path,
                                            raw_width, raw_height);
        if (!ok) throw std::runtime_error("preprocess_raw failed");
        auto step1_end = high_resolution_clock::now();
        step1_time = duration_cast<milliseconds>(step1_end - step1_start);
        log("Step 1/3: RAW -> 512x512 PNG + JSON - " + std::to_string(step1_time.count()) + " ms");

        // 步骤2: 模型预测（适配新的predict_single_image函数）
        auto step2_start = high_resolution_clock::now();
        const std::string pred_mask_path = pred_mask_dir + "/" + base_name + ".png";
        Predict::predict_single_image(session, preprocessed_png_path, pred_mask_path, config, log_file);
        auto step2_end = high_resolution_clock::now();
        step2_time = duration_cast<milliseconds>(step2_end - step2_start);
        log("Step 2/3: Model Inference - " + std::to_string(step2_time.count()) + " ms");

        // 核心流程时间统计
        auto core_end = high_resolution_clock::now();
        const auto core_time = duration_cast<milliseconds>(core_end - core_start).count();

        // 步骤3: 生成轮廓（保持不变）
        auto step3_start = high_resolution_clock::now();
        
        Mask2Polygon::process_single_mask(pred_mask_path, polygon_dir, size_json_path, preprocessed_png_path, base_name);
        auto step3_end = high_resolution_clock::now();
        step3_time = duration_cast<milliseconds>(step3_end - step3_start);
        log("Step 3/3: Polygon Generation - " + std::to_string(step3_time.count()) + " ms (not included in core time)");

        // 处理总结
        log("\n[Processing Summary]");
        log("Core processing time (steps 1-2): " + std::to_string(core_time) + " ms");
        log("Breakdown:");
        log("  Preprocess: " + std::to_string(100.0 * step1_time.count() / core_time) + "% | Inference: " + std::to_string(100.0 * step2_time.count() / core_time) + "%");
        

        // 释放资源（适配新的freeResources函数）
        Predict::freeResources(session, config.engine_type);
        log("\nResources released successfully");

        // 总执行时间
        auto program_end = high_resolution_clock::now();
        log("\n=== Pipeline Completed ===");
        log("Total execution time: " + std::to_string(duration_cast<milliseconds>(program_end - program_start).count()) + " ms");
        log("Results saved to: " + output_dir);

    } catch (const std::exception& e) {
        std::string error = "[ERROR] " + std::string(e.what());
        log(error);
        log_file.close();
        return 1;
    } catch (...) {
        std::string error = "[ERROR] Unknown exception occurred";
        log(error);
        log_file.close();
        return 1;
    }

    log_file.close();
    return 0;
}