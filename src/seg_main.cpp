#include <iostream>
#include <filesystem>
#include <chrono>
#include <cstdlib>
#include <torch/torch.h>
#include <torch/script.h>
#include "raw2png.h"
#include "png_normalize.h"
#include "predict.h"
#include "png_denormalize.h"
#include "mask2polygon.h"

namespace fs = std::filesystem;
using namespace std::chrono;

// 打印命令行参数使用说明
void print_usage(const std::string& program_name) {
    std::cerr << "Usage: " << program_name << " <raw_path> <model_path> <output_dir> <raw_width> <raw_height>" << std::endl;
    std::cerr << "Example:" << std::endl;
    std::cerr << "  " << program_name << " input.raw model.pt output 3072 3072" << std::endl;
    std::cerr << "Parameters:" << std::endl;
    std::cerr << "  <raw_path>    - Input Raw File Path (such as test.raw)" << std::endl;
    std::cerr << "  <model_path>  - Model File Path (such as unet_model.pt)" << std::endl;
    std::cerr << "  <output_dir>  - Output Dir Path (such as output)" << std::endl;
    std::cerr << "  <raw_width>   - RAW Image Width (Int, such as 3072)" << std::endl;
    std::cerr << "  <raw_height>  - RAW Image Height (Int, such as 3072)" << std::endl;
}

int main(int argc, char* argv[]) {
    // 记录程序启动时间（用于计算所有步骤的总时长，包括模型加载和轮廓生成）
    auto program_start = high_resolution_clock::now();

    // 校验命令行参数数量
    if (argc != 6) {
        std::cerr << "Error: Invalid number of parameters!" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // 解析命令行参数
    const std::string raw_path = argv[1];
    const std::string model_path = argv[2];
    const std::string output_dir = argv[3];
    const int raw_width = std::atoi(argv[4]);
    const int raw_height = std::atoi(argv[5]);

    // 校验参数有效性
    try {
        if (!fs::exists(raw_path)) {
            throw std::runtime_error("RAW file not found: " + raw_path);
        }
        if (!fs::exists(model_path)) {
            throw std::runtime_error("Model file not found: " + model_path);
        }
        if (raw_width <= 0 || raw_height <= 0) {
            throw std::runtime_error("Invalid width/height (must be positive integers)");
        }
    } catch (const std::exception& e) {
        std::cerr << "Parameter Validation Error: " << e.what() << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    try {
        // 创建输出目录
        fs::create_directories(output_dir);
        std::cout << "===== Start Processing =====" << std::endl;
        std::cout << "Parameters:" << std::endl;
        std::cout << "  RAW Path: " << raw_path << std::endl;
        std::cout << "  Model Path: " << model_path << std::endl;
        std::cout << "  Output Dir: " << output_dir << std::endl;
        std::cout << "  Image Size: " << raw_width << "x" << raw_height << std::endl;

        // -------------------------- 模型加载（不计入总耗时） --------------------------
        auto model_load_start = high_resolution_clock::now();
        std::cout << "\nLoading Model: " << model_path << std::endl;

        // 初始化设备（保持CPU）
        torch::Device device(torch::kCPU);

        // 加载模型
        torch::jit::script::Module model = torch::jit::load(model_path);
        model.to(device);  // 将模型移动到设备
        model.eval();      // 切换到推理模式

        // 计算模型加载时间（单独统计）
        auto model_load_end = high_resolution_clock::now();
        duration<double> model_load_time = model_load_end - model_load_start;
        std::cout << "Model Load Complete! Time: " << model_load_time.count() << " sec" << std::endl;
        // --------------------------------------------------------------------------

        // 记录核心流程开始时间（从步骤1到步骤4，不含模型加载和步骤5）
        auto core_start = high_resolution_clock::now();

        // 步骤1: RAW转PNG
        auto step1_start = high_resolution_clock::now();
        std::string raw_png_path = output_dir + "/1_raw_png/test.png";
        Raw2Png::raw_to_png(raw_path, raw_png_path, raw_width, raw_height);
        auto step1_end = high_resolution_clock::now();
        duration<double> step1_time = step1_end - step1_start;
        std::cout << "\nStep 1 Time: " << step1_time.count() << " sec" << std::endl;


        // 步骤2: PNG归一化
        auto step2_start = high_resolution_clock::now();
        std::string norm_png_path = output_dir + "/2_normalized_png/test.png";
        std::string size_json_path = output_dir + "/2_normalized_png/original_sizes.json";
        PngNormalize::normalize_single_png(raw_png_path, norm_png_path, size_json_path);
        auto step2_end = high_resolution_clock::now();
        duration<double> step2_time = step2_end - step2_start;
        std::cout << "Step 2 Time: " << step2_time.count() << " sec" << std::endl;


        // 步骤3: 模型预测（传入预加载的模型和设备）
        auto step3_start = high_resolution_clock::now();
        std::string pred_mask_path = output_dir + "/3_pred_masks/test.png";
        Predict::predict_single_image(model, device, norm_png_path, pred_mask_path);  // 传入模型和设备
        auto step3_end = high_resolution_clock::now();
        duration<double> step3_time = step3_end - step3_start;
        std::cout << "Step 3 Time: " << step3_time.count() << " sec" << std::endl;


        // 步骤4: 掩码反归一化
        auto step4_start = high_resolution_clock::now();
        std::string denorm_mask_path = output_dir + "/4_denormalized_mask/test.png";
        PngDenormalize::denormalize_single_png(pred_mask_path, denorm_mask_path, size_json_path);
        auto step4_end = high_resolution_clock::now();
        duration<double> step4_time = step4_end - step4_start;
        std::cout << "Step 4 Time: " << step4_time.count() << " sec" << std::endl;

        // 计算核心流程总耗时（步骤1-4，不含模型加载和步骤5）
        auto core_end = high_resolution_clock::now();
        duration<double> core_time = core_end - core_start;


        // 步骤5: 生成轮廓和覆盖图（不计入总耗时）
        auto step5_start = high_resolution_clock::now();
        std::string contour_json_path = output_dir + "/test.json";
        std::string overlay_path = output_dir + "/test_contour_overlay.png";
        std::string original_png = output_dir + "/1_raw_png/test.png";
        Mask2Polygon::process_single_mask(denorm_mask_path, output_dir, size_json_path, original_png);
        auto step5_end = high_resolution_clock::now();
        duration<double> step5_time = step5_end - step5_start;
        std::cout << "\nStep 5 Time: " << step5_time.count() << " sec (not included in total)" << std::endl;


        // 计算程序总运行时间（含所有步骤，仅用于参考）
        auto program_end = high_resolution_clock::now();
        duration<double> program_total_time = program_end - program_start;

        // 输出核心流程统计（步骤1-4）
        std::cout << "\n===== Core Process Done =====" << std::endl;
        std::cout << "Core Total Time (Steps 1-4): " << core_time.count() << " sec" << std::endl;
        std::cout << "Core Step Percentiles:" << std::endl;
        std::cout << "  Step 1: " << (step1_time / core_time) * 100 << "%" << std::endl;
        std::cout << "  Step 2: " << (step2_time / core_time) * 100 << "%" << std::endl;
        std::cout << "  Step 3: " << (step3_time / core_time) * 100 << "%" << std::endl;
        std::cout << "  Step 4: " << (step4_time / core_time) * 100 << "%" << std::endl;

        // 输出其他时间统计（供参考）
        std::cout << "\n===== Additional Timing =====" << std::endl;
        std::cout << "Model Load Time: " << model_load_time.count() << " sec (not included in core)" << std::endl;
        std::cout << "Step 5 (Contour) Time: " << step5_time.count() << " sec (not included in core)" << std::endl;
        std::cout << "Program Total Time (all steps): " << program_total_time.count() << " sec (for reference)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Processing Failure: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}