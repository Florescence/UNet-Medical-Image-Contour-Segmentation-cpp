#include <iostream>
#include <filesystem>
#include <chrono>
#include "raw2png.h"
#include "png_normalize.h"
#include "predict.h"
#include "png_denormalize.h"
#include "mask2polygon.h"

namespace fs = std::filesystem;
using namespace std::chrono;

int main() {
    // 记录全流程开始时间
    auto total_start = high_resolution_clock::now();

    // 配置参数
    const std::string raw_path = "0-chestap-10000-grid.raw";
    const std::string model_path = "../unet_model.pt";
    const std::string output_dir = "output";
    const int raw_width = 3072;
    const int raw_height = 3072;

    try {
        // 创建输出目录
        fs::create_directories(output_dir);
        std::cout << "===== Start Processing =====" << std::endl;

        // 步骤1: RAW转PNG
        auto step1_start = high_resolution_clock::now();
        std::string raw_png_path = output_dir + "/1_raw_png" + "/test.png";
        Raw2Png::raw_to_png(raw_path, raw_png_path, raw_width, raw_height);
        auto step1_end = high_resolution_clock::now();
        duration<double> step1_time = step1_end - step1_start;
        std::cout << "Step 1 Time: " << step1_time.count() << " sec" << std::endl;


        // 步骤2: PNG归一化
        auto step2_start = high_resolution_clock::now();
        std::string norm_png_path = output_dir + "/2_normalized_png" + "/test.png";
        std::string size_json_path = output_dir + "/2_normalized_png" + "/original_sizes.json";
        PngNormalize::normalize_single_png(raw_png_path, norm_png_path, size_json_path);
        auto step2_end = high_resolution_clock::now();
        duration<double> step2_time = step2_end - step2_start;
        std::cout << "Step 2 Time: " << step2_time.count() << " sec" << std::endl;


        // 步骤3: 模型预测）
        auto step3_start = high_resolution_clock::now();
        std::string pred_mask_path = output_dir + "/3_pred_masks" + "/test.png";
        Predict::predict_single_image(model_path, norm_png_path, pred_mask_path);
        auto step3_end = high_resolution_clock::now();
        duration<double> step3_time = step3_end - step3_start;
        std::cout << "Step 3 Time: " << step3_time.count() << " sec" << std::endl;


        // 步骤4: 掩码反归一化
        auto step4_start = high_resolution_clock::now();
        std::string denorm_mask_path = output_dir + "/4_denormalized_mask" + "/test.png";
        PngDenormalize::denormalize_single_png(pred_mask_path, denorm_mask_path, size_json_path);
        auto step4_end = high_resolution_clock::now();
        duration<double> step4_time = step4_end - step4_start;
        std::cout << "Step 4 Time: " << step4_time.count() << " sec" << std::endl;


        // 步骤5: 生成轮廓和覆盖图
        auto step5_start = high_resolution_clock::now();
        std::string contour_json_path = output_dir + "/test.json";
        std::string overlay_path = output_dir + "/test_contour_overlay.png";
        std::string original_png = output_dir + "/1_raw_png" + "/test.png";
        Mask2Polygon::process_single_mask(denorm_mask_path, output_dir, size_json_path, original_png);
        auto step5_end = high_resolution_clock::now();
        duration<double> step5_time = step5_end - step5_start;
        std::cout << "Step 5 Time: " << step5_time.count() << " sec" << std::endl;


        // 计算全流程总耗时
        auto total_end = high_resolution_clock::now();
        duration<double> total_time = total_end - total_start;

        std::cout << "\n===== Whole Process Done =====" << std::endl;
        std::cout << "Total Time: " << total_time.count() << " sec" << std::endl;
        std::cout << "Percentile per Step:" << std::endl;
        std::cout << "  Step 1: " << (step1_time / total_time) * 100 << "%" << std::endl;
        std::cout << "  Step 2: " << (step2_time / total_time) * 100 << "%" << std::endl;
        std::cout << "  Step 3: " << (step3_time / total_time) * 100 << "%" << std::endl;
        std::cout << "  Step 4: " << (step4_time / total_time) * 100 << "%" << std::endl;
        std::cout << "  Step 5: " << (step5_time / total_time) * 100 << "%" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Processing Failure: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}