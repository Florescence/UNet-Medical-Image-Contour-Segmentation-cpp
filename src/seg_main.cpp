#include <iostream>
#include <filesystem>
#include "raw2png.h"
#include "png_normalize.h"
#include "predict.h"
#include "png_denormalize.h"
#include "mask2polygon.h"

namespace fs = std::filesystem;

int main() {
    // 配置参数
    const std::string raw_path = "test.raw";
    const std::string model_path = "../unet_model.pt";
    const std::string output_dir = "output";
    const int raw_width = 4267;
    const int raw_height = 4267;

    try {
        // 创建输出目录
        fs::create_directories(output_dir);

        // 步骤1: RAW转PNG
        std::string raw_png_path = output_dir + "/raw.png";
        Raw2Png::raw_to_png(raw_path, raw_png_path, raw_width, raw_height);

        // 步骤2: PNG归一化
        std::string norm_png_path = output_dir + "/normalized.png";
        std::string size_json_path = output_dir + "/original_sizes.json";
        PngNormalize::normalize_single_png(raw_png_path, norm_png_path, size_json_path);

        // 步骤3: 模型预测
        std::string pred_mask_path = output_dir + "/pred_mask.png";
        Predict::predict_single_image(model_path, norm_png_path, pred_mask_path);

        // 步骤4: 掩码反归一化
        std::string denorm_mask_path = output_dir + "/denorm_mask.png";
        PngDenormalize::denormalize_single_png(pred_mask_path, denorm_mask_path, size_json_path);

        // 步骤5: 生成轮廓和覆盖图
        std::string contour_json_path = output_dir + "/contour.json";
        std::string overlay_path = output_dir + "/contour_overlay.png";
        Mask2Polygon::process_single_mask(denorm_mask_path, output_dir, size_json_path);

        std::cout << "Thourough Process Complete！" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Processing Failure: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}