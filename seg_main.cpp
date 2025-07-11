#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

// 定义工作目录结构
struct WorkDirs {
    std::string raw_png;
    std::string normalized_png;
    std::string pred_masks;
    std::string denormalized_masks;
    std::string json_results;
};

// 创建工作目录
WorkDirs create_work_dirs(const std::string& root_dir) {
    WorkDirs dirs;
    dirs.raw_png = root_dir + "/1_raw_png";
    dirs.normalized_png = root_dir + "/2_normalized_png";
    dirs.pred_masks = root_dir + "/3_pred_masks";
    dirs.denormalized_masks = root_dir + "/4_denormalized_masks";
    dirs.json_results = root_dir + "/5_json_results";

    for (const auto& dir : {dirs.raw_png, dirs.normalized_png, 
                           dirs.pred_masks, dirs.denormalized_masks, 
                           dirs.json_results}) {
        fs::create_directories(dir);
    }
    return dirs;
}

// 步骤1：RAW转PNG
std::string step_raw_to_png(const std::string& input_raw, 
                           const std::string& output_png_dir, 
                           int width, int height, 
                           int window_width, int window_length) {
    std::cout << "===== 开始步骤1：RAW转PNG =====" << std::endl;
    
    // 读取RAW文件
    std::ifstream raw_file(input_raw, std::ios::binary);
    if (!raw_file) {
        throw std::runtime_error("无法打开RAW文件: " + input_raw);
    }

    // 计算图像大小
    size_t image_size = width * height;
    std::vector<uint16_t> buffer(image_size);
    
    // 读取RAW数据
    raw_file.read(reinterpret_cast<char*>(buffer.data()), image_size * sizeof(uint16_t));
    if (raw_file.gcount() != static_cast<std::streamsize>(image_size * sizeof(uint16_t))) {
        throw std::runtime_error("读取RAW文件失败，数据不完整");
    }

    // 创建窗口并归一化
    int min_val = window_length - window_width / 2;
    int max_val = window_length + window_width / 2;
    
    // 创建输出目录
    fs::create_directories(output_png_dir);
    
    // 生成输出文件名
    std::string base_name = fs::path(input_raw).stem().string();
    std::string output_png_path = output_png_dir + "/" + base_name + ".png";
    
    // 创建OpenCV矩阵
    cv::Mat image(height, width, CV_16UC1, buffer.data());
    
    // 应用窗宽窗位
    cv::Mat normalized_image;
    cv::normalize(image, normalized_image, 0, 65535, cv::NORM_MINMAX);
    
    // 转换为8位图像
    cv::Mat image_8bit;
    normalized_image.convertTo(image_8bit, CV_8U, 1.0 / 256.0);
    
    // 保存PNG
    cv::imwrite(output_png_path, image_8bit);
    
    // 保存原始尺寸JSON
    std::string sizes_json_path = output_png_dir + "/original_sizes.json";
    json sizes_json;
    sizes_json[base_name + ".png"] = {{"width", width}, {"height", height}};
    
    std::ofstream sizes_file(sizes_json_path);
    sizes_file << sizes_json.dump(2);
    
    std::cout << "步骤1完成：PNG保存至 " << output_png_path << std::endl;
    return output_png_dir;
}

// 步骤2：PNG归一化到512x512
std::string step_normalize_png(const std::string& input_png_dir, 
                              const std::string& output_norm_dir) {
    std::cout << "===== 开始步骤2：PNG归一化到512x512 =====" << std::endl;
    
    // 查找输入目录中的PNG文件
    std::vector<std::string> png_files;
    for (const auto& entry : fs::directory_iterator(input_png_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            png_files.push_back(entry.path().string());
        }
    }
    
    if (png_files.empty()) {
        throw std::runtime_error("步骤2未找到PNG文件，终止流程");
    }
    
    // 创建输出目录
    fs::create_directories(output_norm_dir);
    
    // 原始尺寸JSON
    json sizes_json;
    
    // 处理每张PNG
    for (const auto& png_file : png_files) {
        std::string filename = fs::path(png_file).filename().string();
        std::string base_name = fs::path(png_file).stem().string();
        
        // 读取图像
        cv::Mat image = cv::imread(png_file, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "警告: 无法读取图像 " << png_file << std::endl;
            continue;
        }
        
        // 记录原始尺寸
        sizes_json[filename] = {{"width", image.cols}, {"height", image.rows}};
        
        // 计算缩放比例
        double scale = 512.0 / std::max(image.cols, image.rows);
        
        // 缩放图像
        cv::Mat scaled_image;
        cv::resize(image, scaled_image, cv::Size(), scale, scale, cv::INTER_LANCZOS4);
        
        // 创建512x512的空白图像
        cv::Mat normalized_image = cv::Mat::zeros(512, 512, CV_8UC1);
        
        // 计算放置位置
        int x_offset = (512 - scaled_image.cols) / 2;
        int y_offset = (512 - scaled_image.rows) / 2;
        
        // 将缩放后的图像放置在中心
        scaled_image.copyTo(normalized_image(cv::Rect(x_offset, y_offset, 
                                                     scaled_image.cols, scaled_image.rows)));
        
        // 保存归一化后的图像
        std::string output_path = output_norm_dir + "/" + filename;
        cv::imwrite(output_path, normalized_image);
    }
    
    // 保存原始尺寸JSON
    std::string sizes_json_path = output_norm_dir + "/original_sizes.json";
    std::ofstream sizes_file(sizes_json_path);
    sizes_file << sizes_json.dump(2);
    
    std::cout << "步骤2完成：归一化PNG保存至 " << output_norm_dir << std::endl;
    return output_norm_dir;
}

// 步骤3：轮廓预测
std::string step_predict_mask(const std::string& input_norm_dir, 
                             const std::string& output_pred_dir, 
                             const std::string& model_path) {
    std::cout << "===== 开始步骤3：轮廓预测 =====" << std::endl;
    
    // 查找输入目录中的PNG文件
    std::vector<std::string> png_files;
    for (const auto& entry : fs::directory_iterator(input_norm_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            png_files.push_back(entry.path().string());
        }
    }
    
    if (png_files.empty()) {
        throw std::runtime_error("步骤3未找到归一化PNG，终止流程");
    }
    
    // 创建输出目录
    fs::create_directories(output_pred_dir);
    
    // 加载模型
    torch::jit::script::Module model = torch::jit::load(model_path);
    model.eval();
    
    // 预测每张图像
    for (const auto& png_file : png_files) {
        std::string filename = fs::path(png_file).filename().string();
        
        // 读取图像
        cv::Mat image = cv::imread(png_file, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "警告: 无法读取图像 " << png_file << std::endl;
            continue;
        }
        
        // 预处理图像
        torch::Tensor input_tensor = torch::from_blob(
            image.data, {1, 1, image.rows, image.cols}, torch::kFloat32
        );
        input_tensor = input_tensor.div(255.0).to(torch::kCUDA);
        
        // 模型推理
        torch::Tensor output = model.forward({input_tensor}).toTensor();
        
        // 后处理
        torch::Tensor mask = output.argmax(1).squeeze().to(torch::kCPU).to(torch::kUInt8);
        
        // 转换为OpenCV矩阵
        cv::Mat mask_mat(image.rows, image.cols, CV_8UC1, mask.data_ptr());
        
        // 保存掩码
        std::string output_path = output_pred_dir + "/" + filename;
        cv::imwrite(output_path, mask_mat);
    }
    
    std::cout << "步骤3完成：预测掩码保存至 " << output_pred_dir << std::endl;
    return output_pred_dir;
}

// 步骤4：掩码反归一化
std::string step_denormalize_mask(const std::string& input_pred_dir, 
                                 const std::string& output_denorm_dir, 
                                 const std::string& original_sizes_json) {
    std::cout << "===== 开始步骤4：掩码反归一化 =====" << std::endl;
    
    // 加载原始尺寸JSON
    std::ifstream sizes_file(original_sizes_json);
    if (!sizes_file) {
        throw std::runtime_error("无法打开原始尺寸JSON: " + original_sizes_json);
    }
    json sizes_json;
    sizes_file >> sizes_json;
    
    // 查找输入目录中的PNG文件
    std::vector<std::string> mask_files;
    for (const auto& entry : fs::directory_iterator(input_pred_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            mask_files.push_back(entry.path().string());
        }
    }
    
    if (mask_files.empty()) {
        throw std::runtime_error("步骤4未找到预测掩码，终止流程");
    }
    
    // 创建输出目录
    fs::create_directories(output_denorm_dir);
    
    // 处理每张掩码
    for (const auto& mask_file : mask_files) {
        std::string filename = fs::path(mask_file).filename().string();
        
        // 检查是否有原始尺寸
        if (!sizes_json.contains(filename)) {
            std::cerr << "警告: 未找到 " << filename << " 的原始尺寸" << std::endl;
            continue;
        }
        
        int original_width = sizes_json[filename]["width"];
        int original_height = sizes_json[filename]["height"];
        
        // 读取掩码
        cv::Mat mask = cv::imread(mask_file, cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            std::cerr << "警告: 无法读取掩码 " << mask_file << std::endl;
            continue;
        }
        
        // 反归一化
        cv::Mat denormalized_mask = denormalize_single_image(
            mask, original_width, original_height
        );
        
        // 保存反归一化后的掩码
        std::string output_path = output_denorm_dir + "/" + filename;
        cv::imwrite(output_path, denormalized_mask);
    }
    
    std::cout << "步骤4完成：反归一化掩码保存至 " << output_denorm_dir << std::endl;
    return output_denorm_dir;
}

// 步骤5：Mask转Polygon
std::string step_mask_to_polygon(const std::string& input_denorm_mask_dir, 
                                const std::string& output_json_dir, 
                                const std::string& original_sizes_json) {
    std::cout << "===== 开始步骤5：Mask转Polygon =====" << std::endl;
    
    // 加载原始尺寸JSON
    std::ifstream sizes_file(original_sizes_json);
    if (!sizes_file) {
        throw std::runtime_error("无法打开原始尺寸JSON: " + original_sizes_json);
    }
    json sizes_json;
    sizes_file >> sizes_json;
    
    // 查找输入目录中的PNG文件
    std::vector<std::string> mask_files;
    for (const auto& entry : fs::directory_iterator(input_denorm_mask_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            mask_files.push_back(entry.path().string());
        }
    }
    
    if (mask_files.empty()) {
        throw std::runtime_error("步骤5未找到反归一化掩码，终止流程");
    }
    
    // 创建输出目录
    fs::create_directories(output_json_dir);
    
    // 处理每张掩码
    for (const auto& mask_file : mask_files) {
        std::string filename = fs::path(mask_file).filename().string();
        
        // 检查是否有原始尺寸
        if (!sizes_json.contains(filename)) {
            std::cerr << "警告: 未找到 " << filename << " 的原始尺寸" << std::endl;
            continue;
        }
        
        int original_width = sizes_json[filename]["width"];
        int original_height = sizes_json[filename]["height"];
        
        // 处理单张掩码
        process_single_mask(
            mask_file, output_json_dir, original_sizes_json
        );
    }
    
    std::cout << "步骤5完成：轮廓JSON和覆盖图保存至 " << output_json_dir << std::endl;
    return output_json_dir;
}

int main(int argc, char* argv[]) {
    try {
        // 检查命令行参数
        if (argc != 8) {
            std::cerr << "用法: " << argv[0] 
                      << " <input_raw> <output_root> <width> <height> <window_width> <window_length> <model_path>" 
                      << std::endl;
            return 1;
        }
        
        // 解析参数
        std::string input_raw = argv[1];
        std::string output_root = argv[2];
        int width = std::stoi(argv[3]);
        int height = std::stoi(argv[4]);
        int window_width = std::stoi(argv[5]);
        int window_length = std::stoi(argv[6]);
        std::string model_path = argv[7];
        
        std::cout << "===== 开始端到端RAW图像轮廓提取流程 =====" << std::endl;
        
        // 创建工作目录
        WorkDirs work_dirs = create_work_dirs(output_root);
        std::string original_sizes_json = work_dirs.normalized_png + "/original_sizes.json";
        
        // 执行流程
        std::string raw_png_dir = step_raw_to_png(
            input_raw, work_dirs.raw_png, width, height, window_width, window_length
        );
        
        std::string norm_png_dir = step_normalize_png(
            raw_png_dir, work_dirs.normalized_png
        );
        
        std::string pred_mask_dir = step_predict_mask(
            norm_png_dir, work_dirs.pred_masks, model_path
        );
        
        std::string denorm_mask_dir = step_denormalize_mask(
            pred_mask_dir, work_dirs.denormalized_masks, original_sizes_json
        );
        
        std::string json_result_dir = step_mask_to_polygon(
            denorm_mask_dir, work_dirs.json_results, original_sizes_json
        );
        
        std::cout << "===== 全流程完成 =====" << std::endl;
        std::cout << "最终结果目录：" << json_result_dir << std::endl;
        std::cout << "流程成功结束" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "流程失败：" << e.what() << std::endl;
        return 1;
    }
}