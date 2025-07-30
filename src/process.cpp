#include "process.h"
#include "preprocess.h"
#include "mask2polygon.h"
#include "initialize.h"
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <chrono>
#include <filesystem>
#include "postprocess.cpp"

namespace fs = std::filesystem;
namespace MedicalSeg {

// 线程局部上下文缓存
static thread_local TensorRTContext g_context;
// 获取线程局部上下文
TensorRTContext& get_thread_local_context() {
    return g_context;
}

// 图像预处理
std::vector<float> preprocess_image(const cv::Mat& gray_img) {
    const int type = gray_img.type();
    assert(type == CV_8UC1 || type == CV_16UC1 && "Input must be 8-bit or 16-bit grayscale");

    const int total_pixels = gray_img.rows * gray_img.cols;
    std::vector<float> input_data(total_pixels);
    float* dst = input_data.data();

    if (type == CV_16UC1) {
        const uint16_t* src = gray_img.ptr<uint16_t>();
        for (int i = 0; i < total_pixels; ++i) {
            dst[i] = static_cast<float>(src[i]) / 65535.0f;
        }
    } else {
        const uchar* src = gray_img.data;
        for (int i = 0; i < total_pixels; ++i) {
            dst[i] = static_cast<float>(src[i]) / 255.0f;
        }
    }
    return input_data;
}

// 初始化TensorRT上下文（固定512x512输入尺寸）
bool initialize_context(nvinfer1::ICudaEngine* engine, const std::string& input_name, 
                       const std::string& output_name) {
    try {
        // 1. 销毁旧上下文（若存在，确保重新初始化时干净）
        if (g_context.context) {
            g_context.context.reset();
        }
        if (g_context.stream) {
            cudaStreamDestroy(g_context.stream);
        }
        if (g_context.input_buffer) {
            cudaFree(g_context.input_buffer);
            g_context.input_buffer = nullptr;
        }
        if (g_context.output_buffer) {
            cudaFree(g_context.output_buffer);
            g_context.output_buffer = nullptr;
        }
        if (g_context.graph_exec) {
            cudaGraphExecDestroy(g_context.graph_exec);
            g_context.graph_exec = nullptr;
        }

        // 2. 创建执行上下文并固定输入形状（512x512）
        g_context.context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        nvinfer1::Dims4 fixed_dims(1, 1, 512, 512);  // 固定N=1, C=1, H=512, W=512
        g_context.context->setInputShape(input_name.c_str(), fixed_dims);
        
        // 3. 预分配输入内存（基于固定尺寸，一次分配永久复用）
        const size_t input_size  = 1 * 1 * 512 * 512 * sizeof(float);
        const int input_idx = engine->getBindingIndex(input_name.c_str());
        g_context.max_input_size = input_size;
        cudaMalloc(&g_context.input_buffer,  input_size);

        // 4. 获取输出维度并预分配输出内存（基于固定输入尺寸的输出）
        const int output_idx = engine->getBindingIndex(output_name.c_str());
        nvinfer1::Dims output_dims = g_context.context->getBindingDimensions(output_idx);
        const int num_classes = output_dims.d[1];
        const int output_h = output_dims.d[2];  // 通常与输入同尺寸（512）
        const int output_w = output_dims.d[3];  // 通常与输入同尺寸（512）
        const size_t output_size = num_classes * output_h * output_w * sizeof(float);
        g_context.max_output_size = output_size;
        cudaMalloc(&g_context.output_buffer, output_size);

        // 5. 创建CUDA流（复用流减少创建开销）
        cudaStreamCreate(&g_context.stream);

        /* --------  关键：先做一次热身推理  -------- */
        void* buffers[] = {g_context.input_buffer, g_context.output_buffer};
        g_context.context->enqueueV2(buffers, g_context.stream, nullptr);
        cudaStreamSynchronize(g_context.stream);
        /* ----------------------------------------- */

        // 6. 启用CUDA Graph优化（记录一次推理流程，永久复用）
        cudaGraph_t graph;
        cudaStreamBeginCapture(g_context.stream, cudaStreamCaptureModeGlobal);
        g_context.context->enqueueV2(buffers, g_context.stream, nullptr);
        cudaStreamEndCapture(g_context.stream, &graph);
        // 实例化Graph（可多次启动）
        cudaGraphInstantiate(&g_context.graph_exec, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);  // 释放临时Graph

        // 日志：初始化完成
        auto& log_file = get_log_file();
        log_file << "TensorRT context initialized for fixed 512x512 input" << std::endl;
        log_file << "  Input size: " << input_size << " bytes" << std::endl;
        log_file << "  Output size: " << output_size << " bytes (classes=" << num_classes << ")" << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Context initialization failed: " << e.what() << std::endl;
        auto& log_file = get_log_file();
        log_file << "Context initialization failed: " << e.what() << std::endl;
        return false;
    }
}

// 执行TensorRT推理
cv::Mat execute_inference(nvinfer1::ICudaEngine* engine, const cv::Mat& gray_img, 
                         const std::string& input_name, const std::string& output_name) {
    try {
        if (gray_img.rows != 512 || gray_img.cols != 512) {
            throw std::runtime_error("Input size must be 512x512 for fixed context");
        }
        
        // 初始化上下文（首次调用）
        if (!g_context.context){
            if (!initialize_context(engine, input_name, output_name)) {
                throw std::runtime_error("Failed to initialize TensorRT context");
            }
        }
        
        // 预处理输入数据
        static std::vector<float> input_data(1 * 1 * 512 * 512);
        input_data = preprocess_image(gray_img);
        
        // 2. 异步拷贝输入数据到GPU
        const size_t input_size = g_context.max_input_size;
        cudaMemcpyAsync(g_context.input_buffer, input_data.data(), input_size, 
                       cudaMemcpyHostToDevice, g_context.stream);

        // 3. 执行推理（复用CUDA Graph）
        cudaGraphLaunch(g_context.graph_exec, g_context.stream);

        // 4. 异步拷贝输出数据到CPU
        const int output_idx = engine->getBindingIndex(output_name.c_str());
        const size_t output_size = g_context.max_output_size;
        static std::vector<float> output_data(output_size / sizeof(float));  // 预分配输出内存
        cudaMemcpyAsync(output_data.data(), g_context.output_buffer, output_size, 
                       cudaMemcpyDeviceToHost, g_context.stream);
        cudaStreamSynchronize(g_context.stream);  // 等待推理完成
        
        // 后处理：多分类argmax
        cv::Mat pred_mask(512, 512, CV_8UC1);
        cv::Mat max_prob(512, 512, CV_32FC1, cv::Scalar(-FLT_MAX));
        cv::Mat class_idx(512, 512, CV_8UC1, cv::Scalar(0));
        
        for (int c = 0; c < 3; ++c) {
            cv::Mat class_channel(512, 512, CV_32FC1, &output_data[c * 512 * 512]);
            cv::Mat update_mask;
            cv::compare(class_channel, max_prob, update_mask, cv::CMP_GT);
            class_channel.copyTo(max_prob, update_mask);
            cv::Mat class_mask = cv::Mat::ones(512, 512, CV_8UC1) * c;
            class_mask.copyTo(class_idx, update_mask);
        }
        class_idx.copyTo(pred_mask);
        return pred_mask;
    } catch (const std::exception& e) {
        throw std::runtime_error("Inference failed: " + std::string(e.what()));
    }
}

// 掩码可视化
cv::Mat mask_to_image(const cv::Mat& mask) {
    cv::Mat vis_mask(mask.size(), CV_8UC1);
    uchar lut[256] = {0};
    lut[1] = 128;
    lut[2] = 255;
    cv::LUT(mask, cv::Mat(1, 256, CV_8UC1, lut), vis_mask);
    return vis_mask;
}

// 处理单张图像
bool process_single_image(const std::string& raw_path, int width, int height, 
                         const std::string& output_dir) {
    try {
        auto& log_file = get_log_file();
        nvinfer1::ICudaEngine* engine = get_engine();
        
        if (!engine) {
            throw std::runtime_error("Engine not initialized");
        }
        
        log_file << "\n=== Processing Image: " << fs::path(raw_path).filename().string() << " ===" << std::endl;
        
        // 创建输出目录
        const std::string base_name = fs::path(raw_path).stem().string();
        
        // 记录总处理时间
        auto total_start_time = std::chrono::high_resolution_clock::now();
        
        // 预处理RAW文件
        const std::string preprocessed_png_path = output_dir + "/" + base_name + "_normalized.png";
        const std::string size_json_path = output_dir + "/" + base_name + "_original_sizes.json";
        const std::string pred_mask_path = output_dir + "/" + base_name + "_mask.png";
        
        bool ok = Preprocess::preprocess_raw(raw_path, preprocessed_png_path, size_json_path, width, height);
        if (!ok) {
            throw std::runtime_error("Preprocessing failed");
        }
        
        // 读取预处理后的图像
        cv::Mat gray_img = cv::imread(preprocessed_png_path, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
        if (gray_img.empty()) {
            throw std::runtime_error("Failed to read preprocessed image");
        }
        
        // 执行推理（记录推理时间）
        auto inference_start_time = std::chrono::high_resolution_clock::now();
        cv::Mat pred_mask = execute_inference(engine, gray_img, "input", "output");
        auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - inference_start_time).count();
        
        log_file << "Inference time: " << inference_duration << " ms" << std::endl;
        
        // 后处理
        pred_mask = postprocess_mask(pred_mask);

        // 保存掩码
        cv::Mat vis_mask = mask_to_image(pred_mask);
        
        std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
        if (!cv::imwrite(pred_mask_path, vis_mask, params)) {
            throw std::runtime_error("Failed to save mask");
        }
        
        // 生成轮廓
        Mask2Polygon::process_single_mask(pred_mask_path, output_dir, size_json_path, preprocessed_png_path, base_name);
        
        // 计算总处理时间
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - total_start_time).count();
        
        // 打印总处理时间
        log_file << "Total processing time: " << total_duration << " ms" << std::endl;
        log_file << "Processing completed for: " << base_name << std::endl;
        
        // 同时在控制台输出总时间
        std::cout << "Total processing time: " << total_duration << " ms" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Processing error: " << e.what() << std::endl;
        auto& log_file = get_log_file();
        log_file << "Processing error: " << e.what() << std::endl;
        return false;
    }
}

}  // namespace MedicalSeg