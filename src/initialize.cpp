#include "initialize.h"
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <Windows.h>  // 必须包含，用于字符串转换

namespace fs = std::filesystem;
namespace MedicalSeg {

// 1. 字符串转换函数
std::wstring to_wstring(const std::string& str) {
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    if (len == 0) {
        throw std::runtime_error("MultiByteToWideChar failed");
    }
    std::wstring wstr(len - 1, L'\0');  // 减1是为了去掉末尾的null
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], len);
    return wstr;
}

// 全局资源
static std::unique_ptr<Ort::Env> g_env;
static std::unique_ptr<Ort::Session> g_session;
static std::vector<std::string> g_input_names;
static std::vector<std::string> g_output_names;
static std::ofstream g_log_file;
static std::string g_log_path;

// 初始化函数
bool initialize_onnx_runtime(const std::string& onnx_model_path, const std::string& log_dir) {
    try {
        // 创建日志目录
        fs::create_directories(log_dir);
        g_log_path = log_dir + "/segmentation_log.txt";
        g_log_file.open(g_log_path, std::ios::out | std::ios::trunc);
        if (!g_log_file.is_open()) {
            std::cerr << "Failed to create log file: " << g_log_path << std::endl;
            return false;
        }
        g_log_file << "=== Initializing ONNX Runtime (CPU) ===" << std::endl;
        g_log_file << "ONNX Model Path: " << onnx_model_path << std::endl;

        // 检查模型文件
        if (!fs::exists(onnx_model_path)) {
            g_log_file << "Error: ONNX model not found: " << onnx_model_path << std::endl;
            return false;
        }

        // 初始化环境
        g_env = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING);

        // 配置会话选项
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);  // CPU线程数
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 2. 转换模型路径为宽字符
        std::wstring w_model_path = to_wstring(onnx_model_path);

        // 3. 直接构造Session，不使用make_unique（避免参数推导问题）
        g_session = std::unique_ptr<Ort::Session>(
            new Ort::Session(*g_env, w_model_path.c_str(), session_options)
        );

        // 获取输入输出名称
        Ort::AllocatorWithDefaultOptions allocator;
        size_t input_count = g_session->GetInputCount();
        size_t output_count = g_session->GetOutputCount();

        // 获取输入名称
        for (size_t i = 0; i < input_count; ++i) {
#ifdef ORT_API_VERSION>=12
            Ort::AllocatedStringPtr name_ptr = g_session->GetInputNameAllocated(i, allocator);
            g_input_names.push_back(std::string(name_ptr.get()));
            name_ptr.release();  // 释放内存
#else
            const char* name = g_session->GetInputName(i, allocator);
            g_input_names.push_back(std::string(name));
#endif
        }

        // 获取输出名称
        for (size_t i = 0; i < output_count; ++i) {
#ifdef ORT_API_VERSION>=12
            Ort::AllocatedStringPtr name_ptr = g_session->GetOutputNameAllocated(i, allocator);
            g_output_names.push_back(std::string(name_ptr.get()));
            name_ptr.release();  // 释放内存
#else
            const char* name = g_session->GetOutputName(i, allocator);
            g_output_names.push_back(std::string(name));
#endif
        }

        g_log_file << "ONNX Runtime initialized successfully (CPU)" << std::endl;
        g_log_file << "Input nodes: " << input_count << ", Output nodes: " << output_count << std::endl;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Error: " << e.what() << std::endl;
        g_log_file << "ONNX Error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        g_log_file << "Error: " << e.what() << std::endl;
        return false;
    }
}

Ort::Session* get_onnx_session() { return g_session.get(); }
const std::vector<std::string>& get_input_names() { return g_input_names; }
const std::vector<std::string>& get_output_names() { return g_output_names; }
std::ofstream& get_log_file() { return g_log_file; }
std::string get_log_path() { return g_log_path; }

}  // namespace MedicalSeg