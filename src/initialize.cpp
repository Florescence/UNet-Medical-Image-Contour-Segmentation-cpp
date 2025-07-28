#include "initialize.h"
#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
namespace MedicalSeg {

// TensorRT 错误记录器
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << "TensorRT Logger: " << msg << std::endl;
        }
    }
} gLogger;

// 全局资源
std::unique_ptr<nvinfer1::IRuntime> g_runtime;
std::unique_ptr<nvinfer1::ICudaEngine> g_engine;
std::ofstream g_log_file;
std::string g_log_path;

// 初始化TensorRT引擎
bool initialize_engine(const std::string& trt_cache_path, const std::string& log_dir) {
    try {
        // 创建日志目录
        fs::create_directories(log_dir);
        g_log_path = log_dir + "/segmentation_log.txt";
        g_log_file.open(g_log_path, std::ios::out | std::ios::trunc);
        
        if (!g_log_file.is_open()) {
            std::cerr << "Failed to create log file: " << g_log_path << std::endl;
            return false;
        }

        g_log_file << "=== Initializing Medical Image Segmentation Engine ===" << std::endl;
        g_log_file << "TensorRT Engine Cache: " << trt_cache_path << std::endl;

        // 检查缓存文件
        if (!fs::exists(trt_cache_path)) {
            g_log_file << "Error: TensorRT cache file not found - " << trt_cache_path << std::endl;
            return false;
        }

        // 创建运行时并反序列化引擎
        g_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        std::ifstream file(trt_cache_path, std::ios::binary);
        
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        
        std::vector<char> trtModelStream(size);
        file.read(trtModelStream.data(), size);
        file.close();
        
        g_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            g_runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr));
        
        if (!g_engine) {
            g_log_file << "Error: Failed to deserialize TensorRT engine" << std::endl;
            return false;
        }

        g_log_file << "TensorRT engine initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Initialization error: " << e.what() << std::endl;
        if (g_log_file.is_open()) {
            g_log_file << "Initialization error: " << e.what() << std::endl;
        }
        return false;
    }
}

// 获取全局引擎指针
nvinfer1::ICudaEngine* get_engine() {
    return g_engine.get();
}

// 获取日志文件引用
std::ofstream& get_log_file() {
    return g_log_file;
}

// 获取日志路径
std::string get_log_path() {
    return g_log_path;
}

}  // namespace MedicalSeg