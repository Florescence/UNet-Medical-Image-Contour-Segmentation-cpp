// initialize.h
#ifndef INITIALIZE_H
#define INITIALIZE_H

#include <NvInfer.h>
#include <string>
#include <fstream>

namespace MedicalSeg {

// 初始化TensorRT引擎和日志系统
bool initialize_engine(const std::string& trt_cache_path, const std::string& log_dir);

// 获取全局引擎指针
nvinfer1::ICudaEngine* get_engine();

// 获取日志文件引用
std::ofstream& get_log_file();

// 获取日志路径
std::string get_log_path();

// 外部声明全局资源
extern std::unique_ptr<nvinfer1::IRuntime> g_runtime;
extern std::unique_ptr<nvinfer1::ICudaEngine> g_engine;

}  // namespace MedicalSeg

#endif  // INITIALIZE_H