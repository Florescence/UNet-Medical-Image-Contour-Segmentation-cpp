#ifndef INITIALIZE_H
#define INITIALIZE_H

#include <onnxruntime_cxx_api.h>
#include <string>
#include <fstream>
#include <memory>

namespace MedicalSeg {

// 初始化ONNX Runtime环境和模型
bool initialize_onnx_runtime(const std::string& onnx_model_path, const std::string& log_dir);

// 获取ONNX Runtime会话
Ort::Session* get_onnx_session();

// 获取输入/输出节点名称
const std::vector<std::string>& get_input_names();
const std::vector<std::string>& get_output_names();

// 日志相关
std::ofstream& get_log_file();
std::string get_log_path();

}  // namespace MedicalSeg

#endif  // INITIALIZE_H