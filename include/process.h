// process.h
#ifndef PROCESS_H
#define PROCESS_H

#include <string>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime.h>  // 添加CUDA头文件

namespace MedicalSeg {

// TensorRT上下文缓存结构（线程局部存储）
struct TensorRTContext {
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;
    void* input_buffer = nullptr;
    void* output_buffer = nullptr;
    size_t max_input_size = 0;
    size_t max_output_size = 0;
    int fixed_input_h = 512;
    int fixed_input_w = 512;
    cudaGraphExec_t graph_exec = nullptr;
};

// 获取线程局部上下文
TensorRTContext& get_thread_local_context();

// 处理单张RAW图像
bool process_single_image(const std::string& raw_path, int width, int height, 
                         const std::string& output_dir);

}  // namespace MedicalSeg

#endif  // PROCESS_H