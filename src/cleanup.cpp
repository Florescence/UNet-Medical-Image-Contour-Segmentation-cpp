// cleanup.cpp（修正版）
#include "cleanup.h"
#include "initialize.h"
#include "process.h"
#include <NvInfer.h>
#include <iostream>

namespace MedicalSeg {

void cleanup_resources() {
    try {
        auto& log_file = get_log_file();
        log_file << "\n=== Cleaning Up Resources ===" << std::endl;
        
        // 1. 先销毁所有执行上下文（IExecutionContext）
        TensorRTContext& context = get_thread_local_context();
        if (context.context) {
            // 显式销毁执行上下文（必须在引擎销毁前）
            context.context.reset();  // 调用IExecutionContext的析构函数
            log_file << "IExecutionContext destroyed" << std::endl;
        }
        
        // 2. 释放设备内存（输入/输出缓冲区）
        if (context.input_buffer) {
            cudaFree(context.input_buffer);
            context.input_buffer = nullptr;
        }
        if (context.output_buffer) {
            cudaFree(context.output_buffer);
            context.output_buffer = nullptr;
        }
        if (context.stream) {
            cudaStreamDestroy(context.stream);
            context.stream = nullptr;
        }
        
        // 3. 最后销毁引擎（ICudaEngine）
        nvinfer1::ICudaEngine* engine = get_engine();
        if (engine) {
            engine->destroy();  // 引擎必须在所有上下文销毁后再销毁
            log_file << "ICudaEngine destroyed" << std::endl;
        }
        
        // 4. 重置运行时
        if (g_runtime) {
            g_runtime.reset();
            log_file << "IRuntime reset" << std::endl;
        }
        
        // 关闭日志文件
        if (log_file.is_open()) {
            log_file << "All resources cleaned up successfully" << std::endl;
            log_file.close();
        }
        
        std::cout << "Resources cleaned up successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Cleanup error: " << e.what() << std::endl;
        if (get_log_file().is_open()) {
            get_log_file() << "Cleanup error: " << e.what() << std::endl;
            get_log_file().close();
        }
    }
}

}  // namespace MedicalSeg