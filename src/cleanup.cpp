#include "cleanup.h"
#include "initialize.h"
#include "process.h"
#include <iostream>

namespace MedicalSeg {

// 清理ONNX Runtime资源
void cleanup_resources() {
    try {
        auto& log_file = get_log_file();
        log_file << "\n=== Cleaning Up ONNX Runtime Resources ===" << std::endl;
        
        // 释放ONNX会话和环境（通过unique_ptr自动管理）
        // 线程局部上下文清理
        cleanup_ort_context();
        
        // 关闭日志文件
        if (log_file.is_open()) {
            log_file << "Resources cleaned up successfully" << std::endl;
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