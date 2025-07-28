// seg_main.cpp（交互式模式）
#include "initialize.h"
#include "process.h"
#include "cleanup.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace MedicalSeg;

// 打印使用说明
void print_usage() {
    std::cout << "\nMedical Image Segmentation Tool (TensorRT)" << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  init <trt_cache_path>         - Initialize TensorRT engine" << std::endl;
    std::cout << "  process <raw_path> <width> <height> [output_dir] - Process image" << std::endl;
    std::cout << "  exit                          - Cleanup and exit" << std::endl;
}

int main() {
    bool initialized = false;
    std::string command;
    
    std::cout << "Welcome to Medical Image Segmentation Tool" << std::endl;
    print_usage();
    
    while (true) {
        std::cout << "\n> ";
        std::getline(std::cin, command);
        
        // 解析命令
        std::istringstream iss(command);
        std::string cmd;
        iss >> cmd;
        
        if (cmd == "init") {
            std::string trt_cache_path;
            iss >> trt_cache_path;
            
            if (trt_cache_path.empty()) {
                std::cerr << "Error: Missing TRT cache path" << std::endl;
                continue;
            }
            
            const std::string log_dir = fs::path(trt_cache_path).parent_path().string() + "/../log";
            if (initialize_engine(trt_cache_path, log_dir)) {
                std::cout << "Engine initialized successfully" << std::endl;
                initialized = true;
            } else {
                std::cerr << "Engine initialization failed" << std::endl;
            }
        }
        else if (cmd == "process") {
            if (!initialized) {
                std::cerr << "Error: Engine not initialized" << std::endl;
                continue;
            }
            
            std::string raw_path;
            int width, height;
            std::string output_dir;
            
            iss >> raw_path >> width >> height;
            
            if (raw_path.empty() || !iss) {
                std::cerr << "Error: Invalid process command" << std::endl;
                continue;
            }
            
            // 默认输出目录
            if (output_dir.empty()) {
                output_dir = fs::path(raw_path).parent_path().string();
            }
            
            if (process_single_image(raw_path, width, height, output_dir)) {
                std::cout << "Processing completed" << std::endl;
            } else {
                std::cerr << "Processing failed" << std::endl;
            }
        }
        else if (cmd == "exit") {
            if (initialized) {
                cleanup_resources();
            }
            std::cout << "Exiting..." << std::endl;
            break;
        }
        else if (cmd == "help") {
            print_usage();
        }
        else if (!cmd.empty()) {
            std::cerr << "Unknown command: " << cmd << std::endl;
        }
    }
    
    return 0;
}