// seg_main.cpp（增强版 - 支持目录递归处理）
#include "initialize.h"
#include "process.h"
#include "cleanup.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>
#include <sstream>
#include <filesystem>
#include <vector>
#include <regex>

namespace fs = std::filesystem;
using namespace MedicalSeg;

// 检查文件是否为16位图像（基于扩展名）
bool is_16bit_image(const std::string& path) {
    static const std::vector<std::string> extensions = {
        ".raw", ".dcm", ".tif", ".tiff"
    };
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(extensions.begin(), extensions.end(), ext) != extensions.end();
}

// 递归查找目录中的所有16位图像文件
std::vector<std::string> find_16bit_images(const std::string& dir_path, bool recursive) {
    std::vector<std::string> result;
    try {
        if (recursive) {
            for (const auto& entry : fs::recursive_directory_iterator(dir_path)) {
                if (entry.is_regular_file() && is_16bit_image(entry.path().string())) {
                    result.push_back(entry.path().string());
                }
            }
        } else {
            for (const auto& entry : fs::directory_iterator(dir_path)) {
                if (entry.is_regular_file() && is_16bit_image(entry.path().string())) {
                    result.push_back(entry.path().string());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Directory error: " << e.what() << std::endl;
    }
    return result;
}

// 打印使用说明
void print_usage() {
    std::cout << "\nMedical Image Segmentation Tool (TensorRT)" << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  init <onnx_model_path>         - Initialize ONNX Runtime engine" << std::endl;
    std::cout << "  process [-r] <input> <width> <height> [output_dir] - Process file/directory" << std::endl;
    std::cout << "  exit                          - Cleanup and exit" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  -r                            - Recursively process directory" << std::endl;
    std::cout << "  <input>                       - Path to image file or directory" << std::endl;
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
            std::string onnx_model_path;
            iss >> onnx_model_path;
            
            if (onnx_model_path.empty()) {
                std::cerr << "Error: Missing TRT cache path" << std::endl;
                continue;
            }
            
            const std::string log_dir = fs::path(onnx_model_path).parent_path().string() + "/bin/log";
            if (initialize_onnx_runtime(onnx_model_path, log_dir)) {
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
            
            bool recursive = false;
            std::string input_path;
            int width, height;
            std::string output_dir;
            
            // 解析-r选项
            std::string arg;
            iss >> arg;
            if (arg == "-r") {
                recursive = true;
                iss >> input_path;
            } else {
                input_path = arg;
            }
            
            iss >> width >> height;
            
            if (input_path.empty() || !iss) {
                std::cerr << "Error: Invalid process command" << std::endl;
                continue;
            }
            
            // 解析输出目录（可选）
            iss >> output_dir;
            if (output_dir.empty()) {
                output_dir = fs::path(input_path).parent_path().string();
            }
            
            // 确保输出目录存在
            fs::create_directories(output_dir);
            
            // 处理单个文件或目录
            try {
                if (fs::is_directory(input_path)) {
                    std::cout << "Processing directory: " << input_path << std::endl;
                    std::cout << "Recursive: " << (recursive ? "Yes" : "No") << std::endl;
                    
                    auto files = find_16bit_images(input_path, recursive);
                    if (files.empty()) {
                        std::cerr << "No 16-bit images found in directory" << std::endl;
                        continue;
                    }
                    
                    std::cout << "Found " << files.size() << " images to process" << std::endl;
                    int success_count = 0;
                    int fail_count = 0;
                    
                    for (const auto& file : files) {
                        // 为每个文件创建单独的输出子目录
                        std::string file_output_dir = output_dir;
                        if (recursive) {
                            // 保留目录结构
                            std::string rel_path = fs::relative(file, input_path).parent_path().string();
                            file_output_dir = (fs::path(output_dir) / rel_path).string();
                            fs::create_directories(file_output_dir);
                        }
                        
                        std::cout << "\nProcessing: " << file << std::endl;
                        if (process_single_image(file, width, height, file_output_dir)) {
                            success_count++;
                        } else {
                            fail_count++;
                        }
                    }
                    
                    std::cout << "\nDirectory processing completed:" << std::endl;
                    std::cout << "  Success: " << success_count << " files" << std::endl;
                    std::cout << "  Failed: " << fail_count << " files" << std::endl;
                } else if (fs::is_regular_file(input_path)) {
                    std::cout << "Processing file: " << input_path << std::endl;
                    if (process_single_image(input_path, width, height, output_dir)) {
                        std::cout << "Processing completed" << std::endl;
                    } else {
                        std::cerr << "Processing failed" << std::endl;
                    }
                } else {
                    std::cerr << "Error: Input path is not a valid file or directory" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Processing error: " << e.what() << std::endl;
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