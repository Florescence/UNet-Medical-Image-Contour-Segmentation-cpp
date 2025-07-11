cmake_minimum_required(VERSION 3.10)
project(SegmentationPipeline)

set(CMAKE_CXX_STANDARD 17)

# 手动指定OpenCV路径（替换为你的实际路径）
set(OpenCV_DIR "D:/PROGRA~1/OpenCV4.7/install_x64_release/share/OpenCV")

# 查找OpenCV和LibTorch
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# 添加头文件路径
include_directories(${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

# 添加所有源文件
file(GLOB_RECURSE SOURCES "src/*.cpp")

# 生成可执行文件
add_executable(seg_main ${SOURCES})

# 链接库
target_link_libraries(seg_main ${OpenCV_LIBS} ${TORCH_LIBRARIES})