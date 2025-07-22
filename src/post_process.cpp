#include <iostream>
#include <opencv2/opencv.hpp>

// 常量定义（与Python版本保持一致）
const int FOREGROUND_VALUE = 2;       // 前景像素值
const int BACKGROUND_VALUE_0 = 0;     // 背景像素值0
const int BACKGROUND_VALUE_1 = 1;     // 背景像素值1
const int MIN_AREA = 512 * 512 * 0.03;           // 最小连通域面积
const int MORPH_KERNEL_SIZE = 3;      // 形态学操作核大小

/**
 * 去除前景区域内部的非前景连通域
 * 对应Python的remove_internal_regions函数
 */
cv::Mat remove_internal_regions(const cv::Mat& mask) {
    // 创建掩码副本，避免修改原图
    cv::Mat processed_mask = mask.clone();

    // 二值化：前景(2)设为255，其他设为0
    cv::Mat binary_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (int y = 0; y < mask.rows; ++y) {
        const uchar* mask_row = mask.ptr<uchar>(y);
        uchar* bin_row = binary_mask.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (mask_row[x] == FOREGROUND_VALUE) {
                bin_row[x] = 255;
            }
        }
    }

    // 寻找前景轮廓（只保留最外层轮廓）
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 遍历每个前景轮廓
    for (const auto& contour : contours) {
        // 创建轮廓掩码（填充轮廓内部）
        cv::Mat contour_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::drawContours(contour_mask, std::vector<std::vector<cv::Point>>{contour}, -1, 255, -1);

        // 标记轮廓内部的非前景区域
        for (int y = 0; y < mask.rows; ++y) {
            const uchar* mask_row = mask.ptr<uchar>(y);
            const uchar* cnt_row = contour_mask.ptr<uchar>(y);
            uchar* proc_row = processed_mask.ptr<uchar>(y);
            for (int x = 0; x < mask.cols; ++x) {
                // 轮廓内部且为非前景像素（0或1）
                if (cnt_row[x] == 255 && mask_row[x] != FOREGROUND_VALUE) {
                    proc_row[x] = FOREGROUND_VALUE;  // 转为前景
                }
            }
        }
    }

    return processed_mask;
}

/**
 * 完整的掩码后处理流程
 */
cv::Mat postprocess_mask(const cv::Mat& mask) {
    // 1. 去除前景内部的非前景区域
    cv::Mat mask1 = remove_internal_regions(mask);

    // 2. 二值化（只保留前景值2）
    cv::Mat binary_mask = cv::Mat::zeros(mask1.size(), CV_8UC1);
    for (int y = 0; y < mask1.rows; ++y) {
        const uchar* row = mask1.ptr<uchar>(y);
        uchar* bin_row = binary_mask.ptr<uchar>(y);
        for (int x = 0; x < mask1.cols; ++x) {
            if (row[x] == FOREGROUND_VALUE) {
                bin_row[x] = 255;
            }
        }
    }

    // 3. 形态学开运算（去除噪点）
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, 
        cv::Size(MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    );
    cv::Mat opened_mask;
    cv::morphologyEx(binary_mask, opened_mask, cv::MORPH_OPEN, kernel);

    // 4. 连通域分析（去除小面积区域）
    int num_labels;
    cv::Mat labels, stats, centroids;
    num_labels = cv::connectedComponentsWithStats(
        opened_mask, 
        labels, 
        stats, 
        centroids, 
        8  // 8连通域
    );

    cv::Mat processed_binary = cv::Mat::zeros(opened_mask.size(), CV_8UC1);
    for (int i = 1; i < num_labels; ++i) {  // 跳过背景（标签0）
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= MIN_AREA) {
            // 保留大面积连通域
            processed_binary.setTo(255, labels == i);
        }
    }

    // 5. 转回[0,1,2]格式
    cv::Mat processed_mask = mask1.clone();
    for (int y = 0; y < processed_mask.rows; ++y) {
        uchar* proc_row = processed_mask.ptr<uchar>(y);
        const uchar* bin_row = processed_binary.ptr<uchar>(y);
        for (int x = 0; x < processed_mask.cols; ++x) {
            if (bin_row[x] == 0) {
                proc_row[x] = 0;  // 移除的前景区域设为0
            } else {
                proc_row[x] = 2;  // 保留的前景区域设为2
            }
        }
    }

    return processed_mask;
}

// // 测试主函数（硬编码参数）
// int main() {
//     // 硬编码测试参数（根据实际需求修改）
//     const std::string input_mask_path = "input_mask.png";    // 输入掩码路径
//     const std::string output_mask_path = "output_mask.png";  // 输出处理后掩码路径

//     try {
//         // 读取输入掩码（单通道，值为0/1/2）
//         cv::Mat mask = cv::imread(input_mask_path, cv::IMREAD_GRAYSCALE);
//         if (mask.empty()) {
//             throw std::runtime_error("无法读取输入掩码: " + input_mask_path);
//         }

//         // 执行后处理
//         cv::Mat processed_mask = postprocess_mask(mask);

//         // 保存结果
//         cv::imwrite(output_mask_path, processed_mask);
//         std::cout << "后处理完成，结果保存至: " << output_mask_path << std::endl;

//     } catch (const std::exception& e) {
//         std::cerr << "处理失败: " << e.what() << std::endl;
//         return 1;
//     }

//     return 0;
// }