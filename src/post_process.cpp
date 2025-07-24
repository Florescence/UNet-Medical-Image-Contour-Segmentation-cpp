#include <opencv2/opencv.hpp>
#include <vector>

// --------------  可调参数  --------------
constexpr int   FOREGROUND_VALUE      = 2;
constexpr int   BACKGROUND_VALUE_0    = 0;
constexpr int   BACKGROUND_VALUE_1    = 1;
constexpr int   MORPH_KERNEL_SIZE     = 3;
constexpr float MIN_AREA_RATIO        = 0.03f;   // 面积比例，小于该比例的连通域将被删除
// ---------------------------------------

// 前景内部空洞填充
static void fill_holes_inside_foreground(cv::Mat& mask)
{
    CV_Assert(mask.type() == CV_8UC1);

    // 1. 生成二值前景掩码
    cv::Mat bin = (mask == FOREGROUND_VALUE);   // 0/255

    // 2. 求 inv = ~bin
    cv::Mat inv;
    cv::bitwise_not(bin, inv);                  // 0/255

    // 3. 以 inv 做连通域；标签 0 为背景
    cv::Mat labels, stats, centroids;
    int nc = cv::connectedComponentsWithStats(inv, labels, stats, centroids, 8);

    // 4. 只要某个连通域的 bbox 不接触图像边界 → 内部孔
    const int w = mask.cols, h = mask.rows;
    const int min_area = static_cast<int>(w * h * MIN_AREA_RATIO);

    for (int i = 1; i < nc; ++i) {
        const int* s = stats.ptr<int>(i);
        int left   = s[cv::CC_STAT_LEFT];
        int top    = s[cv::CC_STAT_TOP];
        int right  = left + s[cv::CC_STAT_WIDTH]  - 1;
        int bottom = top  + s[cv::CC_STAT_HEIGHT] - 1;
        int area   = s[cv::CC_STAT_AREA];

        if (left > 0 && top > 0 && right < w - 1 && bottom < h - 1 && area < min_area) {
            mask.setTo(FOREGROUND_VALUE, labels == i);   // 填孔
        }
    }
}

// 主后处理函数
cv::Mat postprocess_mask(const cv::Mat& src)
{
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    cv::Mat mask = src.clone();          // 避免修改原图

    // 1. 填孔
    fill_holes_inside_foreground(mask);

    // 2. 二值化 + 开运算
    cv::Mat bin = (mask == FOREGROUND_VALUE);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                               {MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE});
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, kernel);

    // 3. 连通域面积过滤
    cv::Mat labels, stats, centroids;
    int nc = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8);

    const int min_area = static_cast<int>(mask.cols * mask.rows * MIN_AREA_RATIO);

    cv::Mat keep(bin.size(), CV_8UC1, cv::Scalar(0));
    for (int i = 1; i < nc; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= min_area)
            keep.setTo(255, labels == i);
    }

    // 4. 映射回 0/2
    mask.setTo(0);                       // 全部先置 0
    mask.setTo(FOREGROUND_VALUE, keep);  // 保留区域置 2

    return mask;
}

// --------------------  可选测试  --------------------
/*
int main() {
    cv::Mat m = cv::imread("input_mask.png", cv::IMREAD_GRAYSCALE);
    cv::Mat out = postprocess_mask(m);
    cv::imwrite("output_mask.png", out);
    return 0;
}
*/