#pragma once
#include <string>

namespace Preprocess {

/**
 * 一步完成
 * 1. mmap 读取 16-bit RAW
 * 2. 计算真实 min/max
 * 3. 双线性降采样到 512×512 + 8-bit 归一化
 * 4. 写出 512×512 PNG
 * 5. 写出 {原始尺寸, 512×512} 的 JSON
 *
 * @param raw_path      输入 *.raw
 * @param png_path      输出 512×512 png
 * @param json_path     输出 *.json
 * @param w, h          RAW 宽高
 * @return              成功 true / 失败 false
 */
bool preprocess_raw(const std::string& raw_path,
                    const std::string& png_path,
                    const std::string& json_path,
                    int w, int h);

}   // namespace Preprocess