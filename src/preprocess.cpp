#include "preprocess.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <stdexcept>
#include <limits>
#include <fstream>
#include "nlohmann/json.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#include <io.h>
#include <fcntl.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

// ---------- mmap 封装 ----------
class MMapFile {
public:
    MMapFile(const std::string& path, size_t size) : size_(size), data_(nullptr) {
#ifdef _WIN32
        fd_ = ::_open(path.c_str(), _O_RDONLY | _O_BINARY);
        if (fd_ == -1) throw std::runtime_error("open failed");
        data_ = ::MapViewOfFile(
            ::CreateFileMapping((HANDLE)_get_osfhandle(fd_), nullptr, PAGE_READONLY, 0, 0, nullptr),
            FILE_MAP_READ, 0, 0, size_);
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("open failed");
        data_ = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd_, 0);
#endif
    }
    ~MMapFile() {
#ifdef _WIN32
        if (data_) ::UnmapViewOfFile(data_);
        if (fd_ != -1) ::_close(fd_);
#else
        if (data_ && data_ != MAP_FAILED) ::munmap(data_, size_);
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    const uint16_t* data() const { return reinterpret_cast<const uint16_t*>(data_); }
private:
    size_t size_;
    void* data_;
#ifdef _WIN32
    int fd_ = -1;
#else
    int fd_ = -1;
#endif
};

namespace Preprocess {

static void compute_minmax(const uint16_t* src, size_t len, uint16_t& mn, uint16_t& mx) {
    mn = std::numeric_limits<uint16_t>::max();
    mx = std::numeric_limits<uint16_t>::min();
#pragma omp parallel reduction(min:mn) reduction(max:mx)
    for (int i = 0; i < static_cast<int>(len); ++i) {
        uint16_t v = src[i];
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
}

bool preprocess_raw(const std::string& raw_path,
                    const std::string& png_path,
                    const std::string& json_path,
                    int w, int h) {
    try {
        const int outW = 512, outH = 512;
        const double stepX = static_cast<double>(w) / outW;
        const double stepY = static_cast<double>(h) / outH;

        // mmap
        MMapFile file(raw_path, w * h * 2);
        const uint16_t* src = file.data();

        // min/max
        uint16_t mn, mx;
        compute_minmax(src, w * h, mn, mx);
        if (mn == mx) mx = mn + 1;
        const double scale8 = 255.0 / (mx - mn);

        // 512×512 8-bit
        cv::Mat dst8(outH, outW, CV_8UC1);
#pragma omp parallel for
        for (int y = 0; y < outH; ++y) {
            for (int x = 0; x < outW; ++x) {
                double fx = x * stepX, fy = y * stepY;
                int ix = static_cast<int>(fx);
                int iy = static_cast<int>(fy);
                int ix1 = std::min(ix + 1, w - 1);
                int iy1 = std::min(iy + 1, h - 1);
                double dx = fx - ix, dy = fy - iy;

                uint16_t v00 = src[iy * w + ix];
                uint16_t v01 = src[iy * w + ix1];
                uint16_t v10 = src[iy1 * w + ix];
                uint16_t v11 = src[iy1 * w + ix1];

                double v = (1 - dx) * (1 - dy) * v00 +
                           dx      * (1 - dy) * v01 +
                           (1 - dx) * dy      * v10 +
                           dx      * dy      * v11;
                dst8.at<uchar>(y, x) = static_cast<uchar>((v - mn) * scale8 + 0.5);
            }
        }

        // 写 PNG
        fs::create_directories(fs::path(png_path).parent_path());
        if (!cv::imwrite(png_path, dst8, {cv::IMWRITE_PNG_COMPRESSION, 0}))
            throw std::runtime_error("imwrite failed");

        // 写 JSON
        json j;
        j[fs::path(raw_path).filename().string()] = {
            {"original_width", w},
            {"original_height", h},
            {"scaled_width", outW},
            {"scaled_height", outH}
        };
        std::ofstream jf(json_path);
        jf << j << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "preprocess_raw error: " << e.what() << '\n';
        return false;
    }
}

}   // namespace Preprocess