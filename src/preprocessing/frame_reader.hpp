#pragma once
#include <cstdint>
#include <cstdio>
#include <vector>

// Binary frame protocol used between Python adapter and C++ engine.
//
// Each frame on stdin:
//   [0..3]   int32_t  frame_number
//   [4..11]  float64  timestamp_sec
//   [12..]   float32  tensor[3 * H * W]   (CHW, CLIP-normalized)
//
// EOF signals end of stream.

namespace sentinel {

#pragma pack(push, 1)
struct FrameHeader {
    int32_t  frame_number;
    double   timestamp_sec;  // float64, no padding
};
#pragma pack(pop)

static constexpr int FRAME_HW = 224;
static constexpr int FRAME_TENSOR_LEN = 3 * FRAME_HW * FRAME_HW; // 150528

class FrameReader {
public:
    explicit FrameReader(FILE* stream = stdin) : stream_(stream) {}

    // Read next frame. Returns false on EOF or error.
    bool read_next(FrameHeader& header, float* tensor) {
        if (std::fread(&header, sizeof(header), 1, stream_) != 1)
            return false;
        size_t n = std::fread(tensor, sizeof(float), FRAME_TENSOR_LEN, stream_);
        return (n == static_cast<size_t>(FRAME_TENSOR_LEN));
    }

private:
    FILE* stream_;
};

} // namespace sentinel
