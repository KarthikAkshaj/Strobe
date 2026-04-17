#pragma once
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <onnxruntime_cxx_api.h>
#include "../core/math_utils.hpp"

namespace sentinel {

// ONNX Runtime session wrapper for the CLIP visual encoder.
// Input:  (1, 3, 224, 224) float32 CHW, CLIP-normalized
// Output: (1, 512) float32 raw (NOT unit-normalized — we normalize after)
// Note: named ClipSession to avoid clashing with OrtSession typedef in onnxruntime_c_api.h
class ClipSession {
public:
    explicit ClipSession(const std::string& model_path, int input_size = 224);

    // Run inference on a single pre-processed frame tensor.
    // tensor: float32 array of length 3 * input_size * input_size, CHW layout.
    // out:    output float32 array of length EMBEDDING_DIM (512).
    // Returns L2-normalized embedding.
    void run(const float* tensor, float* out); // non-const: ORT Run() is non-const

    int input_size()    const { return input_size_; }
    int embedding_dim() const { return EMBEDDING_DIM; }

private:
    int input_size_;

    Ort::Env         env_;
    Ort::Session     session_{nullptr};
    Ort::MemoryInfo  mem_info_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
};

inline ClipSession::ClipSession(const std::string& model_path, int input_size)
    : input_size_(input_size)
    , env_(ORT_LOGGING_LEVEL_WARNING, "sentinel")
    , mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , input_shape_{1, 3, input_size, input_size}
    , output_shape_{1, EMBEDDING_DIM}
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
    std::wstring wpath(model_path.begin(), model_path.end());
    session_ = Ort::Session(env_, wpath.c_str(), opts);
#else
    session_ = Ort::Session(env_, model_path.c_str(), opts);
#endif
}

inline void ClipSession::run(const float* tensor, float* out) {
    int64_t num_input_elements = 3LL * input_size_ * input_size_;

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_,
        const_cast<float*>(tensor),
        static_cast<size_t>(num_input_elements),
        input_shape_.data(), input_shape_.size()
    );

    const char* input_names[]  = {"image"};
    const char* output_names[] = {"embedding"};

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,  &input_tensor, 1,
        output_names, 1
    );

    const float* raw = output_tensors[0].GetTensorData<float>();

    // Copy and L2-normalize (ONNX model output is NOT normalized)
    std::copy(raw, raw + EMBEDDING_DIM, out);
    l2_normalize(out, EMBEDDING_DIM);
}

} // namespace sentinel
