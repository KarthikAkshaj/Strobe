#pragma once
#include <vector>
#include <cstring>
#include <stdexcept>
#include "../core/math_utils.hpp"

namespace sentinel {

// Direct C++ port of python_tools/change_detector.py :: AnchorChangeDetector
class AnchorDetector {
public:
    AnchorDetector(float similarity_threshold = 0.85f,
                   int   hysteresis_count     = 2,
                   int   embedding_dim        = EMBEDDING_DIM)
        : threshold_(similarity_threshold)
        , hysteresis_(hysteresis_count)
        , dim_(embedding_dim)
        , consecutive_(0)
        , frame_count_(0)
        , trigger_count_(0)
    {
        anchor_.resize(dim_, 0.0f);
        has_anchor_ = false;
    }

    struct Result {
        bool  triggered;
        float similarity;
    };

    Result process(const float* embedding) {
        frame_count_++;

        // Validate (mirrors Python defensive check)
        validate_embedding(embedding, dim_);

        if (!has_anchor_) {
            std::memcpy(anchor_.data(), embedding, dim_ * sizeof(float));
            has_anchor_ = true;
            return {false, 1.0f};
        }

        float sim = cosine_similarity(embedding, anchor_.data(), dim_);
        bool  different = (sim < threshold_);

        if (different) {
            consecutive_++;
            if (consecutive_ >= hysteresis_) {
                // Confirmed scene change — anchor becomes current frame
                trigger_count_++;
                std::memcpy(anchor_.data(), embedding, dim_ * sizeof(float));
                consecutive_ = 0;
                return {true, sim};
            }
        } else {
            consecutive_ = 0;
        }

        return {false, sim};
    }

    void reset() {
        has_anchor_ = false;
        consecutive_ = 0;
        frame_count_ = 0;
        trigger_count_ = 0;
    }

    int frame_count()   const { return frame_count_; }
    int trigger_count() const { return trigger_count_; }

private:
    float             threshold_;
    int               hysteresis_;
    int               dim_;
    std::vector<float> anchor_;
    bool              has_anchor_;
    int               consecutive_;
    int               frame_count_;
    int               trigger_count_;
};

} // namespace sentinel
