#pragma once
#include <cmath>
#include <stdexcept>
#include <string>

namespace sentinel {

static constexpr int EMBEDDING_DIM = 512;

inline float l2_norm(const float* v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return std::sqrt(sum);
}

inline void l2_normalize(float* v, int n) {
    float norm = l2_norm(v, n);
    if (norm > 0.0f)
        for (int i = 0; i < n; i++) v[i] /= norm;
}

// For unit vectors, cosine similarity == dot product.
// We compute the full formula for robustness.
inline float cosine_similarity(const float* a, const float* b, int n) {
    float dot = 0.0f;
    float na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return (denom > 0.0f) ? (dot / denom) : 0.0f;
}

inline void validate_embedding(const float* v, int n, float lo = 0.9f, float hi = 1.1f) {
    float norm = l2_norm(v, n);
    if (norm < lo || norm > hi) {
        throw std::runtime_error(
            "Embedding norm " + std::to_string(norm) +
            " outside valid range [" + std::to_string(lo) + ", " + std::to_string(hi) + "]. "
            "CLIP embeddings must be unit-normalized."
        );
    }
}

} // namespace sentinel
