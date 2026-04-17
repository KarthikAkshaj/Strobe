#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include "../core/math_utils.hpp"
#include "../../deps/json/nlohmann/json.hpp"

namespace sentinel {

struct TopKResult {
    std::string caption;
    float       score;
};

class CaptionBank {
public:
    explicit CaptionBank(const std::string& path);

    // Returns best caption + confidence + top-K alternatives
    TopKResult top1(const float* embedding, int dim) const;
    std::vector<TopKResult> topk(const float* embedding, int dim, int k) const;

    int  size()     const { return static_cast<int>(captions_.size()); }
    bool empty()    const { return captions_.empty(); }

private:
    std::vector<std::string>        captions_;
    std::vector<std::vector<float>> embeddings_; // each is unit-normalized, length=512
};

inline CaptionBank::CaptionBank(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open caption bank: " + path);

    nlohmann::json j;
    f >> j;

    auto& entries = j["captions"];
    captions_.reserve(entries.size());
    embeddings_.reserve(entries.size());

    for (auto& e : entries) {
        captions_.push_back(e["text"].get<std::string>());
        auto vec = e["embedding"].get<std::vector<float>>();
        embeddings_.push_back(std::move(vec));
    }
}

inline std::vector<TopKResult> CaptionBank::topk(
    const float* embedding, int dim, int k) const
{
    int n = size();
    if (n == 0) return {};

    std::vector<TopKResult> results;
    results.reserve(n);
    for (int i = 0; i < n; i++) {
        float score = cosine_similarity(embedding, embeddings_[i].data(), dim);
        results.push_back({captions_[i], score});
    }

    // Partial sort: get top-k by score descending
    int actual_k = std::min(k, n);
    std::partial_sort(
        results.begin(), results.begin() + actual_k, results.end(),
        [](const TopKResult& a, const TopKResult& b) { return a.score > b.score; }
    );
    results.resize(actual_k);
    return results;
}

inline TopKResult CaptionBank::top1(const float* embedding, int dim) const {
    auto k1 = topk(embedding, dim, 1);
    if (k1.empty()) return {"", 0.0f};
    return k1[0];
}

} // namespace sentinel
