#pragma once
#include <string>
#include <array>
#include <fstream>
#include <stdexcept>
#include "../../deps/json/nlohmann/json.hpp"

namespace sentinel {

struct Config {
    // Model
    std::string onnx_path        = "models/clip_visual_vit_b32.onnx";
    int         embedding_dim    = 512;
    int         input_size       = 224;

    // Caption bank
    std::string caption_bank_path = "models/caption_bank.json";

    // Pipeline
    float target_fps        = 1.0f;
    float change_threshold  = 0.85f;
    int   hysteresis_count  = 2;
    float caption_threshold = 0.20f;
    int   top_k             = 3;
    float stability_delta   = 0.02f;

    // CLIP preprocessing
    std::array<float, 3> mean = {0.48145466f, 0.4578275f,  0.40821073f};
    std::array<float, 3> std  = {0.26862954f, 0.26130258f, 0.27577711f};

    static Config from_json(const std::string& path);
    static Config defaults() { return Config{}; }
};

inline Config Config::from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open config: " + path);

    nlohmann::json j;
    f >> j;

    Config c;
    if (j.contains("model")) {
        auto& m = j["model"];
        if (m.contains("onnx_path"))     c.onnx_path     = m["onnx_path"].get<std::string>();
        if (m.contains("embedding_dim")) c.embedding_dim = m["embedding_dim"].get<int>();
        if (m.contains("input_size"))    c.input_size    = m["input_size"].get<int>();
    }
    if (j.contains("caption_bank_path"))
        c.caption_bank_path = j["caption_bank_path"].get<std::string>();

    if (j.contains("pipeline")) {
        auto& p = j["pipeline"];
        if (p.contains("target_fps"))        c.target_fps        = p["target_fps"].get<float>();
        if (p.contains("change_threshold"))  c.change_threshold  = p["change_threshold"].get<float>();
        if (p.contains("hysteresis_count"))  c.hysteresis_count  = p["hysteresis_count"].get<int>();
        if (p.contains("caption_threshold")) c.caption_threshold = p["caption_threshold"].get<float>();
        if (p.contains("top_k"))             c.top_k             = p["top_k"].get<int>();
        if (p.contains("stability_delta"))   c.stability_delta   = p["stability_delta"].get<float>();
    }
    if (j.contains("preprocessing")) {
        auto& pp = j["preprocessing"];
        if (pp.contains("mean") && pp["mean"].size() == 3)
            c.mean = {pp["mean"][0].get<float>(), pp["mean"][1].get<float>(), pp["mean"][2].get<float>()};
        if (pp.contains("std") && pp["std"].size() == 3)
            c.std = {pp["std"][0].get<float>(), pp["std"][1].get<float>(), pp["std"][2].get<float>()};
    }
    return c;
}

} // namespace sentinel
