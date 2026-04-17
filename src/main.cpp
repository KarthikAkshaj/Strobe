// Semantic Sentinel - C++ Runtime Engine (Phase 3)
// Reads pre-processed CLIP-normalized frames from stdin, runs ONNX inference,
// anchor-based change detection, and caption matching.
//
// Usage:
//   python python_tools/encode_video_frames.py <video.mp4> | sentinel_engine [options]
//
// Options:
//   --config  <path>   Path to config/default.json (default: config/default.json)
//   --model   <path>   Override ONNX model path
//   --captions <path>  Override caption bank JSON path
//   --output  <path>   Output JSON file (default: stdout)
//   --threshold <f>    Change threshold override
//   --fps <f>          Target FPS (informational only; frame selection done by Python)

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "core/math_utils.hpp"
#include "utils/config.hpp"
#include "caption_bank/caption_bank.hpp"
#include "change_detection/anchor_detector.hpp"
#include "inference/ort_session.hpp"
#include "preprocessing/frame_reader.hpp"
#include "../deps/json/nlohmann/json.hpp"

using namespace sentinel;
using json = nlohmann::json;

// ─── Caption event ──────────────────────────────────────────────────────────

struct CaptionEvent {
    int32_t frame_number;
    double  timestamp;
    std::string caption;
    float   confidence;
    float   confidence_gap;
    std::vector<TopKResult> alternatives; // top_k minus the winner
    float   change_similarity;
};

// ─── Stability state ─────────────────────────────────────────────────────────

struct StabilityState {
    std::string prev_caption;
    float       prev_score = 0.0f;
    bool        has_prev   = false;
};

// Exact port of process_video.py stability logic.
// Returns the selected (caption, score) after applying the stability constraint.
// Updates state. Always returns something (may be prev caption if suppressed).
std::pair<std::string, float> apply_stability(
    const std::vector<TopKResult>& topk,
    StabilityState& state,
    float delta)
{
    if (topk.empty()) return {"", 0.0f};
    const std::string& top1_text  = topk[0].caption;
    float              top1_score = topk[0].score;

    std::string selected_text  = top1_text;
    float       selected_score = top1_score;

    if (state.has_prev) {
        float score_improvement = top1_score - state.prev_score;
        if (score_improvement < delta) {
            // Keep previous caption (Python: selected_text = prev, selected_score = prev)
            selected_text  = state.prev_caption;
            selected_score = state.prev_score;
        }
    }

    // Always update stability state (mirrors Python)
    state.prev_caption = selected_text;
    state.prev_score   = selected_score;
    state.has_prev     = true;

    return {selected_text, selected_score};
}

// ─── Argument parsing ────────────────────────────────────────────────────────

struct Args {
    std::string config_path   = "config/default.json";
    std::string model_path;
    std::string captions_path;
    std::string output_path;
    float change_threshold    = -1.0f;
    float caption_threshold   = -1.0f;
    float stability_delta     = -1.0f;
    int   hysteresis          = -1;
    int   top_k               = -1;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];
        if ((key == "--config"    || key == "-c") && i+1 < argc) a.config_path     = argv[++i];
        else if (key == "--model"  && i+1 < argc)                 a.model_path      = argv[++i];
        else if (key == "--captions" && i+1 < argc)               a.captions_path   = argv[++i];
        else if (key == "--output" && i+1 < argc)                  a.output_path     = argv[++i];
        else if (key == "--threshold" && i+1 < argc)               a.change_threshold = std::stof(argv[++i]);
        else if (key == "--caption-threshold" && i+1 < argc)       a.caption_threshold= std::stof(argv[++i]);
        else if (key == "--stability-delta" && i+1 < argc)         a.stability_delta  = std::stof(argv[++i]);
        else if (key == "--hysteresis" && i+1 < argc)              a.hysteresis       = std::stoi(argv[++i]);
        else if (key == "--top-k" && i+1 < argc)                   a.top_k            = std::stoi(argv[++i]);
    }
    return a;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
#ifdef _WIN32
    // Set stdin to binary mode on Windows (critical for binary protocol)
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    Args args = parse_args(argc, argv);

    // Load config
    Config cfg;
    try {
        cfg = Config::from_json(args.config_path);
    } catch (...) {
        std::cerr << "[sentinel] Config not found, using defaults\n";
    }

    // Apply CLI overrides
    if (!args.model_path.empty())     cfg.onnx_path          = args.model_path;
    if (!args.captions_path.empty())  cfg.caption_bank_path  = args.captions_path;
    if (args.change_threshold >= 0)   cfg.change_threshold   = args.change_threshold;
    if (args.caption_threshold >= 0)  cfg.caption_threshold  = args.caption_threshold;
    if (args.stability_delta >= 0)    cfg.stability_delta    = args.stability_delta;
    if (args.hysteresis >= 0)         cfg.hysteresis_count   = args.hysteresis;
    if (args.top_k > 0)               cfg.top_k              = args.top_k;

    // Print config summary to stderr
    std::cerr << "============================================================\n";
    std::cerr << "Semantic Sentinel - C++ Runtime Engine\n";
    std::cerr << "============================================================\n";
    std::cerr << "  Model:      " << cfg.onnx_path         << "\n";
    std::cerr << "  Captions:   " << cfg.caption_bank_path << "\n";
    std::cerr << "  Threshold:  " << cfg.change_threshold  << "\n";
    std::cerr << "  Hysteresis: " << cfg.hysteresis_count  << "\n";
    std::cerr << "  Top-K:      " << cfg.top_k             << "\n\n";

    // Load components
    std::cerr << "Loading ONNX model...\n";
    ClipSession ort(cfg.onnx_path, cfg.input_size);

    std::cerr << "Loading caption bank...\n";
    CaptionBank bank(cfg.caption_bank_path);
    std::cerr << "  Loaded " << bank.size() << " captions\n\n";

    AnchorDetector detector(cfg.change_threshold, cfg.hysteresis_count, cfg.embedding_dim);
    StabilityState stability;
    FrameReader reader;

    // Buffers
    std::vector<float> tensor(FRAME_TENSOR_LEN);
    std::vector<float> embedding(cfg.embedding_dim);
    FrameHeader header{};

    std::vector<CaptionEvent> events;
    int frames_processed = 0;

    auto t_start = std::chrono::steady_clock::now();

    std::cerr << "Processing frames...\n";

    while (reader.read_next(header, tensor.data())) {
        frames_processed++;

        // ONNX inference → L2-normalized embedding
        ort.run(tensor.data(), embedding.data());

        // Change detection
        auto result = detector.process(embedding.data());

        if (!result.triggered) continue;

        // Caption matching (top-K)
        auto topk = bank.topk(embedding.data(), cfg.embedding_dim, cfg.top_k);
        if (topk.empty()) continue;

        float conf_gap = (topk.size() >= 2) ? (topk[0].score - topk[1].score) : topk[0].score;

        // Stability constraint: returns selected (caption, score), always updates state
        auto [sel_caption, sel_score] = apply_stability(topk, stability, cfg.stability_delta);

        // Only emit if selected score meets threshold (mirrors Python)
        if (sel_score < cfg.caption_threshold) continue;

        CaptionEvent ev;
        ev.frame_number    = header.frame_number;
        ev.timestamp       = header.timestamp_sec;
        ev.caption         = sel_caption;
        ev.confidence      = sel_score;
        ev.confidence_gap  = conf_gap;
        ev.change_similarity = result.similarity;
        for (size_t i = 1; i < topk.size(); i++)
            ev.alternatives.push_back(topk[i]);

        events.push_back(std::move(ev));

        std::cerr << "  [" << header.frame_number << " @ " << header.timestamp_sec << "s] "
                  << events.back().caption << " (conf=" << sel_score << ")\n";
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Build output JSON
    json out;
    out["metadata"] = {
        {"engine", "c++"},
        {"model", cfg.onnx_path},
        {"change_threshold", cfg.change_threshold},
        {"hysteresis_count", cfg.hysteresis_count},
        {"caption_threshold", cfg.caption_threshold},
        {"top_k", cfg.top_k},
        {"stability_delta", cfg.stability_delta},
        {"frames_processed", frames_processed},
        {"processing_ms", elapsed_ms}
    };

    json caps_arr = json::array();
    for (auto& ev : events) {
        json alt_arr = json::array();
        for (auto& alt : ev.alternatives)
            alt_arr.push_back({{"text", alt.caption}, {"score", alt.score}});

        caps_arr.push_back({
            {"frame",            ev.frame_number},
            {"timestamp",        ev.timestamp},
            {"caption",          ev.caption},
            {"confidence",       ev.confidence},
            {"confidence_gap",   ev.confidence_gap},
            {"alternatives",     alt_arr},
            {"change_similarity",ev.change_similarity}
        });
    }
    out["captions"] = caps_arr;

    std::cerr << "\n============================================================\n";
    std::cerr << "Done. frames=" << frames_processed
              << "  events=" << events.size()
              << "  time=" << elapsed_ms << "ms\n";
    std::cerr << "============================================================\n";

    // Write output
    std::string json_str = out.dump(2);
    if (!args.output_path.empty()) {
        std::ofstream f(args.output_path);
        f << json_str << "\n";
        std::cerr << "Output written to: " << args.output_path << "\n";
    } else {
        // Restore stdout to text mode on Windows before writing JSON
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_TEXT);
#endif
        std::cout << json_str << "\n";
    }

    return 0;
}
