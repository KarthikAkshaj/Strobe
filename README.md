# Semantic Sentinel

An AI system that watches a video, detects when the scene changes, and automatically describes what's happening — then lets you search through any video using plain English.

---

## What it does

Give it a video. It figures out when the scene meaningfully changes (not just motion blur or lighting — actual semantic changes), writes a natural language description of each scene, and outputs timestamps. You can also type a query like *"find me the part with a dog"* and it finds the exact frame.

```
Input:  video.mp4
Output: [5.0s]  "a man sitting on the side of a mountain"
        [9.0s]  "a man walking down a path in the mountains"
        [11.0s] "a woman sitting in the grass with a backpack"
```

---

## How it works

Two AI models work together:

**CLIP (ViT-B/32)** — handles the *when*. It converts every frame into a 512-dimensional semantic embedding. When the embedding drifts far enough from the current scene's anchor, it triggers a caption event. This is fast and runs at 1 FPS by default.

**BLIP** — handles the *what*. When CLIP detects a scene change, BLIP looks at that frame and writes a free-form English description. Unlike a fixed label classifier, BLIP generates natural language — it can describe anything it sees.

For semantic search, CLIP's text encoder converts your query into the same 512D space as the frame embeddings, making text-to-frame matching possible with a single dot product.

```
Video frames
    │
    ▼
[CLIP visual encoder]  ←── runs on every frame at target FPS
    │
    ├── Anchor-based change detection
    │       └── triggers when cosine similarity < threshold (0.85)
    │
    └── On trigger → [BLIP captioning]
                          └── generates free-form description
                                    │
                          ┌─────────┼──────────────┐
                          ▼         ▼               ▼
                      results.json  video.srt   embeddings.npz
                          │                         │
                      Web Viewer              text query search
```

---

## AI domains covered

- **Computer Vision** — frame extraction, preprocessing (resize, normalize, CHW conversion)
- **Vision-Language Models** — CLIP zero-shot semantic embeddings
- **Image Captioning** — BLIP generative free-form descriptions
- **Information Retrieval** — CLIP text encoder for semantic video search
- **Video Understanding** — temporal change detection, scene segmentation

---

## Setup

### Prerequisites

- Python 3.10+
- `pip install torch clip transformers opencv-python Pillow numpy`
- For CLIP: `pip install git+https://github.com/openai/CLIP.git`
- For the C++ engine: MinGW-w64, CMake 3.20+ (Windows)

### Models

Download the CLIP ONNX model (needed for C++ engine only):
```bash
python python_tools/export_onnx.py --output models/clip_visual_vit_b32.onnx
```

BLIP downloads automatically from HuggingFace on first use (~945MB for base, ~1.8GB for large).

### Build C++ engine (optional)

```bash
python setup_cpp_deps.py          # downloads ONNX Runtime + nlohmann/json
cmake -B build -G "MinGW Makefiles"
cmake --build build
```

---

## Usage

### Standard pipeline (Python, recommended)

```bash
# 1. Detect scene changes and generate CLIP captions
python python_tools/process_video.py video.mp4 \
    --caption-bank models/caption_bank.json \
    --output raw.json

# 2. Enhance with BLIP generative captioning
python python_tools/enhance_captions.py video.mp4 \
    --input raw.json \
    --output enhanced.json

# 3. Export as subtitles
python python_tools/export_subtitles.py \
    --input enhanced.json \
    --format srt \
    --output video.srt
```

### C++ engine pipeline (faster inference)

```bash
python python_tools/encode_video_frames.py video.mp4 \
    | ./build/sentinel_engine.exe \
        --config config/default.json \
        --captions models/caption_bank.json \
        --output raw.json
```

### Semantic video search

```bash
# Extract embeddings for all frames
python python_tools/extract_embeddings.py video.mp4 \
    --output video.embeddings.npz

# Search by text query
python python_tools/search_video.py \
    --query "sunset over water" \
    --embeddings video.embeddings.npz \
    --top-n 10 \
    --output search_results.json
```

### Web viewer

Open `web/index.html` in any browser — no server needed.

Drop in your video file and results JSON. Optionally drop a search results JSON to see query matches highlighted on the timeline.

![Viewer shows video with caption overlay, timeline with detection markers, and sidebar with confidence metrics]

---

## Configuration

All pipeline parameters live in `config/default.json`:

| Parameter | Default | Description |
|---|---|---|
| `target_fps` | 1.0 | Frames per second to analyze |
| `change_threshold` | 0.85 | Cosine similarity drop that triggers a new caption |
| `hysteresis_count` | 2 | Consecutive frames needed to confirm a scene change |
| `caption_threshold` | 0.20 | Minimum confidence to emit a caption |
| `top_k` | 3 | Number of alternative captions to include |
| `stability_delta` | 0.02 | Minimum score improvement to switch captions |

Override any parameter at runtime:
```bash
python python_tools/process_video.py video.mp4 \
    --fps 2 --change-threshold 0.80 --caption-bank models/caption_bank.json
```

---

## Output format

```json
{
  "metadata": {
    "model": "ViT-B/32",
    "change_threshold": 0.85,
    "frames_processed": 18,
    "blip_model": "blip-image-captioning-base"
  },
  "captions": [
    {
      "frame": 150,
      "timestamp": 5.0,
      "caption": "a man sitting on the side of a mountain",
      "confidence": 0.2904,
      "confidence_gap": 0.016,
      "change_similarity": 0.823,
      "caption_source": "blip",
      "clip_caption": "a mountain landscape",
      "alternatives": [
        { "text": "a mountain landscape", "score": 0.252 },
        { "text": "a forest path", "score": 0.236 }
      ]
    }
  ]
}
```

---

## Project structure

```
├── python_tools/
│   ├── process_video.py          # Main Python pipeline (change detection + captioning)
│   ├── enhance_captions.py       # BLIP generative captioning post-processor
│   ├── extract_embeddings.py     # Save per-frame CLIP embeddings for search
│   ├── search_video.py           # Semantic text-to-frame search
│   ├── export_subtitles.py       # SRT / WebVTT subtitle export
│   ├── export_onnx.py            # Export CLIP visual encoder to ONNX
│   ├── create_caption_bank.py    # Build caption bank with CLIP text embeddings
│   ├── encode_video_frames.py    # Binary frame encoder for C++ engine
│   └── captions/default_v2.txt  # 55-caption taxonomy (3 tiers)
├── src/                          # C++ runtime engine
│   ├── main.cpp
│   ├── core/math_utils.hpp       # L2 norm, cosine similarity
│   ├── inference/ort_session.hpp # ONNX Runtime wrapper
│   ├── caption_bank/             # Top-K caption matching
│   ├── change_detection/         # Anchor-based change detector
│   ├── preprocessing/            # Binary pipe frame reader
│   └── utils/config.hpp          # JSON config loader
├── web/                          # Static HTML viewer (no server needed)
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── models/
│   └── caption_bank.json         # Pre-encoded 55-caption bank (512D embeddings)
├── config/default.json           # Pipeline configuration
├── tests/test_cpp_parity.py      # C++ vs Python parity test (3/3 pass)
└── validation/                   # Validation suite and scripts
```

---

## Technical notes

- **No averaging of embeddings** — CLIP embeddings lie on a hypersphere; averaging distorts geometry. The anchor is always a real frame embedding.
- **Binary pipe protocol** — Python handles video decoding and CLIP preprocessing; C++ reads raw float tensors over stdin, avoiding OpenCV C++ SDK dependency.
- **Struct packing** — `FrameHeader` uses `#pragma pack(push,1)` so `{int32, double}` reads as 12 bytes matching Python's `struct.pack('<id', ...)`.
- **L2 normalization** — `clip.encode_image()` and `encode_text()` do not normalize; normalization is applied post-inference so dot product equals cosine similarity.
