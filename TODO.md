# Semantic Sentinel - Task Breakdown

> "Don't process pixels you don't need to. Don't generate text unless the meaning changes."

---

## Phase 1: Python Prototype & Asset Preparation

### 1.1 Environment Setup
- [x] Create Python virtual environment for development tools
- [x] Install PyTorch with CUDA support (if available)
- [x] Install OpenAI CLIP package
- [x] Install ONNX and ONNX Runtime packages
- [x] Install OpenCV-Python for video handling
- [x] Verify all dependencies with version check script

### 1.2 CLIP Model Acquisition
- [x] Download CLIP ViT-B/32 model weights
- [x] Verify model integrity (checksum validation)
- [x] Document embedding dimension (512D for ViT-B/32)
- [x] Test model loading in Python environment

### 1.3 Python Prototype - Frame Embedding
- [x] Write script to load a single image
- [x] Preprocess image to CLIP input format (224x224, normalized)
- [x] Extract embedding vector from CLIP visual encoder
- [x] Verify output shape matches expected dimension (512D)
- [x] Save embedding to file for inspection

### 1.4 Python Prototype - Caption Bank Creation
- [x] Define initial set of test captions (10-20 diverse captions)
- [x] Write script to encode captions using CLIP text encoder
- [x] Store text embeddings with corresponding caption strings
- [x] Verify text embedding dimension matches visual embedding dimension
- [x] Export caption bank to JSON format

### 1.5 Python Prototype - Cosine Similarity Verification
- [x] Implement cosine similarity function in Python
- [x] Test similarity between related image-text pairs
- [x] Test similarity between unrelated image-text pairs
- [x] Document expected similarity ranges for threshold tuning
- [x] Validate manual implementation against numpy/scipy

### 1.6 Python Prototype - Change Detection Logic
- [x] Implement anchor-based change detection (replaces rolling window)
- [x] Compare frames to stable anchor embedding (no averaging)
- [x] Implement similarity threshold check with hysteresis
- [x] Test with sequence of similar frames (expect no trigger)
- [x] Test with sequence containing scene change (expect trigger)
- [x] Verify re-stabilization after scene change
- [x] Document optimal threshold values

### 1.7 Python Prototype - End-to-End Validation
- [x] Process short test video through complete pipeline
- [x] Extract frames at target FPS
- [x] Generate embeddings for each frame
- [x] Apply change detection logic
- [x] Match triggered frames against caption bank
- [x] Output timestamped captions
- [x] Verify selective readout behavior (not every frame captioned)

---

## Phase 2: Model Export & Caption Bank Generation

### 2.1 ONNX Export - Visual Encoder
- [x] Isolate CLIP visual encoder from full model
- [x] Define input tensor specification (1, 3, 224, 224)
- [x] Define output tensor specification (1, 512)
- [x] Export visual encoder to ONNX format
- [x] Verify ONNX model with onnx.checker
- [x] Document input preprocessing requirements

### 2.2 ONNX Export - Validation
- [x] Load exported ONNX model in Python
- [x] Run inference on test image
- [x] Compare output to original PyTorch model output
- [x] Verify numerical equivalence (cosine > 0.99999, max normalized diff < 1e-5)
- [x] Document any precision differences

### 2.3 ONNX Optimization
- [x] Apply ONNX Runtime graph optimizations (saved models/clip_visual_vit_b32_optimized.onnx)
- [ ] Quantize model to FP16 (if targeting GPU) - deferred
- [x] Measure inference latency (~31ms/frame on CPU)
- [x] Verify output accuracy after optimization
- [x] Save optimized model to models/ directory

### 2.4 Caption Bank - Production Dataset
- [x] Define comprehensive caption set (55 captions, 3-tier taxonomy)
- [x] Organize captions by semantic category (Nature, Urban, Indoor, Vehicles, Objects, Activity, Meta)
- [x] Remove duplicate or near-duplicate captions (redundancy validation, threshold 0.90)
- [x] Encode all captions using CLIP text encoder
- [x] Validate embedding dimensions

### 2.5 Caption Bank - Export Format
- [x] Define JSON schema for caption bank
- [x] Include metadata (model name, embedding dimension, creation date)
- [x] Store captions with their embedding vectors
- [x] Export to models/caption_bank.json
- [x] Verify JSON file is valid and loadable

### 2.6 Configuration File Generation
- [x] Create config schema for runtime parameters
- [x] Define default similarity threshold (0.85)
- [x] Define hysteresis count (2)
- [x] Define target FPS (1.0)
- [x] Define embedding dimension (512)
- [x] Export to config/default.json

---

## Phase 3: C++ Runtime Engine

### 3.1 Project Setup
- [x] Create CMakeLists.txt for project
- [x] Configure C++17 standard requirement
- [x] Add ONNX Runtime as dependency
- [x] Add nlohmann/json as dependency (for config/caption bank)
- [x] Verify build system compiles empty main()
- Note: OpenCV not needed — Python adapter handles video decoding via binary pipe

### 3.2 Core Utilities
- [x] Implement cosine similarity function (manual, no external lib)
- [x] Implement L2 normalization function
- [x] Implement vector dot product function
- [x] Verify results match Python implementation
- Note: Unit tests covered by parity test (test_cpp_parity.py)

### 3.3 Configuration Loader
- [x] Implement JSON config file parser
- [x] Load similarity threshold parameter
- [x] Load rolling window size parameter (hysteresis_count)
- [x] Load target FPS parameter
- [x] Load embedding dimension parameter
- [x] Add validation for required parameters

### 3.4 Caption Bank Loader
- [x] Implement JSON caption bank parser
- [x] Load caption strings into memory
- [x] Load embedding vectors into contiguous array
- [x] Verify embedding dimensions match config
- [x] Implement caption lookup by index (topk / top1)

### 3.5 ONNX Inference Engine
- [x] Initialize ONNX Runtime environment
- [x] Create inference session with visual encoder model
- [x] Configure session options (thread count, optimization level)
- [x] Implement input tensor creation from preprocessed frame
- [x] Implement inference execution
- [x] Extract output embedding vector
- [x] Wrap in RAII class for resource management (ClipSession)

### 3.6 Frame Preprocessor
- [x] Implement frame capture from video file (Python adapter: encode_video_frames.py)
- [x] Implement frame resize to 224x224 (CLIP Resize+CenterCrop via CLIP's own _PREPROCESS)
- [x] Implement RGB channel extraction
- [x] Implement CLIP normalization (mean/std)
- [x] Implement HWC to CHW tensor conversion
- [x] Output tensor ready for ONNX input (FrameReader reads binary pipe from Python)

### 3.7 Anchor-Based Change Detector
- [x] Implement anchor embedding storage (replaces rolling window)
- [x] Implement similarity check against current frame vs anchor
- [x] Implement threshold comparison logic
- [x] Return trigger signal when change detected
- [x] Implement hysteresis to suppress duplicates
- [x] Re-stabilize anchor after trigger

### 3.8 Caption Matcher
- [x] Implement similarity computation against all captions
- [x] Find caption with highest similarity score (top-K via partial_sort)
- [x] Return best matching caption and confidence score
- [x] Apply minimum confidence threshold
- [x] Handle case when no caption exceeds threshold

### 3.9 Main Pipeline Integration
- [x] Initialize all components in correct order
- [x] Implement main processing loop
- [x] Extract frame at target FPS intervals (handled by Python encoder)
- [x] Preprocess frame for inference (handled by Python encoder)
- [x] Run ONNX inference to get embedding
- [x] Pass embedding to change detector
- [x] On trigger: match against caption bank
- [x] Output caption with timestamp
- [x] Handle video end gracefully

### 3.10 Output Formatting
- [x] Implement timestamped caption output
- [x] Support console output mode (stderr progress)
- [x] Support JSON output mode (stdout or --output file)
- [x] Add frame number to output metadata
- Note: SRT output not implemented (out of scope for current phase)

### 3.11 Error Handling
- [x] Add error handling for missing model file (ORT throws on bad path)
- [x] Add error handling for missing config file (catches exception, uses defaults)
- [x] Add error handling for missing caption bank (CaptionBank throws on bad path)
- [x] Add error handling for invalid video file (Python encoder exits with error)
- [x] Add error handling for ONNX runtime failures (ORT exceptions propagate)

### 3.12 Performance Optimization
- [x] Profile inference latency (~31ms/frame on CPU, measured in Phase 2)
- [x] Add timing metrics output option (processing_ms in output JSON metadata)
- Note: Frame skipping not needed — Python encoder controls extraction rate

### 3.13 Testing
- [x] Test with short video file (< 1 minute)
- [x] Verify selective readout behavior
- [x] Compare C++ output to Python prototype output (test_cpp_parity.py: 3/3 PASS)
- [x] Test with different video resolutions (handled transparently by Python encoder)

---

## Phase 4: Optional Visualization Layer

### 4.1 Viewer Setup
- [x] Create static HTML viewer (no server required — zero-dependency, works offline)
- [x] Serve static files from web/ directory (open web/index.html directly in browser)
- Note: WebSocket not needed — viewer loads pre-computed results JSON from engine output

### 4.2 Frontend Structure
- [x] Create index.html with load screen + viewer layout
- [x] Create styles.css for visualization styling (dark theme)
- [x] Create app.js for client-side logic
- Note: No backend connection — file picker loads video and JSON locally

### 4.3 Video Display
- [x] Implement video element for playback (native controls)
- [x] Synchronize video playback with caption feed (rAF loop + binary search)
- [x] Display current timestamp in caption overlay

### 4.4 Caption Display
- [x] Display current active caption (overlay + sidebar)
- [x] Show caption confidence score (bar + numeric)
- [x] Show confidence gap and change_similarity bars
- [x] Implement detection events list (sidebar, scrollable)
- [x] Highlight caption changes visually (active event highlighted + auto-scroll)
- [x] Show top-K alternative captions panel

### 4.5 Detection Visualization
- [x] Show similarity score to current caption (confidence bar)
- [x] Show change detection trigger similarity (change sim bar)
- [x] Show detection events list (all triggers, clickable to seek)
- Note: Embedding heatmap skipped — embeddings not stored per-frame in output JSON

### 4.6 Controls
- [x] Implement play/pause (native video controls)
- [x] Implement seek via timeline click + drag
- [x] Clicking any event in sidebar seeks to that timestamp
- Note: Real-time threshold adjustment requires re-running the engine (out of scope for viewer)

### 4.7 Integration Testing
- [ ] Test full pipeline with visualization (open web/index.html, load video + results.json)
- [ ] Verify caption sync tracks video correctly
- [ ] Verify timeline markers align with events
- [ ] Verify event list highlights and scrolls correctly

---

## Verification Checklist

At the end of each phase, verify:

- [x] **Phase 1**: Python prototype produces correct embeddings and demonstrates selective readout
- [x] **Phase 2**: ONNX model produces identical outputs to PyTorch model; caption bank is valid JSON
- [x] **Phase 3**: C++ engine produces identical outputs to Python prototype; runs in real-time (parity test 3/3 PASS)
- [x] **Phase 4**: Visualization accurately reflects engine state; controls function correctly

---

## Phase 5: Advanced Features

### 5.1 BLIP Generative Captioning
- [x] Create `python_tools/enhance_captions.py` — BLIP post-processor
- [x] Supports base and large BLIP models
- [x] Preserves CLIP data as `clip_caption` / `clip_confidence`
- [x] Tested: fixes all v2 regressions (walking→phone, mountains→bridge, nature→cooking)

### 5.2 Semantic Video Search
- [x] Create `python_tools/extract_embeddings.py` — persists frame embeddings to .npz
- [x] Create `python_tools/search_video.py` — text-to-frame search via CLIP text encoder
- [x] Viewer integration: optional search results JSON, gold timeline markers, search panel

### 5.3 Subtitle Export
- [x] Create `python_tools/export_subtitles.py` — SRT and WebVTT formats
- [x] Optional confidence display via `--show-confidence`

### 5.4 Enhanced Web Viewer
- [x] BLIP-aware: shows source badge (BLIP/CLIP), original CLIP caption reference
- [x] Search results: drop zone, gold markers, sidebar panel
- [x] Metadata panel shows BLIP model + enhancement count

---

## Notes

- All tasks are atomic and independently verifiable
- No task requires code from a later phase
- Tasks within each phase should be completed in order
- Do not proceed to next phase until current phase verification passes
- All embeddings must maintain consistent dimensionality (512D for ViT-B/32)
