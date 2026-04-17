#!/usr/bin/env python3
"""
ONNX Validation Script for Semantic Sentinel.
Verifies numerical parity between PyTorch CLIP and exported ONNX model
using real video frames as test inputs.

Usage:
    python validate_onnx.py [--onnx-model <path>] [--video <path>] [--num-frames 5]
    python validate_onnx.py --optimize  # Also test optimized model
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import clip
from PIL import Image

# Constants (must match process_video.py)
MODEL_NAME = "ViT-B/32"
EMBEDDING_DIM = 512
INPUT_SIZE = 224

# Parity thresholds
# Raw tolerance is slightly relaxed because unnormalized outputs have larger
# magnitude (~10) so FP32 rounding accumulates more absolute error.
# Normalized tolerance is strict because that's what the pipeline uses.
RAW_ATOL = 5e-5
NORMALIZED_ATOL = 1e-5
COSINE_MIN = 0.9999


def extract_test_frames(video_path: Path, num_frames: int = 5) -> list:
    """Extract evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        num_frames = total_frames

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))

    cap.release()
    return frames


def validate_parity(
    onnx_path: Path,
    video_path: Path,
    num_frames: int = 5,
    device: str = "cpu",
) -> bool:
    """Validate numerical parity between PyTorch and ONNX inference."""

    # Load PyTorch model
    print(f"Loading PyTorch CLIP {MODEL_NAME}...")
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()

    # Load ONNX model
    print(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(str(onnx_path))
    print()

    # Extract test frames
    print(f"Extracting {num_frames} test frames from: {video_path}")
    frames = extract_test_frames(video_path, num_frames)
    print(f"Extracted {len(frames)} frames")
    print()

    all_passed = True
    raw_diffs = []
    norm_diffs = []
    cosine_sims = []

    for i, (frame_idx, frame) in enumerate(frames):
        # Preprocess (identical for both backends)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        tensor = preprocess(pil_image).unsqueeze(0).to(device)
        input_np = tensor.cpu().numpy()

        # PyTorch inference
        with torch.no_grad():
            pytorch_out = model.encode_image(tensor).cpu().numpy().squeeze()

        # ONNX inference
        onnx_out = session.run(["embedding"], {"image": input_np})[0].squeeze()

        # Raw comparison
        raw_diff = np.abs(pytorch_out - onnx_out).max()
        raw_diffs.append(raw_diff)

        # Normalized comparison
        pytorch_norm = pytorch_out / np.linalg.norm(pytorch_out)
        onnx_norm = onnx_out / np.linalg.norm(onnx_out)
        norm_diff = np.abs(pytorch_norm - onnx_norm).max()
        norm_diffs.append(norm_diff)

        # Cosine similarity
        cosine = np.dot(pytorch_out, onnx_out) / (
            np.linalg.norm(pytorch_out) * np.linalg.norm(onnx_out)
        )
        cosine_sims.append(cosine)

        # Per-frame verdict
        raw_ok = raw_diff <= RAW_ATOL
        norm_ok = norm_diff <= NORMALIZED_ATOL
        cos_ok = cosine >= COSINE_MIN
        passed = raw_ok and norm_ok and cos_ok

        status = "PASS" if passed else "FAIL"
        print(f"  Frame {frame_idx:5d} [{i+1}/{len(frames)}]: "
              f"raw_diff={raw_diff:.2e}, norm_diff={norm_diff:.2e}, "
              f"cosine={cosine:.8f} -> {status}")

        if not passed:
            all_passed = False

    print()
    print("-" * 60)
    print("AGGREGATE RESULTS")
    print("-" * 60)
    print(f"  Max raw difference:    {max(raw_diffs):.2e} (threshold: {RAW_ATOL:.0e})")
    print(f"  Max norm difference:   {max(norm_diffs):.2e} (threshold: {NORMALIZED_ATOL:.0e})")
    print(f"  Min cosine similarity: {min(cosine_sims):.8f} (threshold: {COSINE_MIN})")
    print(f"  Mean cosine similarity: {np.mean(cosine_sims):.8f}")
    print()

    if all_passed:
        print("  OVERALL: PASSED - Numerical parity verified")
    else:
        print("  OVERALL: FAILED - Parity check failed on one or more frames")

    return all_passed


def measure_latency(onnx_path: Path, num_runs: int = 20):
    """Measure ONNX inference latency."""
    session = ort.InferenceSession(str(onnx_path))
    dummy = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)

    # Warmup
    for _ in range(3):
        session.run(["embedding"], {"image": dummy})

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(["embedding"], {"image": dummy})
        times.append(time.perf_counter() - start)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    min_ms = np.min(times) * 1000
    max_ms = np.max(times) * 1000

    print(f"  Latency ({num_runs} runs): {mean_ms:.1f} +/- {std_ms:.1f} ms "
          f"(min={min_ms:.1f}, max={max_ms:.1f})")
    return mean_ms


def optimize_model(input_path: Path, output_path: Path):
    """Apply ONNX Runtime graph optimizations."""
    print(f"Optimizing: {input_path}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = str(output_path)

    # Creating session with optimization saves the optimized model
    ort.InferenceSession(str(input_path), sess_options)

    original_size = input_path.stat().st_size / (1024 * 1024)
    optimized_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  Original: {original_size:.1f} MB")
    print(f"  Optimized: {optimized_size:.1f} MB")
    print(f"  Saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Validate ONNX model parity with PyTorch CLIP"
    )
    parser.add_argument(
        "--onnx-model", type=str,
        default="models/clip_visual_vit_b32.onnx",
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--video", type=str,
        default=None,
        help="Video to extract test frames from"
    )
    parser.add_argument(
        "--num-frames", type=int, default=5,
        help="Number of test frames to extract (default: 5)"
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Also optimize the model and validate the optimized version"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for PyTorch (default: cpu)"
    )

    args = parser.parse_args()

    onnx_path = Path(args.onnx_model)
    if not onnx_path.exists():
        print(f"Error: ONNX model not found: {onnx_path}")
        print("Run export_onnx.py first.")
        return 1

    # Find a test video if not specified
    if args.video:
        video_path = Path(args.video)
    else:
        # Look for validation videos
        search_dirs = [
            Path("validation/test_videos/static"),
            Path("validation/test_videos/multi_scene"),
            Path("validation/test_videos/single_scene_dynamic"),
        ]
        video_path = None
        for d in search_dirs:
            if d.exists():
                videos = list(d.glob("*.mp4"))
                if videos:
                    video_path = videos[0]
                    break

        if video_path is None:
            print("Error: No test video found. Use --video to specify one.")
            return 1

    print("=" * 60)
    print("Semantic Sentinel - ONNX Parity Validation")
    print("=" * 60)
    print()

    # Validate original model
    print("--- Original Model ---")
    passed = validate_parity(onnx_path, video_path, args.num_frames, args.device)
    print()

    # Measure latency
    print("--- Inference Latency (Original) ---")
    original_latency = measure_latency(onnx_path)
    print()

    # Optimize if requested
    if args.optimize:
        optimized_path = onnx_path.parent / (onnx_path.stem + "_optimized.onnx")

        print("--- Optimization ---")
        optimize_model(onnx_path, optimized_path)
        print()

        print("--- Optimized Model Validation ---")
        opt_passed = validate_parity(optimized_path, video_path, args.num_frames, args.device)
        print()

        print("--- Inference Latency (Optimized) ---")
        opt_latency = measure_latency(optimized_path)
        print()

        speedup = original_latency / opt_latency if opt_latency > 0 else 0
        print(f"Speedup: {speedup:.2f}x")

        passed = passed and opt_passed

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
