#!/usr/bin/env python3
"""
Frame Embedding Extractor for Semantic Sentinel.
Processes a video and saves per-frame CLIP embeddings for semantic search.

Usage:
    python extract_embeddings.py video.mp4 --output video.embeddings.npz
    python extract_embeddings.py video.mp4 --fps 1 --output video.embeddings.npz

The .npz sidecar file contains:
    embeddings:    float32 array (N, 512) — L2-normalized CLIP frame embeddings
    frame_numbers: int32 array (N,)
    timestamps:    float64 array (N,)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import clip
from PIL import Image

MODEL_NAME = "ViT-B/32"
EMBEDDING_DIM = 512


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP frame embeddings for search")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--output", help="Output .npz file (default: <video>.embeddings.npz)")
    parser.add_argument("--fps", type=float, default=1.0, help="Target FPS for extraction (default: 1.0)")
    parser.add_argument("--device", default=None, help="Device: 'cuda', 'cpu', or auto-detect")
    args = parser.parse_args()

    # Output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(Path(args.video).with_suffix("")) + ".embeddings.npz"

    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP
    print(f"Loading CLIP {MODEL_NAME} on {device}...", file=sys.stderr)
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    frame_interval = int(video_fps / args.fps)
    if frame_interval < 1:
        frame_interval = 1

    expected_count = total_frames // frame_interval if frame_interval > 0 else total_frames

    print(f"Video: {Path(args.video).name}", file=sys.stderr)
    print(f"  FPS: {video_fps:.2f}, Frames: {total_frames}, Duration: {duration:.1f}s", file=sys.stderr)
    print(f"  Target FPS: {args.fps}, Interval: every {frame_interval} frames", file=sys.stderr)
    print(f"  Expected embeddings: ~{expected_count}", file=sys.stderr)

    # Extract embeddings
    embeddings_list = []
    frame_numbers_list = []
    timestamps_list = []

    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps if video_fps > 0 else 0.0

            # Preprocess with CLIP's own transform for exact parity
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            tensor = preprocess(pil_image).unsqueeze(0).to(device)

            # Encode
            with torch.no_grad():
                embedding = model.encode_image(tensor)

            emb_np = embedding.cpu().numpy().squeeze()

            # L2 normalize (CLIP encode_image does not normalize)
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm

            embeddings_list.append(emb_np)
            frame_numbers_list.append(frame_idx)
            timestamps_list.append(timestamp)

            if len(embeddings_list) % 50 == 0:
                print(f"  Processed {len(embeddings_list)} frames...", file=sys.stderr)

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t_start

    # Stack into arrays
    embeddings = np.array(embeddings_list, dtype=np.float32)
    frame_numbers = np.array(frame_numbers_list, dtype=np.int32)
    timestamps = np.array(timestamps_list, dtype=np.float64)

    # Save as compressed .npz
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        frame_numbers=frame_numbers,
        timestamps=timestamps,
    )

    file_size = Path(output_path).stat().st_size
    print(f"\nDone: {len(embeddings)} embeddings in {elapsed:.1f}s", file=sys.stderr)
    print(f"Saved to {output_path} ({file_size / 1024:.0f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
