#!/usr/bin/env python3
"""
Video frame encoder for Semantic Sentinel C++ engine.

Reads a video, extracts frames at target FPS, applies CLIP preprocessing,
and writes binary frame data to stdout for the C++ sentinel_engine.

Binary protocol per frame:
  [0..3]   int32   frame_number
  [4..11]  float64 timestamp_sec
  [12..]   float32 tensor[3*224*224]  (CHW, CLIP-normalized)

Usage:
  python encode_video_frames.py <video.mp4> [--fps 1.0] | sentinel_engine --config config/default.json
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import clip

MODEL_NAME = "ViT-B/32"
INPUT_SIZE = 224

# Load CLIP preprocessing at import time (used for exact parity with process_video.py)
_DEVICE = "cpu"
_, _PREPROCESS = clip.load(MODEL_NAME, device=_DEVICE)


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    BGR OpenCV frame → CLIP-normalized float32 CHW tensor.
    Uses CLIP's own preprocessing pipeline to guarantee exact parity
    with process_video.py::_preprocess_frame().
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    tensor = _PREPROCESS(pil_image)          # torch.Tensor CHW float32
    return tensor.numpy()                    # numpy CHW float32


def main():
    parser = argparse.ArgumentParser(
        description="Encode video frames for sentinel_engine"
    )
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to extract (default: 1.0)")
    parser.add_argument("--info", action="store_true",
                        help="Print video info to stderr and exit")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}", file=sys.stderr)
        return 1

    src_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_fr  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration  = total_fr / src_fps if src_fps > 0 else 0

    print(f"Video: {video_path.name}", file=sys.stderr)
    print(f"  Source FPS:    {src_fps:.2f}", file=sys.stderr)
    print(f"  Total frames:  {total_fr}", file=sys.stderr)
    print(f"  Duration:      {duration:.2f}s", file=sys.stderr)
    print(f"  Target FPS:    {args.fps}", file=sys.stderr)

    if args.info:
        cap.release()
        return 0

    # Frame interval in source frames
    frame_interval = int(src_fps / args.fps)  # int() = floor, matches process_video.py exactly
    if frame_interval < 1:
        frame_interval = 1
    print(f"  Frame interval: every {frame_interval} source frames", file=sys.stderr)
    print(file=sys.stderr)

    # Set stdout to binary mode on Windows
    if sys.platform == "win32":
        import os
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    out = sys.stdout.buffer
    frame_number = 0
    sent = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            timestamp = frame_number / src_fps
            tensor = preprocess_frame(frame)

            # Write header: int32 frame_number + float64 timestamp
            header = struct.pack('<id', frame_number, timestamp)
            out.write(header)

            # Write tensor as little-endian float32
            out.write(tensor.tobytes())
            out.flush()
            sent += 1

        frame_number += 1

    cap.release()
    print(f"Encoded {sent} frames from {frame_number} source frames", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
