#!/usr/bin/env python3
"""
BLIP Caption Enhancer for Semantic Sentinel.
Post-processes pipeline results by generating free-form captions using BLIP
for each triggered frame, replacing the fixed caption bank matches.

Usage:
    python enhance_captions.py video.mp4 --input raw.json --output enhanced.json
    python enhance_captions.py video.mp4 --input raw.json --model large
    python enhance_captions.py video.mp4 --input raw.json --conditional "a photograph of"

First run downloads the BLIP model (~1.8GB) to HuggingFace cache.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODELS = {
    "base": "Salesforce/blip-image-captioning-base",
    "large": "Salesforce/blip-image-captioning-large",
}


def load_blip(model_key: str, device: str):
    """Load BLIP model and processor."""
    from transformers import BlipProcessor, BlipForConditionalGeneration

    model_name = MODELS.get(model_key, model_key)
    print(f"Loading BLIP model: {model_name}", file=sys.stderr)
    print(f"  Device: {device}", file=sys.stderr)

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    print(f"  Model loaded.", file=sys.stderr)
    return processor, model


def extract_frame_at_timestamp(video_path: str, timestamp: float) -> Image.Image:
    """Extract a single frame from video at given timestamp, return as PIL Image."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def generate_caption(
    processor, model, image: Image.Image, device: str,
    conditional_text: str = None, max_length: int = 50
) -> str:
    """Generate a caption for an image using BLIP."""
    if conditional_text:
        inputs = processor(image, conditional_text, return_tensors="pt").to(device)
    else:
        inputs = processor(image, return_tensors="pt").to(device)

    output = model.generate(**inputs, max_length=max_length)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def main():
    parser = argparse.ArgumentParser(description="Enhance captions with BLIP generative model")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--input", required=True, help="Raw results JSON from sentinel engine")
    parser.add_argument("--output", help="Output enhanced JSON (default: stdout)")
    parser.add_argument("--model", choices=["base", "large"], default="large",
                        help="BLIP model size (default: large)")
    parser.add_argument("--device", default=None,
                        help="Device: 'cuda', 'cpu', or auto-detect")
    parser.add_argument("--conditional", default=None,
                        help="Conditional text prefix (e.g. 'a photograph of')")
    parser.add_argument("--max-length", type=int, default=50,
                        help="Max caption token length (default: 50)")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load results
    with open(args.input, "r") as f:
        data = json.load(f)

    captions = data.get("captions", [])
    if not captions:
        print("No captions found in input JSON.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(captions)} caption events to enhance.", file=sys.stderr)

    # Load BLIP
    processor, model = load_blip(args.model, device)

    # Enhance each caption event
    t_start = time.time()
    enhanced = 0
    failed = 0

    for i, event in enumerate(captions):
        timestamp = event["timestamp"]

        # Extract frame
        image = extract_frame_at_timestamp(args.video, timestamp)
        if image is None:
            print(f"  [{i+1}/{len(captions)}] FAILED to extract frame at {timestamp:.2f}s",
                  file=sys.stderr)
            failed += 1
            continue

        # Generate BLIP caption
        blip_caption = generate_caption(
            processor, model, image, device,
            conditional_text=args.conditional,
            max_length=args.max_length,
        )

        # Preserve original CLIP data
        event["clip_caption"] = event["caption"]
        event["clip_confidence"] = event.get("confidence", 0.0)

        # Replace with BLIP caption
        event["caption"] = blip_caption
        event["caption_source"] = "blip"

        enhanced += 1
        print(f"  [{i+1}/{len(captions)}] {timestamp:.1f}s: \"{blip_caption}\"", file=sys.stderr)

    elapsed = time.time() - t_start

    # Update metadata
    meta = data.get("metadata", {})
    meta["blip_model"] = MODELS.get(args.model, args.model)
    meta["blip_enhanced"] = enhanced
    meta["blip_failed"] = failed
    meta["blip_time_sec"] = round(elapsed, 2)
    if args.conditional:
        meta["blip_conditional"] = args.conditional
    data["metadata"] = meta

    print(f"\nDone: {enhanced} enhanced, {failed} failed, {elapsed:.1f}s total",
          file=sys.stderr)

    # Output
    result_json = json.dumps(data, indent=2)
    if args.output:
        Path(args.output).write_text(result_json, encoding="utf-8")
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(result_json)


if __name__ == "__main__":
    main()
