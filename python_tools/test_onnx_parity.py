#!/usr/bin/env python3
"""
Behavioral Parity Test for Semantic Sentinel.
Verifies that PyTorch and ONNX backends produce identical
"when to speak / when not to speak" decisions.

This test runs the full pipeline twice — once with PyTorch, once with ONNX —
and compares trigger frames, selected captions, and confidence scores.

Usage:
    python test_onnx_parity.py [--onnx-model <path>]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
import clip
from PIL import Image

from change_detector import AnchorChangeDetector

# Constants (must match process_video.py exactly)
MODEL_NAME = "ViT-B/32"
EMBEDDING_DIM = 512
INPUT_SIZE = 224

# Pipeline defaults (must match process_video.py)
TARGET_FPS = 1.0
CHANGE_THRESHOLD = 0.85
HYSTERESIS_COUNT = 2
CAPTION_THRESHOLD = 0.20
TOP_K = 3
STABILITY_DELTA = 0.02

# Parity tolerance for confidence scores
CONFIDENCE_TOLERANCE = 0.001


def load_caption_bank(path: Path) -> Tuple[List[str], np.ndarray]:
    """Load caption bank from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    captions = [c["text"] for c in data["captions"]]
    embeddings = np.array([c["embedding"] for c in data["captions"]], dtype=np.float32)
    return captions, embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def match_captions_topk(embedding: np.ndarray, caption_embeddings: np.ndarray,
                         captions: List[str], k: int) -> List[Dict]:
    """Reproduce _match_captions_topk from process_video.py."""
    similarities = np.array([
        cosine_similarity(embedding, cap_emb)
        for cap_emb in caption_embeddings
    ])
    top_indices = np.argsort(similarities)[::-1][:k]
    return [
        {"text": captions[idx], "score": float(similarities[idx]), "index": int(idx)}
        for idx in top_indices
    ]


def run_pipeline(video_path: Path, extract_fn, captions: List[str],
                 caption_embeddings: np.ndarray, preprocess) -> List[Dict]:
    """
    Run the full pipeline with a given embedding extraction function.
    Reproduces process_video.py logic exactly.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / TARGET_FPS)
    if frame_interval < 1:
        frame_interval = 1

    detector = AnchorChangeDetector(
        similarity_threshold=CHANGE_THRESHOLD,
        hysteresis_count=HYSTERESIS_COUNT,
        embedding_dim=EMBEDDING_DIM,
    )

    prev_caption_text = None
    prev_caption_score = 0.0
    results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps

            # Extract embedding using provided function
            embedding = extract_fn(frame, preprocess)

            # Change detection
            triggered, similarity = detector.process(embedding)

            if triggered:
                # Top-K matching
                top_matches = match_captions_topk(
                    embedding, caption_embeddings, captions, TOP_K
                )

                top1_text = top_matches[0]["text"]
                top1_score = top_matches[0]["score"]

                confidence_gap = 0.0
                if len(top_matches) >= 2:
                    confidence_gap = top1_score - top_matches[1]["score"]

                # Stability constraint
                selected_text = top1_text
                selected_score = top1_score

                if prev_caption_text is not None:
                    score_improvement = top1_score - prev_caption_score
                    if score_improvement < STABILITY_DELTA:
                        selected_text = prev_caption_text
                        selected_score = prev_caption_score

                prev_caption_text = selected_text
                prev_caption_score = selected_score

                if selected_score >= CAPTION_THRESHOLD:
                    results.append({
                        "frame": frame_idx,
                        "timestamp": round(timestamp, 3),
                        "caption": selected_text,
                        "confidence": round(float(selected_score), 4),
                        "confidence_gap": round(float(confidence_gap), 4),
                        "change_similarity": round(float(similarity), 4),
                    })

        frame_idx += 1

    cap.release()
    return results


def make_pytorch_extractor(model, device):
    """Create PyTorch embedding extraction function."""
    def extract(frame, preprocess):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        tensor = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(tensor)

        embedding_np = embedding.cpu().numpy().squeeze()
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm
        return embedding_np

    return extract


def make_onnx_extractor(session):
    """Create ONNX embedding extraction function."""
    def extract(frame, preprocess):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        tensor = preprocess(pil_image).unsqueeze(0)
        input_np = tensor.numpy()

        embedding = session.run(["embedding"], {"image": input_np})[0].squeeze()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    return extract


def compare_results(pytorch_results: List[Dict], onnx_results: List[Dict]) -> bool:
    """Compare pipeline results from both backends."""

    print(f"  PyTorch events: {len(pytorch_results)}")
    print(f"  ONNX events:    {len(onnx_results)}")
    print()

    if len(pytorch_results) != len(onnx_results):
        print("  FAILED: Different number of events")
        print()
        # Show details
        pt_frames = {r["frame"] for r in pytorch_results}
        ox_frames = {r["frame"] for r in onnx_results}
        only_pt = pt_frames - ox_frames
        only_ox = ox_frames - pt_frames
        if only_pt:
            print(f"  PyTorch-only triggers at frames: {sorted(only_pt)}")
        if only_ox:
            print(f"  ONNX-only triggers at frames: {sorted(only_ox)}")
        return False

    if len(pytorch_results) == 0:
        print("  Both produced 0 events (identical)")
        return True

    all_passed = True
    for i, (pt, ox) in enumerate(zip(pytorch_results, onnx_results)):
        frame_match = pt["frame"] == ox["frame"]
        caption_match = pt["caption"] == ox["caption"]
        conf_diff = abs(pt["confidence"] - ox["confidence"])
        conf_match = conf_diff <= CONFIDENCE_TOLERANCE

        passed = frame_match and caption_match and conf_match
        status = "PASS" if passed else "FAIL"

        print(f"  Event {i}: frame={pt['frame']}/{ox['frame']} "
              f"caption={'MATCH' if caption_match else 'MISMATCH'} "
              f"conf_diff={conf_diff:.4f} -> {status}")

        if not caption_match:
            print(f"    PyTorch: \"{pt['caption']}\"")
            print(f"    ONNX:    \"{ox['caption']}\"")

        if not passed:
            all_passed = False

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral parity test: PyTorch vs ONNX"
    )
    parser.add_argument(
        "--onnx-model", type=str,
        default="models/clip_visual_vit_b32.onnx",
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--caption-bank", type=str,
        default="models/caption_bank.json",
        help="Path to caption bank"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for PyTorch (default: cpu)"
    )

    args = parser.parse_args()

    onnx_path = Path(args.onnx_model)
    caption_bank_path = Path(args.caption_bank)

    if not onnx_path.exists():
        print(f"Error: ONNX model not found: {onnx_path}")
        return 1
    if not caption_bank_path.exists():
        print(f"Error: Caption bank not found: {caption_bank_path}")
        return 1

    # Find test videos (1 from each category for coverage)
    test_videos = []
    video_base = Path("validation/test_videos")
    for category in ["static", "multi_scene", "gradual_transition"]:
        cat_dir = video_base / category
        if cat_dir.exists():
            videos = sorted(cat_dir.glob("*.mp4"))
            if videos:
                test_videos.append((category, videos[0]))

    if not test_videos:
        print("Error: No test videos found in validation/test_videos/")
        return 1

    print("=" * 60)
    print("Semantic Sentinel - Behavioral Parity Test")
    print("PyTorch vs ONNX: trigger decisions must be identical")
    print("=" * 60)
    print()

    # Load models
    print(f"Loading PyTorch CLIP {MODEL_NAME}...")
    model, preprocess = clip.load(MODEL_NAME, device=args.device)
    model.eval()

    print(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(str(onnx_path))

    print(f"Loading caption bank: {caption_bank_path}")
    captions, caption_embeddings = load_caption_bank(caption_bank_path)
    print(f"  {len(captions)} captions loaded")
    print()

    # Create extractors
    pytorch_extract = make_pytorch_extractor(model, args.device)
    onnx_extract = make_onnx_extractor(session)

    # Test each video
    overall_passed = True

    for category, video_path in test_videos:
        print("=" * 60)
        print(f"Video: {video_path.name} ({category})")
        print("=" * 60)
        print()

        print("Running PyTorch backend...")
        pytorch_results = run_pipeline(
            video_path, pytorch_extract, captions, caption_embeddings, preprocess
        )

        print("Running ONNX backend...")
        onnx_results = run_pipeline(
            video_path, onnx_extract, captions, caption_embeddings, preprocess
        )

        print()
        print("Comparing results:")
        passed = compare_results(pytorch_results, onnx_results)
        print()

        if passed:
            print(f"  {video_path.name}: PASSED")
        else:
            print(f"  {video_path.name}: FAILED")
            overall_passed = False

        print()

    # Final summary
    print("=" * 60)
    print("BEHAVIORAL PARITY SUMMARY")
    print("=" * 60)
    for category, video_path in test_videos:
        print(f"  {category}: {video_path.name}")
    print()

    if overall_passed:
        print("  OVERALL: PASSED")
        print("  PyTorch and ONNX backends produce identical trigger decisions.")
        print("  ONNX model is a valid replacement for the PyTorch pipeline.")
    else:
        print("  OVERALL: FAILED")
        print("  Trigger decisions differ between backends.")
        print("  ONNX model cannot be used as a drop-in replacement.")

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
