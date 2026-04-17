#!/usr/bin/env python3
"""
Semantic Video Search for Semantic Sentinel.
Search through video frames using natural language queries via CLIP.

Usage:
    python search_video.py --query "sunset over water" --embeddings video.embeddings.npz
    python search_video.py --query "person running" --embeddings video.embeddings.npz --top-n 5
    python search_video.py --query "mountain landscape" --embeddings video.embeddings.npz --output results.json
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
import clip

MODEL_NAME = "ViT-B/32"


def main():
    parser = argparse.ArgumentParser(description="Search video frames by text query using CLIP")
    parser.add_argument("--query", required=True, help="Natural language search query")
    parser.add_argument("--embeddings", required=True, help="Frame embeddings .npz file")
    parser.add_argument("--top-n", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--output", help="Output JSON file (default: stdout)")
    parser.add_argument("--device", default=None, help="Device: 'cuda', 'cpu', or auto-detect")
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load frame embeddings
    print(f"Loading embeddings from {args.embeddings}...", file=sys.stderr)
    data = np.load(args.embeddings)
    embeddings = data["embeddings"]       # (N, 512) float32
    frame_numbers = data["frame_numbers"]  # (N,) int32
    timestamps = data["timestamps"]        # (N,) float64

    print(f"  {len(embeddings)} frames loaded.", file=sys.stderr)

    # Load CLIP text encoder
    print(f"Loading CLIP {MODEL_NAME} on {device}...", file=sys.stderr)
    model, _ = clip.load(MODEL_NAME, device=device)
    model.eval()

    # Encode query
    t_start = time.time()
    text_tokens = clip.tokenize([args.query]).to(device)

    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)

    text_emb_np = text_embedding.cpu().numpy().squeeze()

    # L2 normalize (CLIP encode_text does not normalize)
    norm = np.linalg.norm(text_emb_np)
    if norm > 0:
        text_emb_np = text_emb_np / norm

    # Compute cosine similarity (embeddings are already L2-normalized)
    similarities = embeddings @ text_emb_np  # (N,)

    # Top-N results
    top_n = min(args.top_n, len(similarities))
    top_indices = np.argsort(similarities)[::-1][:top_n]

    results = []
    for idx in top_indices:
        results.append({
            "rank": len(results) + 1,
            "frame": int(frame_numbers[idx]),
            "timestamp": round(float(timestamps[idx]), 3),
            "similarity": round(float(similarities[idx]), 5),
        })

    elapsed = time.time() - t_start

    output = {
        "query": args.query,
        "total_frames": len(embeddings),
        "top_n": top_n,
        "search_time_ms": round(elapsed * 1000, 1),
        "results": results,
    }

    result_json = json.dumps(output, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result_json)
        print(f"\nResults saved to {args.output}", file=sys.stderr)
    else:
        print(result_json)

    # Summary to stderr
    print(f"\nQuery: \"{args.query}\"", file=sys.stderr)
    print(f"Top {top_n} results ({elapsed*1000:.0f}ms):", file=sys.stderr)
    for r in results:
        mins = int(r["timestamp"] // 60)
        secs = r["timestamp"] % 60
        print(f"  #{r['rank']}  {mins}:{secs:04.1f}  frame {r['frame']:>5d}  sim={r['similarity']:.4f}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
