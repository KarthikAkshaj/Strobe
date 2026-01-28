#!/usr/bin/env python3
"""
Caption Bank Creation Script for Semantic Sentinel.
Encodes a set of captions using CLIP text encoder and exports to JSON.

Usage:
    python create_caption_bank.py [--output <output_path>] [--captions <captions_file>]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import clip

# Constants
MODEL_NAME = "ViT-B/32"
EMBEDDING_DIM = 512

# Default test captions covering diverse semantic content
DEFAULT_CAPTIONS = [
    # People and actions
    "a person walking down the street",
    "a group of people talking",
    "someone sitting at a desk",
    "a person running",
    "people standing in a line",

    # Indoor scenes
    "an empty room",
    "a living room with furniture",
    "a kitchen with appliances",
    "an office workspace",
    "a bedroom with a bed",

    # Outdoor scenes
    "a city street with buildings",
    "a park with trees",
    "a beach with ocean waves",
    "a mountain landscape",
    "a forest path",

    # Objects
    "a car on the road",
    "a computer screen",
    "a book on a table",
    "food on a plate",

    # Transitions/States
    "a dark scene",
    "a bright sunny day",
    "motion blur",
    "a static scene with no movement",
]


def load_model(device: str):
    """Load CLIP model."""
    model, _ = clip.load(MODEL_NAME, device=device)
    model.eval()
    return model


def encode_captions(captions: List[str], model, device: str) -> np.ndarray:
    """
    Encode a list of captions using CLIP text encoder.

    Args:
        captions: List of caption strings
        model: CLIP model
        device: Computation device

    Returns:
        Array of shape (num_captions, EMBEDDING_DIM)
    """
    # Tokenize all captions
    text_tokens = clip.tokenize(captions, truncate=True).to(device)

    # Encode
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)

    # Convert to numpy
    embeddings_np = text_embeddings.cpu().numpy()

    # Verify shape
    assert embeddings_np.shape == (len(captions), EMBEDDING_DIM), \
        f"Unexpected shape: {embeddings_np.shape}"

    # L2-normalize all embeddings for consistent cosine similarity downstream
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    embeddings_np = embeddings_np / norms

    return embeddings_np


def create_caption_bank(captions: List[str], embeddings: np.ndarray) -> dict:
    """Create caption bank dictionary with metadata."""
    return {
        "metadata": {
            "model": MODEL_NAME,
            "embedding_dimension": EMBEDDING_DIM,
            "num_captions": len(captions),
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
        "captions": [
            {
                "id": i,
                "text": caption,
                "embedding": embeddings[i].tolist(),
            }
            for i, caption in enumerate(captions)
        ]
    }


def load_captions_from_file(filepath: Path) -> List[str]:
    """Load captions from a text file (one caption per line)."""
    with open(filepath, "r", encoding="utf-8") as f:
        captions = [line.strip() for line in f if line.strip()]
    return captions


def main():
    parser = argparse.ArgumentParser(
        description="Create caption bank with CLIP text embeddings"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="caption_bank.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--captions", "-c", type=str, default=None,
        help="Text file with captions (one per line). Uses defaults if not specified."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Computation device (cuda/cpu, default: auto)"
    )

    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load captions
    if args.captions:
        captions_path = Path(args.captions)
        if not captions_path.exists():
            print(f"Error: Captions file not found: {captions_path}")
            return 1
        captions = load_captions_from_file(captions_path)
        print(f"Loaded {len(captions)} captions from: {captions_path}")
    else:
        captions = DEFAULT_CAPTIONS
        print(f"Using {len(captions)} default test captions")

    print(f"Device: {device}")
    print()

    # Remove duplicates while preserving order
    seen = set()
    unique_captions = []
    for c in captions:
        if c not in seen:
            seen.add(c)
            unique_captions.append(c)

    if len(unique_captions) < len(captions):
        print(f"Removed {len(captions) - len(unique_captions)} duplicate captions")
        captions = unique_captions

    # Load model
    print("Loading CLIP model...")
    model = load_model(device)

    # Encode captions
    print("Encoding captions...")
    embeddings = encode_captions(captions, model, device)

    # Verify dimensions match visual embeddings
    print(f"Text embedding dimension: {embeddings.shape[1]}")
    print(f"Expected dimension (matches visual): {EMBEDDING_DIM}")

    if embeddings.shape[1] != EMBEDDING_DIM:
        print("ERROR: Dimension mismatch!")
        return 1

    # Create caption bank
    caption_bank = create_caption_bank(captions, embeddings)

    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(caption_bank, f, indent=2)

    print()
    print(f"Caption bank saved to: {output_path}")
    print(f"Total captions: {len(captions)}")

    # Print sample
    print()
    print("Sample captions and embedding norms:")
    for i, caption in enumerate(captions[:5]):
        norm = np.linalg.norm(embeddings[i])
        print(f"  [{i}] \"{caption[:50]}...\" (norm: {norm:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())