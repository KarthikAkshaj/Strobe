#!/usr/bin/env python3
"""
Frame Embedding Extraction Script for Semantic Sentinel.
Loads an image, preprocesses it for CLIP, and extracts the visual embedding.

Usage:
    python extract_embedding.py <image_path> [--output <output_path>]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image

# Constants
MODEL_NAME = "ViT-B/32"
EMBEDDING_DIM = 512
INPUT_SIZE = 224


def load_model(device: str):
    """Load CLIP model and preprocessing function."""
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()
    return model, preprocess


def extract_embedding(image_path: Path, model, preprocess, device: str) -> np.ndarray:
    """
    Extract visual embedding from an image.

    Args:
        image_path: Path to input image
        model: CLIP model
        preprocess: CLIP preprocessing transform
        device: Computation device

    Returns:
        512-dimensional embedding vector as numpy array
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Verify input shape
    assert image_tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE), \
        f"Unexpected input shape: {image_tensor.shape}"

    # Extract embedding
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)

    # Convert to numpy and verify shape
    embedding_np = embedding.cpu().numpy().squeeze()

    assert embedding_np.shape == (EMBEDDING_DIM,), \
        f"Unexpected output shape: {embedding_np.shape}"

    return embedding_np


def save_embedding(embedding: np.ndarray, output_path: Path, metadata: dict = None):
    """Save embedding to JSON file with optional metadata."""
    data = {
        "embedding": embedding.tolist(),
        "dimension": len(embedding),
        "model": MODEL_NAME,
    }
    if metadata:
        data["metadata"] = metadata

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP visual embedding from an image"
    )
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON file path (default: <image_name>_embedding.json)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Computation device (cuda/cpu, default: auto)"
    )

    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.with_suffix("").with_suffix("_embedding.json")

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Image: {image_path}")
    print(f"Device: {device}")
    print(f"Output: {output_path}")
    print()

    # Load model
    print("Loading CLIP model...")
    model, preprocess = load_model(device)

    # Extract embedding
    print("Extracting embedding...")
    embedding = extract_embedding(image_path, model, preprocess, device)

    # Verify output
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")

    # Save embedding
    metadata = {
        "source_image": str(image_path.name),
        "image_resolution": str(Image.open(image_path).size),
    }
    save_embedding(embedding, output_path, metadata)
    print(f"Embedding saved to: {output_path}")

    # Print first few values for inspection
    print()
    print("First 10 embedding values:")
    print(f"  {embedding[:10]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())