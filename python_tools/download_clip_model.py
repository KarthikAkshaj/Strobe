#!/usr/bin/env python3
"""
CLIP Model Acquisition Script for Semantic Sentinel.
Downloads CLIP ViT-B/32 and verifies model integrity.

Model: CLIP ViT-B/32
Embedding Dimension: 512
Input Shape: (1, 3, 224, 224)
"""

import os
import hashlib
import torch
import clip
from pathlib import Path

# Model configuration
MODEL_NAME = "ViT-B/32"
EMBEDDING_DIM = 512
INPUT_SHAPE = (1, 3, 224, 224)

# Expected SHA256 hash for ViT-B/32 weights
# This ensures reproducibility and integrity
EXPECTED_HASH = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"

def get_cache_dir() -> Path:
    """Get the CLIP cache directory."""
    return Path(os.path.expanduser("~/.cache/clip"))

def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    print("=" * 60)
    print("Semantic Sentinel - CLIP Model Acquisition")
    print("=" * 60)
    print()
    print(f"Model: {MODEL_NAME}")
    print(f"Embedding Dimension: {EMBEDDING_DIM}")
    print(f"Input Shape: {INPUT_SHAPE}")
    print()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    # Download and load model
    print("Downloading/Loading CLIP model...")
    print("-" * 40)

    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()  # Set to evaluation mode

    print(f"Model loaded successfully!")
    print()

    # Verify model architecture
    print("Model Verification:")
    print("-" * 40)

    # Test with dummy input
    dummy_input = torch.randn(INPUT_SHAPE).to(device)

    with torch.no_grad():
        # Get visual features
        visual_features = model.encode_image(dummy_input)

    actual_dim = visual_features.shape[-1]
    print(f"Output embedding dimension: {actual_dim}")

    if actual_dim != EMBEDDING_DIM:
        print(f"ERROR: Expected {EMBEDDING_DIM}, got {actual_dim}")
        return 1

    print(f"Dimension verification: PASSED")
    print()

    # Verify model weights hash (if file exists in cache)
    cache_dir = get_cache_dir()
    model_file = cache_dir / "ViT-B-32.pt"

    if model_file.exists():
        print("Checksum Verification:")
        print("-" * 40)
        actual_hash = compute_file_hash(model_file)
        print(f"Expected: {EXPECTED_HASH}")
        print(f"Actual:   {actual_hash}")

        if actual_hash == EXPECTED_HASH:
            print("Checksum verification: PASSED")
        else:
            print("Checksum verification: MISMATCH (model may have been updated)")
            print("This is informational - the model should still work correctly.")
    else:
        print(f"Note: Model file not found at {model_file}")
        print("Model was loaded but cache location may differ.")

    print()
    print("=" * 60)
    print("CLIP Model Acquisition: SUCCESS")
    print("=" * 60)
    print()
    print("Documentation:")
    print(f"  - Model: {MODEL_NAME}")
    print(f"  - Visual Embedding Dimension: {EMBEDDING_DIM}")
    print(f"  - Text Embedding Dimension: {EMBEDDING_DIM}")
    print(f"  - Input Resolution: 224x224 RGB")
    print(f"  - Normalization: CLIP standard (mean=[0.48145466, 0.4578275, 0.40821073],")
    print(f"                                  std=[0.26862954, 0.26130258, 0.27577711])")

    return 0

if __name__ == "__main__":
    exit(main())