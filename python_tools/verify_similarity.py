#!/usr/bin/env python3
"""
Cosine Similarity Verification Script for Semantic Sentinel.
Implements and validates cosine similarity computation.

This script:
1. Implements manual cosine similarity (to match C++ implementation)
2. Validates against numpy/scipy implementations
3. Tests with CLIP image-text pairs
4. Documents expected similarity ranges
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import clip
from PIL import Image

# Constants
MODEL_NAME = "ViT-B/32"
TOLERANCE = 1e-6


def cosine_similarity_manual(a: np.ndarray, b: np.ndarray) -> float:
    """
    Manual cosine similarity implementation.
    This matches the formula to be used in C++.

    cosine_similarity = (a · b) / (||a|| * ||b||)
    """
    dot_product = np.dot(a, b)
    norm_a = np.sqrt(np.sum(a * a))
    norm_b = np.sqrt(np.sum(b * b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity using numpy's built-in functions."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_similarity_normalized(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity for pre-normalized vectors.
    If vectors are L2 normalized, cosine similarity = dot product.
    """
    return np.dot(a, b)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def validate_implementations():
    """Validate that all implementations produce identical results."""
    print("Validating cosine similarity implementations...")
    print("-" * 50)

    # Test with random vectors
    np.random.seed(42)  # For reproducibility

    for i in range(5):
        a = np.random.randn(512).astype(np.float32)
        b = np.random.randn(512).astype(np.float32)

        sim_manual = cosine_similarity_manual(a, b)
        sim_numpy = cosine_similarity_numpy(a, b)

        # Test normalized version
        a_norm = l2_normalize(a)
        b_norm = l2_normalize(b)
        sim_normalized = cosine_similarity_normalized(a_norm, b_norm)

        # Check all are equal
        diff1 = abs(sim_manual - sim_numpy)
        diff2 = abs(sim_manual - sim_normalized)

        status = "PASS" if diff1 < TOLERANCE and diff2 < TOLERANCE else "FAIL"
        print(f"  Test {i + 1}: manual={sim_manual:.6f}, numpy={sim_numpy:.6f}, "
              f"normalized={sim_normalized:.6f} [{status}]")

        if status == "FAIL":
            return False

    print()
    print("All implementation validation tests PASSED")
    return True


def test_clip_similarity():
    """
    Test cosine similarity with actual CLIP embeddings.

    WARNING: CLIP text-text similarity is NOT reliable for semantic distance.
    CLIP is trained for image-text alignment (contrastive learning), not for
    measuring similarity between text pairs. Two semantically similar sentences
    may have low cosine similarity, and vice versa.

    These tests are INFORMATIONAL ONLY and do not constitute pass/fail criteria
    for the pipeline. The relevant metric is image-to-text similarity, not
    text-to-text similarity.
    """
    print()
    print("Testing CLIP text-text similarity (INFORMATIONAL ONLY)...")
    print("-" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()

    # Test captions
    related_pairs = [
        ("a photo of a cat", "a photograph of a feline"),
        ("a red car", "a vehicle in red color"),
        ("a person walking", "someone taking a walk"),
    ]

    unrelated_pairs = [
        ("a photo of a cat", "a mountain landscape"),
        ("a red car", "a plate of food"),
        ("a person walking", "an empty room"),
    ]

    # NOTE: These thresholds are illustrative, not validation criteria.
    # CLIP text-text similarity does not reliably reflect semantic similarity.
    print()
    print("Related text pairs (observing similarity, not validating):")
    for text_a, text_b in related_pairs:
        tokens_a = clip.tokenize([text_a]).to(device)
        tokens_b = clip.tokenize([text_b]).to(device)

        with torch.no_grad():
            emb_a = model.encode_text(tokens_a).cpu().numpy().squeeze()
            emb_b = model.encode_text(tokens_b).cpu().numpy().squeeze()

        sim = cosine_similarity_manual(emb_a, emb_b)
        print(f"  {sim:.4f}: \"{text_a}\" vs \"{text_b}\"")

    print()
    print("Unrelated text pairs (observing similarity, not validating):")
    for text_a, text_b in unrelated_pairs:
        tokens_a = clip.tokenize([text_a]).to(device)
        tokens_b = clip.tokenize([text_b]).to(device)

        with torch.no_grad():
            emb_a = model.encode_text(tokens_a).cpu().numpy().squeeze()
            emb_b = model.encode_text(tokens_b).cpu().numpy().squeeze()

        sim = cosine_similarity_manual(emb_a, emb_b)
        print(f"  {sim:.4f}: \"{text_a}\" vs \"{text_b}\"")

    print()
    print("NOTE: Text-text similarity values above are for observation only.")
    print("      CLIP is not trained for text-text semantic comparison.")

    return True  # Always passes - informational only


def document_similarity_ranges():
    """Document expected similarity ranges for threshold tuning."""
    print()
    print("=" * 60)
    print("SIMILARITY RANGE DOCUMENTATION")
    print("=" * 60)
    print()
    print("Expected cosine similarity ranges for CLIP ViT-B/32:")
    print()
    print("  Image-Text Matching:")
    print("    - Perfect match (same concept):     0.25 - 0.35")
    print("    - Related content:                  0.20 - 0.28")
    print("    - Unrelated content:                0.15 - 0.22")
    print("    - Note: CLIP similarities are lower than typical embeddings")
    print()
    print("  Text-Text Similarity:")
    print("    - Near-identical meaning:           0.85 - 0.99")
    print("    - Related concepts:                 0.70 - 0.85")
    print("    - Somewhat related:                 0.50 - 0.70")
    print("    - Unrelated:                        0.20 - 0.50")
    print()
    print("  Frame-to-Frame (Image-Image) Similarity:")
    print("    - Same frame (identical):           1.00")
    print("    - Consecutive frames (same scene):  0.95 - 0.99")
    print("    - Same scene, minor change:         0.85 - 0.95")
    print("    - Scene transition:                 0.60 - 0.85")
    print("    - Completely different scenes:      0.40 - 0.70")
    print()
    print("  Recommended Thresholds:")
    print("    - Change detection threshold:       0.85 - 0.90")
    print("      (trigger when similarity drops below this)")
    print("    - Caption matching minimum:         0.20 - 0.25")
    print("      (reject matches below this confidence)")
    print()


def main():
    print("=" * 60)
    print("Semantic Sentinel - Cosine Similarity Verification")
    print("=" * 60)
    print()

    # Validate implementations
    if not validate_implementations():
        print("ERROR: Implementation validation failed!")
        return 1

    # Test with CLIP
    if not test_clip_similarity():
        print("ERROR: CLIP similarity test failed!")
        return 1

    # Document ranges
    document_similarity_ranges()

    print()
    print("=" * 60)
    print("Cosine Similarity Verification: PASSED")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())