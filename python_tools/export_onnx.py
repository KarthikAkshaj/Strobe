#!/usr/bin/env python3
"""
ONNX Export Script for Semantic Sentinel.
Exports CLIP ViT-B/32 visual encoder to ONNX format.

This exports ONLY the visual encoder. The text encoder is not needed
at runtime because caption embeddings are pre-computed and stored
in the caption bank.

Usage:
    python export_onnx.py [--output <output_path>]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import clip

# Constants (must match process_video.py)
MODEL_NAME = "ViT-B/32"
EMBEDDING_DIM = 512
INPUT_SIZE = 224
INPUT_SHAPE = (1, 3, INPUT_SIZE, INPUT_SIZE)
OPSET_VERSION = 14


def export_visual_encoder(output_path: Path, device: str = "cpu"):
    """Export CLIP visual encoder to ONNX."""

    print(f"Loading CLIP {MODEL_NAME} on {device}...")
    model, _ = clip.load(MODEL_NAME, device=device)
    model.eval()

    visual = model.visual
    visual.eval()

    # Create dummy input matching expected shape
    dummy_input = torch.randn(*INPUT_SHAPE, device=device)

    print(f"Input shape: {INPUT_SHAPE}")
    print(f"Expected output shape: (1, {EMBEDDING_DIM})")
    print()

    # Verify PyTorch output before export
    with torch.no_grad():
        pytorch_output = visual(dummy_input)
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"PyTorch output dtype: {pytorch_output.dtype}")
    print(f"PyTorch output norm: {pytorch_output.float().norm().item():.4f}")
    print()

    # Export to ONNX
    print(f"Exporting to ONNX (opset {OPSET_VERSION})...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        visual,
        dummy_input,
        str(output_path),
        opset_version=OPSET_VERSION,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes=None,  # Fixed batch size of 1
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported to: {output_path} ({file_size_mb:.1f} MB)")
    print()

    # Validate with ONNX checker
    print("Validating ONNX model structure...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("  ONNX checker: PASSED")

    # Verify input/output metadata
    graph = onnx_model.graph
    input_info = graph.input[0]
    output_info = graph.output[0]
    print(f"  Input name: {input_info.name}")
    print(f"  Output name: {output_info.name}")
    print()

    # Quick inference test with ONNX Runtime
    print("Testing ONNX Runtime inference...")
    session = ort.InferenceSession(str(output_path))

    input_np = dummy_input.cpu().numpy()
    onnx_output = session.run(["embedding"], {"image": input_np})[0]

    print(f"  ONNX output shape: {onnx_output.shape}")
    print(f"  ONNX output dtype: {onnx_output.dtype}")
    print(f"  ONNX output norm: {np.linalg.norm(onnx_output):.4f}")
    print()

    # Compare PyTorch vs ONNX
    pytorch_np = pytorch_output.cpu().numpy()
    max_diff = np.abs(pytorch_np - onnx_output).max()
    cosine_sim = np.dot(pytorch_np.flatten(), onnx_output.flatten()) / (
        np.linalg.norm(pytorch_np) * np.linalg.norm(onnx_output)
    )

    print("Parity check (PyTorch vs ONNX, same input):")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Cosine similarity: {cosine_sim:.8f}")

    if max_diff <= 1e-5:
        print("  PASSED: Numerical parity within tolerance")
    else:
        print(f"  WARNING: Max diff {max_diff:.2e} exceeds 1e-5 tolerance")
        print("  This may be acceptable for FP32 but should be investigated")

    print()
    print("=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Input: {INPUT_SHAPE} float32")
    print(f"  Output: (1, {EMBEDDING_DIM}) float32 (unnormalized)")
    print(f"  Opset: {OPSET_VERSION}")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Parity: max_diff={max_diff:.2e}, cosine={cosine_sim:.8f}")
    print()
    print("NOTE: Output is NOT L2-normalized. Runtime must normalize")
    print("embeddings after inference (same as Python pipeline).")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Export CLIP visual encoder to ONNX"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="models/clip_visual_vit_b32.onnx",
        help="Output ONNX file path (default: models/clip_visual_vit_b32.onnx)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for export (default: cpu)"
    )

    args = parser.parse_args()
    output_path = Path(args.output)

    print("=" * 60)
    print("Semantic Sentinel - ONNX Visual Encoder Export")
    print("=" * 60)
    print()

    return export_visual_encoder(output_path, args.device)


if __name__ == "__main__":
    sys.exit(main())
