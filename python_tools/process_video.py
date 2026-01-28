#!/usr/bin/env python3
"""
End-to-End Video Processing Pipeline for Semantic Sentinel.
Processes a video through the complete semantic captioning pipeline.

This script:
1. Extracts frames at target FPS
2. Generates CLIP embeddings for each frame
3. Applies change detection logic
4. Matches triggered frames against caption bank
5. Outputs timestamped captions

Usage:
    python process_video.py <video_path> [--fps 1] [--output captions.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import clip
from PIL import Image

from change_detector import AnchorChangeDetector

# Constants
MODEL_NAME = "ViT-B/32"
EMBEDDING_DIM = 512
INPUT_SIZE = 224

# Default caption bank (inline for standalone operation)
DEFAULT_CAPTIONS = [
    "a person walking down the street",
    "a group of people talking",
    "someone sitting at a desk",
    "a person running",
    "people standing in a line",
    "an empty room",
    "a living room with furniture",
    "a kitchen with appliances",
    "an office workspace",
    "a bedroom with a bed",
    "a city street with buildings",
    "a park with trees",
    "a beach with ocean waves",
    "a mountain landscape",
    "a forest path",
    "a car on the road",
    "a computer screen",
    "a book on a table",
    "food on a plate",
    "a dark scene",
    "a bright sunny day",
    "motion blur",
    "a static scene with no movement",
]


class SemanticCaptionPipeline:
    """Complete pipeline for semantic video captioning."""

    def __init__(
        self,
        device: str = None,
        target_fps: float = 1.0,
        change_threshold: float = 0.85,
        hysteresis_count: int = 2,
        caption_threshold: float = 0.20,
    ):
        """
        Initialize the pipeline.

        Args:
            device: Computation device (cuda/cpu)
            target_fps: Target frames per second to extract
            change_threshold: Similarity threshold for change detection
            hysteresis_count: Consecutive frames required to confirm change
            caption_threshold: Minimum similarity for caption matching
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_fps = target_fps
        self.caption_threshold = caption_threshold

        # Load CLIP model
        print(f"Loading CLIP model on {self.device}...")
        self.model, self.preprocess = clip.load(MODEL_NAME, device=self.device)
        self.model.eval()

        # Initialize anchor-based change detector
        self.change_detector = AnchorChangeDetector(
            similarity_threshold=change_threshold,
            hysteresis_count=hysteresis_count,
            embedding_dim=EMBEDDING_DIM,
        )

        # Caption bank (will be populated)
        self.captions: List[str] = []
        self.caption_embeddings: Optional[np.ndarray] = None

    def load_caption_bank(self, caption_bank_path: Optional[Path] = None):
        """Load caption bank from JSON file or use defaults."""
        if caption_bank_path and caption_bank_path.exists():
            with open(caption_bank_path, "r") as f:
                data = json.load(f)

            self.captions = [c["text"] for c in data["captions"]]
            self.caption_embeddings = np.array(
                [c["embedding"] for c in data["captions"]], dtype=np.float32
            )
            print(f"Loaded {len(self.captions)} captions from {caption_bank_path}")
        else:
            # Encode default captions
            print("Encoding default caption bank...")
            self.captions = DEFAULT_CAPTIONS
            tokens = clip.tokenize(self.captions, truncate=True).to(self.device)

            with torch.no_grad():
                self.caption_embeddings = (
                    self.model.encode_text(tokens).cpu().numpy()
                )

            print(f"Encoded {len(self.captions)} default captions")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a video frame for CLIP."""
        # OpenCV uses BGR, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Apply CLIP preprocessing
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        return tensor

    def _extract_embedding(self, frame: np.ndarray) -> np.ndarray:
        """Extract CLIP embedding from a frame."""
        tensor = self._preprocess_frame(frame)

        with torch.no_grad():
            embedding = self.model.encode_image(tensor)

        return embedding.cpu().numpy().squeeze()

    def _match_caption(self, embedding: np.ndarray) -> Tuple[str, float, int]:
        """
        Find the best matching caption for an embedding.

        Returns:
            Tuple of (caption_text, similarity_score, caption_index)
        """
        similarities = np.array([
            self._cosine_similarity(embedding, cap_emb)
            for cap_emb in self.caption_embeddings
        ])

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_caption = self.captions[best_idx]

        return best_caption, best_score, best_idx

    def process_video(self, video_path: Path) -> List[Dict]:
        """
        Process a video and generate timestamped captions.

        Args:
            video_path: Path to input video

        Returns:
            List of caption events with timestamps
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        print(f"Video: {video_path.name}")
        print(f"  FPS: {video_fps:.2f}")
        print(f"  Frames: {total_frames}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Target FPS: {self.target_fps}")
        print()

        # Calculate frame interval
        frame_interval = int(video_fps / self.target_fps)
        if frame_interval < 1:
            frame_interval = 1

        # Process frames
        results = []
        frame_idx = 0
        processed_count = 0
        trigger_count = 0

        self.change_detector.reset()

        print("Processing frames...")
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process at target FPS
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / video_fps

                # Extract embedding
                embedding = self._extract_embedding(frame)

                # Check for change (anchor-based detection)
                triggered, similarity = self.change_detector.process(embedding)

                processed_count += 1

                if triggered:
                    trigger_count += 1

                    # Match against caption bank
                    caption, score, caption_idx = self._match_caption(embedding)

                    # Only output if above threshold
                    if score >= self.caption_threshold:
                        result = {
                            "frame": frame_idx,
                            "timestamp": round(timestamp, 3),
                            "caption": caption,
                            "confidence": round(float(score), 4),
                            "change_similarity": round(float(similarity), 4),
                        }
                        results.append(result)

                        print(f"  [{timestamp:6.2f}s] {caption} (conf: {score:.3f})")

            frame_idx += 1

        cap.release()
        elapsed = time.time() - start_time

        print()
        print(f"Processing complete:")
        print(f"  Total frames: {frame_idx}")
        print(f"  Processed frames: {processed_count}")
        print(f"  Triggers: {trigger_count}")
        print(f"  Captions output: {len(results)}")
        print(f"  Elapsed time: {elapsed:.2f}s")
        print(f"  Processing speed: {frame_idx / elapsed:.1f} fps")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Process video through semantic captioning pipeline"
    )
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument(
        "--fps", type=float, default=1.0,
        help="Target FPS for frame extraction (default: 1.0)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--caption-bank", type=str, default=None,
        help="Path to caption bank JSON file"
    )
    parser.add_argument(
        "--change-threshold", type=float, default=0.85,
        help="Change detection threshold (default: 0.85)"
    )
    parser.add_argument(
        "--hysteresis", type=int, default=2,
        help="Consecutive frames to confirm change (default: 2)"
    )
    parser.add_argument(
        "--caption-threshold", type=float, default=0.20,
        help="Minimum caption confidence (default: 0.20)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Computation device (cuda/cpu)"
    )

    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = video_path.with_suffix(".captions.json")

    print("=" * 60)
    print("Semantic Sentinel - End-to-End Video Processing")
    print("=" * 60)
    print()

    # Initialize pipeline
    pipeline = SemanticCaptionPipeline(
        device=args.device,
        target_fps=args.fps,
        change_threshold=args.change_threshold,
        hysteresis_count=args.hysteresis,
        caption_threshold=args.caption_threshold,
    )

    # Load caption bank
    caption_bank_path = Path(args.caption_bank) if args.caption_bank else None
    pipeline.load_caption_bank(caption_bank_path)

    print()

    # Process video
    results = pipeline.process_video(video_path)

    # Save results
    output_data = {
        "metadata": {
            "video": str(video_path.name),
            "model": MODEL_NAME,
            "target_fps": args.fps,
            "change_threshold": args.change_threshold,
            "hysteresis_count": args.hysteresis,
            "caption_threshold": args.caption_threshold,
        },
        "captions": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print()
    print(f"Results saved to: {output_path}")

    # Verify selective readout
    print()
    print("=" * 60)
    print("SELECTIVE READOUT VERIFICATION")
    print("=" * 60)
    total_processed = pipeline.change_detector.frame_count
    caption_count = len(results)

    if caption_count < total_processed:
        print(f"  Processed frames: {total_processed}")
        print(f"  Captions output: {caption_count}")
        print(f"  Selectivity: {100 * (1 - caption_count / max(1, total_processed)):.1f}% frames skipped")
        print()
        print("  PASSED: Not every frame was captioned")
    else:
        print("  WARNING: Every frame was captioned")
        print("  Consider adjusting change_threshold higher")

    return 0


if __name__ == "__main__":
    sys.exit(main())