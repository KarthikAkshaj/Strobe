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
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter
import math

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

# Phase-2B: Anomaly detection thresholds (observational only, do not affect decisions)
ANOMALY_LOW_CONFIDENCE = 0.25
ANOMALY_TINY_GAP = 0.01
ANOMALY_HIGH_CHANGE_SIMILARITY = 0.88
ANOMALY_CONSECUTIVE_SUPPRESSION = 3
ANOMALY_CAPTION_STUCK = 5
ANOMALY_RAPID_OSCILLATION_SEC = 2.0
ANOMALY_GAP_COMPRESSION = 0.015
ANOMALY_HIGH_SUPPRESSION_RATE = 0.35
ANOMALY_LOW_CAPTION_ENTROPY = 1.0
ANOMALY_CONFIDENCE_COLLAPSE = 0.15

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
        top_k: int = 3,
        stability_delta: float = 0.02,
    ):
        """
        Initialize the pipeline.

        Args:
            device: Computation device (cuda/cpu)
            target_fps: Target frames per second to extract
            change_threshold: Similarity threshold for change detection
            hysteresis_count: Consecutive frames required to confirm change
            caption_threshold: Minimum similarity for caption matching
            top_k: Number of top captions to return per event
            stability_delta: Minimum score improvement to switch captions
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_fps = target_fps
        self.caption_threshold = caption_threshold
        self.top_k = top_k
        self.stability_delta = stability_delta

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

        # Stability state: track previous caption for anti-flapping
        self._prev_caption_text: Optional[str] = None
        self._prev_caption_score: float = 0.0

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
        """Extract CLIP embedding from a frame (L2-normalized)."""
        tensor = self._preprocess_frame(frame)

        with torch.no_grad():
            embedding = self.model.encode_image(tensor)

        embedding_np = embedding.cpu().numpy().squeeze()

        # L2-normalize to unit vector (CLIP encode_image does not normalize)
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm

        return embedding_np

    def _match_captions_topk(self, embedding: np.ndarray, k: int = 3) -> List[Dict]:
        """
        Find the top-K matching captions for an embedding.

        Args:
            embedding: Frame embedding (512D unit-normalized)
            k: Number of top matches to return

        Returns:
            List of dicts with 'text', 'score', 'index', sorted by score descending
        """
        similarities = np.array([
            self._cosine_similarity(embedding, cap_emb)
            for cap_emb in self.caption_embeddings
        ])

        # Get indices of top-K scores (descending order)
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.captions[idx],
                "score": float(similarities[idx]),
                "index": int(idx),
            })

        return results

    def process_video(self, video_path: Path) -> Tuple[List[Dict], Dict]:
        """
        Process a video and generate timestamped captions.

        Args:
            video_path: Path to input video

        Returns:
            Tuple of (caption_events, diagnostics)
            - caption_events: List of caption events with timestamps
            - diagnostics: Dict with events list and summary statistics
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

        # Reset stability state for new video
        self._prev_caption_text = None
        self._prev_caption_score = 0.0

        # Phase-2B: Diagnostic event collection
        diagnostic_events = []
        event_id = 0
        events_since_change = 0
        consecutive_suppressions = 0
        prev_event_timestamp = None
        prev_event_caption_changed = False
        prev_event_confidence = None

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

                    # Match against caption bank (top-K)
                    top_matches = self._match_captions_topk(embedding, k=self.top_k)

                    # Primary caption is top-1
                    top1 = top_matches[0]
                    top1_text = top1["text"]
                    top1_score = top1["score"]

                    # Compute confidence gap (top1 - top2)
                    confidence_gap = 0.0
                    if len(top_matches) >= 2:
                        confidence_gap = top1_score - top_matches[1]["score"]

                    # Phase-2B: Capture pre-stability state for diagnostics
                    diag_prev_caption = self._prev_caption_text
                    diag_prev_score = self._prev_caption_score

                    # Stability constraint: prevent caption oscillation
                    # If previous caption exists and score difference is below delta,
                    # keep previous caption unless new one is clearly better
                    selected_text = top1_text
                    selected_score = top1_score

                    # Phase-2B: Track if stability suppressed
                    stability_suppressed = False
                    score_improvement = 0.0

                    if self._prev_caption_text is not None:
                        score_improvement = top1_score - self._prev_caption_score
                        if score_improvement < self.stability_delta:
                            # New caption not significantly better, keep previous
                            selected_text = self._prev_caption_text
                            selected_score = self._prev_caption_score
                            stability_suppressed = True

                    # Phase-2B: Track caption change
                    caption_changed = (diag_prev_caption is None or selected_text != diag_prev_caption)
                    if caption_changed:
                        events_since_change = 0
                    else:
                        events_since_change += 1

                    # Phase-2B: Track consecutive suppressions
                    if stability_suppressed:
                        consecutive_suppressions += 1
                    else:
                        consecutive_suppressions = 0

                    # Update stability state
                    self._prev_caption_text = selected_text
                    self._prev_caption_score = selected_score

                    # Only output if above threshold
                    if selected_score >= self.caption_threshold:
                        # Build alternatives list (top-K minus the selected caption)
                        alternatives = [
                            {"text": m["text"], "score": round(m["score"], 4)}
                            for m in top_matches[1:]
                        ]

                        result = {
                            "frame": frame_idx,
                            "timestamp": round(timestamp, 3),
                            "caption": selected_text,
                            "confidence": round(float(selected_score), 4),
                            "confidence_gap": round(float(confidence_gap), 4),
                            "alternatives": alternatives,
                            "change_similarity": round(float(similarity), 4),
                        }
                        results.append(result)

                        print(f"  [{timestamp:6.2f}s] {selected_text} (conf: {selected_score:.3f}, gap: {confidence_gap:.3f})")

                        # Phase-2B: Build diagnostic event
                        event_flags: List[str] = []

                        # Per-event anomaly flags
                        if selected_score < ANOMALY_LOW_CONFIDENCE:
                            event_flags.append("LOW_CONFIDENCE")
                        if confidence_gap < ANOMALY_TINY_GAP:
                            event_flags.append("TINY_GAP")
                        if similarity > ANOMALY_HIGH_CHANGE_SIMILARITY:
                            event_flags.append("HIGH_CHANGE_SIMILARITY")
                        if stability_suppressed:
                            event_flags.append("STABILITY_SUPPRESSED")
                        if consecutive_suppressions >= ANOMALY_CONSECUTIVE_SUPPRESSION:
                            event_flags.append("CONSECUTIVE_SUPPRESSION")
                        if events_since_change >= ANOMALY_CAPTION_STUCK:
                            event_flags.append("CAPTION_STUCK")
                        if (prev_event_caption_changed and caption_changed and
                            prev_event_timestamp is not None and
                            (timestamp - prev_event_timestamp) < ANOMALY_RAPID_OSCILLATION_SEC):
                            event_flags.append("RAPID_OSCILLATION")
                        if (prev_event_confidence is not None and
                            (prev_event_confidence - selected_score) > ANOMALY_CONFIDENCE_COLLAPSE):
                            event_flags.append("CONFIDENCE_COLLAPSE")

                        diagnostic_event = {
                            "event_id": event_id,
                            "frame": frame_idx,
                            "timestamp": round(timestamp, 3),
                            "change_similarity": round(float(similarity), 4),
                            "selected": {
                                "caption": selected_text,
                                "score": round(float(selected_score), 4),
                            },
                            "top_k": [
                                {"caption": m["text"], "score": round(m["score"], 4)}
                                for m in top_matches
                            ],
                            "confidence_gap": round(float(confidence_gap), 4),
                            "stability": {
                                "suppressed": stability_suppressed,
                                "prev_caption": diag_prev_caption,
                                "prev_score": round(float(diag_prev_score), 4) if diag_prev_score else None,
                                "score_improvement": round(float(score_improvement), 4),
                            },
                            "caption_changed": caption_changed,
                            "events_since_change": events_since_change,
                            "flags": event_flags,
                        }
                        diagnostic_events.append(diagnostic_event)

                        # Phase-2B: Update tracking state for next event
                        prev_event_timestamp = timestamp
                        prev_event_caption_changed = caption_changed
                        prev_event_confidence = selected_score
                        event_id += 1

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

        # Phase-2B: Compute summary statistics
        diagnostics = self._compute_diagnostics_summary(
            video_path.name, duration, total_frames, processed_count,
            diagnostic_events
        )

        return results, diagnostics

    def _compute_diagnostics_summary(
        self,
        video_name: str,
        duration_sec: float,
        total_frames: int,
        processed_frames: int,
        events: List[Dict],
    ) -> Dict:
        """
        Compute per-video summary statistics from diagnostic events.

        Phase-2B: Observational only, does not affect caption decisions.
        """
        total_events = len(events)

        # Handle empty events case
        if total_events == 0:
            return {
                "events": events,
                "summary": {
                    "video": video_name,
                    "duration_sec": round(duration_sec, 2),
                    "total_frames": total_frames,
                    "processed_frames": processed_frames,
                    "total_events": 0,
                    "confidence": None,
                    "confidence_gap": None,
                    "change_similarity": None,
                    "stability": {
                        "suppression_count": 0,
                        "suppression_rate": 0.0,
                        "max_consecutive_suppressions": 0,
                    },
                    "captions": {
                        "unique_count": 0,
                        "most_frequent": None,
                        "most_frequent_count": 0,
                        "entropy": 0.0,
                    },
                    "timing": None,
                    "anomaly_flags": [],
                    "flag_counts": {},
                },
            }

        # Extract arrays for statistics
        confidences = [e["selected"]["score"] for e in events]
        gaps = [e["confidence_gap"] for e in events]
        change_sims = [e["change_similarity"] for e in events]
        timestamps = [e["timestamp"] for e in events]
        captions = [e["selected"]["caption"] for e in events]

        # Confidence statistics
        conf_mean = sum(confidences) / total_events
        conf_min = min(confidences)
        conf_max = max(confidences)
        conf_std = math.sqrt(sum((c - conf_mean) ** 2 for c in confidences) / total_events)

        # Confidence gap statistics
        gap_mean = sum(gaps) / total_events
        gap_min = min(gaps)
        gap_max = max(gaps)
        gap_std = math.sqrt(sum((g - gap_mean) ** 2 for g in gaps) / total_events)

        # Change similarity statistics
        cs_mean = sum(change_sims) / total_events
        cs_min = min(change_sims)
        cs_max = max(change_sims)

        # Stability statistics
        suppression_count = sum(1 for e in events if e["stability"]["suppressed"])
        suppression_rate = suppression_count / total_events if total_events > 0 else 0.0

        # Max consecutive suppressions
        max_consec = 0
        current_consec = 0
        for e in events:
            if e["stability"]["suppressed"]:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        # Caption statistics
        caption_counts = Counter(captions)
        unique_count = len(caption_counts)
        most_frequent, most_frequent_count = caption_counts.most_common(1)[0]

        # Caption entropy: -sum(p * log2(p))
        entropy = 0.0
        for count in caption_counts.values():
            p = count / total_events
            if p > 0:
                entropy -= p * math.log2(p)

        # Timing statistics
        inter_event_times = []
        for i in range(1, len(timestamps)):
            inter_event_times.append(timestamps[i] - timestamps[i - 1])

        timing_stats = None
        if inter_event_times:
            timing_stats = {
                "mean_inter_event_sec": round(sum(inter_event_times) / len(inter_event_times), 3),
                "min_inter_event_sec": round(min(inter_event_times), 3),
                "max_inter_event_sec": round(max(inter_event_times), 3),
            }

        # Aggregate flag counts
        flag_counts: Dict[str, int] = {}
        for e in events:
            for flag in e["flags"]:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

        # Per-video anomaly flags
        video_flags: List[str] = []
        if gap_mean < ANOMALY_GAP_COMPRESSION:
            video_flags.append("GAP_COMPRESSION")
        if suppression_rate > ANOMALY_HIGH_SUPPRESSION_RATE:
            video_flags.append("HIGH_SUPPRESSION_RATE")
        if entropy < ANOMALY_LOW_CAPTION_ENTROPY and unique_count > 1:
            video_flags.append("LOW_CAPTION_ENTROPY")

        summary = {
            "video": video_name,
            "duration_sec": round(duration_sec, 2),
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "total_events": total_events,
            "confidence": {
                "mean": round(conf_mean, 4),
                "min": round(conf_min, 4),
                "max": round(conf_max, 4),
                "std": round(conf_std, 4),
            },
            "confidence_gap": {
                "mean": round(gap_mean, 4),
                "min": round(gap_min, 4),
                "max": round(gap_max, 4),
                "std": round(gap_std, 4),
            },
            "change_similarity": {
                "mean": round(cs_mean, 4),
                "min": round(cs_min, 4),
                "max": round(cs_max, 4),
            },
            "stability": {
                "suppression_count": suppression_count,
                "suppression_rate": round(suppression_rate, 4),
                "max_consecutive_suppressions": max_consec,
            },
            "captions": {
                "unique_count": unique_count,
                "most_frequent": most_frequent,
                "most_frequent_count": most_frequent_count,
                "entropy": round(entropy, 4),
            },
            "timing": timing_stats,
            "anomaly_flags": video_flags,
            "flag_counts": flag_counts,
        }

        return {
            "events": events,
            "summary": summary,
        }


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
        "--top-k", type=int, default=3,
        help="Number of top captions to return (default: 3)"
    )
    parser.add_argument(
        "--stability-delta", type=float, default=0.02,
        help="Minimum score improvement to switch captions (default: 0.02)"
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
        top_k=args.top_k,
        stability_delta=args.stability_delta,
    )

    # Load caption bank
    caption_bank_path = Path(args.caption_bank) if args.caption_bank else None
    pipeline.load_caption_bank(caption_bank_path)

    print()

    # Process video
    results, diagnostics = pipeline.process_video(video_path)

    # Save results (Phase-2B: diagnostics added under separate key)
    output_data = {
        "metadata": {
            "video": str(video_path.name),
            "model": MODEL_NAME,
            "target_fps": args.fps,
            "change_threshold": args.change_threshold,
            "hysteresis_count": args.hysteresis,
            "caption_threshold": args.caption_threshold,
            "top_k": args.top_k,
            "stability_delta": args.stability_delta,
        },
        "captions": results,
        "diagnostics": diagnostics,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print()
    print(f"Results saved to: {output_path}")

    # Phase-2B: Print diagnostics summary
    summary = diagnostics.get("summary", {})
    if summary.get("total_events", 0) > 0:
        print()
        print("=" * 60)
        print("PHASE-2B DIAGNOSTICS SUMMARY")
        print("=" * 60)
        print(f"  Events: {summary['total_events']}")
        if summary.get("confidence"):
            print(f"  Confidence: mean={summary['confidence']['mean']:.3f}, "
                  f"std={summary['confidence']['std']:.3f}")
        if summary.get("confidence_gap"):
            print(f"  Confidence gap: mean={summary['confidence_gap']['mean']:.3f}")
        if summary.get("stability"):
            print(f"  Stability suppressions: {summary['stability']['suppression_count']} "
                  f"({summary['stability']['suppression_rate']*100:.1f}%)")
        if summary.get("captions"):
            print(f"  Unique captions: {summary['captions']['unique_count']}, "
                  f"entropy={summary['captions']['entropy']:.2f}")
        if summary.get("anomaly_flags"):
            print(f"  Video anomaly flags: {', '.join(summary['anomaly_flags'])}")
        if summary.get("flag_counts"):
            print(f"  Event flag counts: {summary['flag_counts']}")

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