#!/usr/bin/env python3
"""
Change Detection Logic for Semantic Sentinel.
Implements anchor-based change detection using cosine similarity.

This module:
1. Maintains a single anchor embedding (real frame, not averaged)
2. Compares each frame against the anchor
3. Updates anchor only after confirmed scene change
4. Never averages CLIP embeddings (preserves hypersphere geometry)
"""

import sys
from typing import Optional, Tuple

import numpy as np


class AnchorChangeDetector:
    """
    Detects semantic changes using anchor-based comparison.

    The detector maintains a single anchor embedding representing the current
    scene. When a new frame differs significantly from the anchor (confirmed
    by hysteresis), a change is triggered and the anchor updates.

    Key properties:
    - Anchor is always a real frame embedding (unit-normalized)
    - No averaging of CLIP embeddings (avoids geometric collapse)
    - No drift over time
    - Re-stabilizes after scene changes
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        hysteresis_count: int = 2,
        embedding_dim: int = 512,
    ):
        """
        Initialize the anchor-based change detector.

        Args:
            similarity_threshold: Trigger when similarity drops below this value
            hysteresis_count: Number of consecutive low-similarity frames required
                              before confirming a scene change (reduces false triggers)
            embedding_dim: Dimension of embedding vectors (512 for CLIP ViT-B/32)
        """
        self.similarity_threshold = similarity_threshold
        self.hysteresis_count = hysteresis_count
        self.embedding_dim = embedding_dim

        # Anchor embedding (always a real frame, never averaged)
        self.anchor: Optional[np.ndarray] = None

        # Hysteresis counter for consecutive low-similarity frames
        self._consecutive_changes = 0

        # Statistics
        self.frame_count = 0
        self.trigger_count = 0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two unit-normalized vectors.

        For unit vectors: cosine_similarity = dot_product
        We still compute the full formula for safety (handles non-normalized input).
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def process(self, embedding: np.ndarray) -> Tuple[bool, float]:
        """
        Process a new embedding and check for semantic change.

        Args:
            embedding: The new frame embedding (512D unit-normalized vector)

        Returns:
            Tuple of:
                - triggered: True if a semantic change was confirmed
                - similarity: Similarity score to current anchor
        """
        self.frame_count += 1

        # Validate input shape
        if embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"Expected shape ({self.embedding_dim},), got {embedding.shape}"
            )

        # Validate embedding is approximately unit-normalized (defensive check)
        norm = np.linalg.norm(embedding)
        if not (0.9 <= norm <= 1.1):
            raise ValueError(
                f"Embedding norm {norm:.4f} outside valid range [0.9, 1.1]. "
                "CLIP embeddings should be unit-normalized."
            )

        # First frame: set as anchor, no trigger
        if self.anchor is None:
            self.anchor = embedding.copy()
            return False, 1.0

        # Compute similarity to anchor
        similarity = self._cosine_similarity(embedding, self.anchor)

        # Check if this frame differs from anchor
        is_different = similarity < self.similarity_threshold

        if is_different:
            self._consecutive_changes += 1

            # Check hysteresis: require consecutive low-similarity frames
            if self._consecutive_changes >= self.hysteresis_count:
                # Confirmed scene change
                self.trigger_count += 1
                self.anchor = embedding.copy()  # New scene becomes anchor
                self._consecutive_changes = 0
                return True, similarity
        else:
            # Frame matches anchor: reset hysteresis counter
            self._consecutive_changes = 0

        return False, similarity

    def reset(self):
        """Reset the detector state."""
        self.anchor = None
        self._consecutive_changes = 0
        self.frame_count = 0
        self.trigger_count = 0

    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            "frame_count": self.frame_count,
            "trigger_count": self.trigger_count,
            "trigger_rate": self.trigger_count / max(1, self.frame_count),
            "has_anchor": self.anchor is not None,
            "consecutive_changes": self._consecutive_changes,
            "threshold": self.similarity_threshold,
            "hysteresis": self.hysteresis_count,
        }


# Backward compatibility alias
RollingWindowChangeDetector = AnchorChangeDetector


def test_similar_frames():
    """Test with sequence of similar frames (expect no triggers)."""
    print("Test 1: Similar frames (expect NO triggers)")
    print("-" * 50)

    detector = AnchorChangeDetector(similarity_threshold=0.85, hysteresis_count=2)

    # Generate base embedding
    np.random.seed(42)
    base_embedding = np.random.randn(512).astype(np.float32)
    base_embedding = base_embedding / np.linalg.norm(base_embedding)

    # Pre-generate all noise vectors with controlled magnitude
    # In high dimensions, randn(512) has magnitude ≈ sqrt(512) ≈ 22.6
    # So we scale by 0.02 / sqrt(512) to get ~2% perturbation in angle
    noise_scale = 0.02 / np.sqrt(512)
    all_noise = np.random.randn(20, 512).astype(np.float32) * noise_scale

    triggers = []
    for i in range(20):
        # Apply small perturbation (maintains high similarity to base)
        embedding = base_embedding + all_noise[i]
        embedding = embedding / np.linalg.norm(embedding)

        triggered, similarity = detector.process(embedding)
        triggers.append(triggered)
        print(f"  Frame {i + 1:2d}: similarity={similarity:.4f}, triggered={triggered}")

    total_triggers = sum(triggers)
    print()
    print(f"  Total triggers: {total_triggers} (expected: 0)")

    return total_triggers == 0


def test_scene_change():
    """Test with sequence containing scene change (expect exactly 1 trigger)."""
    print()
    print("Test 2: Scene change (expect 1 trigger around frame 10-11)")
    print("-" * 50)

    detector = AnchorChangeDetector(similarity_threshold=0.85, hysteresis_count=2)

    np.random.seed(42)

    # Scene 1: frames 1-9
    scene1_base = np.random.randn(512).astype(np.float32)
    scene1_base = scene1_base / np.linalg.norm(scene1_base)

    # Scene 2: frames 10-20 (different direction)
    scene2_base = np.random.randn(512).astype(np.float32)
    scene2_base = scene2_base / np.linalg.norm(scene2_base)

    # Pre-generate noise with controlled magnitude
    noise_scale = 0.02 / np.sqrt(512)
    all_noise = np.random.randn(20, 512).astype(np.float32) * noise_scale

    triggers = []
    trigger_frames = []

    for i in range(20):
        # Use scene1 for first 9 frames, scene2 after
        if i < 9:
            base = scene1_base
        else:
            base = scene2_base

        embedding = base + all_noise[i]
        embedding = embedding / np.linalg.norm(embedding)

        triggered, similarity = detector.process(embedding)
        triggers.append(triggered)
        if triggered:
            trigger_frames.append(i + 1)
        print(f"  Frame {i + 1:2d}: similarity={similarity:.4f}, triggered={triggered}")

    print()
    print(f"  Triggers at frames: {trigger_frames}")
    print(f"  Expected: exactly 1 trigger around frame 10-11")

    # Should have exactly 1 trigger, around frame 10-11 (accounting for hysteresis)
    return len(trigger_frames) == 1 and trigger_frames[0] in [10, 11, 12]


def test_restabilization():
    """Test that detector re-stabilizes after a scene change."""
    print()
    print("Test 3: Re-stabilization after scene change")
    print("-" * 50)

    detector = AnchorChangeDetector(similarity_threshold=0.85, hysteresis_count=2)

    np.random.seed(42)

    # Scene 1
    scene1_base = np.random.randn(512).astype(np.float32)
    scene1_base = scene1_base / np.linalg.norm(scene1_base)

    # Scene 2
    scene2_base = np.random.randn(512).astype(np.float32)
    scene2_base = scene2_base / np.linalg.norm(scene2_base)

    # Pre-generate noise with controlled magnitude for all 20 frames
    noise_scale = 0.02 / np.sqrt(512)
    all_noise = np.random.randn(20, 512).astype(np.float32) * noise_scale

    trigger_frames = []

    # Phase 1: 5 frames of scene 1
    print("  Phase 1: Scene 1 (frames 1-5)")
    for i in range(5):
        embedding = scene1_base + all_noise[i]
        embedding = embedding / np.linalg.norm(embedding)
        triggered, similarity = detector.process(embedding)
        if triggered:
            trigger_frames.append(i + 1)
        print(f"    Frame {i + 1}: similarity={similarity:.4f}, triggered={triggered}")

    # Phase 2: 10 frames of scene 2 (should trigger once, then stabilize)
    print("  Phase 2: Scene 2 (frames 6-15)")
    for i in range(5, 15):
        embedding = scene2_base + all_noise[i]
        embedding = embedding / np.linalg.norm(embedding)
        triggered, similarity = detector.process(embedding)
        if triggered:
            trigger_frames.append(i + 1)
        print(f"    Frame {i + 1}: similarity={similarity:.4f}, triggered={triggered}")

    # Phase 3: 5 more frames of scene 2 (should NOT trigger - anchor updated)
    print("  Phase 3: Scene 2 continued (frames 16-20)")
    for i in range(15, 20):
        embedding = scene2_base + all_noise[i]
        embedding = embedding / np.linalg.norm(embedding)
        triggered, similarity = detector.process(embedding)
        if triggered:
            trigger_frames.append(i + 1)
        print(f"    Frame {i + 1}: similarity={similarity:.4f}, triggered={triggered}")

    print()
    print(f"  Triggers at frames: {trigger_frames}")
    print(f"  Expected: exactly 1 trigger (at scene transition)")

    # Should trigger exactly once at the scene change, then stabilize
    return len(trigger_frames) == 1


def test_identical_frames():
    """Test that identical frames never trigger."""
    print()
    print("Test 4: Identical frames (expect 0 triggers)")
    print("-" * 50)

    detector = AnchorChangeDetector(similarity_threshold=0.85, hysteresis_count=2)

    np.random.seed(42)
    embedding = np.random.randn(512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)

    triggers = []
    for i in range(20):
        # Exact same embedding every frame
        triggered, similarity = detector.process(embedding.copy())
        triggers.append(triggered)
        print(f"  Frame {i + 1:2d}: similarity={similarity:.4f}, triggered={triggered}")

    total_triggers = sum(triggers)
    print()
    print(f"  Total triggers: {total_triggers} (expected: 0)")

    return total_triggers == 0


def document_thresholds():
    """Document optimal threshold values based on anchor-based detection."""
    print()
    print("=" * 60)
    print("ANCHOR-BASED CHANGE DETECTION DOCUMENTATION")
    print("=" * 60)
    print()
    print("Algorithm:")
    print("  - Maintain a single anchor embedding (real frame)")
    print("  - Compare each frame to anchor via cosine similarity")
    print("  - Trigger only after N consecutive low-similarity frames")
    print("  - On trigger: new frame becomes new anchor")
    print()
    print("Key Properties:")
    print("  - No embedding averaging (preserves CLIP geometry)")
    print("  - No drift over time (anchor is static until change)")
    print("  - Re-stabilizes automatically after scene change")
    print("  - Identical frames never trigger")
    print()
    print("Recommended Configuration:")
    print()
    print("  Similarity Threshold: 0.85")
    print("    - Triggers on significant scene changes")
    print("    - Ignores minor frame-to-frame variations")
    print("    - Tune lower (0.80) for more sensitivity")
    print("    - Tune higher (0.90) for less sensitivity")
    print()
    print("  Hysteresis Count: 2")
    print("    - Requires 2 consecutive different frames to trigger")
    print("    - Reduces false triggers from single-frame glitches")
    print("    - Increase to 3 for noisy video sources")
    print()


def main():
    print("=" * 60)
    print("Semantic Sentinel - Anchor-Based Change Detection")
    print("=" * 60)
    print()

    # Run all tests
    test1_passed = test_similar_frames()
    test2_passed = test_scene_change()
    test3_passed = test_restabilization()
    test4_passed = test_identical_frames()

    # Document thresholds
    document_thresholds()

    all_passed = test1_passed and test2_passed and test3_passed and test4_passed

    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("-" * 60)
    print(f"  Test 1 (similar frames):    {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Test 2 (scene change):      {'PASS' if test2_passed else 'FAIL'}")
    print(f"  Test 3 (re-stabilization):  {'PASS' if test3_passed else 'FAIL'}")
    print(f"  Test 4 (identical frames):  {'PASS' if test4_passed else 'FAIL'}")
    print("-" * 60)
    if all_passed:
        print("Change Detection Verification: PASSED")
    else:
        print("Change Detection Verification: FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
