#!/usr/bin/env python3
"""Run Phase-2B validation on all test videos. Cross-platform replacement for run_validation.bat."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON_TOOLS = PROJECT_ROOT / "python_tools"
TEST_VIDEOS = Path(__file__).parent / "test_videos"
CAPTION_BANK = PROJECT_ROOT / "models" / "caption_bank.json"
RUN_DIR = Path(__file__).parent / "outputs" / "run_v2"

CATEGORIES = ["static", "single_scene_dynamic", "multi_scene", "gradual_transition", "noisy", "adversarial"]

def main():
    total = 0
    success = 0
    failed = 0

    for cat in CATEGORIES:
        cat_dir = TEST_VIDEOS / cat
        out_dir = RUN_DIR / cat
        out_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted(cat_dir.glob("*.mp4"))
        if not videos:
            continue

        print(f"\n=== Category: {cat} ({len(videos)} videos) ===\n")

        for video in videos:
            total += 1
            out_file = out_dir / f"{video.stem}.captions.json"
            print(f"  Processing: {video.name}")

            result = subprocess.run(
                [sys.executable, str(PYTHON_TOOLS / "process_video.py"),
                 str(video), "--output", str(out_file),
                 "--caption-bank", str(CAPTION_BANK)],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                success += 1
                print(f"    -> OK: {out_file.name}")
            else:
                failed += 1
                print(f"    -> FAILED")
                if result.stderr:
                    print(f"    {result.stderr[:200]}")

    print(f"\n{'='*60}")
    print(f"Validation Complete")
    print(f"{'='*60}")
    print(f"  Total:   {total}")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Output:  {RUN_DIR}")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
