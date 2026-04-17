#!/usr/bin/env python3
"""
Phase 3 parity test: Compare C++ engine output against Python prototype.

Usage:
  python tests/test_cpp_parity.py [--engine ./build/sentinel_engine.exe] [--videos ...]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON_TOOLS = PROJECT_ROOT / "python_tools"
DEFAULT_ENGINE = PROJECT_ROOT / "build" / "sentinel_engine.exe"
DEFAULT_CAPTION_BANK = PROJECT_ROOT / "models" / "caption_bank.json"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.json"

TEST_VIDEOS = [
    PROJECT_ROOT / "validation" / "test_videos" / "multi_scene"  / "Travel Edit.mp4",
    PROJECT_ROOT / "validation" / "test_videos" / "static"        / "Building Timelapse.mp4",
    PROJECT_ROOT / "validation" / "test_videos" / "gradual_transition" / "Mountain View Timelapse.mp4",
]


def run_python(video_path: Path) -> dict:
    """Run the Python pipeline and return parsed JSON output."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, str(PYTHON_TOOLS / "process_video.py"),
             str(video_path), "--output", out_path,
             "--caption-bank", str(DEFAULT_CAPTION_BANK)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  Python FAILED: {result.stderr[:200]}")
            return {}
        with open(out_path) as f:
            return json.load(f)
    finally:
        try:
            os.unlink(out_path)
        except Exception:
            pass


def run_cpp(video_path: Path, engine_path: Path) -> dict:
    """Run the C++ engine via Python frame encoder and return parsed JSON."""
    encoder = subprocess.Popen(
        [sys.executable, str(PYTHON_TOOLS / "encode_video_frames.py"),
         str(video_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    engine = subprocess.Popen(
        [str(engine_path),
         "--config", str(DEFAULT_CONFIG),
         "--captions", str(DEFAULT_CAPTION_BANK)],
        stdin=encoder.stdout,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    encoder.stdout.close()

    cpp_out, cpp_err = engine.communicate()
    encoder.wait()

    if engine.returncode != 0:
        print(f"  C++ FAILED: {cpp_err.decode()[:200]}")
        return {}

    try:
        return json.loads(cpp_out.decode())
    except json.JSONDecodeError as e:
        print(f"  C++ JSON parse error: {e}")
        return {}


def compare(py_data: dict, cpp_data: dict, video_name: str) -> bool:
    py_caps  = py_data.get("captions", [])
    cpp_caps = cpp_data.get("captions", [])

    if len(py_caps) != len(cpp_caps):
        print(f"  MISMATCH: event count py={len(py_caps)} cpp={len(cpp_caps)}")
        return False

    all_match = True
    for i, (py_ev, cpp_ev) in enumerate(zip(py_caps, cpp_caps)):
        frame_match   = py_ev["frame"]   == cpp_ev["frame"]
        caption_match = py_ev["caption"] == cpp_ev["caption"]
        conf_diff     = abs(py_ev["confidence"] - cpp_ev["confidence"])
        ok = frame_match and caption_match and conf_diff < 1e-3

        status = "PASS" if ok else "FAIL"
        print(f"  Event {i}: [{status}] frame={cpp_ev['frame']}  "
              f"caption=\"{cpp_ev['caption']}\"  "
              f"conf_diff={conf_diff:.5f}")

        if not ok:
            if not caption_match:
                print(f"    Caption mismatch: py=\"{py_ev['caption']}\" cpp=\"{cpp_ev['caption']}\"")
            if not frame_match:
                print(f"    Frame mismatch: py={py_ev['frame']} cpp={cpp_ev['frame']}")
            all_match = False

    return all_match


def main():
    parser = argparse.ArgumentParser(description="C++ parity test")
    parser.add_argument("--engine", default=str(DEFAULT_ENGINE))
    parser.add_argument("--videos", nargs="+", default=[str(v) for v in TEST_VIDEOS])
    args = parser.parse_args()

    engine_path = Path(args.engine)
    if not engine_path.exists():
        print(f"ERROR: Engine not found: {engine_path}")
        print("Build first: cmake --build build")
        return 1

    print("=" * 60)
    print("Phase 3 Parity Test: C++ vs Python")
    print("=" * 60)

    total = 0
    passed = 0

    for video_str in args.videos:
        video_path = Path(video_str)
        if not video_path.exists():
            print(f"\nSkipping (not found): {video_path.name}")
            continue

        print(f"\n--- {video_path.name} ---")
        total += 1

        py_data  = run_python(video_path)
        cpp_data = run_cpp(video_path, engine_path)

        if not py_data or not cpp_data:
            print("  SKIP (pipeline error)")
            continue

        ok = compare(py_data, cpp_data, video_path.name)
        if ok:
            passed += 1
            print(f"  -> PASS ({len(py_data.get('captions', []))} events matched)")
        else:
            print(f"  -> FAIL")

    print()
    print("=" * 60)
    print(f"Result: {passed}/{total} videos passed parity check")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
