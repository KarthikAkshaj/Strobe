#!/usr/bin/env python3
"""
Subtitle Export for Semantic Sentinel.
Converts caption results JSON to SRT or WebVTT subtitle files.

Usage:
    python export_subtitles.py --input results.json --format srt --output video.srt
    python export_subtitles.py --input results.json --format vtt --output video.vtt
    python export_subtitles.py --input results.json --format srt --show-confidence
"""

import argparse
import json
import sys
from pathlib import Path


def seconds_to_srt_time(sec: float) -> str:
    """Convert seconds to SRT timecode: HH:MM:SS,mmm"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def seconds_to_vtt_time(sec: float) -> str:
    """Convert seconds to WebVTT timecode: HH:MM:SS.mmm"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def export_srt(captions: list, show_confidence: bool, video_duration: float = None) -> str:
    lines = []
    for i, ev in enumerate(captions):
        start = ev["timestamp"]
        # End time = next event's start, or +5s for last event
        if i + 1 < len(captions):
            end = captions[i + 1]["timestamp"]
        else:
            end = start + 5.0
            if video_duration and end > video_duration:
                end = video_duration

        text = ev["caption"]
        if show_confidence:
            text += f" ({ev.get('confidence', 0) * 100:.1f}%)"

        lines.append(f"{i + 1}")
        lines.append(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def export_vtt(captions: list, show_confidence: bool, video_duration: float = None) -> str:
    lines = ["WEBVTT", ""]
    for i, ev in enumerate(captions):
        start = ev["timestamp"]
        if i + 1 < len(captions):
            end = captions[i + 1]["timestamp"]
        else:
            end = start + 5.0
            if video_duration and end > video_duration:
                end = video_duration

        text = ev["caption"]
        if show_confidence:
            text += f" ({ev.get('confidence', 0) * 100:.1f}%)"

        lines.append(f"{seconds_to_vtt_time(start)} --> {seconds_to_vtt_time(end)}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Export Sentinel results as subtitles")
    parser.add_argument("--input", required=True, help="Results JSON from sentinel engine")
    parser.add_argument("--format", choices=["srt", "vtt"], default="srt", help="Subtitle format")
    parser.add_argument("--output", help="Output file (default: stdout)")
    parser.add_argument("--show-confidence", action="store_true", help="Append confidence %% to each caption")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    captions = data.get("captions", [])
    if not captions:
        print("No captions found in input JSON.", file=sys.stderr)
        sys.exit(1)

    # Sort by timestamp
    captions.sort(key=lambda e: e["timestamp"])

    # Try to get video duration from metadata
    duration = data.get("metadata", {}).get("duration_sec")

    if args.format == "srt":
        result = export_srt(captions, args.show_confidence, duration)
    else:
        result = export_vtt(captions, args.show_confidence, duration)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"Exported {len(captions)} subtitles to {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()
