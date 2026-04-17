#!/usr/bin/env python3
"""Compare v1 and v2 caption bank validation results side-by-side."""

import json
import sys
from pathlib import Path

V1_DIR = Path(__file__).parent / "outputs" / "run_1709-02-202611_2266"
V2_DIR = Path(__file__).parent / "outputs" / "run_v2"

CATEGORIES = ["static", "single_scene_dynamic", "multi_scene", "gradual_transition", "noisy", "adversarial"]


def load_result(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def summarize(data: dict) -> dict:
    """Extract key metrics from a pipeline output."""
    captions = data.get("captions", [])
    diag = data.get("diagnostics", {})
    summary = diag.get("summary", {})

    confidences = [c["confidence"] for c in captions]
    gaps = [c.get("confidence_gap", 0) for c in captions]
    caption_texts = [c["caption"] for c in captions]

    return {
        "num_events": len(captions),
        "mean_conf": sum(confidences) / len(confidences) if confidences else 0,
        "max_conf": max(confidences) if confidences else 0,
        "min_conf": min(confidences) if confidences else 0,
        "mean_gap": sum(gaps) / len(gaps) if gaps else 0,
        "unique_captions": len(set(caption_texts)),
        "captions": caption_texts,
        "confidences": confidences,
        "gaps": gaps,
        "suppression_rate": summary.get("stability", {}).get("suppression_rate", 0),
    }


def main():
    print("=" * 80)
    print("CAPTION BANK v1 vs v2 COMPARISON")
    print("=" * 80)
    print()

    all_v1_confs = []
    all_v2_confs = []
    all_v1_gaps = []
    all_v2_gaps = []
    all_v1_events = 0
    all_v2_events = 0

    for cat in CATEGORIES:
        v1_cat = V1_DIR / cat
        v2_cat = V2_DIR / cat

        if not v1_cat.exists() or not v2_cat.exists():
            continue

        v1_files = sorted(v1_cat.glob("*.captions.json"))

        print(f"\n{'='*80}")
        print(f"CATEGORY: {cat.upper()}")
        print(f"{'='*80}")

        for v1_file in v1_files:
            v2_file = v2_cat / v1_file.name
            if not v2_file.exists():
                print(f"\n  {v1_file.stem}: v2 output MISSING")
                continue

            v1_data = load_result(v1_file)
            v2_data = load_result(v2_file)
            v1_s = summarize(v1_data)
            v2_s = summarize(v2_data)

            video_name = v1_file.stem.replace(".captions", "")
            print(f"\n  --- {video_name} ---")
            print(f"  Events:          v1={v1_s['num_events']:3d}   v2={v2_s['num_events']:3d}   delta={v2_s['num_events'] - v1_s['num_events']:+d}")
            print(f"  Mean confidence: v1={v1_s['mean_conf']:.4f}  v2={v2_s['mean_conf']:.4f}  delta={v2_s['mean_conf'] - v1_s['mean_conf']:+.4f}")
            print(f"  Max confidence:  v1={v1_s['max_conf']:.4f}  v2={v2_s['max_conf']:.4f}  delta={v2_s['max_conf'] - v1_s['max_conf']:+.4f}")
            print(f"  Mean gap:        v1={v1_s['mean_gap']:.4f}  v2={v2_s['mean_gap']:.4f}  delta={v2_s['mean_gap'] - v1_s['mean_gap']:+.4f}")
            print(f"  Unique captions: v1={v1_s['unique_captions']:3d}   v2={v2_s['unique_captions']:3d}")
            print(f"  Suppression:     v1={v1_s['suppression_rate']:.2f}   v2={v2_s['suppression_rate']:.2f}")

            # Show v2 captions
            if v2_s['captions']:
                unique_v2 = []
                seen = set()
                for c in v2_s['captions']:
                    if c not in seen:
                        unique_v2.append(c)
                        seen.add(c)
                print(f"  v2 captions: {unique_v2}")

            # Show v1 captions for comparison
            if v1_s['captions']:
                unique_v1 = []
                seen = set()
                for c in v1_s['captions']:
                    if c not in seen:
                        unique_v1.append(c)
                        seen.add(c)
                print(f"  v1 captions: {unique_v1}")

            all_v1_confs.extend(v1_s['confidences'])
            all_v2_confs.extend(v2_s['confidences'])
            all_v1_gaps.extend(v1_s['gaps'])
            all_v2_gaps.extend(v2_s['gaps'])
            all_v1_events += v1_s['num_events']
            all_v2_events += v2_s['num_events']

    # Global summary
    print(f"\n\n{'='*80}")
    print("GLOBAL SUMMARY")
    print(f"{'='*80}")
    print(f"  Total events:       v1={all_v1_events}  v2={all_v2_events}")
    if all_v1_confs:
        print(f"  Mean confidence:    v1={sum(all_v1_confs)/len(all_v1_confs):.4f}  v2={sum(all_v2_confs)/len(all_v2_confs):.4f}")
    if all_v1_gaps:
        print(f"  Mean gap:           v1={sum(all_v1_gaps)/len(all_v1_gaps):.4f}  v2={sum(all_v2_gaps)/len(all_v2_gaps):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
