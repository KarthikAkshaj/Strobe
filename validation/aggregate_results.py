#!/usr/bin/env python3
"""
Phase-2B Validation Aggregator

Reads all output JSONs and review files from a validation run directory,
computes cross-video statistics, and produces an aggregate_summary.json.

Usage:
    python validation/aggregate_results.py <run_directory>

Example:
    python validation/aggregate_results.py validation/outputs/run_20260209_143000
"""

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

CATEGORIES = [
    "static",
    "single_scene_dynamic",
    "multi_scene",
    "gradual_transition",
    "noisy",
    "adversarial",
]


def load_output_jsons(run_dir: Path) -> list:
    """Load all .captions.json files from a run directory."""
    results = []
    for category in CATEGORIES:
        cat_dir = run_dir / category
        if not cat_dir.exists():
            continue
        for json_file in sorted(cat_dir.glob("*.captions.json")):
            with open(json_file, "r") as f:
                data = json.load(f)
            results.append(
                {
                    "file": str(json_file.name),
                    "category": category,
                    "path": str(json_file),
                    "data": data,
                }
            )
    return results


def parse_review_verdict(review_path: Path) -> str:
    """Parse verdict from a review .txt file. Returns PASS/BORDERLINE/FAIL/UNREVIEWED."""
    if not review_path.exists():
        return "UNREVIEWED"
    text = review_path.read_text(encoding="utf-8", errors="replace")
    # Look for [X] PASS, [X] BORDERLINE, [X] FAIL (case-insensitive)
    if re.search(r"\[x\]\s*PASS", text, re.IGNORECASE):
        return "PASS"
    if re.search(r"\[x\]\s*BORDERLINE", text, re.IGNORECASE):
        return "BORDERLINE"
    if re.search(r"\[x\]\s*FAIL", text, re.IGNORECASE):
        return "FAIL"
    return "UNREVIEWED"


def load_review_verdicts(run_dir: Path) -> dict:
    """Load all review verdicts from the reviews/ directory."""
    reviews_dir = run_dir / "reviews"
    verdicts = {}
    if not reviews_dir.exists():
        return verdicts
    for review_file in reviews_dir.glob("*.review.txt"):
        video_name = review_file.stem.replace(".review", "")
        verdicts[video_name] = parse_review_verdict(review_file)
    return verdicts


def safe_mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def safe_std(values: list) -> float:
    if not values:
        return 0.0
    m = safe_mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def aggregate(run_dir: Path) -> dict:
    """Aggregate all results from a validation run."""
    outputs = load_output_jsons(run_dir)
    verdicts = load_review_verdicts(run_dir)

    if not outputs:
        print(f"Error: No output JSONs found in {run_dir}")
        sys.exit(1)

    # Per-category breakdown
    by_category = {}
    for category in CATEGORIES:
        cat_outputs = [o for o in outputs if o["category"] == category]
        if not cat_outputs:
            continue

        cat_verdicts = []
        cat_video_flags = []
        cat_event_flags = Counter()
        cat_confidences = []
        cat_gaps = []
        cat_suppression_rates = []
        cat_events_per_min = []

        for o in cat_outputs:
            video_name = o["file"].replace(".captions.json", "")
            verdict = verdicts.get(video_name, "UNREVIEWED")
            cat_verdicts.append(verdict)

            summary = o["data"].get("diagnostics", {}).get("summary", {})
            if not summary:
                continue

            total_events = summary.get("total_events", 0)
            duration = summary.get("duration_sec", 0)

            if duration > 0:
                cat_events_per_min.append(total_events / (duration / 60.0))

            if total_events > 0:
                conf = summary.get("confidence", {})
                if conf:
                    cat_confidences.append(conf.get("mean", 0))

                gap = summary.get("confidence_gap", {})
                if gap:
                    cat_gaps.append(gap.get("mean", 0))

                stab = summary.get("stability", {})
                if stab:
                    cat_suppression_rates.append(stab.get("suppression_rate", 0))

            video_flags = summary.get("anomaly_flags", [])
            if video_flags:
                cat_video_flags.append({"video": video_name, "flags": video_flags})

            for flag, count in summary.get("flag_counts", {}).items():
                cat_event_flags[flag] += count

        verdict_counts = Counter(cat_verdicts)
        by_category[category] = {
            "count": len(cat_outputs),
            "pass": verdict_counts.get("PASS", 0),
            "borderline": verdict_counts.get("BORDERLINE", 0),
            "fail": verdict_counts.get("FAIL", 0),
            "unreviewed": verdict_counts.get("UNREVIEWED", 0),
            "mean_events_per_minute": round(safe_mean(cat_events_per_min), 2),
            "mean_confidence": (
                round(safe_mean(cat_confidences), 4) if cat_confidences else None
            ),
            "mean_confidence_gap": round(safe_mean(cat_gaps), 4) if cat_gaps else None,
            "mean_suppression_rate": (
                round(safe_mean(cat_suppression_rates), 4)
                if cat_suppression_rates
                else None
            ),
            "videos_with_flags": cat_video_flags,
            "event_flag_totals": dict(cat_event_flags),
        }

    # Overall verdict counts
    all_verdicts = []
    for cat_data in by_category.values():
        all_verdicts.extend(
            ["PASS"] * cat_data["pass"]
            + ["BORDERLINE"] * cat_data["borderline"]
            + ["FAIL"] * cat_data["fail"]
            + ["UNREVIEWED"] * cat_data["unreviewed"]
        )
    overall_counts = Counter(all_verdicts)
    total_videos = len(outputs)
    reviewed = total_videos - overall_counts.get("UNREVIEWED", 0)

    # Cross-video metrics from all summaries
    all_conf_means = []
    all_gap_means = []
    all_suppression_rates = []
    all_max_consec = []
    systemic_video_flags = Counter()
    total_event_flags = Counter()

    for o in outputs:
        summary = o["data"].get("diagnostics", {}).get("summary", {})
        if not summary or summary.get("total_events", 0) == 0:
            continue

        conf = summary.get("confidence", {})
        if conf:
            all_conf_means.append(conf.get("mean", 0))

        gap = summary.get("confidence_gap", {})
        if gap:
            all_gap_means.append(gap.get("mean", 0))

        stab = summary.get("stability", {})
        if stab:
            all_suppression_rates.append(stab.get("suppression_rate", 0))
            all_max_consec.append(stab.get("max_consecutive_suppressions", 0))

        for flag in summary.get("anomaly_flags", []):
            systemic_video_flags[flag] += 1

        for flag, count in summary.get("flag_counts", {}).items():
            total_event_flags[flag] += count

    # Build aggregate
    aggregate_data = {
        "run_id": run_dir.name,
        "total_videos": total_videos,
        "by_category": by_category,
        "overall": {
            "pass": overall_counts.get("PASS", 0),
            "borderline": overall_counts.get("BORDERLINE", 0),
            "fail": overall_counts.get("FAIL", 0),
            "unreviewed": overall_counts.get("UNREVIEWED", 0),
            "pass_rate": (
                round(overall_counts.get("PASS", 0) / max(1, reviewed), 4)
                if reviewed > 0
                else None
            ),
            "pass_plus_borderline_rate": (
                round(
                    (
                        overall_counts.get("PASS", 0)
                        + overall_counts.get("BORDERLINE", 0)
                    )
                    / max(1, reviewed),
                    4,
                )
                if reviewed > 0
                else None
            ),
        },
        "systemic_flags": dict(systemic_video_flags),
        "event_flag_totals": dict(total_event_flags),
        "cross_video_metrics": {
            "confidence_mean_of_means": (
                round(safe_mean(all_conf_means), 4) if all_conf_means else None
            ),
            "confidence_gap_mean_of_means": (
                round(safe_mean(all_gap_means), 4) if all_gap_means else None
            ),
            "suppression_rate_mean": (
                round(safe_mean(all_suppression_rates), 4)
                if all_suppression_rates
                else None
            ),
            "max_consecutive_suppressions_worst": (
                max(all_max_consec) if all_max_consec else 0
            ),
        },
        "go_criteria": {
            "no_systematic_failures": None,
            "suppression_bounded_lte_5": None,
            "mean_confidence_gt_020": None,
            "mean_gap_gt_001": None,
            "spot_checks_gt_50pct_correct": None,
        },
        "nogo_criteria": {
            "mean_confidence_lt_015": None,
            "gt_30pct_gap_compression": None,
            "gt_30pct_high_suppression_rate": None,
            "gt_50pct_spot_checks_wrong": None,
        },
    }

    # Auto-fill criteria that can be determined from data
    videos_with_events = len(all_conf_means)

    # GO criteria
    if all_max_consec:
        aggregate_data["go_criteria"]["suppression_bounded_lte_5"] = (
            max(all_max_consec) <= 5
        )
    if all_conf_means:
        aggregate_data["go_criteria"]["mean_confidence_gt_020"] = (
            safe_mean(all_conf_means) > 0.20
        )
    if all_gap_means:
        aggregate_data["go_criteria"]["mean_gap_gt_001"] = (
            safe_mean(all_gap_means) > 0.01
        )

    # Check systematic failures: >50% of any category with video-level flags
    systematic = True
    for cat, cat_data in by_category.items():
        flagged = len(cat_data["videos_with_flags"])
        total = cat_data["count"]
        if total > 0 and flagged / total > 0.5:
            systematic = False
            break
    aggregate_data["go_criteria"]["no_systematic_failures"] = systematic

    # NO-GO criteria
    if all_conf_means:
        aggregate_data["nogo_criteria"]["mean_confidence_lt_015"] = (
            safe_mean(all_conf_means) < 0.15
        )
    if videos_with_events > 0:
        gap_compression_count = systemic_video_flags.get("GAP_COMPRESSION", 0)
        aggregate_data["nogo_criteria"]["gt_30pct_gap_compression"] = (
            gap_compression_count / videos_with_events > 0.30
        )
        high_supp_count = systemic_video_flags.get("HIGH_SUPPRESSION_RATE", 0)
        aggregate_data["nogo_criteria"]["gt_30pct_high_suppression_rate"] = (
            high_supp_count / videos_with_events > 0.30
        )

    return aggregate_data


def main():
    if len(sys.argv) != 2:
        print("Usage: python aggregate_results.py <run_directory>")
        print(
            "Example: python aggregate_results.py validation/outputs/run_20260209_143000"
        )
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    print(f"Aggregating results from: {run_dir}")
    print()

    result = aggregate(run_dir)

    # Write aggregate summary
    output_path = run_dir / "aggregate_summary.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary to console
    print(f"Total videos: {result['total_videos']}")
    print()

    print("Per-category breakdown:")
    for cat, data in result["by_category"].items():
        print(
            f"  {cat}: {data['count']} videos | "
            f"P:{data['pass']} B:{data['borderline']} F:{data['fail']} U:{data['unreviewed']} | "
            f"events/min: {data['mean_events_per_minute']}"
        )
    print()

    print("Cross-video metrics:")
    m = result["cross_video_metrics"]
    if m["confidence_mean_of_means"] is not None:
        print(f"  Confidence (mean of means): {m['confidence_mean_of_means']}")
    if m["confidence_gap_mean_of_means"] is not None:
        print(f"  Confidence gap (mean of means): {m['confidence_gap_mean_of_means']}")
    if m["suppression_rate_mean"] is not None:
        print(f"  Suppression rate (mean): {m['suppression_rate_mean']}")
    print(
        f"  Worst consecutive suppressions: {m['max_consecutive_suppressions_worst']}"
    )
    print()

    if result["systemic_flags"]:
        print(f"Systemic video-level flags: {result['systemic_flags']}")
    # else:
        print("Systemic video-level flags: NONE")

    if result["event_flag_totals"]:
        print(f"Event flag totals: {result['event_flag_totals']}")
    print()

    print("GO criteria:")
    for k, v in result["go_criteria"].items():
        status = "PASS" if v is True else ("FAIL" if v is False else "PENDING")
        print(f"  {k}: {status}")
    print()

    print("NO-GO criteria (True = triggered = blocking):")
    for k, v in result["nogo_criteria"].items():
        status = "TRIGGERED" if v is True else ("CLEAR" if v is False else "PENDING")
        print(f"  {k}: {status}")
    print()

    print(f"Aggregate summary saved to: {output_path}")


if __name__ == "__main__":
    main()
