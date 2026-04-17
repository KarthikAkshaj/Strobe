
# Phase-2B Exit Report

**Run ID:** {run_id}
**Date:** {date}
**Prepared by:** {name}

---

## Executive Summary

**Decision: [ GO / NO-GO / CONDITIONAL GO ]**

{2-3 sentence summary of findings and recommendation}

---

## Test Corpus

| Category | Videos | Pass | Borderline | Fail |
|----------|--------|------|------------|------|
| static | | | | |
| single_scene_dynamic | | | | |
| multi_scene | | | | |
| gradual_transition | | | | |
| noisy | | | | |
| adversarial | | | | |
| **Total** | **25** | | | |

**Pass + Borderline Rate:** {value}
**Pass Rate (excluding adversarial):** {value}

---

## Go Criteria Checklist

| Criterion | Status | Evidence |
|-----------|--------|----------|
| No systematic failures (>50% of category with flags) | | |
| Suppression bounded (max_consecutive <= 5) | | |
| Mean confidence > 0.20 | | |
| Mean gap > 0.01 | | |
| Static videos < 1 event/min | | |
| Multi-scene entropy > 1.0 | | |
| Spot checks >50% correct | | |

---

## No-Go Criteria Checklist

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Mean confidence < 0.15 | | |
| >30% videos with GAP_COMPRESSION | | |
| >30% videos with HIGH_SUPPRESSION_RATE | | |
| Static videos >1 event/min | | |
| Multi-scene <1 event/2min | | |
| >50% spot checks wrong | | |

---

## Systemic Observations

**Video-Level Flag Distribution:**
{from aggregate_summary.json systemic_flags}

**Event-Level Flag Totals:**
{from aggregate_summary.json event_flag_totals}

**Interpretation:**
{describe what the flag distribution means}

---

## Cross-Video Metrics

| Metric | Value |
|--------|-------|
| Confidence (mean of means) | |
| Confidence gap (mean of means) | |
| Suppression rate (mean) | |
| Worst consecutive suppressions | |

---

## Known Limitations (Accepted)

1. {limitation description and why it is acceptable}
2. {limitation description and why it is acceptable}

---

## Deferred to Phase-2C

| Issue | Observed Evidence | Proposed Investigation |
|-------|-------------------|------------------------|
| | | |
| | | |

---

## Evidence Artifacts

- [ ] All 25 output JSONs archived in run directory
- [ ] All 25 review files completed
- [ ] Aggregate summary JSON generated
- [ ] This exit report completed

---

## Recommendation

**[ GO / NO-GO / CONDITIONAL GO ]**

{Final recommendation with justification}

**If GO:**
- Phase-2A + Phase-2B behavior is frozen
- Python implementation becomes reference implementation
- Proceed to ONNX export

**If NO-GO:**
- Do not proceed to ONNX export
- Document failures for Phase-2C investigation
- Re-run validation after changes

**If CONDITIONAL GO:**
- Proceed with documented limitations
- Phase-2C must address: {specific issues}

---

**Sign-off:**

Reviewer: _________________ Date: _________
