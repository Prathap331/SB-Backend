#!/usr/bin/env python3
"""
Validate pipeline payload for dashboard compatibility (TSS + CSI + CAGS).

Usage:
  python3 dashboard_payload_check.py --input /tmp/pipeline_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def _get(obj: dict[str, Any], key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else default


def check_tss(payload: dict[str, Any]) -> None:
    required = [
        "topic",
        "category",
        "category_layer",
        "tss",
        "band",
        "regime",
        "regime_label",
        "regime_confidence",
        "regime_method",
        "scan_mean",
        "base_score",
        "psych_boost",
        "reliability",
        "methods",
        "weights_used",
        "relative_signals",
        "quality",
        "psych_signals",
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        _fail(f"TSS missing keys: {missing}")

    tss = float(payload.get("tss", 0) or 0)
    if not (0 <= tss <= 100):
        _fail(f"TSS out of range: {tss}")

    band = payload.get("band")
    if band not in {"flat", "emerging", "rising", "peak", "saturating"}:
        _fail(f"TSS band invalid: {band}")

    regime = payload.get("regime")
    if regime not in {"R1", "R2", "R3", "R4"}:
        _fail(f"TSS regime invalid: {regime}")


def check_csi(payload: dict[str, Any]) -> None:
    csi = _get(payload, "csi", {})
    if not csi or not isinstance(csi, dict):
        _fail("CSI block missing")
    if "error" in csi:
        _fail(f"CSI error: {csi.get('error')}")

    required = [
        "csi",
        "label",
        "supply",
        "demand",
        "freshness",
        "redundancy",
        "virality",
        "quality_gap",
        "data_quality",
    ]
    missing = [k for k in required if k not in csi]
    if missing:
        _fail(f"CSI missing keys: {missing}")

    csi_score = float(csi.get("csi", 0) or 0)
    if not (0 <= csi_score <= 100):
        _fail(f"CSI score out of range: {csi_score}")

    label = csi.get("label")
    if label not in {"OPEN", "COMPETITIVE", "CROWDED", "SATURATED"}:
        _fail(f"CSI label invalid: {label}")

    dq = _get(csi, "data_quality", {})
    dq_required = [
        "corpus_stale_warning",
        "engagement_coverage",
        "freshness_ratio",
        "redundancy_embedding_failed",
        "redundancy_used_fallback",
        "engagement_insufficient",
    ]
    missing_dq = [k for k in dq_required if k not in dq]
    if missing_dq:
        _fail(f"CSI data_quality missing keys: {missing_dq}")


def check_cags(payload: dict[str, Any]) -> None:
    cags = _get(payload, "cags", {})
    if not cags or not isinstance(cags, dict):
        _fail("CAGS block missing")
    if "cags_error" in cags:
        _fail(f"CAGS error: {cags.get('cags_error')}")

    required = [
        "topic",
        "perspective_tree",
        "labelled_corpus",
        "scored_angles",
        "gap_angles",
        "briefs",
    ]
    missing = [k for k in required if k not in cags]
    if missing:
        _fail(f"CAGS missing keys: {missing}")

    scored = cags.get("scored_angles", []) or []
    if not isinstance(scored, list):
        _fail("CAGS scored_angles must be a list")

    if scored:
        required_angle_fields = [
            "rank",
            "cags_score",
            "coverage_label",
            "angle_string",
            "who",
            "what",
            "when",
            "scale",
            "how",
            "who_benefits",
            "story_frame",
            "best_video",
            "matched_count",
            "demand_score",
            "best_quality",
        ]
        missing_angle = [k for k in required_angle_fields if k not in scored[0]]
        if missing_angle:
            _fail(f"CAGS scored_angle missing keys: {missing_angle}")

        ranks = [int(a.get("rank", 0) or 0) for a in scored]
        expected = list(range(1, len(scored) + 1))
        if ranks != expected:
            _fail(f"CAGS rank sequence invalid: {ranks[:10]} != {expected[:10]}")

        for a in scored:
            score = float(a.get("cags_score", 0) or 0)
            if not (0 <= score <= 100):
                _fail(f"CAGS score out of range: {score}")


def check_briefs(payload: dict[str, Any], strict: bool = False) -> None:
    cags = _get(payload, "cags", {})
    briefs = cags.get("briefs", []) or []
    if not isinstance(briefs, list):
        _fail("CAGS briefs must be a list")
    if not briefs:
        if strict:
            _fail("CAGS briefs are empty in strict mode")
        return

    valid_urgency = {"now", "within_1_week", "within_1_month", "anytime"}
    for idx, brief in enumerate(briefs, 1):
        if not isinstance(brief, dict):
            _fail(f"Brief {idx} is not an object")
        for key in ("suggested_title", "hook_sentence", "publish_urgency"):
            if key not in brief:
                _fail(f"Brief {idx} missing key: {key}")

        title = str(brief.get("suggested_title") or "").strip()
        hook = str(brief.get("hook_sentence") or "").strip()
        urgency = str(brief.get("publish_urgency") or "").strip()

        if strict and not title:
            _fail(f"Brief {idx} has empty suggested_title in strict mode")
        if strict and not hook:
            _fail(f"Brief {idx} has empty hook_sentence in strict mode")
        if urgency and urgency not in valid_urgency:
            _fail(f"Brief {idx} has invalid publish_urgency: {urgency}")


def print_dashboard_mapping(payload: dict[str, Any]) -> None:
    csi = payload.get("csi", {})
    cags = payload.get("cags", {})
    gaps = cags.get("gap_angles", []) or []
    top_cags = max((float(a.get("cags_score", 0) or 0) for a in gaps), default=0.0)

    print("=== TSS ===")
    print(f"topic: {payload.get('topic')}")
    print(f"category: {payload.get('category')} (layer {payload.get('category_layer')})")
    print(f"tss: {payload.get('tss')}  band: {payload.get('band')}")
    print(f"regime: {payload.get('regime')} ({payload.get('regime_label')})")
    print(f"verdict: {payload.get('verdict')}")

    print("\n=== CSI ===")
    print(f"csi: {csi.get('csi')}  label: {csi.get('label')}")
    print(f"supply score: {round(float(_get(_get(csi,'supply',{}), 'score', 0) or 0)*100)}")
    print(f"demand score: {round(float(_get(_get(csi,'demand',{}), 'score', 0) or 0)*100)}")
    print(f"freshness score: {round(float(_get(_get(csi,'freshness',{}), 'score', 0) or 0)*100)}")
    print(f"virality structural: {_get(_get(csi,'virality',{}), 'structural_score', None)}")
    print(f"quality gap: {round(float(_get(_get(csi,'quality_gap',{}), 'score', 0) or 0)*100)}")

    print("\n=== CAGS ===")
    print(f"angles mapped: {len(cags.get('perspective_tree', []) or [])}")
    print(f"videos labelled: {len(cags.get('labelled_corpus', []) or [])}")
    print(f"scored angles: {len(cags.get('scored_angles', []) or [])}")
    print(f"gap angles: {len(gaps)}")
    print(f"top cags: {top_cags}")
    briefs = cags.get("briefs", []) or []
    print(f"briefs: {len(briefs)}")
    for i, b in enumerate(briefs[:3], 1):
        print(f"  {i}. rank={b.get('rank')} score={b.get('cags_score')} label={b.get('coverage_label')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/tmp/pipeline_metrics.json", help="Path to pipeline JSON payload")
    parser.add_argument(
        "--strict-briefs",
        action="store_true",
        help="Fail if any brief title/hook is empty.",
    )
    args = parser.parse_args()

    path = Path(args.input).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    # Support both:
    # 1) raw run_tss payload
    # 2) adapted frontend payload with raw_payload
    if "raw_payload" in payload and "trend_strength_score" in payload:
        raw = payload.get("raw_payload") or {}
    else:
        raw = payload

    check_tss(raw)
    check_csi(raw)
    check_cags(raw)
    check_briefs(raw, strict=args.strict_briefs)

    print("TSS: PASS")
    print("CSI: PASS")
    print("CAGS: PASS")
    print(f"Briefs: PASS (strict={args.strict_briefs})")
    print_dashboard_mapping(raw)
    print("\nDashboard payload check PASS")


if __name__ == "__main__":
    main()
