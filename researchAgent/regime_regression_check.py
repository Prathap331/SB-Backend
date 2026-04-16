#!/usr/bin/env python3
"""
Regime regression checker for tss_v3.py.

Usage:
  python3 regime_regression_check.py
  python3 regime_regression_check.py --strict
  python3 regime_regression_check.py --limit 10
  python3 regime_regression_check.py --output-json /tmp/regime_report.json
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CASES: list[dict[str, Any]] = [
    {"topic": "who won ipl match yesterday", "expected": ["R1", "R4"], "group": "search/news"},
    {"topic": "best budget smartphone under 20000 india 2026", "expected": ["R1"], "group": "search"},
    {"topic": "sooryavanshi ipl highlights today", "expected": ["R3", "R1"], "group": "creator/search"},
    {"topic": "new iphone launch event date 2026", "expected": ["R4", "R1"], "group": "event/search"},
    {"topic": "israel iran war latest updates", "expected": ["R4", "R1"], "group": "news"},
    {"topic": "earthquake in japan today", "expected": ["R4", "R1"], "group": "breaking"},
    {"topic": "virat kohli retirement news", "expected": ["R4", "R3", "R1"], "group": "news/creator"},
    {"topic": "bigg boss elimination today", "expected": ["R2", "R3", "R1"], "group": "social/creator"},
    {"topic": "how to start dropshipping in india", "expected": ["R1"], "group": "evergreen"},
    {"topic": "ai video generator free tool review", "expected": ["R1", "R3"], "group": "search/creator"},
    {"topic": "tesla stock crash today", "expected": ["R4", "R1"], "group": "news/search"},
    {"topic": "el clasico match highlights", "expected": ["R3", "R1"], "group": "creator/search"},
    {"topic": "chatgpt outage today", "expected": ["R4", "R1"], "group": "breaking"},
    {"topic": "best protein powder for beginners", "expected": ["R1"], "group": "evergreen"},
    {"topic": "met gala 2026 looks", "expected": ["R2", "R4", "R1"], "group": "social/news"},
]


def extract_json_blob(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in stdout.")
    return json.loads(raw[start : end + 1])


def run_topic(
    topic: str,
    python_bin: str,
    tss_script: str,
    timeout: int,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [python_bin, tss_script, topic, "--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"timeout_after_{timeout}s",
            "stderr": "",
            "stdout": "",
        }
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": f"command_failed:{proc.returncode}",
            "stderr": proc.stderr[-800:],
            "stdout": proc.stdout[-800:],
        }
    try:
        payload = extract_json_blob(proc.stdout)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"json_parse_failed:{exc}",
            "stderr": proc.stderr[-800:],
            "stdout": proc.stdout[-800:],
        }

    methods = payload.get("methods", {}) or {}
    method_scores = {k: float((methods.get(k) or {}).get("score", 0) or 0) for k in ("m1", "m2", "m3", "m4")}
    return {
        "ok": True,
        "topic": topic,
        "tss": payload.get("tss"),
        "band": payload.get("band"),
        "regime": payload.get("regime"),
        "regime_method": payload.get("regime_method"),
        "regime_confidence": payload.get("regime_confidence"),
        "regime_flags": payload.get("regime_flags", []),
        "relative_signals": payload.get("relative_signals", {}),
        "method_scores": method_scores,
    }


def check_record(rec: dict[str, Any], expected: list[str]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    regime = rec.get("regime")
    conf = rec.get("regime_confidence")
    tss = rec.get("tss")
    method = rec.get("regime_method")

    if regime not in {"R1", "R2", "R3", "R4"}:
        issues.append(f"invalid_regime:{regime}")
    if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
        issues.append(f"invalid_confidence:{conf}")
    if not isinstance(tss, (int, float)) or tss < 0 or tss > 100:
        issues.append(f"invalid_tss:{tss}")
    if regime and expected and regime not in expected:
        issues.append(f"unexpected_regime:{regime} not in {expected}")

    # Useful signal for "stuck in R1 default".
    if regime == "R1" and method in {"default", "clash_collapse"}:
        scores = rec.get("method_scores", {})
        if all(float(scores.get(k, 0) or 0) < 20 for k in ("m1", "m2", "m3", "m4")):
            issues.append("all_methods_low_with_r1_default")

    return (len(issues) == 0), issues


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Regime regression checker for tss_v3.py")
    p.add_argument("--python", default=sys.executable, help="Python binary to execute tss_v3.py")
    p.add_argument("--tss-script", default="tss_v3.py", help="Path to tss_v3.py")
    p.add_argument("--timeout", type=int, default=180, help="Per-topic timeout (seconds)")
    p.add_argument("--strict", action="store_true", help="Fail on expected-regime mismatches")
    p.add_argument("--limit", type=int, default=0, help="Run only first N topics")
    p.add_argument("--output-json", default="", help="Write full report to this JSON path")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cases = DEFAULT_CASES[: args.limit] if args.limit and args.limit > 0 else DEFAULT_CASES

    passed = 0
    failed = 0
    soft_warn = 0
    rows: list[dict[str, Any]] = []

    print("Running regime regression suite...")
    print(f"Topics: {len(cases)} | strict={args.strict} | script={args.tss_script}")
    print("-" * 110)
    print(f"{'topic':40} {'regime':6} {'method':14} {'conf':6} {'tss':6} {'result'}")
    print("-" * 110)

    for case in cases:
        topic = case["topic"]
        expected = case.get("expected", [])
        rec = run_topic(topic, args.python, args.tss_script, args.timeout)
        if not rec.get("ok"):
            failed += 1
            result = f"FAIL {rec.get('error')}"
            print(f"{topic[:40]:40} {'-':6} {'-':14} {'-':6} {'-':6} {result}")
            rows.append({"topic": topic, "expected": expected, "record": rec, "issues": [rec.get("error")]})
            continue

        ok, issues = check_record(rec, expected)
        hard_issues = [i for i in issues if i.startswith("invalid_") or i.startswith("unexpected_regime:")]
        warn_issues = [i for i in issues if i not in hard_issues]

        if hard_issues and args.strict:
            failed += 1
            result = "FAIL " + ",".join(hard_issues)
        else:
            passed += 1
            if hard_issues:
                soft_warn += 1
                result = "WARN " + ",".join(hard_issues)
            elif warn_issues:
                soft_warn += 1
                result = "WARN " + ",".join(warn_issues)
            else:
                result = "PASS"

        print(
            f"{topic[:40]:40} "
            f"{str(rec.get('regime')):6} "
            f"{str(rec.get('regime_method')):14} "
            f"{str(rec.get('regime_confidence')):6} "
            f"{str(rec.get('tss')):6} "
            f"{result}"
        )
        rows.append({"topic": topic, "expected": expected, "record": rec, "issues": issues})

    print("-" * 110)
    print(f"Passed: {passed} | Failed: {failed} | Warnings: {soft_warn}")

    report = {
        "summary": {
            "total": len(cases),
            "passed": passed,
            "failed": failed,
            "warnings": soft_warn,
            "strict": args.strict,
        },
        "rows": rows,
    }
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report: {out_path}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
