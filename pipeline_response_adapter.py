from __future__ import annotations

from typing import Any


def _to_pct(value: Any) -> int:
    try:
        v = float(value or 0)
    except Exception:
        return 0
    # CSI sub-scores are mostly 0..1; convert to 0..100 if needed.
    if 0 <= v <= 1.0:
        v *= 100.0
    return max(0, min(100, int(round(v))))


def _status_from_score(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "Strong"
    return "Very strong"


def _title_case_label(label: str) -> str:
    return " ".join(part.capitalize() for part in str(label or "").replace("_", " ").split())


def _signal_tag(status: str) -> str:
    s = str(status or "").strip().lower()
    if s in {"strong", "very strong"}:
        return "Leading"
    if s == "moderate":
        return "Normal"
    return "Quiet"


def _platform_note(platform: str, score: float, status: str, m: dict[str, Any]) -> str:
    if platform == "Search":
        ratio = float(m.get("ratio", 0) or 0)
        if ratio >= 1.5:
            return "Search interest rising quickly"
        if ratio >= 1.0:
            return "Search interest stable"
        return "Search interest soft"
    if platform == "Social":
        div = float(m.get("diversity", 0) or 0)
        if div >= 0.7:
            return "Discussion broad across platforms"
        if div >= 0.4:
            return "Discussion active but concentrated"
        return "Limited social spread"
    if platform == "YouTube":
        if score >= 60:
            return "Video activity accelerating"
        if score >= 30:
            return "Video activity stable"
        return "Video activity muted"
    if platform == "News":
        if str(m.get("status", "")).lower() == "ok":
            return "News coverage active"
        return "News coverage limited"
    return f"{status} signal"


def _friendly_angle_title(angle: dict[str, Any]) -> str:
    suggested = str(angle.get("suggested_title") or "").strip()
    if suggested:
        return suggested[:120]
    who = str(angle.get("who") or "Audience").strip()
    frame = _title_case_label(str(angle.get("story_frame") or "angle"))
    when = str(angle.get("when") or "today").strip()
    return f"{who} {frame}: What It Means {when.capitalize()}"[:120]


def _friendly_angle_subtitle(angle: dict[str, Any]) -> str:
    who = str(angle.get("who") or "Audience").strip()
    what = angle.get("what") or []
    if isinstance(what, list):
        what_txt = ", ".join(str(x) for x in what[:3])
    else:
        what_txt = str(what)
    frame = _title_case_label(str(angle.get("story_frame") or "angle"))
    return f"{who} {what_txt} {frame} view".strip()[:120]


def _human_breakout_label(label: str) -> str:
    txt = str(label or "").strip().replace("_", " ").lower()
    if not txt:
        return "No breakout"
    return txt[:1].upper() + txt[1:]


def adapt_pipeline_payload(raw: dict[str, Any], include_raw: bool = False) -> dict[str, Any]:
    methods = raw.get("methods", {}) or {}
    weights = raw.get("weights_used", {}) or {}
    csi = raw.get("csi", {}) or {}
    cags = raw.get("cags", {}) or {}
    verdict = raw.get("verdict", {}) or {}

    m1 = methods.get("m1", {}) or {}
    m2 = methods.get("m2", {}) or {}
    m3 = methods.get("m3", {}) or {}
    m4 = methods.get("m4", {}) or {}

    # ---- TSS block ----
    tss_score = int(round(float(raw.get("tss", 0) or 0)))
    band = str(raw.get("band", "flat") or "flat")
    trend_status = _title_case_label(band)
    trend_verdict = "Monitor closely" if tss_score < 20 else ("Publish soon" if tss_score < 50 else "Publish now")

    # platform signal weights from effective run weights (sum may not be 1 if failures happened)
    w_search = int(round(float(weights.get("w1", 0) or 0) * 100))
    w_social = int(round(float(weights.get("w2", 0) or 0) * 100))
    w_youtube = int(round(float(weights.get("w3", 0) or 0) * 100))
    w_news = int(round(float(weights.get("w4", 0) or 0) * 100))
    w_total = max(1, w_search + w_social + w_youtube + w_news)
    platform_weights = [
        {"platform": "Search", "percentage": f"{int(round(w_search * 100 / w_total))}%"},
        {"platform": "Social", "percentage": f"{int(round(w_social * 100 / w_total))}%"},
        {"platform": "YouTube", "percentage": f"{int(round(w_youtube * 100 / w_total))}%"},
        {"platform": "News", "percentage": f"{int(round(w_news * 100 / w_total))}%"},
    ]

    driver_scores = {
        "Search curiosity": float(m1.get("score", 0) or 0),
        "Social momentum": float(m2.get("score", 0) or 0),
        "YouTube momentum": float(m3.get("score", 0) or 0),
        "News cycle": float(m4.get("score", 0) or 0),
    }
    primary_driver = max(driver_scores, key=driver_scores.get) if driver_scores else "Search curiosity"

    trend_strength_score = {
        "score": tss_score,
        "max": 100,
        "status": trend_status,
        "verdict": trend_verdict,
        "description": f"{trend_status} trend phase based on cross-platform signals",
        "phase": trend_status,
        "composition": {
            "base": int(round(float(raw.get("base_score", 0) or 0))),
            "psych_boost": int(round(float(raw.get("psych_boost", 0) or 0))),
            "reliability": round(float(raw.get("reliability", 0) or 0), 3),
        },
        "why_trending": {
            "primary_driver": primary_driver,
            "headline": f"{primary_driver} is currently the strongest signal",
            "summary": f"Regime {raw.get('regime')} with {trend_status.lower()} strength and reliability {round(float(raw.get('reliability', 0) or 0), 2)}.",
            "platform_weights": platform_weights,
        },
        "platform_signals": [
            {
                "platform": "YouTube",
                "score": int(round(float(m3.get("score", 0) or 0))),
                "barW": int(round(float(m3.get("score", 0) or 0))),
                "tag": _signal_tag(_status_from_score(float(m3.get("score", 0) or 0))),
                "note": _platform_note("YouTube", float(m3.get("score", 0) or 0), _status_from_score(float(m3.get("score", 0) or 0)), m3),
            },
            {
                "platform": "Search",
                "score": int(round(float(m1.get("score", 0) or 0))),
                "barW": int(round(float(m1.get("score", 0) or 0))),
                "tag": _signal_tag(_status_from_score(float(m1.get("score", 0) or 0))),
                "note": _platform_note("Search", float(m1.get("score", 0) or 0), _status_from_score(float(m1.get("score", 0) or 0)), m1),
            },
            {
                "platform": "Social",
                "score": int(round(float(m2.get("score", 0) or 0))),
                "barW": int(round(float(m2.get("score", 0) or 0))),
                "tag": _signal_tag(_status_from_score(float(m2.get("score", 0) or 0))),
                "note": _platform_note("Social", float(m2.get("score", 0) or 0), _status_from_score(float(m2.get("score", 0) or 0)), m2),
            },
            {
                "platform": "News",
                "score": int(round(float(m4.get("score", 0) or 0))),
                "barW": int(round(float(m4.get("score", 0) or 0))),
                "tag": _signal_tag(_status_from_score(float(m4.get("score", 0) or 0))),
                "note": _platform_note("News", float(m4.get("score", 0) or 0), _status_from_score(float(m4.get("score", 0) or 0)), m4),
            },
        ],
        "confidence": {
            "reliability_score": round(float(raw.get("reliability", 0) or 0), 3),
            "sources": [
                {"name": "Search", "detail": f"Live · {int(raw.get('m1_norm', {}).get('weeks_of_data', 52) or 52)} wks"},
                {"name": "Social", "detail": "Live · 4 sources"},
                {"name": "YouTube", "detail": f"Live · {int((raw.get('m3_norm', {}) or {}).get('video_count', 10) or 10)} videos"},
                {"name": "News", "detail": "Live · 90-day"},
            ],
        },
    }

    # ---- CSI block ----
    csi_score = int(round(float(csi.get("csi", 0) or 0)))
    csi_label = _title_case_label(csi.get("label", "competitive"))
    breakout = csi.get("virality", {}) or {}
    threshold_breakdown = breakout.get("threshold_breakdown", {}) or {}
    active_signals = []
    if threshold_breakdown.get("search_breaking"):
        active_signals.append("Search rising")
    if threshold_breakdown.get("yt_breaking"):
        active_signals.append("YouTube growth detected")
    if threshold_breakdown.get("upload_breaking"):
        active_signals.append("Upload surge detected")
    if threshold_breakdown.get("eng_breaking"):
        active_signals.append("Engagement heat rising")
    if threshold_breakdown.get("view_spike_24h"):
        active_signals.append("24h view spike")
    if not active_signals:
        active_signals = ["No breakout signal yet"]

    supply_score = _to_pct((csi.get("supply", {}) or {}).get("score", 0))
    demand_score = _to_pct((csi.get("demand", {}) or {}).get("score", 0))
    freshness_score = _to_pct((csi.get("freshness", {}) or {}).get("score", 0))
    redundancy_score = _to_pct((csi.get("redundancy", {}) or {}).get("score", 0))
    viral_ceiling_score = _to_pct((csi.get("virality", {}) or {}).get("structural_score", 0))
    quality_gap_score = _to_pct((csi.get("quality_gap", {}) or {}).get("score", 0))
    angle_coverage_score = max(0, 100 - redundancy_score)

    content_saturation_index = {
        "score": csi_score,
        "status": csi_label,
        "verdict": "Winnable with right angle" if csi_score < 50 else ("Competitive but selective" if csi_score < 70 else "Hard to enter"),
        "description": "Content saturation computed from supply, demand, freshness, redundancy, virality and quality gap",
        "dimensions": [
            {"name": "Supply pressure", "score": supply_score, "effect": "Closes", "status": _status_from_score(supply_score)},
            {"name": "Audience demand", "score": demand_score, "effect": "Opens", "status": _status_from_score(demand_score)},
            {"name": "Upload freshness", "score": freshness_score, "effect": "Closes", "status": _status_from_score(freshness_score)},
            {"name": "Angle coverage", "score": angle_coverage_score, "effect": "Opens", "status": _status_from_score(angle_coverage_score)},
            {"name": "Viral ceiling", "score": viral_ceiling_score, "effect": "Opens", "status": _status_from_score(viral_ceiling_score)},
            {"name": "Quality gap", "score": quality_gap_score, "effect": "Opens", "status": _status_from_score(quality_gap_score)},
        ],
        "breakout": {
            "score": int(breakout.get("thresholds_fired", 0) or 0),
            "out_of": 5,
            "label": _human_breakout_label(breakout.get("breakout_indicator", "No breakout")),
            "signals": active_signals,
        },
        "incumbent_health": {
            "engagement_gap": _to_pct((csi.get("quality_gap", {}) or {}).get("eng_gap_norm", 0)),
            "creator_density": round(float((csi.get("supply", {}) or {}).get("creator_density", 0) or 0), 3),
            "vpd_decay": _to_pct((csi.get("quality_gap", {}) or {}).get("vpd_decay_norm", 0)),
            "verdict": "Entry viable" if csi_score < 50 else ("Playable with strong execution" if csi_score < 70 else "Entrenched incumbents"),
        },
    }

    # ---- CAGS block ----
    scored_angles = cags.get("scored_angles", []) or []
    gap_angles = cags.get("gap_angles", []) or []
    top_angles = scored_angles[:3]
    top_gaps = gap_angles[:3]

    dist_not = sum(1 for a in scored_angles if a.get("coverage_label") == "NOT_COVERED")
    dist_low = sum(1 for a in scored_angles if a.get("coverage_label") == "COVERED_LOW_QUALITY")
    dist_well = sum(1 for a in scored_angles if a.get("coverage_label") == "COVERED_WELL")

    content_angle_gap_score = {
        "total_angles": len(cags.get("perspective_tree", []) or []),
        "distribution": [
            {"label": "Not covered", "count": dist_not},
            {"label": "Low quality", "count": dist_low},
            {"label": "Well covered", "count": dist_well},
        ],
        "top_angles": [
            {
                "rank": a.get("rank"),
                "title": _friendly_angle_title(a),
                "who": a.get("who"),
                "what": ", ".join(a.get("what", []) or []) if isinstance(a.get("what"), list) else a.get("what"),
                "when": a.get("when"),
                "frame": a.get("story_frame"),
                "coverage": _title_case_label(a.get("coverage_label", "")),
            }
            for a in top_angles
        ],
        "gap_opportunities": [
            {
                "rank": g.get("rank"),
                "score": g.get("cags_score"),
                "title": _friendly_angle_title(g),
                "angle": _friendly_angle_subtitle(g),
                "demand_score": g.get("demand_score"),
            }
            for g in top_gaps
        ],
    }
    if cags.get("cags_error"):
        content_angle_gap_score["reason"] = str(cags.get("cags_error"))

    adapted: dict[str, Any] = {
        "topic": raw.get("topic"),
        "timestamp": raw.get("timestamp"),
        "trend_strength_score": trend_strength_score,
        "content_saturation_index": content_saturation_index,
        "content_angle_gap_score": content_angle_gap_score,
        "final_verdict": {
            "action": verdict.get("verdict"),
            "summary": verdict.get("reason"),
        },
    }
    # Optional debug mode only. Disabled by default so frontend responses
    # don't include huge vector embeddings or internal payload noise.
    if include_raw:
        adapted["raw_payload"] = raw
    return adapted
