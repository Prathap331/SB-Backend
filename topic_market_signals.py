#!/usr/bin/env python3
"""
Run Google Trends, social, YouTube, and news market signals together for one topic.

Usage:
  python3 topic_market_signals.py "Israel Iran War"
  python3 topic_market_signals.py "AI automation" --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re

from google_trends_only import fetch_keyword_trends
from news_market_signals import scan_topic as scan_news_topic
from social_market_signals import scan_topic as scan_social_topic
from youtube_market_signals import normalize_score, scan_topic


def classify_opportunity(score: float) -> str:
    if score >= 80:
        return "breakout"
    if score >= 60:
        return "hot"
    if score >= 40:
        return "warming"
    return "cold"


def classify_tss_stage(score: float) -> dict[str, object]:
    if score < 20:
        return {"stage": "flat", "band_min": 0, "band_max": 20}
    if score < 50:
        return {"stage": "emerging", "band_min": 20, "band_max": 50}
    if score < 75:
        return {"stage": "rising", "band_min": 50, "band_max": 75}
    if score < 90:
        return {"stage": "peak", "band_min": 75, "band_max": 90}
    return {"stage": "saturating", "band_min": 90, "band_max": 100}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def infer_topic_profile(topic: str) -> str:
    lowered = (topic or "").lower()
    news_keywords = {
        "war", "conflict", "attack", "missile", "ceasefire", "election",
        "sanction", "policy", "government", "president", "prime minister",
        "geopolitics", "breaking", "crisis", "strike",
    }
    if any(re.search(rf"\b{re.escape(k)}\b", lowered) for k in news_keywords):
        return "politics_news"
    return "general"


def infer_content_category(topic: str) -> str:
    lowered = (topic or "").lower()
    mapping = {
        "finance_markets": {"stock", "finance", "market", "invest", "crypto", "forex", "economy"},
        "entertainment": {"movie", "music", "celebrity", "show", "anime", "gaming", "trailer"},
        "politics_news": {"war", "conflict", "election", "policy", "government", "geopolitics", "crisis"},
        "technology": {"ai", "software", "tech", "developer", "engineering", "startup", "automation"},
        "sports": {"football", "cricket", "nba", "fifa", "ipl", "sport", "match", "league"},
    }
    for category, keys in mapping.items():
        if any(re.search(rf"\b{re.escape(k)}\b", lowered) for k in keys):
            return category
    return "general"


def get_m5_weights(profile: str) -> dict[str, float]:
    if profile == "politics_news":
        return {
            "m1_weight": 0.25,
            "m2_weight": 0.25,
            "m3_weight": 0.10,
            "m4_weight": 0.40,
        }
    return {
        "m1_weight": 0.20,
        "m2_weight": 0.20,
        "m3_weight": 0.35,
        "m4_weight": 0.25,
    }


def get_category_weight_profile(category: str) -> dict[str, float]:
    table = {
        "general": {"m1": 0.35, "m2": 0.25, "m3": 0.25, "m4": 0.15},
        "finance_markets": {"m1": 0.50, "m2": 0.15, "m3": 0.20, "m4": 0.15},
        "entertainment": {"m1": 0.20, "m2": 0.40, "m3": 0.30, "m4": 0.10},
        "politics_news": {"m1": 0.25, "m2": 0.25, "m3": 0.10, "m4": 0.40},
        "technology": {"m1": 0.35, "m2": 0.20, "m3": 0.30, "m4": 0.15},
        "sports": {"m1": 0.15, "m2": 0.40, "m3": 0.30, "m4": 0.15},
    }
    return table.get(category, table["general"])


def redistribute_missing_weights(
    base: dict[str, float],
    available: dict[str, bool],
) -> dict[str, float]:
    active_total = sum(weight for method, weight in base.items() if available.get(method, False))
    if active_total <= 0:
        return {k: 0.0 for k in base}
    return {
        method: (weight / active_total) if available.get(method, False) else 0.0
        for method, weight in base.items()
    }


def compute_m5_enhanced(topic: str, youtube: dict, trends: dict, social: dict, news: dict) -> dict:
    category = infer_content_category(topic)
    base_weights = get_category_weight_profile(category)

    scores = {
        "m1": float(trends.get("m1_score", 0.0)),
        "m2": float(social.get("m2_score", 0.0)),
        "m3": float(youtube.get("m3_score", 0.0)),
        "m4": float(news.get("m4_score", 0.0)),
    }

    available = {
        "m1": trends.get("trend_direction") != "unknown",
        "m2": int(social.get("sample_size", 0) or 0) > 0,
        "m3": int(youtube.get("sample_size", 0) or 0) > 0,
        "m4": int(news.get("sample_size", 0) or 0) > 0 or float(news.get("m4_raw", 0.0) or 0.0) > 0,
    }
    weights = redistribute_missing_weights(base_weights, available)

    base_score = 0.0
    weighted_components: dict[str, float] = {}
    for method, weight in weights.items():
        component = round(weight * scores[method], 2)
        weighted_components[method] = component
        base_score += component
    base_score = round(base_score, 2)

    # Reliability adjustment from data coverage.
    quality_m1 = 1.0 if available["m1"] else 0.6
    quality_m2 = clamp((int(social.get("sample_size", 0) or 0) / 20.0), 0.4, 1.0)
    quality_m3 = clamp((int(youtube.get("sample_size", 0) or 0) / 20.0), 0.4, 1.0)
    quality_m4 = clamp((int(news.get("sample_size", 0) or 0) / 20.0), 0.4, 1.0)
    quality_factor = round(
        (weights["m1"] * quality_m1)
        + (weights["m2"] * quality_m2)
        + (weights["m3"] * quality_m3)
        + (weights["m4"] * quality_m4),
        3,
    )

    # Psychology adjustment (bounded): arousal + social proof + novelty.
    reaction_ratio = float(((social.get("m2_formula") or {}).get("x") or {}).get("reaction_ratio", 0.0) or 0.0)
    tone_shift = float(((news.get("m4_formula") or {}).get("tone_shift", 0.0) or 0.0))
    social_proof = clamp(
        (float(youtube.get("channel_diversity", 0.0) or 0.0) + float(social.get("source_diversity", 0.0) or 0.0)) / 2.0,
        0.0,
        1.0,
    )
    novelty = clamp(float(trends.get("m1_search_ratio", 0.0) or 0.0) / 3.0, 0.0, 1.0)
    psych_factor = 1.0
    psych_factor += 0.06 * clamp(reaction_ratio / 0.5, 0.0, 1.0)
    psych_factor += 0.06 * clamp(tone_shift / 10.0, 0.0, 1.0)
    psych_factor += 0.05 * social_proof
    psych_factor += 0.05 * novelty
    psych_factor = round(clamp(psych_factor, 0.90, 1.20), 3)

    enhanced_score = round(clamp(base_score * quality_factor * psych_factor, 0.0, 100.0), 2)
    stage = classify_tss_stage(enhanced_score)
    return {
        "architecture": "M5 Enhanced (category + reliability + psych)",
        "category": category,
        "base_weights": base_weights,
        "effective_weights": weights,
        "available_methods": available,
        "scores": scores,
        "weighted_components": weighted_components,
        "base_score": base_score,
        "quality_factor": quality_factor,
        "psych_factor": psych_factor,
        "enhanced_score": enhanced_score,
        "enhanced_label": classify_opportunity(enhanced_score),
        "enhanced_stage": stage["stage"],
        "enhanced_stage_band": {"min": stage["band_min"], "max": stage["band_max"]},
        "psychology_signals": {
            "reaction_ratio": round(reaction_ratio, 3),
            "tone_shift": round(tone_shift, 3),
            "social_proof": round(social_proof, 3),
            "novelty": round(novelty, 3),
        },
    }


def detect_market_regime(m1: float, m2: float, m3: float, m4: float) -> str:
    if m4 >= 60 and m2 >= 45:
        return "event_urgency"
    if m3 >= 60 and m2 >= 45:
        return "creator_viral"
    if m2 >= 50 and m4 >= 40:
        return "social_contagion"
    if m1 >= 55 and m2 < 35:
        return "curiosity_search"
    return "balanced"


def get_regime_weights(regime: str, category: str) -> dict[str, float]:
    by_regime = {
        "event_urgency": {"m1": 0.20, "m2": 0.25, "m3": 0.10, "m4": 0.45},
        "creator_viral": {"m1": 0.20, "m2": 0.25, "m3": 0.45, "m4": 0.10},
        "social_contagion": {"m1": 0.20, "m2": 0.40, "m3": 0.20, "m4": 0.20},
        "curiosity_search": {"m1": 0.45, "m2": 0.15, "m3": 0.25, "m4": 0.15},
        "balanced": get_category_weight_profile(category),
    }
    return by_regime.get(regime, by_regime["balanced"])


def compute_m5_adaptive_v2(topic: str, youtube: dict, trends: dict, social: dict, news: dict) -> dict:
    category = infer_content_category(topic)
    scores = {
        "m1": float(trends.get("m1_score", 0.0)),
        "m2": float(social.get("m2_score", 0.0)),
        "m3": float(youtube.get("m3_score", 0.0)),
        "m4": float(news.get("m4_score", 0.0)),
    }
    regime = detect_market_regime(scores["m1"], scores["m2"], scores["m3"], scores["m4"])
    base_weights = get_regime_weights(regime, category)

    available = {
        "m1": trends.get("trend_direction") != "unknown",
        "m2": int(social.get("sample_size", 0) or 0) > 0,
        "m3": int(youtube.get("sample_size", 0) or 0) > 0,
        "m4": int(news.get("sample_size", 0) or 0) > 0 or float(news.get("m4_raw", 0.0) or 0.0) > 0,
    }
    weights = redistribute_missing_weights(base_weights, available)

    weighted_components: dict[str, float] = {}
    base_score = 0.0
    for method in ("m1", "m2", "m3", "m4"):
        comp = round(weights[method] * scores[method], 2)
        weighted_components[method] = comp
        base_score += comp
    base_score = round(base_score, 2)

    # Reliability from method coverage.
    reliability = round(
        (weights["m1"] * (1.0 if available["m1"] else 0.6))
        + (weights["m2"] * clamp((int(social.get("sample_size", 0) or 0) / 20.0), 0.4, 1.0))
        + (weights["m3"] * clamp((int(youtube.get("sample_size", 0) or 0) / 20.0), 0.4, 1.0))
        + (weights["m4"] * clamp((int(news.get("sample_size", 0) or 0) / 20.0), 0.4, 1.0)),
        3,
    )

    # Psychology layer: novelty, social proof, urgency, creator FOMO.
    novelty = clamp(float(trends.get("m1_search_ratio", 0.0) or 0.0) / 3.0, 0.0, 1.0)
    social_proof = clamp(
        (float(youtube.get("channel_diversity", 0.0) or 0.0) + float(social.get("source_diversity", 0.0) or 0.0)) / 2.0,
        0.0,
        1.0,
    )
    urgency = clamp(float(news.get("m4_raw", 0.0) or 0.0) / 6.0, 0.0, 1.0)
    m3_exact = float(((youtube.get("m3_exact_experimental") or {}).get("m3_exact_ratio", 0.0) or 0.0))
    creator_fomo = clamp(m3_exact / 2.0, 0.0, 1.0)
    reaction_ratio = float((((social.get("m2_formula") or {}).get("x") or {}).get("reaction_ratio", 0.0) or 0.0))
    emotional_heat = clamp(reaction_ratio / 0.5, 0.0, 1.0)

    psych_boost = (
        6.0 * novelty
        + 6.0 * social_proof
        + 7.0 * urgency
        + 7.0 * creator_fomo
        + 4.0 * emotional_heat
    )
    psych_boost = round(clamp(psych_boost, 0.0, 20.0), 2)

    # Slight penalty when one method dominates too much.
    dominance = max(scores.values()) - min(scores.values())
    coherence_penalty = round(clamp((dominance - 75.0) / 10.0, 0.0, 5.0), 2)

    adaptive_score = round(clamp((base_score + psych_boost - coherence_penalty) * reliability, 0.0, 100.0), 2)
    stage = classify_tss_stage(adaptive_score)
    return {
        "architecture": "M5 Adaptive v2 (regime + dynamic weights + psych)",
        "category": category,
        "detected_regime": regime,
        "base_weights": base_weights,
        "effective_weights": weights,
        "available_methods": available,
        "scores": scores,
        "weighted_components": weighted_components,
        "base_score": base_score,
        "psych_boost": psych_boost,
        "coherence_penalty": coherence_penalty,
        "reliability_factor": reliability,
        "adaptive_score": adaptive_score,
        "adaptive_label": classify_opportunity(adaptive_score),
        "adaptive_stage": stage["stage"],
        "adaptive_stage_band": {"min": stage["band_min"], "max": stage["band_max"]},
        "psychology_signals": {
            "novelty": round(novelty, 3),
            "social_proof": round(social_proof, 3),
            "urgency": round(urgency, 3),
            "creator_fomo": round(creator_fomo, 3),
            "emotional_heat": round(emotional_heat, 3),
        },
    }


def get_psych_weights_v3(regime: str, category: str) -> dict[str, float]:
    # b1=novelty, b2=social_proof, b3=urgency, b4=creator_fomo, b5=emotional_heat
    regime_weights = {
        "event_urgency": {"b1": 4.0, "b2": 5.0, "b3": 10.0, "b4": 4.0, "b5": 5.0},
        "creator_viral": {"b1": 5.0, "b2": 6.0, "b3": 4.0, "b4": 10.0, "b5": 5.0},
        "social_contagion": {"b1": 5.0, "b2": 10.0, "b3": 6.0, "b4": 5.0, "b5": 6.0},
        "curiosity_search": {"b1": 10.0, "b2": 5.0, "b3": 5.0, "b4": 5.0, "b5": 4.0},
        "balanced": {"b1": 6.0, "b2": 6.0, "b3": 7.0, "b4": 7.0, "b5": 4.0},
    }
    category_multiplier = {
        "general": {"b1": 1.0, "b2": 1.0, "b3": 1.0, "b4": 1.0, "b5": 1.0},
        "finance_markets": {"b1": 1.15, "b2": 0.9, "b3": 1.1, "b4": 0.85, "b5": 1.0},
        "entertainment": {"b1": 0.95, "b2": 1.2, "b3": 0.8, "b4": 1.15, "b5": 1.05},
        "politics_news": {"b1": 0.9, "b2": 1.05, "b3": 1.35, "b4": 0.8, "b5": 1.2},
        "technology": {"b1": 1.1, "b2": 0.95, "b3": 0.9, "b4": 1.3, "b5": 1.0},
        "sports": {"b1": 0.9, "b2": 1.25, "b3": 0.85, "b4": 1.1, "b5": 1.05},
    }
    base = regime_weights.get(regime, regime_weights["balanced"])
    mult = category_multiplier.get(category, category_multiplier["general"])
    return {k: round(base[k] * mult[k], 3) for k in ("b1", "b2", "b3", "b4", "b5")}


def compute_psych_confidence_v3(
    trends: dict,
    social: dict,
    youtube: dict,
    news: dict,
) -> dict[str, float]:
    conf_novelty = 1.0 if trends.get("trend_direction") != "unknown" else 0.5
    conf_social = clamp((float(social.get("source_diversity", 0.0) or 0.0) + clamp((int(social.get("sample_size", 0) or 0) / 15.0), 0.0, 1.0)) / 2.0, 0.3, 1.0)
    conf_urgency = 1.0 if float(news.get("m4_raw", 0.0) or 0.0) > 0 else clamp((int(news.get("sample_size", 0) or 0) / 10.0), 0.2, 0.7)
    exact = youtube.get("m3_exact_experimental") or {}
    conf_fomo = clamp((int(exact.get("videos_with_both_windows", 0) or 0) / 5.0), 0.2, 1.0)
    reaction_ratio = float((((social.get("m2_formula") or {}).get("x") or {}).get("reaction_ratio", 0.0) or 0.0))
    conf_emotion = 0.4 if reaction_ratio <= 0 else 1.0
    return {
        "novelty": round(conf_novelty, 3),
        "social_proof": round(conf_social, 3),
        "urgency": round(conf_urgency, 3),
        "creator_fomo": round(conf_fomo, 3),
        "emotional_heat": round(conf_emotion, 3),
    }


def compute_m5_adaptive_v3(topic: str, youtube: dict, trends: dict, social: dict, news: dict) -> dict:
    category = infer_content_category(topic)
    scores = {
        "m1": float(trends.get("m1_score", 0.0)),
        "m2": float(social.get("m2_score", 0.0)),
        "m3": float(youtube.get("m3_score", 0.0)),
        "m4": float(news.get("m4_score", 0.0)),
    }
    regime = detect_market_regime(scores["m1"], scores["m2"], scores["m3"], scores["m4"])
    base_weights = get_regime_weights(regime, category)

    available = {
        "m1": trends.get("trend_direction") != "unknown",
        "m2": int(social.get("sample_size", 0) or 0) > 0,
        "m3": int(youtube.get("sample_size", 0) or 0) > 0,
        "m4": int(news.get("sample_size", 0) or 0) > 0 or float(news.get("m4_raw", 0.0) or 0.0) > 0,
    }
    weights = redistribute_missing_weights(base_weights, available)
    weighted_components = {m: round(weights[m] * scores[m], 2) for m in ("m1", "m2", "m3", "m4")}
    base_score = round(sum(weighted_components.values()), 2)

    reliability = round(
        (weights["m1"] * (1.0 if available["m1"] else 0.6))
        + (weights["m2"] * clamp((int(social.get("sample_size", 0) or 0) / 20.0), 0.4, 1.0))
        + (weights["m3"] * clamp((int(youtube.get("sample_size", 0) or 0) / 20.0), 0.4, 1.0))
        + (weights["m4"] * clamp((int(news.get("sample_size", 0) or 0) / 20.0), 0.4, 1.0)),
        3,
    )

    novelty = clamp(float(trends.get("m1_search_ratio", 0.0) or 0.0) / 3.0, 0.0, 1.0)
    social_proof = clamp(
        (float(youtube.get("channel_diversity", 0.0) or 0.0) + float(social.get("source_diversity", 0.0) or 0.0)) / 2.0,
        0.0,
        1.0,
    )
    urgency = clamp(float(news.get("m4_raw", 0.0) or 0.0) / 6.0, 0.0, 1.0)
    m3_exact = float(((youtube.get("m3_exact_experimental") or {}).get("m3_exact_ratio", 0.0) or 0.0))
    creator_fomo = clamp(m3_exact / 2.0, 0.0, 1.0)
    reaction_ratio = float((((social.get("m2_formula") or {}).get("x") or {}).get("reaction_ratio", 0.0) or 0.0))
    emotional_heat = clamp(reaction_ratio / 0.5, 0.0, 1.0)

    psych_weights = get_psych_weights_v3(regime, category)
    psych_conf = compute_psych_confidence_v3(trends, social, youtube, news)
    psych_components = {
        "novelty": psych_weights["b1"] * novelty * psych_conf["novelty"],
        "social_proof": psych_weights["b2"] * social_proof * psych_conf["social_proof"],
        "urgency": psych_weights["b3"] * urgency * psych_conf["urgency"],
        "creator_fomo": psych_weights["b4"] * creator_fomo * psych_conf["creator_fomo"],
        "emotional_heat": psych_weights["b5"] * emotional_heat * psych_conf["emotional_heat"],
    }
    psych_boost_raw = sum(psych_components.values())
    psych_boost_capped = round(clamp(psych_boost_raw, 0.0, 20.0), 2)

    # EMA-like smoothing against v2 if present to reduce oscillation.
    v2_score = float((compute_m5_adaptive_v2(topic, youtube, trends, social, news) or {}).get("adaptive_score", 0.0))
    unsmoothed = round(clamp((base_score + psych_boost_capped) * reliability, 0.0, 100.0), 2)
    adaptive_v3_score = round((0.8 * unsmoothed) + (0.2 * v2_score), 2)
    stage = classify_tss_stage(adaptive_v3_score)

    return {
        "architecture": "M5 Adaptive v3 (regime + dynamic psych weights + confidence)",
        "category": category,
        "detected_regime": regime,
        "base_weights": base_weights,
        "effective_weights": weights,
        "available_methods": available,
        "scores": scores,
        "weighted_components": weighted_components,
        "base_score": base_score,
        "reliability_factor": reliability,
        "psych_weights": psych_weights,
        "psych_confidence": psych_conf,
        "psych_components": {k: round(v, 3) for k, v in psych_components.items()},
        "psych_boost_raw": round(psych_boost_raw, 3),
        "psych_boost_capped": psych_boost_capped,
        "unsmoothed_score": unsmoothed,
        "adaptive_v3_score": adaptive_v3_score,
        "adaptive_v3_label": classify_opportunity(adaptive_v3_score),
        "adaptive_v3_stage": stage["stage"],
        "adaptive_v3_stage_band": {"min": stage["band_min"], "max": stage["band_max"]},
        "psychology_signals": {
            "novelty": round(novelty, 3),
            "social_proof": round(social_proof, 3),
            "urgency": round(urgency, 3),
            "creator_fomo": round(creator_fomo, 3),
            "emotional_heat": round(emotional_heat, 3),
        },
    }


def compute_combined_opportunity_score(youtube: dict, trends: dict) -> dict:
    return {
        "architecture": "M1 + M3 (legacy)",
        "m1_google_trends_score": float(trends.get("m1_score", 0.0)),
        "m3_youtube_score": float(youtube.get("m3_score", 0.0)),
        "opportunity_score": round(
            (0.40 * float(trends.get("m1_score", 0.0)))
            + (0.60 * float(youtube.get("m3_score", 0.0))),
            2,
        ),
        "opportunity_label": classify_opportunity(
            round(
                (0.40 * float(trends.get("m1_score", 0.0)))
                + (0.60 * float(youtube.get("m3_score", 0.0))),
                2,
            )
        ),
    }


def compute_combined_opportunity_score_v2(
    topic: str,
    youtube: dict,
    trends: dict,
    social: dict,
    news: dict,
) -> dict:
    """
    Final StoryBit M5 market score:
      - M1 = Google Trends search-interest velocity
      - M2 = Social discussion velocity
      - M3 = YouTube traction
      - M4 = News coverage velocity

    M3 remains slightly heavier because it is the closest proxy to content
    performance on the target publishing platform.
    """
    youtube_score = float(youtube.get("m3_score", 0.0))
    trends_score = float(trends.get("m1_score", 0.0))
    social_score = float(social.get("m2_score", 0.0))
    news_score = float(news.get("m4_score", 0.0))

    profile = infer_topic_profile(topic)
    weights = get_m5_weights(profile)
    m1_weight = weights["m1_weight"]
    m2_weight = weights["m2_weight"]
    m3_weight = weights["m3_weight"]
    m4_weight = weights["m4_weight"]

    # Only apply snapshot bonus when all snapshot channels are available.
    # This avoids first-scan partial data inflating M5.
    snapshot_bonus = 0.0
    yt_delta = youtube.get("avg_snapshot_delta")
    social_delta = social.get("snapshot_delta_24h")
    news_delta = news.get("snapshot_delta_24h")
    if yt_delta is not None and social_delta is not None and news_delta is not None:
        snapshot_bonus += 0.04 * normalize_score(float(yt_delta), 250000.0)
        snapshot_bonus += 0.03 * normalize_score(float(social_delta), 100.0)
        snapshot_bonus += 0.03 * normalize_score(float(news_delta), 100.0)
    snapshot_bonus = round(snapshot_bonus, 2)

    combined = round(
        (m1_weight * trends_score)
        + (m2_weight * social_score)
        + (m3_weight * youtube_score)
        + (m4_weight * news_score)
        + snapshot_bonus,
        2,
    )
    stage = classify_tss_stage(combined)
    return {
        "architecture": "M1 + M2 + M3 + M4 => M5",
        "topic_profile": profile,
        "m1_google_trends_score": trends_score,
        "m2_social_score": social_score,
        "m3_youtube_score": youtube_score,
        "m4_news_score": news_score,
        "m1_weight": m1_weight,
        "m2_weight": m2_weight,
        "m3_weight": m3_weight,
        "m4_weight": m4_weight,
        "trends_weighted_component": round(m1_weight * trends_score, 2),
        "social_weighted_component": round(m2_weight * social_score, 2),
        "youtube_weighted_component": round(m3_weight * youtube_score, 2),
        "news_weighted_component": round(m4_weight * news_score, 2),
        "snapshot_bonus": snapshot_bonus,
        "google_trends_breakdown": {
            "m1_search_ratio": trends.get("m1_search_ratio"),
            "m1_score": trends.get("m1_score"),
            "trend_direction": trends.get("trend_direction"),
        },
        "social_breakdown": {
            "mentions_24h": social.get("mentions_24h"),
            "mentions_48h": social.get("mentions_48h"),
            "mentions_7d": social.get("mentions_7d"),
            "m2_score": social.get("m2_score"),
        },
        "news_breakdown": {
            "articles_24h": news.get("articles_24h"),
            "articles_48h": news.get("articles_48h"),
            "articles_7d": news.get("articles_7d"),
            "m4_score": news.get("m4_score"),
        },
        "opportunity_score": combined,
        "opportunity_label": classify_opportunity(combined),
        "opportunity_stage": stage["stage"],
        "opportunity_stage_band": {"min": stage["band_min"], "max": stage["band_max"]},
    }


async def gather_signals(
    topic: str,
    *,
    velocity_pool_size: int = 10,
    velocity_avg_count: int = 5,
    diversity_sample: int = 20,
) -> dict:
    youtube_task = scan_topic(
        topic,
        velocity_pool_size=velocity_pool_size,
        velocity_avg_count=velocity_avg_count,
        diversity_sample=diversity_sample,
    )
    trends_task = asyncio.to_thread(fetch_keyword_trends, topic)
    social_task = asyncio.to_thread(scan_social_topic, topic)
    news_task = asyncio.to_thread(scan_news_topic, topic)
    youtube_result, trends_result, social_result, news_result = await asyncio.gather(
        youtube_task,
        trends_task,
        social_task,
        news_task,
    )
    combined = compute_combined_opportunity_score_v2(
        topic,
        youtube_result,
        trends_result,
        social_result,
        news_result,
    )
    m5_enhanced = compute_m5_enhanced(
        topic,
        youtube_result,
        trends_result,
        social_result,
        news_result,
    )
    m5_adaptive_v2 = compute_m5_adaptive_v2(
        topic,
        youtube_result,
        trends_result,
        social_result,
        news_result,
    )
    m5_adaptive_v3 = compute_m5_adaptive_v3(
        topic,
        youtube_result,
        trends_result,
        social_result,
        news_result,
    )
    return {
        "topic": topic,
        "youtube": youtube_result,
        "google_trends": trends_result,
        "social": social_result,
        "news": news_result,
        "combined": {
            **combined,
            "m5_enhanced": m5_enhanced,
            "m5_adaptive_v2": m5_adaptive_v2,
            "m5_adaptive_v3": m5_adaptive_v3,
        },
        "m5_enhanced": m5_enhanced,
        "m5_adaptive_v2": m5_adaptive_v2,
        "m5_adaptive_v3": m5_adaptive_v3,
    }


def print_human(payload: dict) -> None:
    youtube = payload["youtube"]
    trends = payload["google_trends"]
    social = payload["social"]
    news = payload["news"]
    combined = payload["combined"]
    enhanced = payload["m5_enhanced"]
    adaptive = payload["m5_adaptive_v2"]
    adaptive_v3 = payload["m5_adaptive_v3"]

    print("=" * 72)
    print("Topic Market Signals")
    print("=" * 72)
    print(f"Topic:                    {payload['topic']}")
    print()
    print("YouTube")
    print(f"  Upload surge ratio:     {youtube['upload_surge_ratio']}")
    print(f"  Avg view velocity:      {youtube['avg_view_velocity']}")
    print(f"  Channel diversity:      {youtube['channel_diversity']}")
    print(f"  Avg snapshot delta:     {youtube['avg_snapshot_delta']}")
    print(f"  M3 score:               {youtube['m3_score']}")
    print(f"  Quota estimate:         {youtube['quota_estimate_units']} units")
    print()
    print("Google Trends")
    print(f"  Query used:             {trends['query_used']}")
    print(f"  Trend direction:        {trends['trend_direction']}")
    print(f"  Current week index:     {trends['current_week_index']}")
    print(f"  12m avg index:          {trends['twelve_month_avg_index']}")
    print(f"  M1 search ratio:        {trends['m1_search_ratio']}")
    print(f"  M1 score:               {trends['m1_score']}")
    print(f"  Current score:          {trends['current_score']}")
    print(f"  Peak score:             {trends['peak_score']}")
    print(f"  Search surge ratio:     {trends['search_surge_ratio']}")
    print(f"  Avg recent interest:    {trends['avg_recent_interest']}")
    print(f"  Region diversity:       {trends['region_diversity']}")
    print(f"  Snapshot delta:         {trends['snapshot_delta']}")
    print(f"  Google M3 diagnostics:  {trends['m3_score']}")
    print(f"  Trending now:           {trends['is_trending_now']}")
    print(f"  Queries tried:          {', '.join(trends.get('queries_tried', []))}")
    print()
    print("Social")
    print(f"  Mentions 24h:           {social['mentions_24h']}")
    print(f"  Mentions 48h:           {social['mentions_48h']}")
    print(f"  Mentions 7d:            {social['mentions_7d']}")
    print(f"  24h vs 48h accel:       {social['accel_24h_vs_48h']}")
    print(f"  48h vs 7d accel:        {social['accel_48h_vs_7d']}")
    print(f"  Source diversity:       {social['source_diversity']}")
    print(f"  Snapshot delta 24h:     {social['snapshot_delta_24h']}")
    print(f"  M2 score:               {social['m2_score']}")
    print()
    print("News")
    print(f"  Articles 24h:           {news['articles_24h']}")
    print(f"  Articles 48h:           {news['articles_48h']}")
    print(f"  Articles 7d:            {news['articles_7d']}")
    print(f"  24h vs 48h accel:       {news['accel_24h_vs_48h']}")
    print(f"  48h vs 7d accel:        {news['accel_48h_vs_7d']}")
    print(f"  Publisher diversity:    {news['publisher_diversity']}")
    print(f"  Snapshot delta 24h:     {news['snapshot_delta_24h']}")
    print(f"  M4 score:               {news['m4_score']}")
    print()
    print("Combined")
    print(f"  Architecture:           {combined['architecture']}")
    print(f"  M1 score:               {combined['m1_google_trends_score']}")
    print(f"  M2 score:               {combined['m2_social_score']}")
    print(f"  M3 score:               {combined['m3_youtube_score']}")
    print(f"  M4 score:               {combined['m4_news_score']}")
    print(f"  Opportunity score:      {combined['opportunity_score']}")
    print(f"  Opportunity label:      {combined['opportunity_label']}")
    print(f"  YouTube weighted:       {combined['youtube_weighted_component']}")
    print(f"  Trends weighted:        {combined['trends_weighted_component']}")
    print(f"  Social weighted:        {combined['social_weighted_component']}")
    print(f"  News weighted:          {combined['news_weighted_component']}")
    print(f"  Snapshot bonus:         {combined['snapshot_bonus']}")
    print(f"  Trends M1 ratio:        {combined['google_trends_breakdown'].get('m1_search_ratio')}")
    print(f"  Trends M1 score:        {combined['google_trends_breakdown'].get('m1_score')}")
    print(f"  Trends direction:       {combined['google_trends_breakdown'].get('trend_direction')}")
    print()
    print("M5 Enhanced")
    print(f"  Architecture:           {enhanced['architecture']}")
    print(f"  Category:               {enhanced['category']}")
    print(f"  Base score:             {enhanced['base_score']}")
    print(f"  Quality factor:         {enhanced['quality_factor']}")
    print(f"  Psych factor:           {enhanced['psych_factor']}")
    print(f"  Enhanced score:         {enhanced['enhanced_score']}")
    print(f"  Enhanced label:         {enhanced['enhanced_label']}")
    print()
    print("M5 Adaptive v2")
    print(f"  Regime:                 {adaptive['detected_regime']}")
    print(f"  Base score:             {adaptive['base_score']}")
    print(f"  Psych boost:            {adaptive['psych_boost']}")
    print(f"  Coherence penalty:      {adaptive['coherence_penalty']}")
    print(f"  Reliability factor:     {adaptive['reliability_factor']}")
    print(f"  Adaptive score:         {adaptive['adaptive_score']}")
    print(f"  Adaptive label:         {adaptive['adaptive_label']}")
    print()
    print("M5 Adaptive v3")
    print(f"  Regime:                 {adaptive_v3['detected_regime']}")
    print(f"  Base score:             {adaptive_v3['base_score']}")
    print(f"  Psych boost (capped):   {adaptive_v3['psych_boost_capped']}")
    print(f"  Reliability factor:     {adaptive_v3['reliability_factor']}")
    print(f"  Adaptive v3 score:      {adaptive_v3['adaptive_v3_score']}")
    print(f"  Adaptive v3 stage:      {adaptive_v3['adaptive_v3_stage']}")
    print()
    print("Top YouTube videos by velocity")
    for item in sorted(youtube["videos"], key=lambda row: row["view_velocity"], reverse=True)[:5]:
        print(
            f"  - {item['title']} | views={item['view_count']} | "
            f"velocity={item['view_velocity']} | channel={item['channel_title']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", help="Topic to scan")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--velocity-pool-size", type=int, default=10, help="Top recent videos considered for velocity")
    parser.add_argument("--velocity-avg-count", type=int, default=5, help="Top velocity videos averaged into M3")
    parser.add_argument("--diversity-sample", type=int, default=20, help="Videos used for diversity score")
    args = parser.parse_args()

    payload = asyncio.run(
        gather_signals(
            args.topic,
            velocity_pool_size=max(1, args.velocity_pool_size),
            velocity_avg_count=max(1, args.velocity_avg_count),
            diversity_sample=max(1, args.diversity_sample),
        )
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_human(payload)


if __name__ == "__main__":
    main()
