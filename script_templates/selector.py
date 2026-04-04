from __future__ import annotations

from typing import Any

from .registry import TEMPLATE_REGISTRY


SELECTION_RULES: list[tuple[dict[str, set[str]], str]] = [
    ({"story_frame": {"hidden_angle"}, "system_dynamic": {"feedback_loop", "trade_off"}}, "dialectic_debate"),
    ({"story_frame": {"hidden_angle", "opportunity", "conflict"}, "cat_id": {"CAT-08", "CAT-07"}}, "philosophical_inquiry"),
    ({"story_frame": {"data_driven"}, "system_dynamic": {"cause_effect"}, "search_intent": {"educational"}}, "psych_concept"),
    ({"cat_id": {"CAT-07"}, "story_frame": {"human_story", "crisis"}}, "historical_event"),
    ({"cat_id": {"CAT-01"}, "story_frame": {"conflict"}, "system_dynamic": {"trade_off"}}, "scientific_controversy"),
    ({"cat_id": {"CAT-04"}, "story_frame": {"hidden_angle", "data_driven"}}, "economic_phenomenon"),
    ({"story_frame": {"data_driven"}, "system_dynamic": {"feedback_loop", "cause_effect"}}, "systems_breakdown"),
    ({"cat_id": {"CAT-03"}, "story_frame": {"crisis"}, "search_intent": {"news_driven"}}, "news_explainer"),
    ({"cat_id": {"CAT-03"}, "story_frame": {"conflict"}, "search_intent": {"news_driven", "educational"}}, "policy_breakdown"),
    ({"story_frame": {"opportunity"}, "system_dynamic": {"risk_scenario"}}, "future_scenario"),
    ({"cat_id": {"CAT-02", "CAT-08"}, "story_frame": {"human_story"}, "search_intent": {"inspirational"}}, "evidence_based_motivational"),
]


def select_template_key(
    scored_angle: dict[str, Any],
    tss_scores: dict[str, Any],
    seo_output: dict[str, Any],
    template_key_override: str | None = None,
) -> tuple[str, str]:
    if template_key_override:
        if template_key_override not in TEMPLATE_REGISTRY:
            raise ValueError("invalid_template_key")
        return template_key_override, "override"

    signals = {
        "story_frame": str(scored_angle.get("story_frame", "")),
        "system_dynamic": str(scored_angle.get("how", "")),
        "cat_id": str(tss_scores.get("cat_id", "CAT-08")),
        "search_intent": str(seo_output.get("search_intent_type", "")),
        "who": str(scored_angle.get("who", "")),
    }

    for conditions, key in SELECTION_RULES:
        if all(signals.get(field, "") in values for field, values in conditions.items()):
            return key, "rule_match"

    scores: dict[str, int] = {}
    for key, template in TEMPLATE_REGISTRY.items():
        about = str(template.get("about", "")).lower()
        score = 0
        for val in signals.values():
            if val and val.lower() in about:
                score += 1
        if signals["cat_id"] == "CAT-03" and "news" in about:
            score += 2
        if signals["cat_id"] == "CAT-01" and ("tech" in about or "science" in about):
            score += 2
        scores[key] = score
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best, "fallback_score"
    return "systems_breakdown", "default"
