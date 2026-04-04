from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field


class AgentPipelineContext(BaseModel):
    topic: str
    selected_idea_id: str
    selected_angle_id: str
    selected_idea: dict[str, Any]
    gap_context: dict[str, Any]
    db_context: str = ""
    web_context: str = ""
    social_data: list[dict[str, Any]] = Field(default_factory=list)
    news_data: list[dict[str, Any]] = Field(default_factory=list)
    tss_scores: dict[str, Any] = Field(default_factory=dict)
    csi_scores: dict[str, Any] = Field(default_factory=dict)
    csi_quality: dict[str, Any] = Field(default_factory=dict)
    pipeline_assembled_at: dt.datetime
    seo_output: dict[str, Any] | None = None


def extract_angle_for_prompt(scored_angle: dict[str, Any]) -> dict[str, Any]:
    return {
        "who": scored_angle.get("who", ""),
        "what": ", ".join(str(x) for x in (scored_angle.get("what") or []) if str(x).strip()),
        "when": scored_angle.get("when", "present"),
        "scale": scored_angle.get("scale", "national"),
        "system_dynamic": scored_angle.get("how", ""),
        "power_layer": scored_angle.get("who_benefits", ""),
        "story_frame": scored_angle.get("story_frame", ""),
        "angle_string": scored_angle.get("angle_string", ""),
        "hook_sentence": scored_angle.get("hook_sentence", ""),
        "cags_score": scored_angle.get("cags_score", 0),
    }


def staleness_hours(pipeline_assembled_at: dt.datetime, now_utc: dt.datetime | None = None) -> float:
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    ts = pipeline_assembled_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return max((now - ts.astimezone(dt.timezone.utc)).total_seconds() / 3600.0, 0.0)
