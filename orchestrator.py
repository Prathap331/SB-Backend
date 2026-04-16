from __future__ import annotations

from typing import Any

from tss_v3 import run_tss


def synthesize_verdict(tss: float, csi: float, top_cags: float) -> dict[str, str]:
    if tss > 30 and csi < 65 and top_cags > 40:
        return {
            "verdict": "GO",
            "reason": "Strong demand, low saturation, clear angle available",
        }
    if csi > 75 and top_cags < 35:
        return {"verdict": "SKIP", "reason": "Market is saturated, all angles covered"}
    if tss < 20:
        return {"verdict": "MONITOR", "reason": "Too early — check back in 48 hours"}
    return {"verdict": "CAUTION", "reason": "Mixed signals — review angles before committing"}


async def run_pipeline(topic: str, region: str = "IN") -> dict[str, Any]:
    try:
        result = await run_tss(topic, region=region) 
    except TypeError:
        result = await run_tss(topic)

    tss_score = float(result.get("tss", 0) or 0)
    csi_score = float(result.get("csi", {}).get("csi", 50) or 50)
    gap_angles = result.get("cags", {}).get("gap_angles", []) or []
    top_cags = max((float(a.get("cags_score", 0) or 0) for a in gap_angles), default=0.0)

    result["verdict"] = synthesize_verdict(tss_score, csi_score, top_cags)
    return result
