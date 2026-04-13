from __future__ import annotations

import datetime as dt
import hashlib
import math
import random
import re
import threading
import time
from typing import Any


__all__ = ["CorpusStalenessError", "calculate_csi", "normalise_corpus", "build_exclusive_cohorts"]

_EMBED_CACHE: dict[str, list[float]] = {}
_EMBED_CACHE_MAX = 5000
_GEMINI_EMBED_LOCK = threading.Lock()
_GEMINI_LAST_CALL_TS = 0.0
_GEMINI_MIN_INTERVAL_SEC = 0.35
_GEMINI_BLOCKED_UNTIL_TS = 0.0
_GEMINI_BATCH_MAX_ITEMS = 100


def _embed_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _extract_retry_delay_seconds(error_text: str) -> float | None:
    if not error_text:
        return None
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", error_text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _is_daily_quota_error(error_text: str) -> bool:
    lowered = (error_text or "").lower()
    return "resource_exhausted" in lowered and "perday" in lowered


def _embed_texts_with_gemini_batches(
    client: Any,
    embedding_model: str,
    inputs: list[str],
    *,
    task_type: str = "SEMANTIC_SIMILARITY",
    output_dimensionality: int = 768,
) -> list[list[float]]:
    """Embed texts in Gemini-safe batches to avoid the 100-request batch cap."""
    if not inputs:
        return []
    models = getattr(client, "models", None)
    if models is None or not hasattr(models, "embed_content"):
        raise AttributeError("gemini client lacks embed_content")
    config_cls = getattr(models, "EmbedContentConfig", None)
    config = (
        config_cls(
            task_type=task_type,
            output_dimensionality=output_dimensionality,
        )
        if config_cls
        else None
    )
    embeddings: list[list[float]] = []
    for start in range(0, len(inputs), _GEMINI_BATCH_MAX_ITEMS):
        batch = inputs[start : start + _GEMINI_BATCH_MAX_ITEMS]
        resp = models.embed_content(model=embedding_model, contents=batch, config=config)
        batch_embeddings = [list(e.values) for e in getattr(resp, "embeddings", []) or []]
        if len(batch_embeddings) != len(batch):
            raise ValueError("insufficient embeddings returned for batch")
        embeddings.extend(batch_embeddings)
    return embeddings


class CorpusStalenessError(RuntimeError):
    pass


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def normalise_corpus(corpus: list[dict[str, Any]], corpus_fetched_at: dt.datetime) -> None:
    fetch_ts = corpus_fetched_at.timestamp()
    for video in corpus:
        published_ts = float(video.get("published_ts") or 0)
        hours_live = max((fetch_ts - published_ts) / 3600.0, 1.0)
        days_live = hours_live / 24.0
        view_count = int(video.get("view_count") or 0)
        video["_vpd"] = view_count / max(days_live, 1.0)
        likes_disabled = bool(video.get("likes_disabled"))
        comments_disabled = bool(video.get("comments_disabled"))
        video["_eng_valid"] = not (likes_disabled or comments_disabled)
        if video["_eng_valid"]:
            likes = int(video.get("like_count") or 0)
            comments = int(video.get("comment_count") or 0)
            video["_eng_rate"] = (likes + comments) / max(view_count, 1)
        else:
            video["_eng_rate"] = None


def build_exclusive_cohorts(
    corpus: list[dict[str, Any]], corpus_fetched_at: dt.datetime
) -> dict[str, list[dict[str, Any]]]:
    windows = {
        "w_24h": (0, 24),
        "w_24_48h": (24, 48),
        "w_48_72h": (48, 72),
        "w_7d": (72, 168),
        "w_30d": (168, 720),
        "w_old": (720, float("inf")),
    }
    cohorts = {key: [] for key in windows}
    now_ts = corpus_fetched_at.timestamp()
    for video in corpus:
        age_hours = max((now_ts - float(video.get("published_ts") or 0)) / 3600.0, 0.0)
        for key, (lo, hi) in windows.items():
            if lo <= age_hours < hi:
                cohorts[key].append(video)
                break
    return cohorts


def calculate_csi(
    corpus: list[dict[str, Any]],
    corpus_fetched_at: dt.datetime,
    total_results: int,
    tss_search_score: float,
    m1_norm: dict[str, Any],
    m3_norm: dict[str, Any],
    m4_norm: dict[str, Any],
    gemini_client: Any,
    region_code: str = "US",
    language_code: str = "en",
    creator_subs: int | None = None,
    embedding_model: str = "gemini-embedding-001",
) -> dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    staleness_hours = (now - corpus_fetched_at).total_seconds() / 3600.0
    if staleness_hours > 2.0:
        raise CorpusStalenessError(f"Corpus is {staleness_hours:.1f}h old — refetch required.")
    data_quality = {
        "corpus_stale_warning": 1.0 < staleness_hours <= 2.0,
        "engagement_insufficient": False,
        "engagement_coverage": 0.0,
        "redundancy_embedding_failed": False,
        "redundancy_used_fallback": False,
    }

    normalise_corpus(corpus, corpus_fetched_at)
    cohorts = build_exclusive_cohorts(corpus, corpus_fetched_at)

    supply_f = compute_supply(corpus, total_results, cohorts, data_quality)
    demand_f = compute_demand(corpus, tss_search_score, data_quality)
    fresh_f = compute_freshness(cohorts, data_quality)
    virality_f = compute_virality(corpus, cohorts, m1_norm, m3_norm, m4_norm, supply_f["gini"])
    quality_f = compute_quality_gap(corpus, corpus_fetched_at, data_quality)
    redundancy_f = compute_redundancy_score(corpus, gemini_client, embedding_model, data_quality)

    csi_score = calculate_csi_score(
        S=supply_f["score"],
        D=demand_f["score"],
        F=fresh_f["score"],
        R=redundancy_f["score"] / 100.0,
        V=virality_f["structural_score"] / 100.0,
        QG=quality_f["score"],
    )
    label = csi_label(csi_score, creator_subs)

    return build_output(
        csi_score,
        label,
        supply_f,
        demand_f,
        fresh_f,
        redundancy_f,
        virality_f,
        quality_f,
        data_quality,
    )


def compute_supply(
    corpus: list[dict[str, Any]],
    total_results: int,
    cohorts: dict[str, list[dict[str, Any]]],
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    vol_norm = min(total_results / 100_000, 1.0)
    w30 = len(cohorts.get("w_30d", []))
    upload_freq = w30 / 30.0
    freq_norm = min(upload_freq / 20.0, 1.0)
    total = max(len(corpus), 1)

    def _creator_key(v: dict[str, Any]) -> str | None:
        for key in ("channel_id", "channelId", "channel_title", "channel", "author", "uploader"):
            val = v.get(key)
            if val is not None:
                s = str(val).strip()
                if s:
                    return s
        return None

    creator_keys = {_creator_key(v) for v in corpus}
    unique_channels = len({k for k in creator_keys if k})
    creator_density = unique_channels / total
    density_norm = 1.0 - creator_density
    vpd_vals = sorted(v.get("_vpd", 0.0) for v in corpus)
    n = len(vpd_vals)
    total_vpd = sum(vpd_vals) or 1.0
    gini = sum((2 * i - n - 1) * v for i, v in enumerate(vpd_vals, 1)) / (n * total_vpd) if n else 0.0
    gini = round(max(gini, 0.0), 3)
    supply_score = (vol_norm + freq_norm + density_norm + gini) / 4.0
    return {
        "score": round(min(max(supply_score, 0.0), 1.0), 3),
        "vol_norm": round(vol_norm, 3),
        "upload_freq_per_day": round(upload_freq, 3),
        "creator_density": round(creator_density, 3),
        "gini": gini,
        "creator_count": unique_channels,
    }


def compute_demand(
    corpus: list[dict[str, Any]],
    tss_search_score: float,
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    total = max(len(corpus), 1)
    avg_vpd = sum(v.get("_vpd", 0.0) for v in corpus) / total
    view_demand_norm = min(avg_vpd / 10_000, 1.0)
    valid_eng = [v["_eng_rate"] for v in corpus if v.get("_eng_valid")]
    if len(valid_eng) >= 3:
        eng_norm = min((sum(valid_eng) / len(valid_eng)) / 0.05, 1.0)
    else:
        eng_norm = 0.5
        data_quality["engagement_insufficient"] = True
    coverage = round(len(valid_eng) / total, 2)
    data_quality["engagement_coverage"] = coverage
    search_norm = min(max(tss_search_score / 100.0, 0.0), 1.0)
    demand_score = (view_demand_norm + eng_norm + search_norm) / 3.0
    return {
        "score": round(min(max(demand_score, 0.0), 1.0), 3),
        "avg_vpd": round(avg_vpd, 1),
        "engagement_coverage": coverage,
        "search_trend_score": round(tss_search_score, 2),
    }


def compute_freshness(
    cohorts: dict[str, list[dict[str, Any]]], data_quality: dict[str, Any]
) -> dict[str, Any]:
    windows = ["w_24h", "w_24_48h", "w_48_72h", "w_7d", "w_30d"]
    window_counts = {key: len(cohorts.get(key, [])) for key in windows}
    total = max(sum(len(cohorts.get(key, [])) for key in windows) + len(cohorts.get("w_old", [])), 1)
    ratios = {key: window_counts[key] / total for key in windows}
    freshness_score = (
        ratios["w_24h"] * 0.35
        + ratios["w_24_48h"] * 0.25
        + ratios["w_48_72h"] * 0.15
        + ratios["w_7d"] * 0.15
        + ratios["w_30d"] * 0.10
    )
    freshness_ratio = sum(window_counts.values()) / total
    data_quality["freshness_ratio"] = round(freshness_ratio, 3)
    return {
        "score": round(min(max(freshness_score, 0.0), 1.0), 3),
        "window_counts": window_counts,
        "freshness_ratio": round(freshness_ratio, 3),
    }


def compute_virality(
    corpus: list[dict[str, Any]],
    cohorts: dict[str, list[dict[str, Any]]],
    m1_norm: dict[str, Any],
    m3_norm: dict[str, Any],
    m4_norm: dict[str, Any],
    gini: float,
) -> dict[str, Any]:
    def avg_vpd(cohort: list[dict[str, Any]]) -> float:
        if not cohort:
            return 0.0
        return sum(v.get("_vpd", 0.0) for v in cohort) / len(cohort)

    top_vpd = sorted((v.get("_vpd", 0.0) for v in corpus), reverse=True)
    n = len(top_vpd)
    top5_count = max(1, n // 20)
    top5_avg = sum(top_vpd[:top5_count]) / top5_count if top5_count else 0.0
    median_vpd = top_vpd[n // 2] if n else 1.0
    view_conc = min(top5_avg / (median_vpd or 1.0), 50.0) / 50.0
    market_openness = max(0.0, min(1.0 - gini, 1.0))
    eng_ceiling = min(max(m3_norm.get("engagement_rate", 0.0) / 0.10, 0.0), 1.0)
    vpd_ceiling = min(max(max((v.get("_vpd", 0.0) for v in corpus), default=0.0) / 50_000, 0.0), 1.0)
    structural = (
        view_conc * 0.35 + market_openness * 0.25 + eng_ceiling * 0.20 + vpd_ceiling * 0.20
    )
    structural_score = round(min(max(structural, 0.0), 1.0) * 100.0, 1)

    w24 = cohorts.get("w_24h", [])
    w24_48 = cohorts.get("w_24_48h", [])
    w48_72 = cohorts.get("w_48_72h", [])
    w7d = cohorts.get("w_7d", [])
    w30d = cohorts.get("w_30d", [])
    w_old = cohorts.get("w_old", [])

    # def avg_eng(cohort: list[dict[str, Any]]) -> float:
    #     vals = [v["_eng_rate"] for v in cohort if v.get("_eng_valid")]
    #     return sum(vals) / len(vals) if vals else 0.0

    old_baseline = avg_vpd(w_old) or avg_vpd(w30d) or 1.0
    m3_ratio = m3_norm.get("acceleration", 1.0)
    accel_24h = min(avg_vpd(w24) / max(old_baseline, 1.0) / 3.0, 1.0)
    accel_24_48 = min(avg_vpd(w24_48) / max(old_baseline, 1.0) / 3.0, 1.0)
    accel_48_72 = min(avg_vpd(w48_72) / max(old_baseline, 1.0) / 3.0, 1.0)
    view_signal = (
        min(m3_ratio / 4.0, 1.0) * 0.40
        + (accel_24h * 0.50 + accel_24_48 * 0.30 + accel_48_72 * 0.20) * 0.60
    )
    daily_7d_avg = max(len(w7d) / 7.0, 0.1)
    surge_24h = min(len(w24) / daily_7d_avg / 4.0, 1.0)
    surge_24_48 = min(len(w24_48) / daily_7d_avg / 4.0, 1.0)
    surge_48_72 = min(len(w48_72) / daily_7d_avg / 4.0, 1.0)
    upload_signal = surge_24h * 0.50 + surge_24_48 * 0.30 + surge_48_72 * 0.20
    def avg_eng_ratio(cohort: list[dict[str, Any]]) -> float:
        rates = [v["_eng_rate"] for v in cohort if v.get("_eng_valid")]
        return sum(rates) / len(rates) if rates else 0.0
    eng_heat = min((avg_eng_ratio(w7d) / max(avg_eng_ratio(w_old), 0.001)) / 2.0, 1.0)
    gdelt_tone = m4_norm.get("gdelt_tone_avg", 0.0)
    controversy = min(abs(gdelt_tone) / 4.0, 1.0)
    eng_signal = eng_heat * 0.70 + controversy * 0.30
    news_signal = (
        min(max(m4_norm.get("velocity", 1.0) / 6.0, 0.0), 1.0) * 0.70
        + min(max(m4_norm.get("source_count", 1) / 20.0, 0.0), 1.0) * 0.30
    )
    search_velocity = m1_norm.get("velocity", 1.0)
    search_slope = 1.0 if m1_norm.get("slope_dir", "flat") == "up" else 0.5
    search_signal = min(search_velocity / 3.0, 1.0) * search_slope
    momentum_score = round(
        (
            search_signal * 0.25
            + view_signal * 0.30
            + upload_signal * 0.20
            + eng_signal * 0.15
            + news_signal * 0.10
        )
        * 100.0,
        1,
    )
    thresholds = {
        "search_breaking": search_velocity >= 2.0,
        "yt_breaking": m3_ratio >= 2.0,
        "upload_breaking": surge_24h >= 0.5,
        "eng_breaking": eng_heat >= 0.6,
        "view_spike_24h": accel_24h >= 0.5,
    }
    t_count = sum(thresholds.values())
    if t_count >= 3:
        breakout_indicator = "BREAKING OUT"
    elif t_count == 2:
        breakout_indicator = "WATCH CLOSELY"
    elif t_count == 1:
        breakout_indicator = "EARLY SIGNAL"
    else:
        breakout_indicator = "NO BREAKOUT"
    return {
        "structural_score": structural_score,
        "momentum_score": momentum_score,
        "breakout_indicator": breakout_indicator,
        "thresholds_fired": t_count,
        "threshold_breakdown": thresholds,
    }


def compute_quality_gap(
    corpus: list[dict[str, Any]],
    corpus_fetched_at: dt.datetime,
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    total = max(len(corpus), 1)
    valid_eng = [v["_eng_rate"] for v in corpus if v.get("_eng_valid")]
    if len(valid_eng) >= 3:
        eng_gap_norm = min(sum(v for v in valid_eng) / len(valid_eng), 1.0)
        # actual gap normalized by 60% underperformance: need to compute per doc
        gap_signals = []
        for v in corpus:
            if not v.get("_eng_valid"):
                continue
            expected = expected_eng_rate(v.get("view_count", 0))
            if expected <= 0:
                continue
            gap = max(expected - v["_eng_rate"], 0.0) / expected
            gap_signals.append(gap)
        if gap_signals:
            eng_gap_norm = min(sum(gap_signals) / len(gap_signals) / 0.60, 1.0)
        else:
            eng_gap_norm = 0.5
    else:
        eng_gap_norm = 0.5
        data_quality["engagement_insufficient"] = True
    data_quality["engagement_coverage"] = round(len(valid_eng) / total, 2)

    top_20pct = sorted(corpus, key=lambda v: v.get("view_count", 0), reverse=True)[: max(1, total // 5)]
    decay_signals = []
    for video in top_20pct:
        views = video.get("view_count", 0)
        if views < 10_000:
            continue
        expected_vpd = views / 365.0
        actual_vpd = video.get("_vpd", 0.0)
        decay = max(1.0 - (actual_vpd / max(expected_vpd, 1.0)), 0.0)
        decay_signals.append(decay)
    vpd_decay_norm = sum(decay_signals) / max(len(decay_signals), 1) if decay_signals else 0.5

    score = (eng_gap_norm * 0.60) + (vpd_decay_norm * 0.40)
    return {
        "score": round(min(max(score, 0.0), 1.0), 3),
        "eng_gap_norm": round(min(max(eng_gap_norm, 0.0), 1.0), 3),
        "vpd_decay_norm": round(min(max(vpd_decay_norm, 0.0), 1.0), 3),
    }


def expected_eng_rate(views: int) -> float:
    if views < 10_000:
        return 0.030
    if views < 100_000:
        return 0.020
    if views < 1_000_000:
        return 0.012
    return 0.007


def _average_cosine_similarity(embeddings: list[list[float]]) -> float:
    norms = [math.sqrt(sum(val * val for val in vec)) or 1.0 for vec in embeddings]
    n = len(embeddings)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dot = sum((embeddings[i][k] or 0.0) * (embeddings[j][k] or 0.0) for k in range(len(embeddings[i])))
            total += dot / (norms[i] * norms[j])
            count += 1
    return total / count if count else 0.0


def _embed_via_lm_studio(texts: list[str]) -> list[list[float]] | None:
    """
    LM Studio embedding fallback.

    Important: LM Studio has two "API styles":
    - OpenAI-compatible:   POST /v1/embeddings
    - LM Studio API:       /api/v1/* (does NOT currently expose embeddings)

    If you hit `/api/v1/embeddings` you'll see "Unexpected endpoint" in LM Studio logs.
    """

    import os

    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234").rstrip("/")
    embed_url = os.getenv("LMSTUDIO_EMBEDDINGS_URL", f"{base_url}/v1/embeddings")
    model_id = os.getenv("LMSTUDIO_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")

    try:
        import httpx

        response = httpx.post(
            embed_url,
            json={"model": model_id, "input": texts},
            timeout=30.0,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        vecs = [item.get("embedding") for item in data if item.get("embedding")]
        if not vecs:
            raise ValueError("no embeddings returned")

        target_dim = len(vecs[0])
        if target_dim != 768:
            raise ValueError(f"unexpected embedding dimension {target_dim} (expected 768)")
        if any(len(vec) != target_dim for vec in vecs):
            raise ValueError("inconsistent embedding dimensions")
        return vecs
    except Exception as exc:
        print(f"LM Studio embedding failed: {exc}")
        print(
            "LM Studio hint: enable the OpenAI-compatible server and ensure the embedding model is loaded. "
            f"Expected endpoint: {base_url}/v1/embeddings"
        )
        return None


def compute_redundancy_score(
    corpus: list[dict[str, Any]],
    gemini_client: Any,
    embedding_model: str,
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    global _GEMINI_LAST_CALL_TS, _GEMINI_BLOCKED_UNTIL_TS

    def build_embed_input(video: dict[str, Any]) -> str:
        desc = (video.get("description") or "")[:200].strip()
        title = video.get("title") or video.get("video_id") or ""
        return f"{title}. {desc}" if desc else title

    if len(corpus) < 3:
        return {"score": 50.0, "avg_cosine_similarity": 0.0, "embeddings": None}

    inputs = [build_embed_input(v) for v in corpus]
    embeddings: list[list[float]] | None = None
    cache_keys = [_embed_cache_key(text) for text in inputs]
    cached_vectors = [_EMBED_CACHE.get(k) for k in cache_keys]
    if all(vec is not None for vec in cached_vectors):
        embeddings = [list(vec or []) for vec in cached_vectors]

    # Try Gemini first
    if gemini_client and embeddings is None and time.time() >= _GEMINI_BLOCKED_UNTIL_TS:
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                with _GEMINI_EMBED_LOCK:
                    now = time.time()
                    sleep_for = _GEMINI_MIN_INTERVAL_SEC - (now - _GEMINI_LAST_CALL_TS)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    _GEMINI_LAST_CALL_TS = time.time()
                    embeddings = _embed_texts_with_gemini_batches(
                        gemini_client,
                        embedding_model,
                        inputs,
                        task_type="SEMANTIC_SIMILARITY",
                        output_dimensionality=768,
                    )
                if len(embeddings) < 2:
                    raise ValueError("insufficient embeddings")

                for key, vec in zip(cache_keys, embeddings):
                    if len(_EMBED_CACHE) >= _EMBED_CACHE_MAX:
                        _EMBED_CACHE.clear()
                    _EMBED_CACHE[key] = vec
                break
            except Exception as exc:
                error_text = str(exc)
                if _is_daily_quota_error(error_text):
                    _GEMINI_BLOCKED_UNTIL_TS = time.time() + 3600.0
                    embeddings = None
                    break
                if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                    delay = _extract_retry_delay_seconds(error_text)
                    if delay is None:
                        delay = min(2 ** attempt + random.random(), 30.0)
                    time.sleep(delay)
                    continue
                embeddings = None
                break

    if embeddings is None:
        embeddings = _embed_via_lm_studio(inputs)

    if embeddings:
        avg_sim = _average_cosine_similarity(embeddings)
        normalised = (0.90 - avg_sim) / 0.60
        # Inline clamp to avoid any accidental name shadowing / import weirdness.
        score = max(0.0, min(normalised * 100.0, 100.0))
        return {"score": round(score, 1), "avg_cosine_similarity": avg_sim, "embeddings": embeddings}

    data_quality["redundancy_embedding_failed"] = True
    data_quality["redundancy_used_fallback"] = True
    return {"score": 50.0, "avg_cosine_similarity": 0.0, "embeddings": None}


def calculate_csi_score(S: float, D: float, F: float, R: float, V: float, QG: float) -> float:
    return (
        0.25 * S
        + 0.25 * (1.0 - D)
        + 0.15 * F
        + 0.15 * R
        + 0.10 * (1.0 - V)
        + 0.10 * (1.0 - QG)
    ) * 100.0


def csi_label(csi: float, creator_subs: int | None) -> str:
    if csi <= 25.0:
        return "OPEN"
    if csi <= 50.0:
        return "COMPETITIVE"
    if csi <= 70.0:
        return "CROWDED"
    return "SATURATED"


def build_output(
    csi_score: float,
    label: str,
    supply_f: dict[str, Any],
    demand_f: dict[str, Any],
    fresh_f: dict[str, Any],
    redundancy_f: dict[str, Any],
    virality_f: dict[str, Any],
    quality_f: dict[str, Any],
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    return {
        "csi": round(min(max(csi_score, 0.0), 100.0), 2),
        "label": label,
        "redundancy_score": float(redundancy_f.get("score", 0.0)),
        "freshness_ratio": fresh_f.get("freshness_ratio", 0.0),
        "embeddings": redundancy_f.get("embeddings"),
        "supply": supply_f,
        "demand": demand_f,
        "freshness": fresh_f,
        "redundancy": redundancy_f,
        "virality": {
            "structural_score": virality_f.get("structural_score"),
            "momentum_score": virality_f.get("momentum_score"),
            "breakout_indicator": virality_f.get("breakout_indicator"),
            "thresholds_fired": virality_f.get("thresholds_fired"),
            "threshold_breakdown": virality_f.get("threshold_breakdown"),
        },
        "quality_gap": quality_f,
        "data_quality": data_quality,
    }
