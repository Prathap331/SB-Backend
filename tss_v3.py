#!/usr/bin/env python3
"""
Trend Strength Score (TSS) V3 — exact spec implementation.

Usage:
  python3 tss_v3.py "Israel Iran War" --json
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import os
import re
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    load_dotenv = lambda: None

from news_market_signals import (
    fetch_newsapi_window,
    fetch_gdelt_artlist,
    get_newsapi_key,
    parse_gdelt_seendate,
)
from social_market_signals import scan_topic as scan_social_topic
from youtube_market_signals import normalize_score as normalize_ratio_score, scan_topic as scan_youtube_topic
from cags import calculate_cags
from csi import calculate_csi, CorpusStalenessError

try:
    from google import genai
except ImportError:
    genai = None
try:
    from groq import AsyncGroq
except ImportError:
    AsyncGroq = None


PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CENTROIDS_PATH = CACHE_DIR / "tss_category_centroids.json"
CATEGORY_REVIEW_LOG = CACHE_DIR / "tss_category_review.log"

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
NEWSAPI_URL = "https://newsapi.org/v2/everything"

load_dotenv()

GEMINI_CLIENT: Any | None = None
GROQ_CLIENT: Any | None = None


def get_gemini_client() -> Any | None:
    global GEMINI_CLIENT
    if GEMINI_CLIENT is not None:
        return GEMINI_CLIENT
    if genai is None:
        return None
    api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY") or "").strip()
    if not api_key:
        return None
    try:
        GEMINI_CLIENT = genai.Client(api_key=api_key)
    except Exception:
        GEMINI_CLIENT = None
    return GEMINI_CLIENT


def get_groq_client() -> Any | None:
    global GROQ_CLIENT
    if GROQ_CLIENT is not None:
        return GROQ_CLIENT
    if AsyncGroq is None:
        return None
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key:
        return None
    try:
        GROQ_CLIENT = AsyncGroq(api_key=key)
    except Exception:
        GROQ_CLIENT = None
    return GROQ_CLIENT


def get_region_code() -> str:
    return (os.getenv("DEFAULT_REGION_CODE") or "US").strip().upper()


def get_language_code() -> str:
    return (os.getenv("DEFAULT_LANGUAGE_CODE") or "en").strip().lower()


def _ensure_datetime(value: str | dt.datetime | None) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str):
        parsed = parse_api_datetime(value)
        if parsed:
            return parsed
    return utc_now()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(value: dt.datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_topic_key(topic: str) -> str:
    return re.sub(r"\s+", " ", (topic or "").strip().lower())


def _check_staleness(
    fetched_at: dt.datetime | float | int | None, *, threshold_hours: float | None = None
) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    if fetched_at is None:
        return "ok"
    try:
        threshold = STALE_HOURS if threshold_hours is None else threshold_hours
        if isinstance(fetched_at, (int, float)):
            age_hours = (now.timestamp() - float(fetched_at)) / 3600.0
        elif isinstance(fetched_at, dt.datetime):
            candidate = fetched_at
            if candidate.tzinfo is None:
                candidate = candidate.replace(tzinfo=dt.timezone.utc)
            age_hours = (now - candidate).total_seconds() / 3600.0
        else:
            return "ok"
        return "stale" if age_hours > threshold else "ok"
    except Exception:
        return "ok"


def parse_api_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def ensure_pytrends_retry_compat() -> None:
    try:
        from urllib3.util import Retry
    except Exception:
        return
    if "method_whitelist" in Retry.__init__.__code__.co_varnames:
        return

    original_init = Retry.__init__

    def patched_init(self, *args, **kwargs):
        if "method_whitelist" in kwargs and "allowed_methods" not in kwargs:
            kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
        return original_init(self, *args, **kwargs)

    Retry.__init__ = patched_init


CATEGORY_LABELS = {
    "CAT-01": "Technology",
    "CAT-02": "Entertainment",
    "CAT-03": "Politics / News",
    "CAT-04": "Finance / Markets",
    "CAT-05": "Sports",
    "CAT-06": "Fashion",
    "CAT-07": "History",
    "CAT-08": "General",
}

# --- Spec constants (TSS Spec V3) ---
DOMINANCE_THRESHOLD = 1.30
DOMINANCE_SOFT_HIGH = 1.40
ABSOLUTE_GUARD = 40.0
R4_ACCEL_THRESHOLD = 0.50
R3_RATIO_THRESHOLD = 2.00
R2_DIVERSITY_THRESHOLD = 0.50
CLASH_CLEAR_MARGIN = 0.15
AMBIGUITY_PENALTY = 0.10
B1_B4_CORR_THRESHOLD = 0.15
B1_B4_DAMPEN = 0.88
CATEGORY_SIM_THRESHOLD = 0.72
B1_DENOM = 2.0
B4_DENOM = 2.0
RELIABILITY_FLOOR = 0.40
STALE_HOURS = 4

SEED_DICTIONARY = {
    "CAT-01": {
        "ai", "artificial intelligence", "llm", "chatgpt", "claude", "developer tools", "software engineering",
        "automation", "saas", "cloud computing", "cybersecurity", "data science",
    },
    "CAT-02": {
        "movie", "netflix", "marvel", "hollywood", "taylor swift", "music video", "celebrity",
        "anime", "gaming", "trailer", "bollywood",
    },
    "CAT-03": {
        "war", "conflict", "election", "ceasefire", "government policy", "bill", "regulation",
        "geopolitics", "sanctions",
    },
    "CAT-04": {
        "stock", "etf", "yield", "interest rate", "crypto", "bitcoin", "forex", "inflation",
        "earnings", "market crash",
    },
    "CAT-05": {
        "football", "cricket", "nba", "ipl", "fifa", "world cup", "formula 1", "tennis",
    },
    "CAT-06": {
        "fashion", "aesthetic", "mob wife aesthetic", "style trend", "outfit", "runway", "beauty", "skincare",
    },
    "CAT-07": {
        "history", "ww2", "ancient rome", "cold war", "biography", "documentary",
    },
    "CAT-08": set(),
}


def log_for_review(keyword: str, scores: dict[str, float]) -> None:
    CATEGORY_REVIEW_LOG.parent.mkdir(exist_ok=True)
    with CATEGORY_REVIEW_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{iso_utc(utc_now())}\t{keyword}\t{json.dumps(scores, ensure_ascii=True)}\n")


def get_embedding_client():
    load_dotenv()
    api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_embedding_client()
    if client is None:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY not configured for embeddings.")
    model = "models/embedding-001"
    response = client.embed_content(model=model, content=texts, task_type="retrieval_document")
    embeddings = [e.values for e in response.embeddings]
    return embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_or_build_centroids() -> dict[str, list[float]]:
    if CENTROIDS_PATH.exists():
        return json.loads(CENTROIDS_PATH.read_text(encoding="utf-8"))
    centroids: dict[str, list[float]] = {}
    for cat_id, seeds in SEED_DICTIONARY.items():
        if not seeds:
            continue
        try:
            embeddings = embed_texts(sorted(seeds))
        except Exception:
            return {}
        centroid = [sum(vals) / len(vals) for vals in zip(*embeddings)]
        centroids[cat_id] = centroid
    if centroids:
        CENTROIDS_PATH.write_text(json.dumps(centroids, ensure_ascii=True), encoding="utf-8")
    return centroids


def classify_keyword(keyword: str) -> tuple[str, float, int]:
    cleaned = normalize_topic_key(keyword)
    # Layer 1: exact seed match
    kw_lower = cleaned.lower()
    for cat_id, seeds in SEED_DICTIONARY.items():
        for seed in seeds:
            if seed and seed.lower() in kw_lower:
                return cat_id, 1.0, 1

    # Overrides
    if re.search(r"\b(policy|regulation|bill)\b", cleaned):
        return "CAT-03", 1.0, 1
    if re.search(r"\b(stock|etf|price|yield)\b", cleaned):
        return "CAT-04", 1.0, 1

    # Single-word heuristic
    if len(cleaned.split()) == 1:
        suffixes = ["news", "stock", "movie", "music", "team", "history", "fashion", "technology"]
        for suffix in suffixes:
            candidate = f"{cleaned} {suffix}"
            for cat_id, seeds in SEED_DICTIONARY.items():
                if candidate in seeds:
                    return cat_id, 1.0, 1

    # Layer 2: embedding similarity
    centroids = load_or_build_centroids()
    if centroids:
        try:
            kw_emb = embed_texts([keyword])[0]
            scores = {cat: cosine_similarity(kw_emb, centroid) for cat, centroid in centroids.items()}
            best = max(scores, key=scores.get)
            best_score = scores[best]
            if best_score >= CATEGORY_SIM_THRESHOLD:
                sorted_scores = sorted(scores.values(), reverse=True)
                if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < 0.03:
                    log_for_review(keyword, scores)
                    return "CAT-08", 0.0, 3
                return best, round(best_score, 3), 2
            log_for_review(keyword, scores)
        except Exception:
            pass
    return "CAT-08", 0.0, 3


def effective_score(score: float, status: str) -> tuple[float, bool]:
    if status in ("ok", "new_topic"):
        return score, True
    if status == "stale":
        return score * 0.80, True
    return 0.0, False


def _slope_direction(values: list[float]) -> str:
    if len(values) < 2:
        return "flat"
    xs = list(range(len(values)))
    mean_x = sum(xs) / len(xs)
    mean_y = sum(values) / len(values)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return "flat"
    slope = num / den
    if slope > 0.05:
        return "up"
    if slope < -0.05:
        return "down"
    return "flat"


def compute_m1(keyword: str) -> dict[str, Any]:
    """
    M1 (Search interest velocity).

    Primary: pytrends (free, but can break depending on urllib3/requests versions).
    Fallback: SerpAPI Google Trends (paid) via `google_trends_only.py`.
    """
    try:
        ensure_pytrends_retry_compat()
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=0, timeout=(5, 12), retries=1, backoff_factor=0.25)
        pytrends.build_payload([keyword], timeframe="today 12-m", geo="")
        df = pytrends.interest_over_time()
        if df.empty or keyword not in df.columns:
            raise ValueError("pytrends returned empty series")
        series = df[keyword].dropna()
        weeks = len(series)
        current_index = float(series.iloc[-1]) if weeks else 0.0
        baseline_avg = float(series.mean()) if weeks else 0.0
        ratio = (current_index / baseline_avg) if baseline_avg > 0 else 0.0
        m1_score = round(normalize_ratio_score(ratio, 5.0), 2)
        last_ts = series.index[-1] if weeks else None
        last_ts = last_ts.to_pydatetime().replace(tzinfo=dt.timezone.utc) if last_ts is not None else None
        slope_dir = _slope_direction(series.tolist())
        status = "ok" if weeks >= 2 else "fail"
        if status == "ok":
            # Pytrends interest_over_time() is typically weekly and can lag a few days.
            status = _check_staleness(last_ts, threshold_hours=14 * 24)
        return {
            "score": m1_score,
            "ratio": round(ratio, 3) if ratio else 0.0,
            "velocity": round(ratio, 3) if ratio else 0.0,
            "slope_dir": slope_dir,
            "weekly_values": series.tolist(),
            "status": status,
            "weeks_of_data": weeks,
            "current_index": round(current_index, 2),
            "baseline_avg": round(baseline_avg, 2),
            "last_data_ts": iso_utc(last_ts) if last_ts else None,
        }
    except Exception:
        # Paid fallback (SerpAPI) or its own internal pytrends fallback.
        try:
            from google_trends_only import fetch_keyword_trends

            r = fetch_keyword_trends(keyword)
            ratio = float(r.get("m1_search_ratio", 0.0) or 0.0)
            score = float(r.get("m1_score", 0.0) or 0.0)
            trend_dir = (r.get("trend_direction") or "unknown").lower()
            slope_dir = "up" if trend_dir == "rising" else ("down" if trend_dir == "falling" else "flat")
            status = "ok" if score > 0 else "fail"
            return {
                "score": round(score, 2),
                "ratio": round(ratio, 3) if ratio else 0.0,
                "velocity": round(ratio, 3) if ratio else 0.0,
                "slope_dir": slope_dir,
                "weekly_values": [],
                "status": status,
                "weeks_of_data": 0,
                "current_index": float(r.get("current_week_index", 0.0) or 0.0),
                "baseline_avg": float(r.get("twelve_month_avg_index", 0.0) or 0.0),
                "last_data_ts": r.get("scan_timestamp"),
                "provider_used": r.get("provider_used"),
                "query_used": r.get("query_used"),
            }
        except Exception:
            return {
                "score": 0.0,
                "ratio": 0.0,
                "velocity": 0.0,
                "slope_dir": "flat",
                "weekly_values": [],
                "status": "fail",
                "weeks_of_data": 0,
                "current_index": 0.0,
                "baseline_avg": 0.0,
                "last_data_ts": None,
                "provider_used": "fail",
            }


def compute_m2(topic: str) -> dict[str, Any]:
    try:
        social = scan_social_topic(topic)
    except Exception as exc:
        print(f"[warn] social scan failed ({exc}); falling back to empty signal")
        social = {}
    posts_48h = int(social.get("mentions_48h", 0) or 0)
    daily_avg_30d = float(social.get("reddit_month_total_results", 0) or 0) / 30.0
    if daily_avg_30d <= 0:
        daily_avg_30d = (float(social.get("mentions_7d", 0) or 0) / 7.0) if social.get("mentions_7d") else 0.0
    post_ratio = ((posts_48h / 2.0) / daily_avg_30d) if daily_avg_30d > 0 else (posts_48h / 2.0)
    upvote_ratio = float(((social.get("m2_formula") or {}).get("reddit") or {}).get("avg_upvote_ratio", 0.0) or 0.0)
    comments_avg = float(((social.get("m2_formula") or {}).get("reddit") or {}).get("avg_comments", 0.0) or 0.0)
    eng_mult = 0.7 + (0.3 * upvote_ratio) + (0.2 * min(comments_avg / 200.0, 1.5))
    raw_ratio = post_ratio * eng_mult
    m2_score = round(normalize_ratio_score(raw_ratio, 8.0), 2)
    m2_diversity = float(social.get("m2_diversity", 0.0) or 0.0)
    platform_diversity_count = int(social.get("platform_diversity_count", 0) or 0)
    latest_post_ts = parse_api_datetime(social.get("latest_post_ts"))
    if latest_post_ts is None:
        for item in social.get("sample_posts", []) or []:
            candidate = parse_api_datetime(item.get("date"))
            if candidate and (latest_post_ts is None or candidate > latest_post_ts):
                latest_post_ts = candidate
    status = "ok" if posts_48h > 0 else "fail"
    if status == "ok":
        status = _check_staleness(latest_post_ts, threshold_hours=24)
    return {
        "score": m2_score,
        "raw_ratio": round(raw_ratio, 3),
        "diversity": round(m2_diversity, 3),
        "platform_count": platform_diversity_count,
        "status": status,
        "posts_48h": posts_48h,
        "daily_avg_30d": round(daily_avg_30d, 2),
        "eng_mult": round(eng_mult, 3),
        "last_data_ts": iso_utc(latest_post_ts) if latest_post_ts else None,
        "source_payload": social,
    }


def fetch_youtube_top10(keyword: str, api_key: str) -> list[dict]:
    import httpx

    params = {
        "key": api_key,
        "q": keyword,
        "part": "snippet",
        "type": "video",
        "order": "viewCount",
        "maxResults": 5,
    }
    with httpx.Client(timeout=20.0) as client:
        search_resp = client.get(YOUTUBE_SEARCH_URL, params=params)
        search_resp.raise_for_status()
        items = search_resp.json().get("items", []) or []
        video_ids = [item.get("id", {}).get("videoId") for item in items if item.get("id", {}).get("videoId")]
        if not video_ids:
            return []
        stats_resp = client.get(
            YOUTUBE_VIDEOS_URL,
            params={"key": api_key, "part": "statistics,snippet", "id": ",".join(video_ids)},
        )
        stats_resp.raise_for_status()
        videos = []
        for item in stats_resp.json().get("items", []) or []:
            stats = item.get("statistics") or {}
            snippet = item.get("snippet") or {}
            videos.append(
                {
                    "video_id": item.get("id"),
                    "title": snippet.get("title"),
                    "channel_id": snippet.get("channelId"),
                    "channel_title": snippet.get("channelTitle"),
                    "published_at": snippet.get("publishedAt"),
                    "view_count": int(stats.get("viewCount", 0) or 0),
                    "like_count": int(stats.get("likeCount", 0) or 0),
                    "comment_count": int(stats.get("commentCount", 0) or 0),
                    "likes_disabled": "likeCount" not in stats,
                    "comments_disabled": "commentCount" not in stats,
                }
            )
        return videos


def _safe_fetch_youtube_market_signals(keyword: str) -> dict[str, Any] | None:
    # run_tss() is async, so this function may execute inside an active event loop.
    # asyncio.run() is illegal in that context and can leave an un-awaited coroutine warning.
    try:
        asyncio.get_running_loop()
        return None
    except RuntimeError:
        pass
    try:
        return asyncio.run(scan_youtube_topic(keyword))
    except Exception:
        return None


def _build_m3_from_market_signals(payload: dict[str, Any]) -> dict[str, Any]:
    exact = payload.get("m3_exact_experimental") or {}
    views_last = float(exact.get("views_last_7d") or 0.0)
    views_prior = float(exact.get("views_prior_7d") or 0.0)
    if views_prior > 0 and exact.get("m3_exact_ratio"):
        ratio_value = float(exact["m3_exact_ratio"])
    elif views_last > 0:
        ratio_value = float(views_last)
    else:
        ratio_value = 0.0
    ratio_value = min(ratio_value, 4.0)
    # If exact windows aren't available but score exists, infer a consistent ratio from score.
    if ratio_value <= 0.0:
        inferred = float(payload.get("m3_score", 0.0) or 0.0)
        if inferred > 0.0:
            ratio_value = min((inferred / 100.0) * 4.0, 4.0)
    videos = []
    for video in payload.get("videos", []) or []:
        video_copy = dict(video)
        published_at = video_copy.get("published_at")
        published_dt = parse_api_datetime(published_at)
        video_copy["published_ts"] = published_dt.timestamp() if published_dt else 0.0
        videos.append(video_copy)
    fetched_at = parse_api_datetime(payload.get("scan_timestamp")) or utc_now()
    latest_published = None
    for video in videos:
        dt_p = parse_api_datetime(video.get("published_at"))
        if dt_p and (latest_published is None or dt_p > latest_published):
            latest_published = dt_p
    status = "ok" if videos else "fail"
    if status == "ok":
        # Staleness should reflect *fetch* age (cached results), not the newest publish time.
        status = _check_staleness(fetched_at, threshold_hours=24)
    engagement_values = []
    for video in videos:
        view_count = int(video.get("view_count", 0) or 0)
        likes = video.get("like_count")
        comments = video.get("comment_count")
        if view_count > 0 and likes is not None and comments is not None:
            engagement_values.append((likes + comments) / view_count)
    engagement_rate = round(sum(engagement_values) / len(engagement_values), 3) if engagement_values else 0.0
    total_results = int(max(payload.get("uploads_last_7d", 0) + payload.get("uploads_prior_30d", 0), len(videos)))
    return {
        "score": payload.get("m3_score", 0.0),
        "ratio": round(ratio_value, 3) if ratio_value else 0.0,
        "status": status,
        "views_L7d": round(views_last, 2),
        "views_P7d": round(views_prior, 2),
        "video_count": len(videos),
        "engagement_rate": float(engagement_rate),
        "last_data_ts": payload.get("scan_timestamp"),
        "videos": videos,
        "fetched_at": fetched_at,
        "total_results": total_results,
        "upload_surge_ratio": payload.get("upload_surge_ratio", 0.0),
    }


def _legacy_compute_m3(keyword: str) -> dict[str, Any]:
    import httpx

    load_dotenv()
    api_key = (os.getenv("YOUTUBE_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        return {
            "score": 0.0,
            "ratio": 0.0,
            "status": "fail",
            "views_L7d": 0.0,
            "views_P7d": 0.0,
            "video_count": 0,
            "videos": [],
            "fetched_at": utc_now(),
            "total_results": 0,
            "upload_surge_ratio": 0.0,
        }
    videos = fetch_youtube_top10(keyword, api_key)
    now = utc_now()
    views_L7d = 0.0
    views_P7d = 0.0
    latest_published = None
    emb_eng_sum = 0.0
    emb_eng_count = 0
    enriched_videos = []
    for vid in videos:
        published = vid.get("published_at")
        if not published:
            continue
        try:
            published_dt = dt.datetime.fromisoformat(published.replace("Z", "+00:00"))
        except Exception:
            continue
        if latest_published is None or published_dt > latest_published:
            latest_published = published_dt
        age_days = max((now - published_dt).total_seconds() / 86400.0, 0.01)
        total_views = float(vid.get("view_count", 0) or 0)
        likes = vid.get("like_count")
        comments = vid.get("comment_count")
        if likes is not None and comments is not None and total_views > 0:
            emb_eng_sum += (likes + comments) / total_views
            emb_eng_count += 1
        if age_days <= 7.0:
            views_L7d += total_views
        else:
            daily_rate = total_views / age_days
            views_L7d += daily_rate * 7.0
            views_P7d += daily_rate * 7.0
        enriched_videos.append({
            **vid,
            "published_ts": published_dt.timestamp(),
        })
    if views_P7d > 0:
        ratio = views_L7d / views_P7d
    else:
        ratio = 4.0 if views_L7d > 0 else 0.0
    ratio = min(ratio, 4.0)
    m3_score = round(normalize_ratio_score(ratio, 4.0), 2)
    status = "ok"
    # Staleness should reflect fetch age; this path always fetches "now".
    status = _check_staleness(now, threshold_hours=24)
    return {
        "score": m3_score,
        "ratio": round(ratio, 3) if ratio else 0.0,
        "status": status,
        "views_L7d": round(views_L7d, 2),
        "views_P7d": round(views_P7d, 2),
        "video_count": len(enriched_videos),
        "engagement_rate": round(emb_eng_sum / emb_eng_count, 3) if emb_eng_count else 0.0,
        "last_data_ts": iso_utc(latest_published) if latest_published else None,
        "videos": enriched_videos,
        "fetched_at": now,
        "total_results": len(enriched_videos),
        "upload_surge_ratio": 0.0,
    }


def compute_m3(keyword: str) -> dict[str, Any]:
    signals = _safe_fetch_youtube_market_signals(keyword)
    if signals:
        return _build_m3_from_market_signals(signals)
    return _legacy_compute_m3(keyword)


def compute_m4(keyword: str) -> dict[str, Any]:
    load_dotenv()
    now = utc_now()
    # NewsAPI is the primary provider. We keep SerpAPI-shaped fields
    # in output for backwards compatibility with existing consumers.
    serpapi_7d = serpapi_90d = 0
    serpapi_sample: list[dict[str, Any]] = []
    provider_used = "newsapi"
    serpapi_failed = False
    newsapi_7d = newsapi_90d = 0
    newsapi_key = get_newsapi_key() or (os.getenv("NEWS_API_KEY") or "").strip() or None
    if not newsapi_key:
        return {"score": 0.0, "ratio": 0.0, "accel": 0.0, "status": "fail", "provider_used": "missing_news_sources"}
    try:
        newsapi_7d, newsapi_sample = fetch_newsapi_window(
            keyword, now - dt.timedelta(days=7), now, page_size=50
        )
        newsapi_90d, _ = fetch_newsapi_window(
            keyword, now - dt.timedelta(days=90), now, page_size=50
        )
        # Keep legacy key name to avoid downstream breakage.
        serpapi_sample = newsapi_sample
    except Exception:
        newsapi_7d = 0
        newsapi_90d = 0
        serpapi_sample = []
    gdelt_failed = False
    try:
        gdelt_7d_articles = fetch_gdelt_artlist(keyword, now - dt.timedelta(days=7), now, maxrecords=100)
        gdelt_90d_articles = fetch_gdelt_artlist(keyword, now - dt.timedelta(days=90), now, maxrecords=100)
    except Exception:
        # GDELT is "best-effort" and must never crash the pipeline.
        gdelt_failed = True
        gdelt_7d_articles = []
        gdelt_90d_articles = []
    gdelt_7d = len(gdelt_7d_articles)
    gdelt_90d = len(gdelt_90d_articles)
    primary_7d = newsapi_7d
    primary_90d = newsapi_90d
    combined_daily = (primary_7d * 0.4 + gdelt_7d * 0.06) / 7.0
    daily_avg_90d = (primary_90d * 0.4 + gdelt_90d * 0.06) / 90.0 if (primary_90d or gdelt_90d) else 0.0
    ratio = (combined_daily / daily_avg_90d) if daily_avg_90d > 0 else 0.0
    m4_score = round(normalize_ratio_score(ratio, 6.0), 2)
    tone_7d_vals = [float(a.get("tone", 0.0) or 0.0) for a in gdelt_7d_articles if a.get("tone") is not None]
    tone_today_vals = [
        float(a.get("tone", 0.0) or 0.0)
        for a in gdelt_7d_articles
        if (parse_gdelt_seendate(a.get("seendate")) or now - dt.timedelta(days=9999)) >= (now - dt.timedelta(days=1))
        and a.get("tone") is not None
    ]
    tone_7d_avg = (sum(tone_7d_vals) / len(tone_7d_vals)) if tone_7d_vals else 0.0
    tone_today_avg = (sum(tone_today_vals) / len(tone_today_vals)) if tone_today_vals else 0.0
    m4_accel = None if not tone_7d_vals else clamp(abs(tone_today_avg - tone_7d_avg) / 10.0, 0.0, 1.0)
    status = "ok" if (primary_7d > 0 or gdelt_7d > 0) else "fail"
    latest_gdelt_ts = None
    for art in gdelt_7d_articles:
        candidate = parse_gdelt_seendate(art.get("seendate"))
        if candidate and (latest_gdelt_ts is None or candidate > latest_gdelt_ts):
            latest_gdelt_ts = candidate
    if status == "ok" and latest_gdelt_ts is not None:
        status = _check_staleness(latest_gdelt_ts, threshold_hours=24)
    source_total = primary_7d if primary_7d > 0 else primary_90d
    source_count = min(max(int(source_total // 20), 1), 50) if source_total > 0 else 0
    return {
        "score": m4_score,
        "ratio": round(ratio, 3) if ratio else 0.0,
        "accel": round(m4_accel, 3) if m4_accel is not None else None,
        "status": status,
        "provider_used": provider_used,
        "serpapi_7d": serpapi_7d,
        "serpapi_90d": serpapi_90d,
        "newsapi_7d": newsapi_7d,
        "newsapi_90d": newsapi_90d,
        "gdelt_7d": gdelt_7d,
        "gdelt_90d": gdelt_90d,
        "combined_daily": round(combined_daily, 3),
        "daily_avg_90d": round(daily_avg_90d, 3),
        "tone_today_avg": round(tone_today_avg, 3),
        "tone_7d_avg": round(tone_7d_avg, 3),
        "last_data_ts": iso_utc(latest_gdelt_ts) if latest_gdelt_ts else None,
        "serpapi_sample_7d": serpapi_sample,
        "serpapi_failed": serpapi_failed,
        "source_count": source_count,
        "gdelt_tone_avg": None if gdelt_failed else round(tone_7d_avg, 3),
        "gdelt_failed": gdelt_failed,
    }


def compute_relative_signals(m1: dict, m2: dict, m3: dict, m4: dict) -> tuple[dict, float]:
    m1_eff, m1_p = effective_score(m1["score"], m1["status"])
    m2_eff, m2_p = effective_score(m2["score"], m2["status"])
    m3_eff, m3_p = effective_score(m3["score"], m3["status"])
    m4_eff, m4_p = effective_score(m4["score"], m4["status"])
    present = [s for s, p in [(m1_eff, m1_p), (m2_eff, m2_p), (m3_eff, m3_p), (m4_eff, m4_p)] if p]
    scan_mean = sum(present) / len(present) if present else 0.0
    rs = {
        "rs1": (m1_eff / scan_mean) if (m1_p and scan_mean > 0) else 0.0,
        "rs2": (m2_eff / scan_mean) if (m2_p and scan_mean > 0) else 0.0,
        "rs3": (m3_eff / scan_mean) if (m3_p and scan_mean > 0) else 0.0,
        "rs4": (m4_eff / scan_mean) if (m4_p and scan_mean > 0) else 0.0,
        "scan_mean": scan_mean,
    }
    return rs, scan_mean


def detect_regime(rs: dict, m1: dict, m2: dict, m3: dict, m4: dict, category_id: str) -> dict[str, Any]:
    labels = {
        "R1": "Curiosity / Search",
        "R2": "Social Contagion",
        "R3": "Creator Momentum",
        "R4": "Event Urgency",
    }
    rs1, rs2, rs3, rs4 = rs["rs1"], rs["rs2"], rs["rs3"], rs["rs4"]
    flags: list[str] = []

    statuses = {
        "m1": m1["status"],
        "m2": m2["status"],
        "m3": m3["status"],
        "m4": m4["status"],
    }
    if m1["status"] == "stale":
        flags.append("stale_m1")
    if m2["status"] == "stale":
        flags.append("stale_m2")
    if m3["status"] == "stale":
        flags.append("stale_m3")
    if m4["status"] == "stale":
        flags.append("stale_m4")

    present = [k for k, st in statuses.items() if st == "ok"]
    base_confidence = 0.80 if all(st == "ok" for st in statuses.values()) else 0.60

    def soft_conf(rs: float) -> float:
        if rs >= DOMINANCE_SOFT_HIGH:
            return 1.0
        return base_confidence

    m4_accel = m4.get("accel")
    m4_accel_proxy_used = False
    if m4_accel is None and m4.get("score", 0.0) >= 55 and rs4 >= 1.5:
        m4_accel = 0.30
        m4_accel_proxy_used = True
        flags.append("m4_accel_proxy")

    if len(present) == 1:
        regime = "R1"
        method = "single_method"
        if statuses["m4"] == "ok" and m4.get("score", 0.0) >= ABSOLUTE_GUARD and (m4_accel or 0.0) >= R4_ACCEL_THRESHOLD:
            regime = "R4"
        elif statuses["m3"] == "ok" and m3.get("ratio", 0.0) >= R3_RATIO_THRESHOLD and m3.get("score", 0.0) >= ABSOLUTE_GUARD:
            regime = "R3"
        elif statuses["m2"] == "ok" and m2.get("diversity", 0.0) >= R2_DIVERSITY_THRESHOLD and m2.get("score", 0.0) >= ABSOLUTE_GUARD:
            regime = "R2"
        confidence = base_confidence
        if regime == "R3" and statuses["m2"] != "ok":
            confidence -= 0.15
            flags.append("guard_waived_m2")
        if m4_accel_proxy_used:
            confidence -= 0.15
        confidence = clamp(confidence, 0.35, 1.00)
        return {
            "regime": regime,
            "regime_label": labels[regime],
            "confidence": confidence,
            "method": method,
            "second_regime": "",
            "second_weight": 0.0,
            "rs1": rs1,
            "rs2": rs2,
            "rs3": rs3,
            "rs4": rs4,
            "scan_mean": rs.get("scan_mean", 0.0),
            "flags": flags,
        }

    social_guard = True if statuses["m2"] != "ok" else (rs2 > 0.70)
    if statuses["m2"] != "ok":
        flags.append("guard_waived_m2")
    news_guard = True if statuses["m4"] != "ok" else (rs4 < 1.20)
    if statuses["m4"] != "ok":
        flags.append("guard_waived_m4")

    r4_fires = (
        statuses["m4"] == "ok"
        and rs4 > DOMINANCE_THRESHOLD
        and rs4 > rs3
        and (m4_accel or 0.0) >= R4_ACCEL_THRESHOLD
        and m4.get("score", 0.0) >= ABSOLUTE_GUARD
    )
    r3_fires = (
        statuses["m3"] == "ok"
        and rs3 > DOMINANCE_THRESHOLD
        and m3.get("ratio", 0.0) >= R3_RATIO_THRESHOLD
        and social_guard
        and m3.get("score", 0.0) >= ABSOLUTE_GUARD
    )
    r2_fires = (
        statuses["m2"] == "ok"
        and rs2 > DOMINANCE_THRESHOLD
        and m2.get("diversity", 0.0) >= R2_DIVERSITY_THRESHOLD
        and news_guard
        and m2.get("score", 0.0) >= ABSOLUTE_GUARD
    )

    fired = []
    if r4_fires:
        fired.append(("R4", rs4))
    if r3_fires:
        fired.append(("R3", rs3))
    if r2_fires:
        fired.append(("R2", rs2))

    if len(fired) == 0:
        return {
            "regime": "R1",
            "regime_label": labels["R1"],
            "confidence": clamp(base_confidence, 0.35, 1.00),
            "method": "default",
            "second_regime": "",
            "second_weight": 0.0,
            "rs1": rs1,
            "rs2": rs2,
            "rs3": rs3,
            "rs4": rs4,
            "scan_mean": rs.get("scan_mean", 0.0),
            "flags": flags,
        }

    if len(fired) == 3:
        flags.append("clash_collapse_all_three")
        return {
            "regime": "R1",
            "regime_label": labels["R1"],
            "confidence": 0.50,
            "method": "clash_collapse",
            "second_regime": "",
            "second_weight": 0.0,
            "rs1": rs1,
            "rs2": rs2,
            "rs3": rs3,
            "rs4": rs4,
            "scan_mean": rs.get("scan_mean", 0.0),
            "flags": flags,
        }

    if len(fired) == 1:
        regime = fired[0][0]
        confidence = soft_conf(fired[0][1])
        if regime == "R3" and statuses["m2"] != "ok":
            confidence -= 0.15
            flags.append("guard_waived_m2")
        if m4_accel_proxy_used:
            confidence -= 0.15
        confidence = clamp(confidence, 0.35, 1.00)
        method = "full" if base_confidence >= 0.80 else "degraded"
        return {
            "regime": regime,
            "regime_label": labels[regime],
            "confidence": confidence,
            "method": method,
            "second_regime": "",
            "second_weight": 0.0,
            "rs1": rs1,
            "rs2": rs2,
            "rs3": rs3,
            "rs4": rs4,
            "scan_mean": rs.get("scan_mean", 0.0),
            "flags": flags,
        }

    # Two-regime clash
    fired.sort(key=lambda item: item[1], reverse=True)
    (regime_a, rs_a), (regime_b, rs_b) = fired[0], fired[1]
    margin = (rs_a - rs_b) / rs_a if rs_a else 0.0

    category_primary = {
        "CAT-01": "R3",
        "CAT-02": "R2",
        "CAT-03": "R4",
        "CAT-04": "R4",
        "CAT-05": "R2",
        "CAT-06": "R2",
        "CAT-07": "R1",
        "CAT-08": "R1",
    }

    confidence = soft_conf(rs_a)
    if "R3" in (regime_a, regime_b) and statuses["m2"] != "ok":
        confidence -= 0.15
        flags.append("guard_waived_m2")
    if m4_accel_proxy_used:
        confidence -= 0.15
    confidence = clamp(confidence, 0.35, 1.00)

    if margin >= CLASH_CLEAR_MARGIN:
        second_w = round((1.0 - margin) * 0.5, 3)
        return {
            "regime": regime_a,
            "regime_label": labels[regime_a],
            "confidence": confidence,
            "method": "clash_blend",
            "second_regime": regime_b,
            "second_weight": second_w,
            "rs1": rs1,
            "rs2": rs2,
            "rs3": rs3,
            "rs4": rs4,
            "scan_mean": rs.get("scan_mean", 0.0),
            "flags": flags,
        }

    primary = category_primary.get(category_id, "R1")
    if primary not in (regime_a, regime_b):
        primary = regime_a
    secondary = regime_b if primary == regime_a else regime_a
    second_w = round(rs_b / (rs_a + rs_b), 3) if (rs_a + rs_b) else 0.0
    confidence = clamp(confidence - AMBIGUITY_PENALTY, 0.35, 1.00)
    return {
        "regime": primary,
        "regime_label": labels[primary],
        "confidence": confidence,
        "method": "clash_category",
        "second_regime": secondary,
        "second_weight": second_w,
        "rs1": rs1,
        "rs2": rs2,
        "rs3": rs3,
        "rs4": rs4,
        "scan_mean": rs.get("scan_mean", 0.0),
        "flags": flags,
    }


WEIGHT_MATRIX = {
    "R1": {
        "CAT-01": (0.45, 0.20, 0.25, 0.10),
        "CAT-02": (0.30, 0.30, 0.25, 0.15),
        "CAT-03": (0.35, 0.20, 0.05, 0.40),
        "CAT-04": (0.50, 0.15, 0.10, 0.25),
        "CAT-05": (0.25, 0.35, 0.20, 0.20),
        "CAT-06": (0.35, 0.30, 0.25, 0.10),
        "CAT-07": (0.50, 0.15, 0.20, 0.15),
        "CAT-08": (0.35, 0.25, 0.25, 0.15),
    },
    "R2": {
        "CAT-01": (0.20, 0.35, 0.30, 0.15),
        "CAT-02": (0.15, 0.55, 0.20, 0.10),
        "CAT-03": (0.20, 0.30, 0.05, 0.45),
        "CAT-04": (0.25, 0.40, 0.10, 0.25),
        "CAT-05": (0.15, 0.55, 0.15, 0.15),
        "CAT-06": (0.15, 0.55, 0.25, 0.05),
        "CAT-07": (0.30, 0.40, 0.20, 0.10),
        "CAT-08": (0.20, 0.45, 0.25, 0.10),
    },
    "R3": {
        "CAT-01": (0.20, 0.20, 0.45, 0.15),
        "CAT-02": (0.15, 0.30, 0.45, 0.10),
        "CAT-03": (0.15, 0.20, 0.20, 0.45),
        "CAT-04": (0.25, 0.20, 0.30, 0.25),
        "CAT-05": (0.15, 0.35, 0.35, 0.15),
        "CAT-06": (0.15, 0.30, 0.50, 0.05),
        "CAT-07": (0.25, 0.20, 0.45, 0.10),
        "CAT-08": (0.20, 0.25, 0.45, 0.10),
    },
    "R4": {
        "CAT-01": (0.20, 0.15, 0.10, 0.55),
        "CAT-02": (0.15, 0.25, 0.15, 0.45),
        "CAT-03": (0.15, 0.15, 0.05, 0.65),
        "CAT-04": (0.20, 0.10, 0.05, 0.65),
        "CAT-05": (0.15, 0.20, 0.10, 0.55),
        "CAT-06": (0.20, 0.25, 0.20, 0.35),
        "CAT-07": (0.30, 0.15, 0.10, 0.45),
        "CAT-08": (0.20, 0.15, 0.10, 0.55),
    },
}

PSYCH_WEIGHTS = {
    "R1": {"B1": 0.5333, "B2": 0.2000, "B3": 0.1333, "B4": 0.1333},
    "R2": {"B1": 0.2381, "B2": 0.4286, "B3": 0.1429, "B4": 0.1905},
    "R3": {"B1": 0.1739, "B2": 0.2609, "B3": 0.1739, "B4": 0.3913},
    "R4": {"B1": 0.1500, "B2": 0.1500, "B3": 0.5000, "B4": 0.2000},
}


def compute_psych_signals(m1: dict, m2: dict, m3: dict, m4: dict) -> dict[str, Any]:
    b1 = clamp((m1["ratio"] - 1.0) / B1_DENOM, 0.0, 1.0)
    b2 = clamp(m2["diversity"], 0.0, 1.0)
    accel = m4["accel"] if m4["accel"] is not None else 0.0
    b3 = clamp(accel * (1.0 + (m4["score"] / 100.0)), 0.0, 1.0)
    b4 = clamp((m3["ratio"] - 1.0) / B4_DENOM, 0.0, 1.0)
    dampened = False
    if abs(b1 - b4) < B1_B4_CORR_THRESHOLD:
        b1 *= B1_B4_DAMPEN
        b4 *= B1_B4_DAMPEN
        dampened = True
    return {"B1": b1, "B2": b2, "B3": b3, "B4": b4, "correlation_damped": dampened}


def compute_reliability(m1: dict, m2: dict, m3: dict, m4: dict) -> tuple[float, dict]:
    methods_ok = sum(1 for st in [m1["status"], m2["status"], m3["status"], m4["status"]] if st == "ok")
    coverage_factor = methods_ok / 4.0
    source_health = 1.0
    if m1.get("weeks_of_data", 0) < 8:
        source_health *= 0.85
    if m2.get("platform_count", 0) < 3:
        source_health *= 0.90
    if m3.get("video_count", 0) < 5:
        source_health *= 0.90
    if any(st == "stale" for st in [m1["status"], m2["status"], m3["status"], m4["status"]]):
        source_health *= 0.80
    source_health = clamp(source_health, RELIABILITY_FLOOR, 1.00)
    return coverage_factor * source_health, {
        "coverage_factor": round(coverage_factor, 3),
        "source_health": round(source_health, 3),
        "methods_ok": methods_ok,
    }


def normalise_m1(m1: dict[str, Any]) -> dict[str, Any]:
    return {
        "velocity": m1.get("ratio", 0.0),
        "slope_dir": m1.get("slope_dir", "flat"),
        "score": m1.get("score", 0.0),
        "ratio": m1.get("ratio", 0.0),
        "status": m1.get("status", "fail"),
    }


def normalise_m2(m2: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": m2.get("score", 0.0),
        "diversity": m2.get("m2_diversity") or m2.get("diversity", 0.0),
        "status": m2.get("status", "fail"),
    }


def normalise_m3(m3: dict[str, Any]) -> dict[str, Any]:
    eng = m3.get("engagement_rate", 0.0) or 0.0
    if eng > 1.0:
        eng = eng / 100.0
    raw_ratio = float(
        m3.get("m3_ratio") or m3.get("ratio") or m3.get("acceleration") or 0.0
    )
    ratio = min(max(raw_ratio, 0.0), 4.0)
    default_score = round(ratio / 4.0 * 100.0, 2) if ratio else 0.0
    score = float(m3.get("score", default_score))
    return {
        "acceleration": ratio,
        "engagement_rate": float(eng),
        "score": score,
        "ratio": ratio,
        "status": m3.get("status", "fail"),
        "video_count": m3.get("video_count", 0),
    }


def normalise_m4(m4: dict[str, Any]) -> dict[str, Any]:
    return {
        "velocity": m4.get("ratio", 0.0),
        "source_count": int(m4.get("source_count", 0) or 0),
        "gdelt_tone_avg": m4.get("gdelt_tone_avg"),
        "score": m4.get("score", 0.0),
        "accel": m4.get("accel"),
        "status": m4.get("status", "fail"),
    }


def classify_band(score: float) -> str:
    if score < 20:
        return "flat"
    if score < 50:
        return "emerging"
    if score < 75:
        return "rising"
    if score < 90:
        return "peak"
    return "saturating"


async def run_tss(keyword: str) -> dict[str, Any]:
    category_id, category_conf, category_layer = classify_keyword(keyword)
    m1 = compute_m1(keyword)
    m2 = compute_m2(keyword)
    m3 = compute_m3(keyword)
    m4 = compute_m4(keyword)

    rs, scan_mean = compute_relative_signals(m1, m2, m3, m4)
    regime_result = detect_regime(rs, m1, m2, m3, m4, category_id)
    regime = regime_result["regime"]
    regime_label = regime_result["regime_label"]

    primary_weights = WEIGHT_MATRIX[regime][category_id]
    w1, w2, w3, w4 = primary_weights

    # Step 4 weight blending modes
    if regime_result["method"] in ("clash_blend", "clash_category") and regime_result["second_regime"]:
        secondary_weights = WEIGHT_MATRIX[regime_result["second_regime"]][category_id]
        if regime_result["method"] == "clash_blend":
            secondary_alpha = regime_result["second_weight"]
            primary_alpha = 1.0 - secondary_alpha
        else:
            secondary_alpha = min(regime_result["second_weight"], 0.35)
            primary_alpha = max(0.65, 1.0 - secondary_alpha)
        w1 = primary_alpha * w1 + secondary_alpha * secondary_weights[0]
        w2 = primary_alpha * w2 + secondary_alpha * secondary_weights[1]
        w3 = primary_alpha * w3 + secondary_alpha * secondary_weights[2]
        w4 = primary_alpha * w4 + secondary_alpha * secondary_weights[3]
    elif regime != "R1" and regime_result["confidence"] < 1.0:
        # Soft-threshold blend toward R1 when confidence is below 1.0.
        r1_weights = WEIGHT_MATRIX["R1"][category_id]
        alpha = clamp(regime_result["confidence"], 0.35, 1.0)
        w1 = alpha * w1 + (1.0 - alpha) * r1_weights[0]
        w2 = alpha * w2 + (1.0 - alpha) * r1_weights[1]
        w3 = alpha * w3 + (1.0 - alpha) * r1_weights[2]
        w4 = alpha * w4 + (1.0 - alpha) * r1_weights[3]
    # redistribute failed methods
    if m1["status"] != "ok":
        w1 = 0.0
    if m2["status"] != "ok":
        w2 = 0.0
    if m3["status"] != "ok":
        w3 = 0.0
    if m4["status"] != "ok":
        w4 = 0.0
    ssum = w1 + w2 + w3 + w4
    if ssum > 0:
        w1, w2, w3, w4 = (w1 / ssum, w2 / ssum, w3 / ssum, w4 / ssum)

    base_score = (w1 * m1["score"]) + (w2 * m2["score"]) + (w3 * m3["score"]) + (w4 * m4["score"])

    psych_signals = compute_psych_signals(m1, m2, m3, m4)
    w = PSYCH_WEIGHTS[regime]
    psych_raw = (
        w["B1"] * psych_signals["B1"]
        + w["B2"] * psych_signals["B2"]
        + w["B3"] * psych_signals["B3"]
        + w["B4"] * psych_signals["B4"]
    )
    psych_boost = clamp(psych_raw * 20.0, 0.0, 20.0)

    reliability, quality = compute_reliability(m1, m2, m3, m4)
    tss = clamp(base_score * reliability + psych_boost * math.sqrt(reliability), 0.0, 100.0)

    payload = {
        "topic": keyword,
        "timestamp": iso_utc(utc_now()),
        "category": CATEGORY_LABELS[category_id],
        "category_layer": category_layer,
        "tss": round(tss, 2),
        "band": classify_band(tss),
        "regime": regime,
        "regime_label": regime_label,
        "regime_confidence": round(regime_result["confidence"], 3),
        "regime_method": regime_result["method"],
        "second_regime": regime_result.get("second_regime") or "",
        "second_weight": regime_result.get("second_weight", 0.0),
        "regime_flags": regime_result.get("flags", []),
        "scan_mean": round(scan_mean, 3),
        "relative_signals": {k: round(v, 3) for k, v in rs.items() if k in ("rs1", "rs2", "rs3", "rs4")},
        "base_score": round(base_score, 2),
        "psych_boost": round(psych_boost, 2),
        "reliability": round(reliability, 3),
        "weights_used": {"w1": round(w1, 3), "w2": round(w2, 3), "w3": round(w3, 3), "w4": round(w4, 3)},
        "psych_signals": {
            "B1_novelty": round(psych_signals["B1"], 3),
            "B2_social_proof": round(psych_signals["B2"], 3),
            "B3_urgency": round(psych_signals["B3"], 3),
            "B4_creator_fomo": round(psych_signals["B4"], 3),
            "B1_contribution": round(w["B1"] * psych_signals["B1"], 3),
            "B2_contribution": round(w["B2"] * psych_signals["B2"], 3),
            "B3_contribution": round(w["B3"] * psych_signals["B3"], 3),
            "B4_contribution": round(w["B4"] * psych_signals["B4"], 3),
            "correlation_damped": psych_signals["correlation_damped"],
        },
        "quality": quality,
        "methods": {
            "m1": {"score": m1["score"], "ratio": m1["ratio"], "status": m1["status"]},
            "m2": {"score": m2["score"], "diversity": m2["diversity"], "status": m2["status"]},
            "m3": {"score": m3["score"], "ratio": m3["ratio"], "status": m3["status"]},
            "m4": {"score": m4["score"], "accel": m4["accel"], "status": m4["status"]},
        },
        "m1_norm": normalise_m1(m1),
        "m3_norm": normalise_m3(m3),
        "m4_norm": normalise_m4(m4),
    }
    gemini_client = get_gemini_client()
    groq_client = get_groq_client()
    if m3.get("videos"):
        try:
            csi_result = calculate_csi(
                corpus=m3["videos"],
                corpus_fetched_at=_ensure_datetime(m3.get("fetched_at")),
                total_results=int(m3.get("total_results", len(m3.get("videos", [])))),
                tss_search_score=payload["methods"]["m1"]["score"],
                m1_norm=payload["m1_norm"],
                m3_norm=payload["m3_norm"],
                m4_norm=payload["m4_norm"],
                gemini_client=gemini_client,
                region_code=get_region_code(),
                language_code=get_language_code(),
            )
            payload["csi"] = csi_result
        except CorpusStalenessError as exc:
            payload["csi_error"] = str(exc)
        except Exception as exc:  # pragma: no cover - best-effort wrapper
            payload["csi_error"] = str(exc)
    else:
        payload["csi"] = {"error": "insufficient youtube corpus"}
    social_posts = m2.get("source_payload") or []
    if isinstance(social_posts, dict):
        social_posts = social_posts.get("sample_posts") or social_posts.get("posts") or []
    if not isinstance(social_posts, list):
        social_posts = []
    normalized_social_posts: list[dict[str, Any]] = []
    for post in social_posts:
        if isinstance(post, dict):
            normalized_social_posts.append(
                {
                    "title": str(post.get("title") or post.get("headline") or ""),
                    "body": str(post.get("body") or post.get("snippet") or post.get("text") or ""),
                }
            )
        else:
            normalized_social_posts.append({"title": "", "body": str(post)})
    social_posts = normalized_social_posts
    if not social_posts:
        social_posts = []
    news_articles = m4.get("serpapi_sample_7d") or []
    corpus = m3.get("videos", [])
    corpus_embeddings = payload.get("csi", {}).get("embeddings")
    if len(corpus) >= 3:
        try:
            cags_result = await calculate_cags(
                topic=keyword,
                corpus=corpus,
                corpus_embeddings=corpus_embeddings,
                social_data=social_posts,
                news_data=news_articles,
                tss_score=payload["tss"],
                groq_client=groq_client,
                gemini_client=gemini_client,
            )
            payload["cags"] = cags_result
        except Exception as exc:
            payload["cags"] = {"cags_error": str(exc), "topic": keyword}
    else:
        payload["cags"] = {
            "cags_error": "corpus_too_small",
            "topic": keyword,
        }
    tss_score = float(payload.get("tss", 0) or 0)
    csi_score = float(payload.get("csi", {}).get("csi", 50) or 50)
    gap_angles = payload.get("cags", {}).get("gap_angles", []) or []
    top_cags = max((float(a.get("cags_score", 0) or 0) for a in gap_angles), default=0.0)

    if tss_score > 30 and csi_score < 65 and top_cags > 40:
        verdict = "GO"
        reason = "Strong demand, low saturation, clear angle available"
    elif csi_score > 75 and top_cags < 35:
        verdict = "SKIP"
        reason = "Market is saturated, all angles covered"
    elif tss_score < 20:
        verdict = "MONITOR"
        reason = "Too early — check back in 48 hours"
    else:
        verdict = "CAUTION"
        reason = "Mixed signals — review angles before committing"

    payload["verdict"] = {"verdict": verdict, "reason": reason}
    if category_layer == 2:
        payload["category_confidence"] = category_conf
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", help="Topic to analyze")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument(
        "--out",
        default="",
        help="Write the full JSON payload to a file (useful when terminal truncates large output).",
    )
    args = parser.parse_args()

    payload = asyncio.run(run_tss(args.topic))
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
