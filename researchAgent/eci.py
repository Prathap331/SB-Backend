import os
from dotenv import load_dotenv 
import requests


YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"
BASE_URL = "https://www.googleapis.com/youtube/v3"

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

from serpapi import GoogleSearch
import numpy as np


def get_google_trends_serpapi(topic: str):
    params = {
        "engine": "google_trends",
        "q": topic,
        "data_type": "TIMESERIES",
        "api_key": SERPAPI_KEY
    }

    search = GoogleSearch(params)
    result = search.get_dict()

    timeline = result.get("interest_over_time", {}).get("timeline_data", [])

    if not timeline:
        return {
            "demand_score": 0,
            "trend_direction": "unknown",
            "volatility": 0,
            "seasonality": False,
            "breakout_signal": False
        }

    values = [point["values"][0]["extracted_value"] for point in timeline]

    values = np.array(values)

    # 1. Demand Score
    demand_score = float(np.mean(values))

    # 2. Trend Direction
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]

    if slope > 0.2:
        trend_direction = "rising"
    elif slope < -0.2:
        trend_direction = "declining"
    else:
        trend_direction = "stable"

    # 3. Volatility
    volatility = float(np.std(values) / (np.mean(values) + 1e-6))
    volatility = min(volatility, 1.0)

    # 4. Seasonality
    diffs = np.diff(values)
    spike_ratio = np.sum(diffs > np.std(values)) / len(values)
    seasonality = spike_ratio > 0.15

    # 5. Breakout detection
    recent = np.mean(values[-10:])
    past = np.mean(values[:10]) if len(values) > 20 else np.mean(values)

    breakout_signal = recent > past * 1.3

    return {
        "demand_score":    round(demand_score, 2),
        "trend_direction": trend_direction,
        "volatility":      round(volatility, 2),
        "seasonality":     bool(seasonality),        
        "breakout_signal": bool(breakout_signal)    
    }


def get_youtube_data(topic: str, max_results=25,category: str = "General"):
    from datetime import datetime, timezone
    import re

    # 1. Search videos
    search_params = {
        "part": "snippet",
        "q": topic,
        "maxResults": max_results,
        "type": "video",
        "order": "viewCount",
        "key": YOUTUBE_API_KEY
    }
    search_res = requests.get(f"{BASE_URL}/search", params=search_params).json()
    video_ids = [item["id"]["videoId"] for item in search_res.get("items", [])]

    if not video_ids:
        return None

    # 2. Get video stats
    stats_params = {
        "part": "statistics,snippet",
        "id": ",".join(video_ids),
        "key": YOUTUBE_API_KEY
    }
    stats_res = requests.get(f"{BASE_URL}/videos", params=stats_params).json()

    views, likes, comments, ages_days = [], [], [], []
    titles = []
    now = datetime.now(timezone.utc)

    for item in stats_res.get("items", []):
        stats   = item.get("statistics", {})
        snippet = item.get("snippet", {})

        view    = int(stats.get("viewCount",    0))
        like    = int(stats.get("likeCount",    0))
        comment = int(stats.get("commentCount", 0))

        views.append(view)
        likes.append(like)
        comments.append(comment)
        titles.append(snippet.get("title", ""))

        published_at = snippet.get("publishedAt")
        if published_at:
            try:
                pub_dt   = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                age_days = max((now - pub_dt).days, 1)
            except Exception:
                age_days = 180
        else:
            age_days = 180
        ages_days.append(age_days)

    if not views:
        return None

    n         = len(views)
    avg_views = sum(views) / n

    # ── Engagement rate ───────────────────────────────────────────────────────
    engagement = [(likes[i] + comments[i]) / (views[i] + 1) for i in range(n)]
    engagement_rate = sum(engagement) / n

    # ── Competition score ─────────────────────────────────────────────────────
    competition_score = min(avg_views / 1_000_000, 1.0)

    # ── Authority score ───────────────────────────────────────────────────────
    max_views     = max(views)
    authority_score = max_views / (avg_views + 1)

    # ── Upload frequency ──────────────────────────────────────────────────────
    upload_frequency = n / max_results

    # ── Version sensitivity — count version keywords in titles ────────────────
    version_pattern = re.compile(
        r'\b(v\d+|\d{4}|version\s*\d+|part\s*\d+|updated?|new|latest|reboot|remake|season\s*\d+)\b',
        re.IGNORECASE
    )
    version_hits = sum(1 for t in titles if version_pattern.search(t))
    version_sensitivity = version_hits  # raw count; label below

    if version_sensitivity <= 2:
        version_sensitivity_label = "Low"
    elif version_sensitivity <= 6:
        version_sensitivity_label = "Medium"
    else:
        version_sensitivity_label = "High"

    # ── Old-to-new view ratio ─────────────────────────────────────────────────
    old_views = [views[i] for i in range(n) if ages_days[i] > 90]
    new_views = [views[i] for i in range(n) if ages_days[i] <= 90]
    old_total  = sum(old_views) or 0
    new_total  = sum(new_views) or 1
    old_to_new_ratio = round(old_total / new_total, 2)

    # ── Foundational stability ────────────────────────────────────────────────
    # True when majority of top videos are older (evergreen content dominates)
    foundational_stability = old_total > new_total

    # ── Incumbent decay — complaint proxy from comment ratio ──────────────────
    # High comment-to-view ratio on old videos signals dissatisfaction/churn
    old_indices    = [i for i in range(n) if ages_days[i] > 90]
    complaint_rate = 0.0
    if old_indices:
        old_comment_ratios = [comments[i] / (views[i] + 1) for i in old_indices]
        complaint_rate     = sum(old_comment_ratios) / len(old_comment_ratios)
    incumbent_decay_pct = round(min(complaint_rate * 1000, 100), 1)  # scale to 0–100%

    # ── YouTube score (0–100) ─────────────────────────────────────────────────
    youtube_score = (
        0.4 * (avg_views / 1_000_000) +
        0.3 * engagement_rate +
        0.2 * (1 - competition_score) +
        0.1 * upload_frequency
    )
    youtube_score = round(max(0, min(youtube_score * 100, 100)), 2)

    CATEGORY_RPM = {
        "Technology":    {"base": 8.0,  "low": 5.0,  "high": 18.0},
        "Finance":       {"base": 12.0, "low": 8.0,  "high": 25.0},
        "Entertainment": {"base": 4.0,  "low": 2.0,  "high": 8.0},
        "Politics":      {"base": 6.0,  "low": 3.0,  "high": 12.0},
        "Sports":        {"base": 5.0,  "low": 3.0,  "high": 10.0},
        "Fashion":       {"base": 6.0,  "low": 3.5,  "high": 12.0},
        "History":       {"base": 4.0,  "low": 2.0,  "high": 8.0},
        "General":       {"base": 5.0,  "low": 2.5,  "high": 10.0},
    }

    rpm_bench = CATEGORY_RPM.get(category, CATEGORY_RPM["General"])
    base_rpm  = rpm_bench["base"]

    # ── Engagement multiplier ─────────────────────────────────────────────────
    like_rate      = sum(likes) / (sum(views) + 1)          # overall like rate
    eng_multiplier = 1.0 + min(like_rate * 10, 1.0)        # 1.0x – 2.0x

    est_rpm        = round(base_rpm * eng_multiplier, 2)
    rpm_low        = round(rpm_bench["low"]  * eng_multiplier, 2)
    rpm_high       = round(rpm_bench["high"] * eng_multiplier, 2)

    # ── Revenue at 100K views/month ───────────────────────────────────────────
    views_per_month   = 100_000
    ad_revenue_mo     = round((views_per_month / 1000) * est_rpm, 2)

    # Brand deal estimate — category-based flat rate scaled by engagement
    CATEGORY_BRAND = {
        "Technology":    800,  "Finance":       1200,
        "Entertainment": 500,  "Politics":       600,
        "Sports":        700,  "Fashion":        900,
        "History":       300,  "General":        500,
    }
    brand_base     = CATEGORY_BRAND.get(category, 500)
    brand_deal_mo  = round(brand_base * eng_multiplier)
    total_est_mo   = round(ad_revenue_mo + brand_deal_mo)

    # ── Revenue potential score (0–100) ──────────────────────────────────────
    revenue_score  = round(min((est_rpm / 25.0) * 100, 100), 1)

    # ── Revenue payload ───────────────────────────────────────────────────────
    revenue_potential = {
        "revenue_score":     revenue_score,
        "est_rpm":           est_rpm,
        "rpm_low":           rpm_low,
        "rpm_high":          rpm_high,
        "rpm_range":         f"${rpm_low}–${rpm_high}",
        "like_rate_pct":     round(like_rate * 100, 2),
        "engagement_adj":    f"{round(like_rate * 100, 1)}% like rate",
        "eng_multiplier":    round(eng_multiplier, 2),
        "ad_revenue_mo":     ad_revenue_mo,
        "brand_deal_est_mo": brand_deal_mo,
        "total_est_mo":      total_est_mo,
        "views_basis":       "100K views/month",
        "rpm_source":        "YT category benchmark",
    }

    return {
        "avg_views":                 int(avg_views),
        "engagement_rate":           round(engagement_rate, 4),
        "competition_score":         round(competition_score, 2),
        "upload_frequency":          round(upload_frequency, 2),
        "authority_score":           round(authority_score, 2),
        "youtube_score":             youtube_score,
        "version_sensitivity":       version_sensitivity,
        "version_sensitivity_label": version_sensitivity_label,
        "old_to_new_ratio":          old_to_new_ratio,
        "foundational_stability":    foundational_stability,
        "incumbent_decay_pct":       incumbent_decay_pct,
        "revenue_potential":         revenue_potential,   
    }


print(
    get_youtube_data("indian elections")
)