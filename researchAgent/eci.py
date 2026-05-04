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
        "demand_score": round(demand_score, 2),
        "trend_direction": trend_direction,
        "volatility": round(volatility, 2),
        "seasonality": seasonality,
        "breakout_signal": breakout_signal
    }


print(get_google_trends_serpapi("westbengal elections"))


def get_youtube_data(topic: str, max_results=25):
    # 1. Search videos
    search_url = f"{BASE_URL}/search"
    
    search_params = {
        "part": "snippet",
        "q": topic,
        "maxResults": max_results,
        "type": "video",
        "order": "viewCount",
        "key": YOUTUBE_API_KEY
    }

    search_res = requests.get(search_url, params=search_params).json()

    video_ids = [
        item["id"]["videoId"]
        for item in search_res.get("items", [])
    ]

    if not video_ids:
        return None

    # 2. Get video stats
    stats_url = f"{BASE_URL}/videos"

    stats_params = {
        "part": "statistics,snippet",
        "id": ",".join(video_ids),
        "key": YOUTUBE_API_KEY
    }

    stats_res = requests.get(stats_url, params=stats_params).json()

    views, likes, comments, ages = [], [], [], []

    for item in stats_res.get("items", []):
        stats = item.get("statistics", {})

        view = int(stats.get("viewCount", 0))
        like = int(stats.get("likeCount", 0))
        comment = int(stats.get("commentCount", 0))

        views.append(view)
        likes.append(like)
        comments.append(comment)

    if not views:
        return None

    # 3. Compute metrics
    avg_views = sum(views) / len(views)

    # Engagement rate
    engagement = [
        (likes[i] + comments[i]) / (views[i] + 1)
        for i in range(len(views))
    ]
    engagement_rate = sum(engagement) / len(engagement)

    # Competition (simple proxy)
    competition_score = min(avg_views / 1_000_000, 1.0)  

    # Authority (top-heavy dominance)
    max_views = max(views)
    authority_score = max_views / (avg_views + 1)

    # Upload frequency proxy (based on count)
    upload_frequency = len(views) / max_results

    # 4. Final YouTube Score (0–100)
    youtube_score = (
        0.4 * (avg_views / 1_000_000) +
        0.3 * engagement_rate +
        0.2 * (1 - competition_score) +
        0.1 * upload_frequency
    )

    youtube_score = max(0, min(youtube_score * 100, 100))

    return {
        "avg_views": int(avg_views),
        "engagement_rate": round(engagement_rate, 4),
        "competition_score": round(competition_score, 2),
        "upload_frequency": round(upload_frequency, 2),
        "authority_score": round(authority_score, 2),
        "youtube_score": round(youtube_score, 2)
    }


print(
    get_youtube_data("indian elections")
)