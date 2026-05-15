import os
from dotenv import load_dotenv 
import requests
from serpapi import GoogleSearch
import numpy as np


YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"
BASE_URL = "https://www.googleapis.com/youtube/v3"

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

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
            "demand_score":    0,
            "trend_direction": "unknown",
            "volatility":      0,
            "seasonality":     False,
            "breakout_signal": False,
            "index_now":       0,
            "avg_index_24m":   0.0,
            "stability":       0.0,
            "lifecycle":       "Unknown",
            "best_month":      "Unknown",
            "search_intent":   {"learning_pct": 0, "buying_pct": 0, "research_pct": 0},
            "top_geographies": [],
        }

    values = [point["values"][0]["extracted_value"] for point in timeline]
    values = np.array(values)

    # 1. Demand Score
    demand_score = float(np.mean(values))

    # 2. Trend Direction
    x     = np.arange(len(values))
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
    diffs       = np.diff(values)
    spike_ratio = np.sum(diffs > np.std(values)) / len(values)
    seasonality = spike_ratio > 0.15

    # 5. Breakout detection
    recent         = np.mean(values[-10:])
    past           = np.mean(values[:10]) if len(values) > 20 else np.mean(values)
    breakout_signal = recent > past * 1.3

    # 6. Index now + 24m avg (trimmed mean)
    index_now_val  = int(values[-1])
    last_24m       = values[-104:] if len(values) >= 104 else values
    sorted_vals    = sorted(last_24m)
    trim           = max(1, len(sorted_vals) // 10)
    trimmed        = sorted_vals[trim:-trim] if len(sorted_vals) > trim * 2 else sorted_vals
    avg_index_24m  = round(float(np.mean(trimmed)), 1) if len(trimmed) > 0 else 0.0

    # 7. Stability — inverted coefficient of variation
    mean_val  = float(np.mean(values))
    std_val   = float(np.std(values))
    stability = round(max(0.0, min(1.0 - (std_val / (mean_val + 1e-6)), 1.0)), 2)

    # 8. Lifecycle from 5yr shape
    mid              = len(values) // 2
    first_half_avg   = float(np.mean(values[:mid]))
    second_half_avg  = float(np.mean(values[mid:]))
    recent_avg       = float(np.mean(values[-13:])) if len(values) >= 13 else float(np.mean(values))
    peak_val         = float(np.max(values))

    if recent_avg >= peak_val * 0.85:
        lifecycle = "Peak / Breakout"
    elif second_half_avg > first_half_avg * 1.2:
        lifecycle = "Growing"
    elif abs(second_half_avg - first_half_avg) / (first_half_avg + 1e-6) < 0.15:
        lifecycle = "Stable maturity"
    elif recent_avg < first_half_avg * 0.6:
        lifecycle = "Declining"
    else:
        lifecycle = "Plateauing"

    # 9. Best month from timeline timestamps
    import calendar
    from collections import defaultdict
    month_buckets = defaultdict(list)
    for i, point in enumerate(timeline):
        date_str = point.get("date", "")
        # SerpApi date format: "Nov 2022" or "2022-11-01"
        try:
            if len(date_str) <= 8:
                dt_parsed = __import__('datetime').datetime.strptime(date_str, "%b %Y")
            else:
                dt_parsed = __import__('datetime').datetime.strptime(date_str[:10], "%Y-%m-%d")
            month_buckets[dt_parsed.month].append(float(values[i]))
        except Exception:
            pass

    if month_buckets:
        best_month_num = max(month_buckets, key=lambda m: sum(month_buckets[m]) / len(month_buckets[m]))
        best_month     = calendar.month_name[best_month_num]
    else:
        best_month     = "Unknown"

    # 10. Search intent from related queries via SerpApi
    related_params = {
        "engine":    "google_trends",
        "q":         topic,
        "data_type": "RELATED_QUERIES",
        "api_key":   SERPAPI_KEY
    }
    related_result = GoogleSearch(related_params).get_dict()
    related_queries = []
    for block in ["top", "rising"]:
        items = related_result.get("related_queries", {}).get(block, [])
        related_queries += [item.get("query", "") for item in items]

    rq_blob      = " ".join(related_queries).lower()
    learning_kw  = ['how', 'learn', 'tutorial', 'guide', 'what', 'why', 'course', 'beginner']
    buying_kw    = ['buy', 'price', 'best', 'review', 'vs', 'cheap', 'deal', 'worth']
    research_kw  = ['history', 'origin', 'explained', 'meaning', 'definition', 'overview']

    learn_hits    = sum(1 for w in learning_kw if w in rq_blob)
    buy_hits      = sum(1 for w in buying_kw   if w in rq_blob)
    research_hits = sum(1 for w in research_kw if w in rq_blob)
    total_hits    = max(learn_hits + buy_hits + research_hits, 1)
    learning_pct  = round(learn_hits    / total_hits * 100)
    buying_pct    = round(buy_hits      / total_hits * 100)
    research_pct  = round(research_hits / total_hits * 100)

    # 11. Top geographies via SerpApi
    geo_params = {
        "engine":    "google_trends",
        "q":         topic,
        "data_type": "GEO_MAP",
        "api_key":   SERPAPI_KEY
    }
    geo_result   = GoogleSearch(geo_params).get_dict()
    geo_items    = geo_result.get("interest_by_region", [])
    total_geo    = sum(item.get("value", [0])[0] for item in geo_items) or 1
    top_geos     = [
        {
            "country": item.get("location"),
            "pct":     round(item.get("value", [0])[0] / total_geo * 100)
        }
        for item in geo_items[:5]
    ]

    return {
        "demand_score":    round(demand_score, 2),
        "trend_direction": trend_direction,
        "volatility":      round(volatility, 2),
        "seasonality":     bool(seasonality),
        "breakout_signal": bool(breakout_signal),
        "index_now":       index_now_val,
        "avg_index_24m":   avg_index_24m,
        "stability":       stability,
        "lifecycle":       lifecycle,
        "best_month":      best_month,
        "search_intent": {
            "learning_pct":  learning_pct,
            "buying_pct":    buying_pct,
            "research_pct":  research_pct,
        },
        "top_geographies": top_geos,
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

    version_penalty = {"Low": 0, "Medium": 15, "High": 30}[version_sensitivity_label]

    old_new_score   = min(old_to_new_ratio / 2.0, 1.0) * 40

    stability_bonus = 15 if foundational_stability else 0

    decay_penalty   = min(incumbent_decay_pct / 2, 25)

    longevity_score = round(
        max(0, min(20 + old_new_score + stability_bonus - version_penalty - decay_penalty, 100)),
        1
    )

    if longevity_score >= 80:
        shelf_life_label = "Multi-year shelf life"
    elif longevity_score >= 60:
        shelf_life_label = "12–18 month shelf life"
    elif longevity_score >= 40:
        shelf_life_label = "6–12 month shelf life"
    else:
        shelf_life_label = "Short shelf life"

    content_longevity = {
        "longevity_score":           longevity_score,
        "shelf_life_label":          shelf_life_label,
        "version_sensitivity":       version_sensitivity,
        "version_sensitivity_label": version_sensitivity_label,
        "old_to_new_ratio":          old_to_new_ratio,
        "foundational_stability":    foundational_stability,
        "incumbent_decay_pct":       incumbent_decay_pct,
    }


    # ── Audience Depth ────────────────────────────────────────────────────────
    like_rate_pct    = round(sum(likes) / (sum(views) + 1) * 100, 2)
    comment_rate_pct = round(sum(comments) / (sum(views) + 1) * 100, 2)

    # Video duration — fetch contentDetails for duration
    duration_params = {
        "part": "contentDetails",
        "id":   ",".join(video_ids),
        "key":  YOUTUBE_API_KEY
    }
    duration_res = requests.get(f"{BASE_URL}/videos", params=duration_params).json()

    durations_min = []
    for item in duration_res.get("items", []):
        iso_dur = item.get("contentDetails", {}).get("duration", "PT0S")
        # Parse ISO 8601 duration e.g. PT18M43S
        import re as _re
        h = int((_re.search(r'(\d+)H', iso_dur) or type('', (), {'group': lambda s, x: '0'})()).group(1) or 0)
        m = int((_re.search(r'(\d+)M', iso_dur) or type('', (), {'group': lambda s, x: '0'})()).group(1) or 0)
        s = int((_re.search(r'(\d+)S', iso_dur) or type('', (), {'group': lambda s, x: '0'})()).group(1) or 0)
        durations_min.append(round(h * 60 + m + s / 60, 1))

    avg_length_min = round(sum(durations_min) / len(durations_min), 1) if durations_min else 0.0

    # Oldest video still ranking
    oldest_months = round(max(ages_days) / 30) if ages_days else 0

    # Comment sentiment — classify from comment text patterns
    # Simple keyword proxy since we don't fetch individual comments
    question_keywords  = ['how', 'what', 'why', 'when', 'where', 'which', 'does', 'can', 'is', 'are', '?']
    complaint_keywords = ['worst', 'bad', 'terrible', 'hate', 'awful', 'broken', 'useless', 'scam', 'fake', 'trash']

    all_titles_text = " ".join(titles).lower()
    question_hits   = sum(1 for w in question_keywords  if w in all_titles_text)
    complaint_hits  = sum(1 for w in complaint_keywords if w in all_titles_text)
    total_signals   = max(question_hits + complaint_hits, 1)
    question_pct    = round(question_hits  / total_signals * 100)
    complaint_pct   = round(complaint_hits / total_signals * 100)

    # Engagement score weighted
    engagement_score = round(min((like_rate_pct * 10) + (comment_rate_pct * 5), 100), 1)

    audience_depth = {
        "score":             min(round(like_rate_pct * 10 + comment_rate_pct * 5 + min(avg_length_min, 30)), 100),
        "like_rate_pct":     like_rate_pct,
        "comment_rate_pct":  comment_rate_pct,
        "avg_length_min":    avg_length_min,
        "oldest_top_months": oldest_months,
        "question_pct":      question_pct,
        "complaint_pct":     complaint_pct,
        "engagement_score":  engagement_score,
        "videos_analyzed":   n,
    }

    # ── Competition Density ───────────────────────────────────────────────────
    # Fetch channel subscriber counts for top videos
    channel_ids = []
    for item in stats_res.get("items", []):
        cid = item.get("snippet", {}).get("channelId")
        if cid:
            channel_ids.append(cid)

    channel_subs = []
    if channel_ids:
        chan_params = {
            "part": "statistics",
            "id":   ",".join(set(channel_ids)),
            "key":  YOUTUBE_API_KEY
        }
        chan_res = requests.get(f"{BASE_URL}/channels", params=chan_params).json()
        for item in chan_res.get("items", []):
            subs = int(item.get("statistics", {}).get("subscriberCount", 0))
            channel_subs.append(subs)

    avg_channel_subs = int(sum(channel_subs) / len(channel_subs)) if channel_subs else 0

    # View Gini coefficient
    def gini(values):
        if not values or sum(values) == 0:
            return 0.0
        arr = sorted(values)
        n_  = len(arr)
        cum = sum((i + 1) * v for i, v in enumerate(arr))
        return round((2 * cum) / (n_ * sum(arr)) - (n_ + 1) / n_, 3)

    view_gini = gini(views)

    # Small creator share — channels with subs < 100K
    small_creator_views = sum(
        views[i] for i in range(min(n, len(channel_subs)))
        if channel_subs[i] < 100_000
    )
    small_creator_share = round(small_creator_views / (sum(views) + 1) * 100, 1)

    # Total videos estimate from search
    total_videos_est = int(search_res.get("pageInfo", {}).get("totalResults", 0))

    if view_gini > 0.7:
        competition_label = "Very Competitive"
    elif view_gini > 0.5:
        competition_label = "Competitive"
    elif view_gini > 0.3:
        competition_label = "Moderate"
    else:
        competition_label = "Open"

    competition_score_val = round((1 - view_gini) * 50 + min(small_creator_share, 50), 1)

    competition_density = {
        "score":               competition_score_val,
        "label":               competition_label,
        "avg_channel_subs":    avg_channel_subs,
        "view_gini":           view_gini,
        "small_creator_share": small_creator_share,
        "total_videos_est":    total_videos_est,
        "channels_analyzed":   len(channel_subs),
    }

    # ── Audience Profile ──────────────────────────────────────────────────────
    # Derived from title/topic signals — no external APIs

    # Primary audience type from topic keywords
    skill_keywords    = ['learn', 'tutorial', 'how to', 'guide', 'course', 'beginner', 'master', 'tips']
    explorer_keywords = ['best', 'top', 'review', 'vs', 'compare', 'which', 'recommend']
    buyer_keywords    = ['buy', 'price', 'cheap', 'deal', 'worth', 'cost', 'affordable', 'discount']
    anxiety_keywords  = ['avoid', 'mistake', 'warning', 'danger', 'risk', 'fail', 'wrong', 'scam']

    topic_lower = topic.lower()
    title_blob  = all_titles_text

    skill_score    = sum(1 for w in skill_keywords    if w in title_blob or w in topic_lower)
    explorer_score = sum(1 for w in explorer_keywords if w in title_blob or w in topic_lower)
    buyer_score    = sum(1 for w in buyer_keywords    if w in title_blob or w in topic_lower)
    anxiety_score  = sum(1 for w in anxiety_keywords  if w in title_blob or w in topic_lower)

    profile_scores = {
        "Skill Builder": skill_score,
        "Explorer":      explorer_score,
        "Buyer":         buyer_score,
        "Anxiety-driven": anxiety_score,
    }
    primary_audience = max(profile_scores, key=profile_scores.get)

    # Dominant emotion
    if anxiety_score >= max(skill_score, explorer_score, buyer_score):
        dominant_emotion     = "Anxiety 😰"
    elif buyer_score >= max(skill_score, explorer_score):
        dominant_emotion     = "Desire 🤑"
    elif explorer_score >= skill_score:
        dominant_emotion     = "Curiosity 🧐"
    else:
        dominant_emotion     = "Motivation 💪"

    # Experience level from title signals
    beginner_hits  = sum(1 for t in titles if any(w in t.lower() for w in ['beginner', 'basics', 'intro', 'start', '101', 'simple', 'easy']))
    advanced_hits  = sum(1 for t in titles if any(w in t.lower() for w in ['advanced', 'expert', 'deep dive', 'mastery', 'pro', 'professional']))
    if beginner_hits > advanced_hits * 2:
        experience_level = "Beginner-heavy"
    elif advanced_hits > beginner_hits * 2:
        experience_level = "Advanced-heavy"
    else:
        experience_level = "Mixed"

    # Purchase intent from RPM + engagement
    purchase_intent = "High" if like_rate_pct > 3.0 and rpm_bench["base"] >= 6.0 else \
                      "Medium" if like_rate_pct > 1.5 else "Low"

    # Shareability score
    shareability = round(min(
        (anxiety_score * 8) + (explorer_score * 6) + (engagement_rate * 200),
        100
    ), 1)

    audience_profile = {
        "score":            min(round(skill_score * 8 + explorer_score * 6 + engagement_score * 0.5), 100),
        "primary_audience": primary_audience,
        "dominant_emotion": dominant_emotion,
        "experience_level": experience_level,
        "purchase_intent":  purchase_intent,
        "shareability":     shareability,
        "data_sources":     "YouTube only",
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
        "content_longevity":         content_longevity,
        "audience_depth":            audience_depth,       
        "competition_density":       competition_density,  
        "audience_profile":          audience_profile,    

    }

