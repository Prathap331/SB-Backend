from pytrends.request import TrendReq
from datetime import datetime, timedelta, timezone
import requests
import os
from dotenv import load_dotenv 
import datetime as dt
from pathlib import Path
import re
import json
import httpx

pytrends = TrendReq()

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
NEWSAPI_URL        = "https://newsapi.org/v2/everything"
SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

print(YOUTUBE_API_KEY)


SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")


def fetch_serpapi_counts(query, start, end):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "tbs": f"cdr:1,cd_min:{start},cd_max:{end}",
    }

    resp = httpx.get(SERPAPI_SEARCH_URL, params=params, timeout=20)
    resp.raise_for_status()

    data = resp.json()
    return int(data.get("search_information", {}).get("total_results", 0))


def get_trends_serpapi(keyword):
    now = dt.datetime.utcnow()

    last_1d = (now - dt.timedelta(days=1)).strftime("%m/%d/%Y")
    last_2d = (now - dt.timedelta(days=2)).strftime("%m/%d/%Y")
    last_7d = (now - dt.timedelta(days=7)).strftime("%m/%d/%Y")
    last_30d = (now - dt.timedelta(days=30)).strftime("%m/%d/%Y")
    today = now.strftime("%m/%d/%Y")

    count_24h = fetch_serpapi_counts(keyword, last_1d, today)
    count_48h = fetch_serpapi_counts(keyword, last_2d, today)
    count_7d = fetch_serpapi_counts(keyword, last_7d, today)
    count_30d = fetch_serpapi_counts(keyword, last_30d, today)

    return {
        "24h": count_24h,
        "48h": count_48h,
        "7d": count_7d,
        "30d": count_30d
    }

def build_trend_dashboard(data):
    weekly_total = data["7d"]
    weekly_avg = data["30d"] / 4 if data["30d"] else weekly_total

    vs_normal = weekly_total / weekly_avg if weekly_avg else 0

    last_week = weekly_avg  
    wow_growth = ((weekly_total - last_week) / last_week * 100) if last_week else 0

    max_val = max(data["30d"], weekly_total, 1)
    index_now = int((weekly_total / max_val) * 100)
    avg_index = (weekly_avg / max_val) * 100
    last_week_index = (last_week / max_val) * 100

    if vs_normal > 1.2:
        trend = "Rising"
    elif vs_normal < 0.8:
        trend = "Falling"
    else:
        trend = "Stable"

    return {
        "Searches this week": f"{round(weekly_total/1_000_000,1)}M",

        "vs avg / wk": f"{round(weekly_avg/1000)}K",

        "vs normal week": f"{round(vs_normal,1)}×",

        "Week-on-week": f"{round(wow_growth)}%",

        "Trend direction": trend,

        "Index now": index_now,

        "52-week avg index": round(avg_index, 1),

        "Last week index": round(last_week_index, 1),
    }


data = get_trends_serpapi("Trump")
summary = build_trend_dashboard(data)
print(summary)

#  yotube 
CATEGORY_MIN_VIEWS = {
    "Technology":    100_000,
    "Entertainment": 200_000,
    "Politics":       50_000,
    "Finance":        30_000,
    "Sports":        150_000,
    "Fashion":        80_000,
    "History":        10_000,
    "General":        50_000,
}


def get_youtube_video_ids(keyword, api_key=YOUTUBE_API_KEY, max_results=10):
    two_weeks_ago = (datetime.now(timezone.utc) - timedelta(weeks=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "key":           api_key,
        "q":             keyword,
        "part":          "id",
        "type":          "video",
        "order":         "viewCount",
        "maxResults":    max_results,
        "publishedAfter": two_weeks_ago,
    }

    resp = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("items", [])

    if not items:
        four_weeks_ago = (datetime.now(timezone.utc) - timedelta(weeks=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params["publishedAfter"] = four_weeks_ago
        resp  = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])

    return [item["id"]["videoId"] for item in items]


def get_youtube_video_stats(video_ids, api_key=YOUTUBE_API_KEY):
    if not video_ids:
        return []

    params = {
        "key":   api_key,
        "id":    ",".join(video_ids),
        "part":  "statistics,snippet",
    }

    resp = requests.get(YOUTUBE_VIDEOS_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("items", [])


def estimate_weekly_views(total_views, age_days):
    """
    Derive last-7-day and prior-7-day view estimates from a cumulative total.
    YouTube doesn't expose per-week view data in the public API.
    """
    age_days   = max(age_days, 1)
    daily_rate = total_views / age_days

    if age_days <= 7:
        return total_views, 0        

    views_l7d = int(daily_rate * 7)
    views_p7d = int(daily_rate * 7)  
    return views_l7d, views_p7d


def creator_competition_label(new_videos_7d):
    if new_videos_7d == 0:   return "None"
    if new_videos_7d <= 3:   return "Low"
    if new_videos_7d <= 15:  return "Moderate"
    if new_videos_7d <= 50:  return "High"
    return "Very high"


def band_for_score(score):
    if score < 20:  return "Flat"
    if score < 50:  return "Emerging"
    if score < 75:  return "Strong"
    if score < 90:  return "Very strong"
    return "Peak"


def build_youtube_summary(keyword, category="General", api_key=YOUTUBE_API_KEY):
    """
    Full YouTube activity scan for a keyword.
    Returns a dict with all engagement metrics + activity score.
    """
    video_ids = get_youtube_video_ids(keyword, api_key)
    if not video_ids:
        return {"status": "no_recent_content", "score": 0, "band": "Flat"}

    videos = get_youtube_video_stats(video_ids, api_key)
    if not videos:
        return {"status": "no_data", "score": 0, "band": "Flat"}

    now = datetime.now(timezone.utc)

    views_l7d_total  = 0
    views_p7d_total  = 0
    likes_total      = 0
    comments_total   = 0
    new_videos_7d    = 0
    channel_ids      = set()
    valid_videos     = 0
    all_new          = True

    for v in videos:
        snippet = v.get("snippet", {})
        stats   = v.get("statistics", {})

        if snippet.get("liveBroadcastContent") == "live":
            continue

        if "viewCount" not in stats:
            continue

        view_count    = int(stats["viewCount"])
        like_count    = int(stats.get("likeCount",    0))
        comment_count = int(stats.get("commentCount", 0))
        channel_id    = snippet.get("channelId", "")

        published_at_str = snippet.get("publishedAt", "")
        try:
            published_at = datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))
            age_days     = max((now - published_at).days, 1)
        except (ValueError, TypeError):
            age_days = 30

        l7d, p7d = estimate_weekly_views(view_count, age_days)
        views_l7d_total += l7d
        views_p7d_total += p7d
        likes_total     += like_count
        comments_total  += comment_count

        if age_days <= 7:
            new_videos_7d += 1
        else:
            all_new = False

        if channel_id:
            channel_ids.add(channel_id)

        valid_videos += 1

    if valid_videos < 3:
        return {"status": "insufficient_data", "score": 0, "band": "Flat"}

    wow_ratio       = min(views_l7d_total / max(views_p7d_total, 1), 4.0)
    engagement_rate = min(
        (likes_total + comments_total) / max(views_l7d_total, 1),
        0.25
    )

    raw_score   = wow_ratio / 4.0 * 100
    min_vol     = CATEGORY_MIN_VIEWS.get(category, CATEGORY_MIN_VIEWS["General"])
    low_volume  = views_l7d_total < min_vol
    if low_volume:
        raw_score = min(raw_score, 35)

    score            = round(raw_score, 1)
    distinct_channels = len(channel_ids)

    return {
        "score":               score,
        "band":                band_for_score(score),
        "low_volume":          low_volume,
        "views_this_week":     views_l7d_total,
        "views_last_week":     views_p7d_total,
        "wow_ratio":           round(wow_ratio, 2),
        "all_new_videos":      all_new,
        "likes_total":         likes_total,
        "comments_total":      comments_total,
        "engagement_rate":     round(engagement_rate, 4),
        "new_videos_7d":       new_videos_7d,
        "distinct_channels":   distinct_channels,
        "creator_competition": creator_competition_label(new_videos_7d),
        "videos_tracked":      valid_videos,
        "status":              "live",
        "updated_at":          datetime.utcnow().isoformat() + "Z",
    }


print(build_youtube_summary("Israel Iran War"))


# reddit/socials
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
TWITTER_COUNTS_URL = "https://api.twitter.com/2/tweets/counts/recent"
TWITTER_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_SEARCH_URL = "https://oauth.reddit.com/search"
SNAPSHOT_DB = CACHE_DIR / "social_market_snapshots.sqlite3"
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
SOCIAL_DOMAINS = (
    "reddit.com",
    "x.com",
    "twitter.com",
    "quora.com",
    "news.ycombinator.com",
    "medium.com",
    "linkedin.com",
)



import sqlite3
from pathlib import Path
from urllib.parse import urlparse
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None

try:
    from ddgs import DDGS
except Exception: 
    DDGS = None

def normalize_score(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return max(0.0, min(value / cap, 1.0)) * 100.0


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(value: dt.datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_topic_key(topic: str) -> str:
    return re.sub(r"\s+", " ", (topic or "").strip().lower())


def init_db() -> None:
    conn = sqlite3.connect(SNAPSHOT_DB)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS social_market_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_key TEXT NOT NULL,
                topic TEXT NOT NULL,
                query_used TEXT NOT NULL,
                provider_used TEXT NOT NULL,
                scan_timestamp TEXT NOT NULL,
                mentions_24h INTEGER NOT NULL,
                mentions_48h INTEGER NOT NULL,
                mentions_7d INTEGER NOT NULL,
                accel_24h_vs_48h REAL NOT NULL,
                accel_48h_vs_7d REAL NOT NULL,
                weekly_daily_velocity REAL NOT NULL,
                source_diversity REAL NOT NULL,
                snapshot_delta_24h REAL,
                m2_score REAL NOT NULL,
                raw_payload TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_social_market_scans_lookup
            ON social_market_scans(topic_key, scan_timestamp DESC)
            """
        )
        conn.commit()
    finally:
        conn.close()


def load_previous_scan(topic: str) -> dict | None:
    init_db()
    conn = sqlite3.connect(SNAPSHOT_DB)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT scan_timestamp, mentions_24h
            FROM social_market_scans
            WHERE topic_key = ?
            ORDER BY scan_timestamp DESC
            LIMIT 1
            """,
            [normalize_topic_key(topic)],
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def persist_scan(topic: str, payload: dict) -> None:
    init_db()
    conn = sqlite3.connect(SNAPSHOT_DB)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO social_market_scans (
                topic_key, topic, query_used, provider_used, scan_timestamp,
                mentions_24h, mentions_48h, mentions_7d,
                accel_24h_vs_48h, accel_48h_vs_7d, weekly_daily_velocity,
                source_diversity, snapshot_delta_24h, m2_score, raw_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalize_topic_key(topic),
                topic,
                payload["query_used"],
                payload["provider_used"],
                payload["scan_timestamp"],
                payload["mentions_24h"],
                payload["mentions_48h"],
                payload["mentions_7d"],
                payload["accel_24h_vs_48h"],
                payload["accel_48h_vs_7d"],
                payload["weekly_daily_velocity"],
                payload["source_diversity"],
                payload.get("snapshot_delta_24h"),
                payload["m2_score"],
                json.dumps(payload, ensure_ascii=True),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def build_social_query(topic: str) -> str:
    site_clause = " OR ".join(f"site:{domain}" for domain in SOCIAL_DOMAINS)
    return f"({topic}) ({site_clause})"


def format_google_date(value: dt.datetime) -> str:
    return value.strftime("%m/%d/%Y")


def google_date_range_tbs(start: dt.datetime, end: dt.datetime) -> str:
    return f"cdr:1,cd_min:{format_google_date(start)},cd_max:{format_google_date(end)}"


def domain_from_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return None


def get_serpapi_key() -> str | None:
    load_dotenv()
    return (os.getenv("SERPAPI_API_KEY") or "").strip() or None


def get_twitter_bearer_token() -> str | None:
    load_dotenv()
    return (
        (os.getenv("TWITTER_BEARER_TOKEN") or "").strip()
        or (os.getenv("X_BEARER_TOKEN") or "").strip()
        or None
    )


def get_reddit_credentials() -> tuple[str | None, str | None, str]:
    load_dotenv()
    client_id = (os.getenv("REDDIT_CLIENT_ID") or "").strip() or None
    client_secret = (os.getenv("REDDIT_CLIENT_SECRET") or "").strip() or None
    user_agent = (os.getenv("REDDIT_USER_AGENT") or "").strip() or "storybit-trend-scanner/1.0"
    return client_id, client_secret, user_agent


def build_twitter_query(topic: str) -> str:
    # Keep query broad but filter obvious noise/duplication.
    return f"({topic}) lang:en -is:retweet"


def build_reddit_query(topic: str) -> str:
    return topic.strip()


def parse_api_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def fetch_twitter_counts_and_posts(
    topic: str,
    start_24h: dt.datetime,
    start_48h: dt.datetime,
    start_7d: dt.datetime,
    now: dt.datetime,
    *,
    sample_size: int = 20,
) -> tuple[int, int, int, list[dict]]:
    token = get_twitter_bearer_token()
    if not token:
        raise RuntimeError("TWITTER_BEARER_TOKEN not configured.")

    query = build_twitter_query(topic)
    headers = {"Authorization": f"Bearer {token}"}
    import httpx
    timeout = httpx.Timeout(20.0)

    count_params = {
        "query": query,
        "start_time": iso_utc(start_7d),
        "end_time": iso_utc(now),
        "granularity": "hour",
    }
    count_resp = httpx.get(TWITTER_COUNTS_URL, headers=headers, params=count_params, timeout=timeout)
    count_resp.raise_for_status()
    count_payload = count_resp.json()
    buckets = count_payload.get("data", []) or []

    mentions_24h = 0
    mentions_48h = 0
    mentions_7d = 0
    for item in buckets:
        bucket_start = parse_api_datetime(item.get("start"))
        bucket_end = parse_api_datetime(item.get("end"))
        bucket_count = int(item.get("tweet_count", 0) or 0)
        if bucket_start is None or bucket_end is None:
            continue
        if bucket_end > start_7d:
            mentions_7d += bucket_count
        if bucket_end > start_48h:
            mentions_48h += bucket_count
        if bucket_end > start_24h:
            mentions_24h += bucket_count

    search_params = {
        "query": query,
        "max_results": max(10, min(sample_size, 100)),
        "tweet.fields": "created_at,public_metrics,lang,author_id,referenced_tweets",
    }
    search_resp = httpx.get(TWITTER_SEARCH_URL, headers=headers, params=search_params, timeout=timeout)
    search_resp.raise_for_status()
    search_payload = search_resp.json()
    posts = []
    for item in search_payload.get("data", []) or []:
        metrics = item.get("public_metrics") or {}
        post_id = item.get("id")
        posts.append(
            {
                "title": (item.get("text") or "")[:120],
                "url": f"https://x.com/i/web/status/{post_id}" if post_id else None,
                "snippet": item.get("text"),
                "date": item.get("created_at"),
                "domain": "x.com",
                "author_id": item.get("author_id"),
                "like_count": int(metrics.get("like_count", 0) or 0),
                "retweet_count": int(metrics.get("retweet_count", 0) or 0),
                "reply_count": int(metrics.get("reply_count", 0) or 0),
                "referenced_tweets": item.get("referenced_tweets") or [],
            }
        )

    return mentions_24h, mentions_48h, mentions_7d, posts


def fetch_reddit_posts(
    topic: str,
    *,
    max_pages: int = 8,
    page_limit: int = 100,
) -> list[dict]:
    client_id, client_secret, user_agent = get_reddit_credentials()
    if not client_id or not client_secret:
        raise RuntimeError("REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET not configured.")

    import httpx
    timeout = httpx.Timeout(20.0)
    token_resp = httpx.post(
        REDDIT_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        headers={"User-Agent": user_agent},
        timeout=timeout,
    )
    token_resp.raise_for_status()
    access_token = token_resp.json().get("access_token")
    if not access_token:
        raise RuntimeError("Failed to fetch reddit access token.")

    headers = {
        "Authorization": f"bearer {access_token}",
        "User-Agent": user_agent,
    }
    query = build_reddit_query(topic)
    after: str | None = None
    posts: list[dict] = []

    for _ in range(max_pages):
        params = {
            "q": query,
            "sort": "new",
            "t": "week",
            "limit": max(1, min(page_limit, 100)),
            "type": "link",
        }
        if after:
            params["after"] = after
        resp = httpx.get(REDDIT_SEARCH_URL, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or {}
        children = data.get("children") or []
        if not children:
            break
        for child in children:
            row = (child or {}).get("data") or {}
            created_utc = row.get("created_utc")
            created_iso = None
            if created_utc is not None:
                try:
                    created_iso = iso_utc(dt.datetime.fromtimestamp(float(created_utc), tz=dt.timezone.utc))
                except Exception:
                    created_iso = None
            permalink = row.get("permalink")
            posts.append(
                {
                    "title": row.get("title"),
                    "url": f"https://www.reddit.com{permalink}" if permalink else row.get("url"),
                    "snippet": row.get("selftext"),
                    "date": created_iso,
                    "domain": "reddit.com",
                    "author_id": row.get("author"),
                    "comment_count": int(row.get("num_comments", 0) or 0),
                    "score": int(row.get("score", 0) or 0),
                    "upvote_ratio": float(row.get("upvote_ratio", 0.0) or 0.0),
                    "subreddit": row.get("subreddit"),
                }
            )
        after = data.get("after")
        if not after:
            break

    return posts


def fetch_reddit_posts_month(
    topic: str,
    *,
    max_pages: int = 10,
    page_limit: int = 100,
) -> list[dict]:
    client_id, client_secret, user_agent = get_reddit_credentials()
    if not client_id or not client_secret:
        raise RuntimeError("REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET not configured.")

    import httpx
    timeout = httpx.Timeout(20.0)
    token_resp = httpx.post(
        REDDIT_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        headers={"User-Agent": user_agent},
        timeout=timeout,
    )
    token_resp.raise_for_status()
    access_token = token_resp.json().get("access_token")
    if not access_token:
        raise RuntimeError("Failed to fetch reddit access token.")

    headers = {"Authorization": f"bearer {access_token}", "User-Agent": user_agent}
    query = build_reddit_query(topic)
    after: str | None = None
    posts: list[dict] = []
    for _ in range(max_pages):
        params = {
            "q": query,
            "sort": "new",
            "t": "month",
            "limit": max(1, min(page_limit, 100)),
            "type": "link",
        }
        if after:
            params["after"] = after
        resp = httpx.get(REDDIT_SEARCH_URL, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or {}
        children = data.get("children") or []
        if not children:
            break
        for child in children:
            row = (child or {}).get("data") or {}
            posts.append({"id": row.get("id")})
        after = data.get("after")
        if not after:
            break
    return posts


def fetch_serpapi_window(query: str, start: dt.datetime, end: dt.datetime, *, num: int = 20) -> tuple[int, list[dict]]:
    api_key = get_serpapi_key()
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY not configured.")

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": num,
        "hl": "en",
        "gl": "us",
        "tbs": google_date_range_tbs(start, end),
    }
    import httpx
    response = httpx.get(SERPAPI_SEARCH_URL, params=params, timeout=20.0)
    response.raise_for_status()
    payload = response.json()
    total = int(payload.get("search_information", {}).get("total_results", 0) or 0)
    items = []
    for item in payload.get("organic_results", []) or []:
        items.append(
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet") or item.get("snippet_highlighted_words"),
                "date": item.get("date"),
                "domain": domain_from_url(item.get("link")),
            }
        )
    return total, items


def fetch_ddgs_window(query: str, *, timelimit: str, max_results: int = 50) -> list[dict]:
    if DDGS is None:
        raise RuntimeError("ddgs is not available.")
    with DDGS(timeout=8) as ddgs:
        results = ddgs.text(query, timelimit=timelimit, max_results=max_results)
    normalized = []
    for item in results or []:
        url = item.get("href")
        normalized.append(
            {
                "title": item.get("title"),
                "url": url,
                "snippet": item.get("body"),
                "date": item.get("date"),
                "domain": domain_from_url(url),
            }
        )
    return normalized


def compute_interval_metrics(count_24h: int, count_48h: int, count_7d: int) -> dict:
    daily_24h = float(count_24h)
    daily_48h = float(count_48h) / 2.0 if count_48h else 0.0
    daily_7d = float(count_7d) / 7.0 if count_7d else 0.0

    accel_24h_vs_48h = round(daily_24h / daily_48h, 2) if daily_48h > 0 else (round(daily_24h, 2) if daily_24h > 0 else 0.0)
    accel_48h_vs_7d = round(daily_48h / daily_7d, 2) if daily_7d > 0 else (round(daily_48h, 2) if daily_48h > 0 else 0.0)
    weekly_daily_velocity = round(daily_7d, 2)

    accel_24_component = 0.0
    accel_48_component = round(0.25 * normalize_score(accel_48h_vs_7d, 3.0), 2)
    weekly_scale_component = round(0.60 * normalize_score(weekly_daily_velocity, 50000.0), 2)

    return {
        "daily_24h": round(daily_24h, 2),
        "daily_48h": round(daily_48h, 2),
        "daily_7d": weekly_daily_velocity,
        "accel_24h_vs_48h": accel_24h_vs_48h,
        "accel_48h_vs_7d": accel_48h_vs_7d,
        "weekly_daily_velocity": weekly_daily_velocity,
        "accel_24_component": accel_24_component,
        "accel_48_component": accel_48_component,
        "weekly_scale_component": weekly_scale_component,
    }


def compute_reddit_raw_score(posts_48h: list[dict], monthly_total_results: int) -> tuple[float, dict]:
    posts_48h_count = len(posts_48h)
    daily_avg_30d = (monthly_total_results / 30.0) if monthly_total_results > 0 else 0.0
    post_ratio = ((posts_48h_count / 2.0) / daily_avg_30d) if daily_avg_30d > 0 else (posts_48h_count / 2.0)

    upvotes = [float(p.get("upvote_ratio", 0.0) or 0.0) for p in posts_48h]
    comments = [int(p.get("comment_count", 0) or 0) for p in posts_48h]
    avg_upvote = sum(upvotes) / len(upvotes) if upvotes else 0.0
    avg_comments = sum(comments) / len(comments) if comments else 0.0
    comment_norm = min(avg_comments / 200.0, 1.5)
    eng_mult = 0.7 + (0.3 * avg_upvote) + (0.2 * comment_norm)

    unique_subs = len({p.get("subreddit") for p in posts_48h if p.get("subreddit")})
    spread_bonus = min(unique_subs / 10.0, 1.3) if unique_subs > 0 else 1.0
    reddit_raw = post_ratio * eng_mult * spread_bonus
    return reddit_raw, {
        "posts_48h": posts_48h_count,
        "daily_avg_30d": round(daily_avg_30d, 2),
        "post_ratio": round(post_ratio, 3),
        "avg_upvote_ratio": round(avg_upvote, 3),
        "avg_comments": round(avg_comments, 2),
        "comment_norm": round(comment_norm, 3),
        "eng_mult": round(eng_mult, 3),
        "unique_subreddits": unique_subs,
        "spread_bonus": round(spread_bonus, 3),
        "reddit_raw": round(reddit_raw, 3),
    }


def compute_x_raw_score(
    tweets_48h: int,
    tweets_7d: int,
    posts_48h: list[dict],
) -> tuple[float, dict]:
    daily_avg_7d = (tweets_7d / 7.0) if tweets_7d > 0 else 0.0
    tweet_ratio = ((tweets_48h / 2.0) / daily_avg_7d) if daily_avg_7d > 0 else (tweets_48h / 2.0)

    total_tweets = len(posts_48h)
    quote_replies = 0
    for post in posts_48h:
        refs = post.get("referenced_tweets") or []
        if any((r.get("type") in ("quoted", "replied_to")) for r in refs if isinstance(r, dict)):
            quote_replies += 1
    reaction_ratio = (quote_replies / total_tweets) if total_tweets > 0 else 0.0
    reaction_mult = 1.0 + (reaction_ratio * 0.5)

    unique_authors = len({p.get("author_id") for p in posts_48h if p.get("author_id")})
    diversity = (unique_authors / total_tweets) if total_tweets > 0 else 0.0
    diversity_mult = 0.8 + (diversity * 0.4)

    x_raw = tweet_ratio * reaction_mult * diversity_mult
    return x_raw, {
        "tweets_48h": tweets_48h,
        "daily_avg_7d_rolling": round(daily_avg_7d, 2),
        "tweet_ratio": round(tweet_ratio, 3),
        "quote_replies": quote_replies,
        "reaction_ratio": round(reaction_ratio, 3),
        "reaction_mult": round(reaction_mult, 3),
        "unique_authors": unique_authors,
        "diversity": round(diversity, 3),
        "diversity_mult": round(diversity_mult, 3),
        "x_raw": round(x_raw, 3),
    }


def scan_topic(topic: str) -> dict:
    query = build_social_query(topic)
    now = utc_now()
    scan_timestamp = iso_utc(now)
    previous = load_previous_scan(topic)

    start_24h = now - dt.timedelta(days=1)
    start_48h = now - dt.timedelta(days=2)
    start_7d = now - dt.timedelta(days=7)

    provider_used = "ddgs"
    sample_items: list[dict] = []

    direct_errors: list[str] = []
    direct_provider_parts: list[str] = []
    count_24h = 0
    count_48h = 0
    count_7d = 0
    direct_items: list[dict] = []
    m2_raw = 0.0
    m2_formula = {}
    twitter_posts_48h: list[dict] = []
    reddit_posts_48h: list[dict] = []
    reddit_month_total_results = 0
    x_counts = {"24h": 0, "48h": 0, "7d": 0}
    m2_diversity = 0.0
    platform_diversity_count = 0
    subreddit_diversity_count = 0
    latest_post_ts: dt.datetime | None = None

    twitter_token = get_twitter_bearer_token()
    if twitter_token:
        try:
            tw24, tw48, tw7, tw_items = fetch_twitter_counts_and_posts(
                topic,
                start_24h,
                start_48h,
                start_7d,
                now,
                sample_size=20,
            )
            count_24h += tw24
            count_48h += tw48
            count_7d += tw7
            direct_items.extend(tw_items)
            direct_provider_parts.append("twitter_api")
            x_counts = {"24h": tw24, "48h": tw48, "7d": tw7}
            twitter_posts_48h = [
                p for p in tw_items
                if (parse_api_datetime(p.get("date")) or dt.datetime.min.replace(tzinfo=dt.timezone.utc)) >= start_48h
            ]
        except Exception as exc:
            direct_errors.append(f"twitter_api_failed: {exc}")

    reddit_client_id, reddit_client_secret, _ = get_reddit_credentials()
    if reddit_client_id and reddit_client_secret:
        try:
            reddit_items = fetch_reddit_posts(topic, max_pages=8, page_limit=100)
            r24 = 0
            r48 = 0
            r7 = 0
            for item in reddit_items:
                published_at = parse_api_datetime(item.get("date"))
                if published_at is None:
                    continue
                if published_at >= start_7d:
                    r7 += 1
                if published_at >= start_48h:
                    r48 += 1
                if published_at >= start_24h:
                    r24 += 1
            count_24h += r24
            count_48h += r48
            count_7d += r7
            direct_items.extend(reddit_items[:20])
            direct_provider_parts.append("reddit_api")
            reddit_posts_48h = [
                p for p in reddit_items
                if (parse_api_datetime(p.get("date")) or dt.datetime.min.replace(tzinfo=dt.timezone.utc)) >= start_48h
            ]
            month_items = fetch_reddit_posts_month(topic, max_pages=10, page_limit=100)
            reddit_month_total_results = len(month_items)
        except Exception as exc:
            direct_errors.append(f"reddit_api_failed: {exc}")

    if direct_provider_parts and count_7d > 0:
        provider_used = "+".join(direct_provider_parts)
        sample_items = direct_items[:20]
    else:
        try:
            count_24h, _ = fetch_serpapi_window(query, start_24h, now, num=10)
            count_48h, _ = fetch_serpapi_window(query, start_48h, now, num=10)
            count_7d, sample_items = fetch_serpapi_window(query, start_7d, now, num=20)
            provider_used = "serpapi"
        except Exception:
            week_items = fetch_ddgs_window(query, timelimit="w", max_results=50)
            day_items = fetch_ddgs_window(query, timelimit="d", max_results=50)
            sample_items = week_items[:20]
            count_24h = len(day_items)
            count_7d = len(week_items)
            count_48h = max(count_24h, min(count_24h * 2, count_7d))
            provider_used = "ddgs_approx"

    metrics = compute_interval_metrics(count_24h, count_48h, count_7d)

    if "twitter_api" in provider_used or "reddit_api" in provider_used:
        source_diversity = round(
            len({item.get("author_id") for item in sample_items if item.get("author_id")}) / len(sample_items),
            2,
        ) if sample_items else 0.0
    else:
        source_diversity = round(
            len({item.get("domain") for item in sample_items if item.get("domain")}) / len(sample_items),
            2,
        ) if sample_items else 0.0
    diversity_component = round(0.15 * (source_diversity * 100.0), 2)

    if "twitter_api" in provider_used or "reddit_api" in provider_used:
        reddit_raw, reddit_meta = compute_reddit_raw_score(reddit_posts_48h, reddit_month_total_results)
        x_raw, x_meta = compute_x_raw_score(x_counts["48h"], x_counts["7d"], twitter_posts_48h)
        m2_raw = (0.55 * reddit_raw) + (0.45 * x_raw)
        m2_formula = {
            "reddit": reddit_meta,
            "x": x_meta,
            "blend": {
                "reddit_weight": 0.55,
                "x_weight": 0.45,
                "m2_raw": round(m2_raw, 3),
            },
        }
        m2_score = round(normalize_score(m2_raw, 8.0), 2)
    else:
        m2_score = round(
            metrics["accel_24_component"]
            + metrics["accel_48_component"]
            + metrics["weekly_scale_component"]
            + diversity_component,
            2,
        )

    subreddit_counts: dict[str, int] = {}
    for post in reddit_posts_48h:
        sub = post.get("subreddit")
        if not sub:
            continue
        subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1
    subreddit_diversity_count = sum(1 for count in subreddit_counts.values() if count >= 3)

    platform_counts: dict[str, int] = {}
    if twitter_posts_48h:
        platform_counts["x"] = len(twitter_posts_48h)
    if reddit_posts_48h:
        platform_counts["reddit"] = len(reddit_posts_48h)
    if not platform_counts:
        for item in sample_items:
            domain = (item.get("domain") or "").lower()
            if not domain:
                continue
            platform_counts[domain] = platform_counts.get(domain, 0) + 1
    platform_diversity_count = sum(1 for count in platform_counts.values() if count >= 3)
    diversity_count = max(platform_diversity_count, subreddit_diversity_count)
    m2_diversity = min(diversity_count / 5.0, 1.0) if diversity_count else 0.0

    for item in (direct_items or sample_items):
        candidate = parse_api_datetime(item.get("date"))
        if candidate and (latest_post_ts is None or candidate > latest_post_ts):
            latest_post_ts = candidate

    snapshot_delta_24h = None
    if previous:
        snapshot_delta_24h = round(float(count_24h) - float(previous.get("mentions_24h", 0)), 2)

    payload = {
        "topic": topic,
        "query_used": (
            f"twitter:{build_twitter_query(topic)} | reddit:{build_reddit_query(topic)}"
            if ("twitter_api" in provider_used or "reddit_api" in provider_used)
            else query
        ),
        "provider_used": provider_used,
        "scan_timestamp": scan_timestamp,
        "latest_post_ts": iso_utc(latest_post_ts) if latest_post_ts else None,
        "mentions_24h": count_24h,
        "mentions_48h": count_48h,
        "mentions_7d": count_7d,
        "daily_24h_velocity": metrics["daily_24h"],
        "daily_48h_velocity": metrics["daily_48h"],
        "daily_7d_velocity": metrics["daily_7d"],
        "accel_24h_vs_48h": metrics["accel_24h_vs_48h"],
        "accel_48h_vs_7d": metrics["accel_48h_vs_7d"],
        "weekly_daily_velocity": metrics["weekly_daily_velocity"],
        "source_diversity": source_diversity,
        "m2_diversity": round(m2_diversity, 3),
        "platform_diversity_count": platform_diversity_count,
        "subreddit_diversity_count": subreddit_diversity_count,
        "snapshot_delta_24h": snapshot_delta_24h,
        "m2_components": {
            "accel_24h_component": metrics["accel_24_component"],
            "accel_48h_component": metrics["accel_48_component"],
            "weekly_scale_component": metrics["weekly_scale_component"],
            "diversity_component": diversity_component,
        },
        "m2_raw": round(m2_raw, 3) if m2_raw > 0 else 0.0,
        "m2_score": m2_score,
        "m2_formula": m2_formula,
        "reddit_month_total_results": reddit_month_total_results,
        "x_counts": x_counts,
        "sample_size": len(sample_items),
        "sample_posts": sample_items[:10],
        "provider_errors": direct_errors,
    }

    payload["reddit_posts_48h_count"] = len(reddit_posts_48h)
    persist_scan(topic, payload)
    payload["dashboard"] = build_serpapi_dashboard(payload)
    return payload


def build_serpapi_dashboard(payload: dict) -> dict:
    mentions_48h = payload.get("mentions_48h", 0)
    mentions_7d = payload.get("mentions_7d", 0)

    daily_avg = mentions_7d / 7 if mentions_7d else max(mentions_48h / 2, 1)
    vs_avg = round((mentions_48h / 2) / daily_avg, 2)

    sample = payload.get("sample_posts", [])
    domains = [p.get("domain") for p in sample if p.get("domain")]
    unique_domains = list(set(domains))[:2]

    spreading = "spreading" if len(unique_domains) > 3 else "not yet spreading"

    engagement_quality = round(
        min((vs_avg * 0.6) + 0.3, 1.2),
        2
    )

    return {
        "Posts in last 48h": mentions_48h,
        "vs daily avg": round(daily_avg, 0),
        "Active communities": len(unique_domains),
        "Community status": spreading,
        "Post sentiment": "Neutral",
        "Communities": unique_domains,
        "vs daily average": f"{vs_avg}×",
        "Engagement quality": engagement_quality,
    }


result = scan_topic("Israel Iran war")
print(result["dashboard"])





# news



 
import datetime as dt
import os
from urllib.parse import urlparse
 
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
 
try:
    import httpx
except ImportError:
    raise SystemExit("pip install httpx")
 
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None
 
 
NEWSAPI_URL   = "https://newsapi.org/v2/everything"
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
 
TIER1_DOMAINS = {
    "reuters.com", "bbc.com", "apnews.com", "nytimes.com",
    "wsj.com", "bloomberg.com", "theguardian.com", "ft.com",
}
 
 
# =============================================================================
# Helpers
# =============================================================================
 
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)
 
 
def gdelt_ts(d: dt.datetime) -> str:
    return d.strftime("%Y%m%d%H%M%S")
 
 
def domain_from_url(url):
    if not url:
        return None
    try:
        h = urlparse(url).netloc.lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return None
 
 
def parse_gdelt_seendate(value):
    if not value:
        return None
    try:
        return dt.datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None
 
 
def safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default
 
 
def tone_label(shift: float) -> str:
    if shift < 1.5:  return "Stable"
    if shift < 4.0:  return "Escalating"
    return "Rapidly escalating"
 
 
def band_for_score(score: float) -> str:
    if score < 20: return "Flat"
    if score < 50: return "Emerging"
    if score < 75: return "Strong"
    if score < 90: return "Very strong"
    return "Peak"
 
 
# =============================================================================
# NewsAPI
# =============================================================================
 
def fetch_newsapi(topic: str, start: dt.datetime, end: dt.datetime, page_size: int = 100):
    api_key = (os.getenv("NEWSAPI_KEY") or "").strip()
    if not api_key:
        return 0, []
 
    resp = httpx.get(NEWSAPI_URL, params={
        "apiKey":   api_key,
        "q":        topic,
        "language": "en",
        "sortBy":   "publishedAt",
        "from":     start.isoformat(),
        "to":       end.isoformat(),
        "pageSize": page_size,
        "page":     1,
    }, timeout=20.0)
    resp.raise_for_status()
    data = resp.json()
 
    articles = [
        {
            "title":        a.get("title"),
            "url":          a.get("url"),
            "source":       (a.get("source") or {}).get("name"),
            "domain":       domain_from_url(a.get("url")),
            "published_at": a.get("publishedAt"),
        }
        for a in (data.get("articles") or [])
    ]
    return int(data.get("totalResults") or 0), articles
 
 
# =============================================================================
# GDELT  — errors are now PRINTED, not silently swallowed
# =============================================================================
 
def fetch_gdelt_artlist(topic: str, start: dt.datetime, end: dt.datetime, maxrecords: int = 250) -> list:
    try:
        resp = httpx.get(GDELT_DOC_URL, params={
            "query":         topic,
            "mode":          "ArtList",
            "format":        "json",
            "maxrecords":    maxrecords,
            "startdatetime": gdelt_ts(start),
            "enddatetime":   gdelt_ts(end),
        }, timeout=20.0)
        resp.raise_for_status()
        return resp.json().get("articles") or []
    except Exception as e:
        print(f"  [GDELT ArtList] {e}")   # visible error — not silent
        return []
 
 
def fetch_gdelt_daily_avg_90d(topic: str, now: dt.datetime) -> float:
    try:
        resp = httpx.get(GDELT_DOC_URL, params={
            "query":         topic,
            "mode":          "TimelineVolRaw",
            "format":        "json",
            "startdatetime": gdelt_ts(now - dt.timedelta(days=90)),
            "enddatetime":   gdelt_ts(now),
        }, timeout=20.0)
        resp.raise_for_status()
        timeline = resp.json().get("timeline") or []
        values   = [safe_float(row.get("value")) for row in timeline if row.get("value") is not None]
        return sum(values) / len(values) if values else 0.0
    except Exception as e:
        print(f"  [GDELT Timeline] {e}")  # visible error — not silent
        return 0.0
 
 
# =============================================================================
# NewsAPI 90-day baseline fallback
# Used when GDELT is blocked / unavailable
# =============================================================================
 
def fetch_newsapi_daily_avg_90d(topic: str, now: dt.datetime) -> float:
    """
    Fetch 30-day and 7-day NewsAPI counts and derive a rough daily average.
    NewsAPI free plan only goes back 30 days — we use that as our baseline.
    """
    try:
        start_30d = now - dt.timedelta(days=30)
        total_30d, _ = fetch_newsapi(topic, start_30d, now, page_size=1)
        return float(total_30d) / 30.0 if total_30d else 0.0
    except Exception as e:
        print(f"  [NewsAPI baseline] {e}")
        return 0.0
 
 
# =============================================================================
# DuckDuckGo fallback
# =============================================================================
 
def fetch_ddgs(topic: str, timelimit: str, max_results: int = 50) -> list:
    if DDGS is None:
        return []
    try:
        with DDGS(timeout=15) as ddgs:
            results = ddgs.news(topic, timelimit=timelimit, max_results=max_results)
        return [
            {
                "title":        r.get("title"),
                "url":          r.get("url"),
                "source":       r.get("source"),
                "domain":       domain_from_url(r.get("url")),
                "published_at": r.get("date"),
            }
            for r in (results or [])
        ]
    except Exception as e:
        print(f"  [DDGS] {e}")
        return []
 
 
# =============================================================================
# Main scanner
# =============================================================================
 
def scan(topic: str) -> dict:
    now       = utc_now()
    start_7d  = now - dt.timedelta(days=7)
    start_24h = now - dt.timedelta(days=1)
 
    # ── NewsAPI (primary) ─────────────────────────────────────────────────────
    newsapi_total, newsapi_articles = fetch_newsapi(topic, start_7d, now)
 
    if newsapi_total == 0:
        ddgs_week        = fetch_ddgs(topic, timelimit="w")
        newsapi_articles = ddgs_week
        newsapi_total    = len(ddgs_week)
 
    articles_7d = newsapi_total
    publishers  = len({
        a.get("source") or a.get("domain")
        for a in newsapi_articles
        if a.get("source") or a.get("domain")
    })
 
    # ── GDELT (best source for baseline + tone) ───────────────────────────────
    gdelt_7d      = fetch_gdelt_artlist(topic, start_7d, now, maxrecords=250)
    gdelt_events  = len(gdelt_7d)
    gdelt_avg_90d = fetch_gdelt_daily_avg_90d(topic, now)
 
    # ── Fallback: use NewsAPI 30-day avg when GDELT is blocked ───────────────
    gdelt_available = gdelt_avg_90d > 0 or gdelt_events > 0
    if not gdelt_available:
        print("  [info] GDELT unavailable — using NewsAPI 30d avg as baseline")
        gdelt_avg_90d = fetch_newsapi_daily_avg_90d(topic, now)
 
    avg_weekly_baseline = round(gdelt_avg_90d * 7, 1)
    daily_this_week     = articles_7d / 7.0
    vs_normal_week      = round(daily_this_week / gdelt_avg_90d, 1) if gdelt_avg_90d > 0 else 0.0
 
    # ── Tone (from GDELT if available, else neutral) ──────────────────────────
    gdelt_90d     = fetch_gdelt_artlist(topic, now - dt.timedelta(days=90), now, maxrecords=250) if gdelt_available else []
 
    tone_7d_vals  = [safe_float(a.get("tone")) for a in gdelt_7d  if a.get("tone") is not None]
    tone_90d_vals = [safe_float(a.get("tone")) for a in gdelt_90d if a.get("tone") is not None]
 
    tone_7d_avg  = sum(tone_7d_vals)  / len(tone_7d_vals)  if tone_7d_vals  else 0.0
    tone_90d_avg = sum(tone_90d_vals) / len(tone_90d_vals) if tone_90d_vals else tone_7d_avg
    tone_shift   = round(abs(tone_7d_avg - tone_90d_avg), 2)
 
    return {
        "articles_7d":          articles_7d,
        "avg_weekly_baseline":  avg_weekly_baseline,
        "publishers":           publishers,
        "vs_normal_week":       vs_normal_week,
        "coverage_tone":        tone_label(tone_shift),
        "newsapi_articles":     newsapi_total,
        "gdelt_events":         gdelt_events,
        "gdelt_available":      gdelt_available,
        "tone_shift":           tone_shift,
        "tone_7d_avg":          round(tone_7d_avg,  3),
        "tone_90d_avg":         round(tone_90d_avg, 3),
        "gdelt_daily_avg_90d":  round(gdelt_avg_90d, 3),
        "scan_timestamp":       now.isoformat().replace("+00:00", "Z"),
        "topic":                topic,
        "_gdelt_7d":            gdelt_7d,   # reused in build_news_summary
    }
 
 
# =============================================================================
# CLI
# =============================================================================
 
def print_human(d: dict) -> None:
    print("=" * 50)
    print(f"  {d['topic']}")
    print("=" * 50)
    print(f"  Articles / week       {d['articles_7d']:,}  ↑ vs {d['avg_weekly_baseline']:,.0f} avg / week")
    print(f"  Publishers covering   {d['publishers']}")
    print(f"  vs normal week        {d['vs_normal_week']}×  above 90-day avg")
    print(f"  Coverage tone         {d['coverage_tone']}")
    print(f"  NewsAPI articles      {d['newsapi_articles']:,}  editorial press")
    print(f"  GDELT events          {d['gdelt_events']:,}  global news events")
    print(f"  Tone shift            {d['tone_shift']}  {'low · story not escalating' if d['tone_shift'] < 1.5 else 'elevated · story escalating'}")
    if not d.get("gdelt_available"):
        print(f"  [baseline source]     NewsAPI 30-day avg (GDELT blocked on this host)")
    print("=" * 50)
 
 
# =============================================================================
# build_news_summary
# Single entry point — mirrors build_youtube_summary / build_social_summary
# =============================================================================
 
CATEGORY_MIN_ARTICLES = {
    "Technology":    30,
    "Entertainment": 50,
    "Politics":      20,
    "Finance":       15,
    "Sports":        40,
    "Fashion":       25,
    "History":        5,
    "General":       20,
}
 
 
def build_news_summary(keyword: str, category: str = "General") -> dict:
    """
    Full news activity scan for a keyword.
    Returns a flat dict matching the news sub-object of the platform output schema.
 
    Usage:
        result = build_news_summary("Israel Iran War", category="Politics")
        print(result["score"])           # 72.4
        print(result["articles_7d"])     # 1302
        print(result["vs_normal_week"])  # 6.8
        print(result["coverage_tone"])   # "Stable"
    """
    raw      = scan(keyword)
    gdelt_7d = raw.pop("_gdelt_7d", [])
 
    # ── Tier-1 authority ──────────────────────────────────────────────────────
    tier1_count   = sum(
        1 for a in gdelt_7d
        if (domain_from_url(a.get("url")) or "").lower() in TIER1_DOMAINS
    )
    total_gdelt   = max(len(gdelt_7d), 1)
    authority_pct = round(tier1_count / total_gdelt * 100, 1)
 
    # ── Volume floor ──────────────────────────────────────────────────────────
    min_articles = CATEGORY_MIN_ARTICLES.get(category, CATEGORY_MIN_ARTICLES["General"])
    low_volume   = raw["articles_7d"] < min_articles
 
    # ── Score (0–100) ─────────────────────────────────────────────────────────
    raw_score  = min(raw["vs_normal_week"] / 8.0, 1.0) * 100
    raw_score += authority_pct * 0.1          # up to +10 pts for Tier-1 coverage
    raw_score  = min(raw_score, 100.0)
    if low_volume:
        raw_score = min(raw_score, 35.0)
 
    score = round(raw_score, 1)
 
    return {
        # score
        "score":               score,
        "band":                band_for_score(score),
        "low_volume":          low_volume,
        "articles_7d":         raw["articles_7d"],
        "newsapi_articles":    raw["newsapi_articles"],
        "gdelt_events":        raw["gdelt_events"],
        "publishers":          raw["publishers"],
        "tier1_count":         tier1_count,
        "authority_pct":       authority_pct,
        "coverage_tone":       raw["coverage_tone"],
        # "avg_weekly_baseline": raw["avg_weekly_baseline"],
        # "vs_normal_week":      raw["vs_normal_week"],
        # "tone_shift":          raw["tone_shift"],
        # "tone_7d_avg":         raw["tone_7d_avg"],
        # "tone_90d_avg":        raw["tone_90d_avg"],
        "gdelt_available":     raw["gdelt_available"],
        "status":              "live",
        "updated_at":          raw["scan_timestamp"],
    }


print(build_news_summary("Israel Iran War", category="Politics"))
