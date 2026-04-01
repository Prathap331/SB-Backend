#!/usr/bin/env python3
"""
Standalone social market-signal scanner.

Implements an M2-style architecture around three interval windows:
  - past 24 hours
  - past 48 hours
  - past 7 days

Primary sources:
  - X/Twitter API (counts + recent tweets) when TWITTER_BEARER_TOKEN is present
  - Reddit API (OAuth search) when REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET are present
  - SerpAPI Google Search fallback
  - DuckDuckGo text search fallback when no paid provider is available

Usage:
  python3 social_market_signals.py "AI automation"
  python3 social_market_signals.py "Israel Iran War" --json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
from pathlib import Path
from urllib.parse import urlparse
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None

try:
    from ddgs import DDGS
except Exception:  # pragma: no cover - optional runtime dependency
    DDGS = None


PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
SNAPSHOT_DB = CACHE_DIR / "social_market_snapshots.sqlite3"
SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"
TWITTER_COUNTS_URL = "https://api.twitter.com/2/tweets/counts/recent"
TWITTER_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_SEARCH_URL = "https://oauth.reddit.com/search"
SOCIAL_DOMAINS = (
    "reddit.com",
    "x.com",
    "twitter.com",
    "quora.com",
    "news.ycombinator.com",
    "medium.com",
    "linkedin.com",
)


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

    # One 7d hourly series lets us derive 24h/48h/7d counts consistently.
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

    # 24h often undercounts because of timezone/day-part effects. Keep it diagnostic only.
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

    # M2 diversity: count distinct platforms/subreddits with >=3 posts in 48h.
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

    # Freshness signal for TSS v3 (latest known post timestamp).
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

    persist_scan(topic, payload)
    return payload


def print_human(payload: dict) -> None:
    print("=" * 68)
    print("Social Market Signals")
    print("=" * 68)
    print(f"Topic:                  {payload['topic']}")
    print(f"Provider:               {payload['provider_used']}")
    print(f"Mentions past 24h:      {payload['mentions_24h']}")
    print(f"Mentions past 48h:      {payload['mentions_48h']}")
    print(f"Mentions past 7d:       {payload['mentions_7d']}")
    print(f"24h velocity:           {payload['daily_24h_velocity']}")
    print(f"48h daily velocity:     {payload['daily_48h_velocity']}")
    print(f"7d daily velocity:      {payload['daily_7d_velocity']}")
    print(f"24h vs 48h accel:       {payload['accel_24h_vs_48h']}")
    print(f"48h vs 7d accel:        {payload['accel_48h_vs_7d']}")
    print(f"Source diversity:       {payload['source_diversity']}")
    print(f"Snapshot delta 24h:     {payload['snapshot_delta_24h']}")
    print(f"M2 score:               {payload['m2_score']}")
    print()
    print("Sample posts")
    for item in payload["sample_posts"][:5]:
        print(f"- {item.get('title')} | {item.get('domain')} | {item.get('url')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", help="Topic to scan")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    payload = scan_topic(args.topic)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_human(payload)


if __name__ == "__main__":
    main()
