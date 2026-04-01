#!/usr/bin/env python3
"""
Standalone news market-signal scanner.

Implements an M4-style architecture around three interval windows:
  - past 24 hours
  - past 48 hours
  - past 7 days

Primary sources:
  - NewsAPI when NEWSAPI_KEY is present
  - DuckDuckGo news fallback otherwise

Usage:
  python3 news_market_signals.py "AI automation"
  python3 news_market_signals.py "Israel Iran War" --json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None

try:
    from duckduckgo_search import DDGS
except Exception:  # pragma: no cover - optional runtime dependency
    DDGS = None


PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
SNAPSHOT_DB = CACHE_DIR / "news_market_snapshots.sqlite3"
NEWSAPI_URL = "https://newsapi.org/v2/everything"
SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
TIER1_DOMAINS = {
    "reuters.com",
    "bbc.com",
    "apnews.com",
    "nytimes.com",
    "wsj.com",
    "bloomberg.com",
    "theguardian.com",
    "ft.com",
}


def normalize_score(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return max(0.0, min(value / cap, 1.0)) * 100.0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(value: dt.datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_topic_key(topic: str) -> str:
    return " ".join((topic or "").strip().lower().split())


def init_db() -> None:
    conn = sqlite3.connect(SNAPSHOT_DB)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS news_market_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_key TEXT NOT NULL,
                topic TEXT NOT NULL,
                provider_used TEXT NOT NULL,
                scan_timestamp TEXT NOT NULL,
                articles_24h INTEGER NOT NULL,
                articles_48h INTEGER NOT NULL,
                articles_7d INTEGER NOT NULL,
                accel_24h_vs_48h REAL NOT NULL,
                accel_48h_vs_7d REAL NOT NULL,
                weekly_daily_velocity REAL NOT NULL,
                publisher_diversity REAL NOT NULL,
                snapshot_delta_24h REAL,
                m4_score REAL NOT NULL,
                raw_payload TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_news_market_scans_lookup
            ON news_market_scans(topic_key, scan_timestamp DESC)
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
            SELECT scan_timestamp, articles_24h
            FROM news_market_scans
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
            INSERT INTO news_market_scans (
                topic_key, topic, provider_used, scan_timestamp,
                articles_24h, articles_48h, articles_7d,
                accel_24h_vs_48h, accel_48h_vs_7d, weekly_daily_velocity,
                publisher_diversity, snapshot_delta_24h, m4_score, raw_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalize_topic_key(topic),
                topic,
                payload["provider_used"],
                payload["scan_timestamp"],
                payload["articles_24h"],
                payload["articles_48h"],
                payload["articles_7d"],
                payload["accel_24h_vs_48h"],
                payload["accel_48h_vs_7d"],
                payload["weekly_daily_velocity"],
                payload["publisher_diversity"],
                payload.get("snapshot_delta_24h"),
                payload["m4_score"],
                json.dumps(payload, ensure_ascii=True),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def parse_flexible_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        pass
    try:
        return parsedate_to_datetime(value)
    except Exception:
        return None


def domain_from_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return None


def get_newsapi_key() -> str | None:
    load_dotenv()
    return (os.getenv("NEWSAPI_KEY") or "").strip() or None


def get_serpapi_key() -> str | None:
    load_dotenv()
    return (os.getenv("SERPAPI_API_KEY") or "").strip() or None


def format_google_date(value: dt.datetime) -> str:
    return value.strftime("%m/%d/%Y")


def google_date_range_tbs(start: dt.datetime, end: dt.datetime) -> str:
    return f"cdr:1,cd_min:{format_google_date(start)},cd_max:{format_google_date(end)}"


def build_news_queries(topic: str) -> list[str]:
    raw = " ".join((topic or "").split())
    if not raw:
        return []
    fixes = {
        "enginner": "engineer",
        "sofware": "software",
    }
    fixed = raw
    for bad, good in fixes.items():
        fixed = fixed.replace(bad, good).replace(bad.title(), good.title())
    variants = [raw]
    if fixed.lower() != raw.lower():
        variants.append(fixed)
    # Keep one broad intent-preserving backup.
    variants.append(f"{fixed} impact")
    deduped = []
    seen = set()
    for item in variants:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item.strip())
    return deduped[:3]


def fetch_newsapi_window(topic: str, start: dt.datetime, end: dt.datetime, *, page_size: int = 100) -> tuple[int, list[dict]]:
    api_key = get_newsapi_key()
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY not configured.")

    params = {
        "apiKey": api_key,
        "q": topic,
        "language": "en",
        "sortBy": "publishedAt",
        "from": start.isoformat(),
        "to": end.isoformat(),
        "pageSize": page_size,
        "page": 1,
    }
    import httpx
    response = httpx.get(NEWSAPI_URL, params=params, timeout=20.0)
    response.raise_for_status()
    payload = response.json()
    total = int(payload.get("totalResults", 0) or 0)
    articles = []
    for item in payload.get("articles", []) or []:
        articles.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("description") or item.get("content"),
                "source": (item.get("source") or {}).get("name"),
                "domain": domain_from_url(item.get("url")),
                "published_at": item.get("publishedAt"),
            }
        )
    return total, articles


def fetch_serpapi_news_window(topic: str, start: dt.datetime, end: dt.datetime, *, num: int = 50) -> tuple[int, list[dict]]:
    api_key = get_serpapi_key()
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY not configured.")

    params = {
        "engine": "google",
        "tbm": "nws",
        "q": topic,
        "api_key": api_key,
        "num": max(1, min(num, 100)),
        "hl": "en",
        "gl": "us",
        "tbs": google_date_range_tbs(start, end),
    }
    import httpx
    response = httpx.get(SERPAPI_SEARCH_URL, params=params, timeout=20.0)
    response.raise_for_status()
    payload = response.json()
    total = int(payload.get("search_information", {}).get("total_results", 0) or 0)
    articles = []
    for item in payload.get("news_results", []) or []:
        articles.append(
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "source": item.get("source"),
                "domain": domain_from_url(item.get("link")),
                "published_at": item.get("date"),
            }
        )
    return total, articles


def fetch_ddgs_news_window(topic: str, *, timelimit: str, max_results: int = 50) -> list[dict]:
    if DDGS is None:
        raise RuntimeError("duckduckgo_search is not available.")
    with DDGS(timeout=15) as ddgs:
        results = ddgs.news(topic, timelimit=timelimit, max_results=max_results)
    normalized = []
    for item in results or []:
        normalized.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("body"),
                "source": item.get("source"),
                "domain": domain_from_url(item.get("url")),
                "published_at": item.get("date"),
            }
        )
    return normalized


def gdelt_datetime(value: dt.datetime) -> str:
    return value.strftime("%Y%m%d%H%M%S")


def parse_gdelt_seendate(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None


def fetch_gdelt_timeline_daily_avg_90d(topic: str, now: dt.datetime) -> float:
    start_90d = now - dt.timedelta(days=90)
    params = {
        "query": topic,
        "mode": "TimelineVolRaw",
        "format": "json",
        "startdatetime": gdelt_datetime(start_90d),
        "enddatetime": gdelt_datetime(now),
    }
    try:
        import httpx
        response = httpx.get(GDELT_DOC_URL, params=params, timeout=20.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        try:
            total, _ = fetch_serpapi_news_window(topic, start_90d, now, num=20)
            return float(total / 90.0) if total else 0.0
        except Exception:
            return 0.0
    timeline = payload.get("timeline") or []
    values: list[float] = []
    for row in timeline:
        val = row.get("value")
        if val is None:
            continue
        try:
            values.append(float(val))
        except Exception:
            continue
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def fetch_gdelt_artlist(topic: str, start: dt.datetime, end: dt.datetime, *, maxrecords: int = 250) -> list[dict]:
    params = {
        "query": topic,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max(1, min(maxrecords, 250)),
        "startdatetime": gdelt_datetime(start),
        "enddatetime": gdelt_datetime(end),
    }
    try:
        import httpx
        response = httpx.get(GDELT_DOC_URL, params=params, timeout=20.0)
        response.raise_for_status()
        payload = response.json()
        return payload.get("articles", []) or []
    except Exception:
        try:
            total, serp_articles = fetch_serpapi_news_window(topic, start, end, num=maxrecords)
            return serp_articles
        except Exception:
            return []


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def compute_interval_metrics(count_24h: int, count_48h: int, count_7d: int) -> dict:
    daily_24h = float(count_24h)
    daily_48h = float(count_48h) / 2.0 if count_48h else 0.0
    daily_7d = float(count_7d) / 7.0 if count_7d else 0.0

    accel_24h_vs_48h = round(daily_24h / daily_48h, 2) if daily_48h > 0 else (round(daily_24h, 2) if daily_24h > 0 else 0.0)
    accel_48h_vs_7d = round(daily_48h / daily_7d, 2) if daily_7d > 0 else (round(daily_48h, 2) if daily_48h > 0 else 0.0)
    weekly_daily_velocity = round(daily_7d, 2)

    # 24h can be sparse depending on crawl timing; keep as diagnostic only.
    accel_24_component = 0.0
    accel_48_component = round(0.25 * normalize_score(accel_48h_vs_7d, 3.0), 2)
    weekly_scale_component = round(0.60 * normalize_score(weekly_daily_velocity, 500.0), 2)

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


def scan_topic(topic: str) -> dict:
    now = utc_now()
    scan_timestamp = iso_utc(now)
    previous = load_previous_scan(topic)

    start_24h = now - dt.timedelta(days=1)
    start_48h = now - dt.timedelta(days=2)
    start_7d = now - dt.timedelta(days=7)

    provider_used = "ddgs"
    sample_items: list[dict] = []
    queries = build_news_queries(topic)
    provider_errors: list[str] = []
    count_24h = 0
    count_48h = 0
    count_7d = 0
    m4_raw = 0.0
    m4_formula: dict = {}

    for query in queries:
        try:
            q24, _ = fetch_newsapi_window(query, start_24h, now, page_size=20)
            q48, _ = fetch_newsapi_window(query, start_48h, now, page_size=20)
            q7, qsample = fetch_newsapi_window(query, start_7d, now, page_size=100)
            if q7 > count_7d:
                count_24h, count_48h, count_7d = q24, q48, q7
                sample_items = qsample
                provider_used = "newsapi"
            if count_7d > 0:
                break
        except Exception as exc:
            provider_errors.append(f"newsapi_failed[{query}]: {exc}")

    if count_7d == 0:
        for query in queries:
            try:
                q24, _ = fetch_serpapi_news_window(query, start_24h, now, num=20)
                q48, _ = fetch_serpapi_news_window(query, start_48h, now, num=20)
                q7, qsample = fetch_serpapi_news_window(query, start_7d, now, num=50)
                if q7 > count_7d:
                    count_24h, count_48h, count_7d = q24, q48, q7
                    sample_items = qsample
                    provider_used = "serpapi_news"
                if count_7d > 0:
                    break
            except Exception as exc:
                provider_errors.append(f"serpapi_failed[{query}]: {exc}")

    if count_7d == 0:
        fallback_query = queries[0] if queries else topic
        week_items = fetch_ddgs_news_window(fallback_query, timelimit="w", max_results=50)
        day_items = fetch_ddgs_news_window(fallback_query, timelimit="d", max_results=50)
        sample_items = week_items[:20]
        count_24h = len(day_items)
        count_7d = len(week_items)
        count_48h = max(count_24h, min(count_24h * 2, count_7d))
        provider_used = "ddgs_approx"

    metrics = compute_interval_metrics(count_24h, count_48h, count_7d)

    publisher_diversity = round(
        len({(item.get("source") or item.get("domain")) for item in sample_items if item.get("source") or item.get("domain")}) / len(sample_items),
        2,
    ) if sample_items else 0.0
    diversity_component = round(0.15 * (publisher_diversity * 100.0), 2)

    # Preferred M4: NewsAPI + GDELT multipliers
    # m4_raw = news_ratio * authority_mult * geo_mult * sentiment_mult
    try:
        articles_7d = count_7d
        gdelt_daily_avg_90d = fetch_gdelt_timeline_daily_avg_90d(topic, now)
        news_ratio = ((articles_7d / 7.0) / gdelt_daily_avg_90d) if gdelt_daily_avg_90d > 0 else ((articles_7d / 7.0) if articles_7d > 0 else 0.0)

        tier1_count = sum(
            1 for a in sample_items
            if (a.get("domain") or "").lower() in TIER1_DOMAINS
        )
        authority_mult = 1.0 + (((tier1_count / len(sample_items)) * 0.5) if sample_items else 0.0)

        gdelt_7d_articles = fetch_gdelt_artlist(topic, start_7d, now, maxrecords=250)
        gdelt_90d_start = now - dt.timedelta(days=90)
        gdelt_90d_articles = fetch_gdelt_artlist(topic, gdelt_90d_start, now, maxrecords=250)

        countries = [
            (a.get("sourcecountry") or "").strip()
            for a in gdelt_7d_articles
            if (a.get("sourcecountry") or "").strip()
        ]
        unique_countries = len(set(countries))
        geo_mult = min(1.0 + (unique_countries / 20.0), 1.4)

        tone_7d_vals = [safe_float(a.get("tone"), 0.0) for a in gdelt_7d_articles if a.get("tone") is not None]
        tone_90d_vals = [safe_float(a.get("tone"), 0.0) for a in gdelt_90d_articles if a.get("tone") is not None]
        tone_7d_avg = (sum(tone_7d_vals) / len(tone_7d_vals)) if tone_7d_vals else 0.0
        tone_90d_avg = (sum(tone_90d_vals) / len(tone_90d_vals)) if tone_90d_vals else tone_7d_avg
        tone_shift = abs(tone_7d_avg - tone_90d_avg)
        sentiment_mult = 1.0 + min(tone_shift / 10.0, 0.3)

        tone_today_vals = [
            safe_float(a.get("tone"), 0.0)
            for a in gdelt_7d_articles
            if (parse_gdelt_seendate(a.get("seendate")) or now - dt.timedelta(days=9999)) >= (now - dt.timedelta(days=1))
            and a.get("tone") is not None
        ]
        tone_today_avg = (sum(tone_today_vals) / len(tone_today_vals)) if tone_today_vals else 0.0
        m4_accel = round(clamp(abs(tone_today_avg - tone_7d_avg) / 10.0, 0.0, 1.0), 3)

        m4_raw = news_ratio * authority_mult * geo_mult * sentiment_mult
        m4_score = round(normalize_score(m4_raw, 6.0), 2)
        m4_formula = {
            "articles_7d": articles_7d,
            "gdelt_daily_avg_90d": round(gdelt_daily_avg_90d, 3),
            "news_ratio": round(news_ratio, 3),
            "tier1_count": tier1_count,
            "authority_mult": round(authority_mult, 3),
            "unique_countries": unique_countries,
            "geo_mult": round(geo_mult, 3),
            "tone_7d_avg": round(tone_7d_avg, 3),
            "tone_90d_avg": round(tone_90d_avg, 3),
            "tone_today_avg": round(tone_today_avg, 3),
            "tone_shift": round(tone_shift, 3),
            "sentiment_mult": round(sentiment_mult, 3),
            "m4_raw": round(m4_raw, 3),
        }
    except Exception as exc:
        provider_errors.append(f"gdelt_formula_failed: {exc}")
        # Fallback to interval model
        m4_score = round(
            metrics["accel_24_component"]
            + metrics["accel_48_component"]
            + metrics["weekly_scale_component"]
            + diversity_component,
            2,
        )

    snapshot_delta_24h = None
    if previous:
        snapshot_delta_24h = round(float(count_24h) - float(previous.get("articles_24h", 0)), 2)

    payload = {
        "topic": topic,
        "provider_used": provider_used,
        "scan_timestamp": scan_timestamp,
        "articles_24h": count_24h,
        "articles_48h": count_48h,
        "articles_7d": count_7d,
        "daily_24h_velocity": metrics["daily_24h"],
        "daily_48h_velocity": metrics["daily_48h"],
        "daily_7d_velocity": metrics["daily_7d"],
        "accel_24h_vs_48h": metrics["accel_24h_vs_48h"],
        "accel_48h_vs_7d": metrics["accel_48h_vs_7d"],
        "weekly_daily_velocity": metrics["weekly_daily_velocity"],
        "publisher_diversity": publisher_diversity,
        "snapshot_delta_24h": snapshot_delta_24h,
        "m4_components": {
            "accel_24h_component": metrics["accel_24_component"],
            "accel_48h_component": metrics["accel_48_component"],
            "weekly_scale_component": metrics["weekly_scale_component"],
            "publisher_diversity_component": diversity_component,
        },
        "m4_raw": round(m4_raw, 3) if m4_raw > 0 else 0.0,
        "m4_score": m4_score,
        "m4_accel": m4_accel if "m4_accel" in locals() else None,
        "m4_formula": m4_formula,
        "sample_size": len(sample_items),
        "sample_articles": sample_items[:10],
        "query_candidates": queries,
        "provider_errors": provider_errors,
    }

    persist_scan(topic, payload)
    return payload


def print_human(payload: dict) -> None:
    print("=" * 68)
    print("News Market Signals")
    print("=" * 68)
    print(f"Topic:                  {payload['topic']}")
    print(f"Provider:               {payload['provider_used']}")
    print(f"Articles past 24h:      {payload['articles_24h']}")
    print(f"Articles past 48h:      {payload['articles_48h']}")
    print(f"Articles past 7d:       {payload['articles_7d']}")
    print(f"24h velocity:           {payload['daily_24h_velocity']}")
    print(f"48h daily velocity:     {payload['daily_48h_velocity']}")
    print(f"7d daily velocity:      {payload['daily_7d_velocity']}")
    print(f"24h vs 48h accel:       {payload['accel_24h_vs_48h']}")
    print(f"48h vs 7d accel:        {payload['accel_48h_vs_7d']}")
    print(f"Publisher diversity:    {payload['publisher_diversity']}")
    print(f"Snapshot delta 24h:     {payload['snapshot_delta_24h']}")
    print(f"M4 score:               {payload['m4_score']}")
    print()
    print("Sample articles")
    for item in payload["sample_articles"][:5]:
        print(f"- {item.get('title')} | {item.get('source') or item.get('domain')} | {item.get('url')}")


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
