#!/usr/bin/env python3
"""
Standalone YouTube market-signal scanner.

Implements the M3-style architecture:
  1. Upload surge ratio
  2. View velocity on recent videos
  3. Channel diversity
  4. Snapshot delta (persisted locally)
  5. Combined M3 score

Usage:
  python3 youtube_market_signals.py "Israel Iran War"
  python3 youtube_market_signals.py "AI automation" --json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None


YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
SNAPSHOT_DB = CACHE_DIR / "youtube_market_snapshots.sqlite3"


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(value: dt.datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_yt_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def normalize_topic_key(topic: str) -> str:
    return re.sub(r"\s+", " ", (topic or "").strip().lower())


def normalize_score(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return max(0.0, min(value / cap, 1.0)) * 100.0


def init_db() -> None:
    conn = sqlite3.connect(SNAPSHOT_DB)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS youtube_topic_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_key TEXT NOT NULL,
                topic TEXT NOT NULL,
                video_id TEXT NOT NULL,
                channel_id TEXT,
                title TEXT,
                views_at_scan INTEGER NOT NULL,
                published_at TEXT,
                scan_timestamp TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_yt_topic_snapshots_lookup
            ON youtube_topic_snapshots(topic_key, video_id, scan_timestamp DESC)
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS youtube_topic_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_key TEXT NOT NULL,
                topic TEXT NOT NULL,
                scan_timestamp TEXT NOT NULL,
                upload_surge_ratio REAL NOT NULL,
                avg_view_velocity REAL NOT NULL,
                channel_diversity REAL NOT NULL,
                avg_snapshot_delta REAL,
                m3_score REAL NOT NULL,
                raw_payload TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def load_previous_snapshots(topic: str, video_ids: list[str]) -> dict[str, dict]:
    if not video_ids:
        return {}
    init_db()
    conn = sqlite3.connect(SNAPSHOT_DB)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in video_ids)
        rows = conn.execute(
            f"""
            SELECT video_id, views_at_scan, scan_timestamp
            FROM youtube_topic_snapshots
            WHERE topic_key = ?
              AND video_id IN ({placeholders})
            ORDER BY scan_timestamp DESC
            """,
            [normalize_topic_key(topic), *video_ids],
        ).fetchall()
    finally:
        conn.close()

    latest: dict[str, dict] = {}
    for row in rows:
        if row["video_id"] not in latest:
            latest[row["video_id"]] = dict(row)
    return latest


def load_snapshot_history(topic: str, video_ids: list[str]) -> dict[str, list[dict]]:
    if not video_ids:
        return {}
    init_db()
    conn = sqlite3.connect(SNAPSHOT_DB)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in video_ids)
        rows = conn.execute(
            f"""
            SELECT video_id, views_at_scan, scan_timestamp
            FROM youtube_topic_snapshots
            WHERE topic_key = ?
              AND video_id IN ({placeholders})
            ORDER BY scan_timestamp DESC
            """,
            [normalize_topic_key(topic), *video_ids],
        ).fetchall()
    finally:
        conn.close()
    history: dict[str, list[dict]] = {vid: [] for vid in video_ids}
    for row in rows:
        history[row["video_id"]].append(dict(row))
    return history


def persist_scan(topic: str, payload: dict, videos: list[dict]) -> None:
    init_db()
    topic_key = normalize_topic_key(topic)
    conn = sqlite3.connect(SNAPSHOT_DB)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO youtube_topic_scans (
                topic_key, topic, scan_timestamp, upload_surge_ratio,
                avg_view_velocity, channel_diversity, avg_snapshot_delta,
                m3_score, raw_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_key,
                topic,
                payload["scan_timestamp"],
                payload["upload_surge_ratio"],
                payload["avg_view_velocity"],
                payload["channel_diversity"],
                payload.get("avg_snapshot_delta"),
                payload["m3_score"],
                json.dumps(payload, ensure_ascii=True),
            ),
        )
        cur.executemany(
            """
            INSERT INTO youtube_topic_snapshots (
                topic_key, topic, video_id, channel_id, title,
                views_at_scan, published_at, scan_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    topic_key,
                    topic,
                    item["video_id"],
                    item.get("channel_id"),
                    item.get("title"),
                    int(item.get("view_count", 0)),
                    item.get("published_at"),
                    payload["scan_timestamp"],
                )
                for item in videos
            ],
        )
        conn.commit()
    finally:
        conn.close()


async def youtube_search_total_results(
    client: httpx.AsyncClient,
    api_key: str,
    query: str,
    *,
    published_after: dt.datetime,
    published_before: dt.datetime | None = None,
) -> int:
    params = {
        "key": api_key,
        "q": query,
        "part": "snippet",
        "type": "video",
        "order": "date",
        "publishedAfter": iso_utc(published_after),
        "maxResults": 1,
    }
    if published_before is not None:
        params["publishedBefore"] = iso_utc(published_before)
    response = await client.get(YOUTUBE_SEARCH_URL, params=params)
    response.raise_for_status()
    return int(response.json().get("pageInfo", {}).get("totalResults", 0) or 0)


async def youtube_search_window_count(
    client: "httpx.AsyncClient",
    api_key: str,
    query: str,
    *,
    published_after: dt.datetime,
    published_before: dt.datetime | None = None,
    max_pages: int = 3,
    max_results: int = 50,
) -> tuple[int, int]:
    """
    Count actual returned items over bounded pagination.
    pageInfo.totalResults is only an estimate and can be inflated.
    """
    counted = 0
    page_token: str | None = None
    calls_used = 0

    for _ in range(max_pages):
        params = {
            "key": api_key,
            "q": query,
            "part": "snippet",
            "type": "video",
            "order": "date",
            "publishedAfter": iso_utc(published_after),
            "maxResults": max(1, min(max_results, 50)),
        }
        if published_before is not None:
            params["publishedBefore"] = iso_utc(published_before)
        if page_token:
            params["pageToken"] = page_token

        response = await client.get(YOUTUBE_SEARCH_URL, params=params)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("items", []) or []
        counted += len(items)
        calls_used += 1
        page_token = payload.get("nextPageToken")
        if not page_token or not items:
            break

    return counted, calls_used


async def youtube_search_recent_videos(
    client: "httpx.AsyncClient",
    api_key: str,
    query: str,
    *,
    published_after: dt.datetime,
    max_results: int = 20,
) -> list[dict]:
    params = {
        "key": api_key,
        "q": query,
        "part": "snippet",
        "type": "video",
        "order": "viewCount",
        "publishedAfter": iso_utc(published_after),
        "maxResults": max_results,
    }
    response = await client.get(YOUTUBE_SEARCH_URL, params=params)
    response.raise_for_status()
    return response.json().get("items", [])


async def youtube_video_statistics(
    client: "httpx.AsyncClient",
    api_key: str,
    video_ids: list[str],
) -> dict[str, dict]:
    if not video_ids:
        return {}
    response = await client.get(
        YOUTUBE_VIDEOS_URL,
        params={
            "key": api_key,
            "part": "statistics,snippet",
            "id": ",".join(video_ids),
        },
    )
    response.raise_for_status()
    return {item["id"]: item for item in response.json().get("items", [])}


async def scan_topic(
    topic: str,
    *,
    velocity_pool_size: int = 10,
    velocity_avg_count: int = 5,
    diversity_sample: int = 20,
) -> dict:
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY not found in environment.")

    now = utc_now()
    scan_timestamp = iso_utc(now)
    seven_days_ago = now - dt.timedelta(days=7)
    thirty_days_ago = now - dt.timedelta(days=30)
    thirty_seven_days_ago = now - dt.timedelta(days=37)
    search_calls_used = 0

    import httpx
    async with httpx.AsyncClient(timeout=20.0) as client:
        recent_uploads, calls_recent = await youtube_search_window_count(
            client,
            api_key,
            topic,
            published_after=seven_days_ago,
            max_pages=3,
            max_results=50,
        )
        prior_uploads, calls_prior = await youtube_search_window_count(
            client,
            api_key,
            topic,
            published_after=thirty_seven_days_ago,
            published_before=seven_days_ago,
            max_pages=3,
            max_results=50,
        )
        search_calls_used += calls_recent + calls_prior
        recent_results = await youtube_search_recent_videos(
            client,
            api_key,
            topic,
            published_after=thirty_days_ago,
            max_results=max(diversity_sample, velocity_pool_size),
        )
        search_calls_used += 1

        video_ids = []
        recent_videos = []
        for item in recent_results:
            vid = item.get("id", {}).get("videoId")
            if not vid:
                continue
            video_ids.append(vid)
            recent_videos.append(
                {
                    "video_id": vid,
                    "channel_id": item["snippet"].get("channelId"),
                    "channel_title": item["snippet"].get("channelTitle"),
                    "title": item["snippet"].get("title"),
                    "published_at": item["snippet"].get("publishedAt"),
                }
            )

        details = await youtube_video_statistics(client, api_key, video_ids)

    previous_snapshots = load_previous_snapshots(topic, video_ids)
    history_snapshots = load_snapshot_history(topic, video_ids)
    enriched = []
    delta_values = []
    for item in recent_videos:
        detail = details.get(item["video_id"], {})
        stats = detail.get("statistics", {})
        snippet = detail.get("snippet", {})
        published_at = item.get("published_at") or snippet.get("publishedAt")
        published_dt = parse_yt_datetime(published_at)
        age_days = max((now - published_dt).total_seconds() / 86400.0, 1.0) if published_dt else 1.0
        view_count = int(stats.get("viewCount", "0") or "0")
        view_velocity = round(view_count / age_days, 2)

        snapshot_delta = None
        previous = previous_snapshots.get(item["video_id"])
        if previous:
            previous_scan = parse_yt_datetime(previous.get("scan_timestamp"))
            if previous_scan is not None:
                elapsed_days = max((now - previous_scan).total_seconds() / 86400.0, 0.0001)
                snapshot_delta = round((view_count - int(previous["views_at_scan"])) / elapsed_days, 2)
                delta_values.append(snapshot_delta)

        enriched.append(
            {
                **item,
                "published_at": published_at,
                "view_count": view_count,
                "like_count": int(stats.get("likeCount", "0") or "0"),
                "comment_count": int(stats.get("commentCount", "0") or "0"),
                "age_days": round(age_days, 2),
                "view_velocity": view_velocity,
                "snapshot_delta": snapshot_delta,
            }
        )

    def _views_at_or_before(series: list[dict], target: dt.datetime) -> int | None:
        for row in series:
            snap = parse_yt_datetime(row.get("scan_timestamp"))
            if snap is None:
                continue
            if snap <= target:
                return int(row.get("views_at_scan", 0))
        return None

    velocity_pool = enriched[:velocity_pool_size]
    top_velocity = sorted(velocity_pool, key=lambda row: row["view_velocity"], reverse=True)[:velocity_avg_count]
    avg_view_velocity = round(
        sum(item["view_velocity"] for item in top_velocity) / len(top_velocity), 2
    ) if top_velocity else 0.0

    upload_surge_ratio = 0.0
    if prior_uploads > 0:
        upload_surge_ratio = round((recent_uploads / 7.0) / (prior_uploads / 30.0), 2)
    elif recent_uploads > 0:
        upload_surge_ratio = round(recent_uploads / 7.0, 2)

    diversity_source = enriched[:diversity_sample]
    channel_diversity = round(
        len({item["channel_id"] for item in diversity_source if item.get("channel_id")}) / len(diversity_source),
        2,
    ) if diversity_source else 0.0

    avg_snapshot_delta = round(sum(delta_values) / len(delta_values), 2) if delta_values else None

    upload_surge_component = round(0.40 * normalize_score(upload_surge_ratio, 5.0), 2)
    view_velocity_component = round(0.40 * normalize_score(avg_view_velocity, 250000.0), 2)
    channel_diversity_component = round(0.20 * (channel_diversity * 100.0), 2)
    m3_score = round(
        upload_surge_component + view_velocity_component + channel_diversity_component,
        2,
    )

    snapshot_delta_score = None
    if avg_snapshot_delta is not None:
        snapshot_delta_score = round(normalize_score(avg_snapshot_delta, 250000.0), 2)

    # Experimental sidecar M3: top-10 views acceleration using two historical windows.
    top10_for_exact = sorted(enriched, key=lambda row: row["view_count"], reverse=True)[:10]
    target_7d = now - dt.timedelta(days=7)
    target_14d = now - dt.timedelta(days=14)
    sum_views_last_7d = 0
    sum_views_prior_7d = 0
    videos_with_7d = 0
    videos_with_14d = 0
    videos_with_both = 0
    for row in top10_for_exact:
        vid = row["video_id"]
        current_views = int(row.get("view_count", 0))
        series = history_snapshots.get(vid, [])
        views_7d = _views_at_or_before(series, target_7d)
        views_14d = _views_at_or_before(series, target_14d)
        if views_7d is not None:
            videos_with_7d += 1
            sum_views_last_7d += max(0, current_views - views_7d)
        if views_14d is not None:
            videos_with_14d += 1
        if views_7d is not None and views_14d is not None:
            videos_with_both += 1
            sum_views_prior_7d += max(0, views_7d - views_14d)

    if sum_views_prior_7d > 0:
        m3_exact_ratio = round(sum_views_last_7d / sum_views_prior_7d, 3)
    elif sum_views_last_7d > 0:
        m3_exact_ratio = round(float(sum_views_last_7d), 3)
    else:
        m3_exact_ratio = 0.0
    m3_exact_score = round(normalize_score(m3_exact_ratio, 4.0), 2)
    exact_status = "ok" if videos_with_both >= 3 else "insufficient_history"

    payload = {
        "topic": topic,
        "scan_timestamp": scan_timestamp,
        "uploads_last_7d": recent_uploads,
        "uploads_prior_30d": prior_uploads,
        "upload_surge_ratio": upload_surge_ratio,
        "avg_view_velocity": avg_view_velocity,
        "channel_diversity": channel_diversity,
        "avg_snapshot_delta": avg_snapshot_delta,
        "velocity_pool_size": velocity_pool_size,
        "velocity_avg_count": velocity_avg_count,
        "diversity_sample_size": diversity_sample,
        "m3_components": {
            "upload_surge_component": upload_surge_component,
            "view_velocity_component": view_velocity_component,
            "channel_diversity_component": channel_diversity_component,
            "snapshot_delta_score": snapshot_delta_score,
        },
        "m3_score": m3_score,
        "m3_exact_experimental": {
            "status": exact_status,
            "cap": 4.0,
            "top_videos_count": len(top10_for_exact),
            "videos_with_7d_history": videos_with_7d,
            "videos_with_14d_history": videos_with_14d,
            "videos_with_both_windows": videos_with_both,
            "views_last_7d": sum_views_last_7d,
            "views_prior_7d": sum_views_prior_7d,
            "m3_exact_ratio": m3_exact_ratio,
            "m3_exact_score": m3_exact_score,
        },
        "top_velocity_video_ids": [item["video_id"] for item in top_velocity],
        "sample_size": len(enriched),
        "upload_count_method": "bounded_pagination_count",
        "upload_count_pages": 3,
        "quota_estimate_units": (search_calls_used * 100) + 1,
        "videos": enriched,
    }

    persist_scan(topic, payload, enriched)
    return payload


def print_human(payload: dict) -> None:
    print("=" * 68)
    print("YouTube Market Signals")
    print("=" * 68)
    print(f"Topic:                {payload['topic']}")
    print(f"Scan timestamp:       {payload['scan_timestamp']}")
    print(f"Uploads last 7d:      {payload['uploads_last_7d']}")
    print(f"Uploads prior 30d:    {payload['uploads_prior_30d']}")
    print(f"Upload surge ratio:   {payload['upload_surge_ratio']}")
    print(f"Avg view velocity:    {payload['avg_view_velocity']}")
    print(f"Channel diversity:    {payload['channel_diversity']}")
    print(f"Avg snapshot delta:   {payload['avg_snapshot_delta']}")
    print(f"Velocity pool size:   {payload['velocity_pool_size']}")
    print(f"Velocity avg count:   {payload['velocity_avg_count']}")
    print(f"Diversity sample:     {payload['diversity_sample_size']}")
    print("M3 components:")
    print(f"  - upload surge:     {payload['m3_components']['upload_surge_component']}")
    print(f"  - view velocity:    {payload['m3_components']['view_velocity_component']}")
    print(f"  - channel diversity:{payload['m3_components']['channel_diversity_component']}")
    print(f"  - snapshot delta:   {payload['m3_components']['snapshot_delta_score']}")
    print(f"M3 score:             {payload['m3_score']}")
    print(f"Quota estimate:       {payload['quota_estimate_units']} units")
    print()
    print("Top velocity videos")
    for item in sorted(payload["videos"], key=lambda row: row["view_velocity"], reverse=True)[:10]:
        print(
            f"- {item['title']} | views={item['view_count']} | "
            f"velocity={item['view_velocity']} | channel={item['channel_title']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", help="Topic to scan")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--velocity-pool-size", type=int, default=10, help="Top recent videos considered for velocity")
    parser.add_argument("--velocity-avg-count", type=int, default=5, help="Top velocity videos averaged into M3")
    parser.add_argument("--diversity-sample", type=int, default=20, help="Videos used for diversity score")
    args = parser.parse_args()

    import asyncio

    payload = asyncio.run(
        scan_topic(
            args.topic,
            velocity_pool_size=max(1, args.velocity_pool_size),
            velocity_avg_count=max(1, args.velocity_avg_count),
            diversity_sample=max(1, args.diversity_sample),
        )
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_human(payload)


if __name__ == "__main__":
    main()
