#!/usr/bin/env python3
"""
Standalone Google Trends probe with alternate-query retries.

Usage:
  python3 google_trends_only.py "Israel Iran War"
  python3 google_trends_only.py "AI automation" --json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pytrends.request import TrendReq


QUERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at", "by",
    "with", "from", "vs", "is", "are", "was", "were", "today", "latest", "news",
}

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
TRENDS_DB = CACHE_DIR / "google_trends_snapshots.sqlite3"
load_dotenv()


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
    conn = sqlite3.connect(TRENDS_DB)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS google_trends_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_key TEXT NOT NULL,
                topic TEXT NOT NULL,
                query_used TEXT NOT NULL,
                scan_timestamp TEXT NOT NULL,
                recent_7d_avg REAL NOT NULL,
                prior_30d_avg REAL NOT NULL,
                search_surge_ratio REAL NOT NULL,
                avg_recent_interest REAL NOT NULL,
                region_diversity REAL NOT NULL,
                snapshot_delta REAL,
                m3_score REAL NOT NULL,
                raw_payload TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_google_trends_scans_lookup
            ON google_trends_scans(topic_key, scan_timestamp DESC)
            """
        )
        conn.commit()
    finally:
        conn.close()


def get_paid_trends_provider() -> tuple[str | None, str | None]:
    provider = (os.getenv("TRENDS_API_PROVIDER") or "").strip().lower()
    if provider == "scaleserp":
        return "scaleserp", (os.getenv("SCALESERP_API_KEY") or os.getenv("TRENDS_API_KEY") or "").strip() or None
    if provider == "serpapi":
        return "serpapi", (os.getenv("SERPAPI_API_KEY") or os.getenv("TRENDS_API_KEY") or "").strip() or None
    if os.getenv("SCALESERP_API_KEY"):
        return "scaleserp", os.getenv("SCALESERP_API_KEY")
    if os.getenv("SERPAPI_API_KEY"):
        return "serpapi", os.getenv("SERPAPI_API_KEY")
    if os.getenv("TRENDS_API_KEY"):
        return "scaleserp", os.getenv("TRENDS_API_KEY")
    return None, None


def extract_numeric_series_from_payload(payload: dict) -> list[float]:
    candidate_paths = [
        payload.get("interest_over_time", {}).get("timeline_data"),
        payload.get("interest_over_time", {}).get("timeline"),
        payload.get("timeline_data"),
        payload.get("timeline"),
        payload.get("data"),
    ]

    def _from_timeline(items) -> list[float]:
        values: list[float] = []
        if not isinstance(items, list):
            return values
        for item in items:
            if isinstance(item, dict):
                raw = item.get("values") or item.get("value")
                if isinstance(raw, list) and raw:
                    first = raw[0]
                    if isinstance(first, dict):
                        first = first.get("value")
                    try:
                        values.append(float(first))
                        continue
                    except Exception:
                        pass
                try:
                    if "value" in item and not isinstance(item["value"], (list, dict)):
                        values.append(float(item["value"]))
                except Exception:
                    pass
        return values

    for path in candidate_paths:
        values = _from_timeline(path)
        if len(values) >= 2:
            return values

    found: list[float] = []

    def _walk(obj):
        nonlocal found
        if found:
            return
        if isinstance(obj, list):
            values = []
            for item in obj:
                if isinstance(item, (int, float)):
                    values.append(float(item))
                elif isinstance(item, dict):
                    raw = item.get("value")
                    if isinstance(raw, (int, float)):
                        values.append(float(raw))
                    elif isinstance(raw, list) and raw and isinstance(raw[0], (int, float)):
                        values.append(float(raw[0]))
            if len(values) >= 2:
                found = values
                return
            for item in obj:
                _walk(item)
        elif isinstance(obj, dict):
            for value in obj.values():
                _walk(value)

    _walk(payload)
    return found


def extract_related_queries_from_payload(payload: dict) -> tuple[list[str], list[str]]:
    rising: list[str] = []
    top: list[str] = []
    related = payload.get("related_queries") or payload.get("queries") or {}

    def _pluck(items) -> list[str]:
        values: list[str] = []
        if not isinstance(items, list):
            return values
        for item in items:
            if isinstance(item, dict):
                query = item.get("query") or item.get("title")
                if isinstance(query, str) and query.strip():
                    values.append(query.strip())
        return values

    if isinstance(related, dict):
        rising = _pluck(related.get("rising"))[:10]
        top = _pluck(related.get("top"))[:10]
    return rising, top


def extract_regions_from_payload(payload: dict) -> list[dict]:
    region_candidates = (
        payload.get("interest_by_region")
        or payload.get("regions")
        or payload.get("geo_map")
        or []
    )
    regions: list[dict] = []
    if isinstance(region_candidates, list):
        for item in region_candidates:
            if not isinstance(item, dict):
                continue
            country = item.get("location") or item.get("country") or item.get("geoName")
            score = item.get("value") or item.get("score")
            if isinstance(score, list) and score:
                score = score[0]
            try:
                if country and score is not None:
                    regions.append({"country": str(country), "score": int(float(score))})
            except Exception:
                continue
    return regions[:5]


def fetch_paid_trends_payload(provider: str, api_key: str, query: str, date_window: str) -> dict:
    timeout = httpx.Timeout(20.0)
    if provider == "scaleserp":
        response = httpx.get(
            "https://api.scaleserp.com/trends",
            params={"api_key": api_key, "q": query, "date": date_window},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    if provider == "serpapi":
        response = httpx.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_trends",
                "q": query,
                "data_type": "TIMESERIES",
                "date": date_window,
                "api_key": api_key,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    raise ValueError(f"Unsupported paid trends provider: {provider}")


def load_previous_scan(topic: str) -> dict | None:
    init_db()
    conn = sqlite3.connect(TRENDS_DB)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT query_used, scan_timestamp, avg_recent_interest
            FROM google_trends_scans
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
    conn = sqlite3.connect(TRENDS_DB)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO google_trends_scans (
                topic_key, topic, query_used, scan_timestamp, recent_7d_avg,
                prior_30d_avg, search_surge_ratio, avg_recent_interest,
                region_diversity, snapshot_delta, m3_score, raw_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalize_topic_key(topic),
                topic,
                payload["query_used"],
                payload["scan_timestamp"],
                payload["recent_7d_avg"],
                payload["prior_30d_avg"],
                payload["search_surge_ratio"],
                payload["avg_recent_interest"],
                payload["region_diversity"],
                payload.get("snapshot_delta"),
                payload["m3_score"],
                json.dumps(payload, ensure_ascii=True),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def build_entity_preserving_query_variants(topic: str) -> list[str]:
    original = re.sub(r"\s+", " ", (topic or "").strip())
    lowered = original.lower()
    tokens = original.split()
    variants = [original]

    capitalized = re.findall(r"[A-Z][a-z]+|[A-Z]{2,}", original)
    entity_phrase = " ".join(capitalized) if capitalized else original
    if entity_phrase and entity_phrase not in variants:
        variants.append(entity_phrase)

    if "war" in lowered:
        variants.append(re.sub(r"\bwar\b", "conflict", original, flags=re.IGNORECASE))
        variants.append(f"{original} analysis")
    if "conflict" not in lowered:
        variants.append(f"{entity_phrase} conflict")
    if "escalation" not in lowered:
        variants.append(f"{entity_phrase} escalation")
    if len(tokens) >= 2:
        variants.append(f"{tokens[1]} {tokens[0]} conflict")

    deduped = []
    seen = set()
    for item in variants:
        candidate = re.sub(r"\s+", " ", item).strip()
        if candidate and candidate.lower() not in seen:
            seen.add(candidate.lower())
            deduped.append(candidate)
    return deduped[:6]


def score_result(item: dict) -> tuple:
    return (
        1 if item.get("trend_direction") != "unknown" else 0,
        1 if item.get("current_score", 0) > 0 else 0,
        len(item.get("rising_queries", [])),
        len(item.get("top_regions", [])),
        item.get("peak_score", 0),
    )


def fetch_keyword_trends(topic: str) -> dict:
    candidates = build_entity_preserving_query_variants(topic)
    now = utc_now()
    paid_provider, paid_api_key = get_paid_trends_provider()
    best = {
        "query_used": topic,
        "queries_tried": [],
        "scan_timestamp": iso_utc(now),
        "provider_used": paid_provider or "pytrends",
        "current_week_index": 0.0,
        "twelve_month_avg_index": 0.0,
        "m1_search_ratio": 0.0,
        "m1_score": 0.0,
        "trend_direction": "unknown",
        "current_score": 0,
        "peak_score": 0,
        "recent_7d_avg": 0.0,
        "prior_30d_avg": 0.0,
        "search_surge_ratio": 0.0,
        "avg_recent_interest": 0.0,
        "region_diversity": 0.0,
        "snapshot_delta": None,
        "m3_components": {
            "surge_component": 0.0,
            "interest_component": 0.0,
            "diversity_component": 0.0,
            "snapshot_delta_score": None,
        },
        "m3_score": 0.0,
        "rising_queries": [],
        "top_queries": [],
        "top_regions": [],
        "is_trending_now": False,
    }
    best_score = score_result(best)
    tried = []

    for candidate in candidates:
        tried.append(candidate)
        result = {
            "query_used": candidate,
            "queries_tried": [],
            "scan_timestamp": iso_utc(now),
            "provider_used": paid_provider or "pytrends",
            "current_week_index": 0.0,
            "twelve_month_avg_index": 0.0,
            "m1_search_ratio": 0.0,
            "m1_score": 0.0,
            "trend_direction": "unknown",
            "current_score": 0,
            "peak_score": 0,
            "recent_7d_avg": 0.0,
            "prior_30d_avg": 0.0,
            "search_surge_ratio": 0.0,
            "avg_recent_interest": 0.0,
            "region_diversity": 0.0,
            "snapshot_delta": None,
            "m3_components": {
                "surge_component": 0.0,
                "interest_component": 0.0,
                "diversity_component": 0.0,
                "snapshot_delta_score": None,
            },
            "m3_score": 0.0,
            "rising_queries": [],
            "top_queries": [],
            "top_regions": [],
            "is_trending_now": False,
        }
        try:
            if paid_provider and paid_api_key:
                year_payload = fetch_paid_trends_payload(paid_provider, paid_api_key, candidate, "today 12-m")
                year_series = extract_numeric_series_from_payload(year_payload)
                if len(year_series) >= 2:
                    current_week_index = float(year_series[-1])
                    twelve_month_avg = float(sum(year_series) / len(year_series))
                    result["current_week_index"] = round(current_week_index, 2)
                    result["twelve_month_avg_index"] = round(twelve_month_avg, 2)
                    result["current_score"] = int(round(current_week_index))
                    result["peak_score"] = int(round(max(year_series)))
                    if twelve_month_avg > 0:
                        result["m1_search_ratio"] = round(current_week_index / twelve_month_avg, 3)
                    elif current_week_index > 0:
                        result["m1_search_ratio"] = round(current_week_index / 10.0, 3)
                    result["m1_score"] = round(normalize_score(result["m1_search_ratio"], 5.0), 2)
                    if result["m1_search_ratio"] >= 1.5:
                        result["trend_direction"] = "rising"
                    elif current_week_index > 0:
                        result["trend_direction"] = "stable"
                    result["rising_queries"], result["top_queries"] = extract_related_queries_from_payload(year_payload)
                    result["top_regions"] = extract_regions_from_payload(year_payload)

                quarter_payload = fetch_paid_trends_payload(paid_provider, paid_api_key, candidate, "today 3-m")
                quarter_series = extract_numeric_series_from_payload(quarter_payload)
                if len(quarter_series) >= 2:
                    recent_slice = quarter_series[-7:] if len(quarter_series) >= 7 else quarter_series
                    baseline_slice = quarter_series[-37:-7] if len(quarter_series) >= 37 else quarter_series[:-7]
                    recent_avg = float(sum(recent_slice) / len(recent_slice)) if recent_slice else 0.0
                    earlier_avg = float(sum(baseline_slice) / len(baseline_slice)) if baseline_slice else 0.0
                    result["recent_7d_avg"] = round(recent_avg, 2)
                    result["prior_30d_avg"] = round(earlier_avg, 2)
                    result["avg_recent_interest"] = round(recent_avg, 2)
                    if earlier_avg > 0:
                        result["search_surge_ratio"] = round(recent_avg / earlier_avg, 2)
                    elif recent_avg > 0:
                        result["search_surge_ratio"] = round(recent_avg / 10.0, 2)
                    if result["trend_direction"] == "unknown" and earlier_avg > 0 and recent_avg > earlier_avg * 1.15:
                        result["trend_direction"] = "rising"
                    elif result["trend_direction"] == "unknown" and earlier_avg > 0 and recent_avg < earlier_avg * 0.85:
                        result["trend_direction"] = "declining"
                    elif result["trend_direction"] == "unknown" and recent_avg > 0:
                        result["trend_direction"] = "stable"
                    if not result["top_regions"]:
                        result["top_regions"] = extract_regions_from_payload(quarter_payload)
            else:
                pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=2, backoff_factor=0.5)
                kw_list = [candidate]

                pytrends.build_payload(kw_list, timeframe="today 12-m", geo="")
                year_df = pytrends.interest_over_time()
                if not year_df.empty and candidate in year_df.columns:
                    year_series = year_df[candidate].dropna()
                    if len(year_series) >= 2:
                        current_week_index = float(year_series.iloc[-1])
                        twelve_month_avg = float(year_series.mean())
                        result["current_week_index"] = round(current_week_index, 2)
                        result["twelve_month_avg_index"] = round(twelve_month_avg, 2)
                        result["current_score"] = int(year_series.iloc[-1])
                        result["peak_score"] = int(year_series.max())
                        if twelve_month_avg > 0:
                            result["m1_search_ratio"] = round(current_week_index / twelve_month_avg, 3)
                        elif current_week_index > 0:
                            result["m1_search_ratio"] = round(current_week_index / 10.0, 3)
                        result["m1_score"] = round(normalize_score(result["m1_search_ratio"], 5.0), 2)
                        if result["m1_search_ratio"] >= 1.5:
                            result["trend_direction"] = "rising"
                        elif current_week_index > 0:
                            result["trend_direction"] = "stable"

                pytrends.build_payload(kw_list, timeframe="today 3-m", geo="")
                iot_df = pytrends.interest_over_time()
                if not iot_df.empty and candidate in iot_df.columns:
                    series = iot_df[candidate].dropna()
                    if len(series) >= 2:
                        recent_avg = float(series.tail(7).mean()) if len(series) >= 7 else float(series.mean())
                        baseline_window = series.iloc[-37:-7] if len(series) >= 37 else series.iloc[:-7]
                        earlier_avg = float(baseline_window.mean()) if len(baseline_window) > 0 else 0.0
                        result["recent_7d_avg"] = round(recent_avg, 2)
                        result["prior_30d_avg"] = round(earlier_avg, 2)
                        result["avg_recent_interest"] = round(recent_avg, 2)
                        if earlier_avg > 0:
                            result["search_surge_ratio"] = round(recent_avg / earlier_avg, 2)
                        elif recent_avg > 0:
                            result["search_surge_ratio"] = round(recent_avg / 10.0, 2)
                        if result["trend_direction"] == "unknown" and earlier_avg > 0 and recent_avg > earlier_avg * 1.15:
                            result["trend_direction"] = "rising"
                        elif result["trend_direction"] == "unknown" and earlier_avg > 0 and recent_avg < earlier_avg * 0.85:
                            result["trend_direction"] = "declining"
                        elif result["trend_direction"] == "unknown" and recent_avg > 0:
                            result["trend_direction"] = "stable"
                        elif result["trend_direction"] == "unknown":
                            result["trend_direction"] = "stable"

                related = pytrends.related_queries()
                topic_data = related.get(candidate, {})
                if topic_data:
                    rising_df = topic_data.get("rising")
                    top_df = topic_data.get("top")
                    if rising_df is not None and not rising_df.empty:
                        result["rising_queries"] = rising_df["query"].head(10).tolist()
                    if top_df is not None and not top_df.empty:
                        result["top_queries"] = top_df["query"].head(10).tolist()

                pytrends.build_payload(kw_list, timeframe="today 3-m", geo="")
                region_df = pytrends.interest_by_region(resolution="COUNTRY", inc_low_vol=False)
                if not region_df.empty and candidate in region_df.columns:
                    top_regions = region_df[candidate].sort_values(ascending=False).head(5)
                    result["top_regions"] = [
                        {"country": country, "score": int(score)}
                        for country, score in top_regions.items()
                        if score > 0
                    ]

                try:
                    trending_df = pytrends.trending_searches(pn="united_states")
                    col = trending_df.columns[0] if not trending_df.empty else None
                    if col is not None:
                        trending_list = trending_df[col].str.lower().tolist()
                        result["is_trending_now"] = candidate.lower() in trending_list
                except Exception:
                    pass

            region_count = len(result["top_regions"])
            result["region_diversity"] = round(region_count / 5.0, 2) if region_count else 0.0
        except Exception as exc:
            result["error"] = str(exc)

        surge_component = round(0.40 * normalize_score(float(result["search_surge_ratio"]), 5.0), 2)
        interest_component = round(0.40 * normalize_score(float(result["avg_recent_interest"]), 100.0), 2)
        diversity_component = round(0.20 * (float(result["region_diversity"]) * 100.0), 2)
        result["m3_components"] = {
            "surge_component": surge_component,
            "interest_component": interest_component,
            "diversity_component": diversity_component,
            "snapshot_delta_score": None,
        }
        result["m3_score"] = round(surge_component + interest_component + diversity_component, 2)

        current_score = (
            1 if result.get("m1_score", 0) > 0 else 0,
            1 if result.get("m1_search_ratio", 0) >= 1.0 else 0,
            len(result.get("rising_queries", [])),
            len(result.get("top_regions", [])),
            result.get("peak_score", 0),
        )
        if current_score > best_score:
            best = result
            best_score = current_score
        if best_score[0] == 1 and (best_score[1] == 1 or best_score[2] > 0 or best_score[3] > 0):
            break

    previous = load_previous_scan(topic)
    if previous:
        previous_interest = float(previous.get("avg_recent_interest", 0.0))
        snapshot_delta = round(best["avg_recent_interest"] - previous_interest, 2)
        best["snapshot_delta"] = snapshot_delta
        best["m3_components"]["snapshot_delta_score"] = round(normalize_score(snapshot_delta, 100.0), 2)

    best["queries_tried"] = tried
    persist_scan(topic, best)
    return best


def print_human(payload: dict) -> None:
    print("=" * 68)
    print("Google Trends Probe")
    print("=" * 68)
    print(f"Query used:          {payload['query_used']}")
    print(f"Queries tried:       {', '.join(payload.get('queries_tried', []))}")
    print(f"Trend direction:     {payload['trend_direction']}")
    print(f"Current week index:  {payload['current_week_index']}")
    print(f"12m avg index:       {payload['twelve_month_avg_index']}")
    print(f"M1 search ratio:     {payload['m1_search_ratio']}")
    print(f"M1 score:            {payload['m1_score']}")
    print(f"Current score:       {payload['current_score']}")
    print(f"Peak score:          {payload['peak_score']}")
    print(f"Recent 7d avg:       {payload['recent_7d_avg']}")
    print(f"Prior 30d avg:       {payload['prior_30d_avg']}")
    print(f"Search surge ratio:  {payload['search_surge_ratio']}")
    print(f"Avg recent interest: {payload['avg_recent_interest']}")
    print(f"Region diversity:    {payload['region_diversity']}")
    print(f"Snapshot delta:      {payload['snapshot_delta']}")
    print("M3 components:")
    print(f"  - surge:           {payload['m3_components']['surge_component']}")
    print(f"  - interest:        {payload['m3_components']['interest_component']}")
    print(f"  - diversity:       {payload['m3_components']['diversity_component']}")
    print(f"  - snapshot delta:  {payload['m3_components']['snapshot_delta_score']}")
    print(f"M3 score:            {payload['m3_score']}")
    print(f"Trending now:        {payload['is_trending_now']}")
    print(f"Rising queries:      {payload['rising_queries']}")
    print(f"Top queries:         {payload['top_queries']}")
    print(f"Top regions:         {payload['top_regions']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", help="Topic to probe")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    payload = fetch_keyword_trends(args.topic)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_human(payload)


if __name__ == "__main__":
    main()
