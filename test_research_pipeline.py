"""
test_research_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
StoryBit  —  Research Data Collection Pipeline  v4.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ARCHITECTURE
    Step 0a  LLM keyword expansion   →  precise, date-aware search strings
    Step 0b  [parallel] YouTube search + Trends (raw)
    Step 0c  LLM semantic filter     →  scores + re-ranks YouTube results
    Step 1   YouTube details         →  stats, tags, comments, transcripts
    Step 2   Google Trends           →  6 methods with 429-resistant strategy

USAGE
    python test_research_pipeline.py "Israel Iran war"
    python test_research_pipeline.py            # uses DEFAULT_TOPIC

OUTPUT  →  ./test_outputs/
    YYYY-MM-DD_HH-MM-SS_<slug>_raw.json
    YYYY-MM-DD_HH-MM-SS_<slug>_report.md

INSTALL
    pip install httpx pandas python-dotenv pytrends openai youtube-transcript-api

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALTERNATIVES FOR GOOGLE TRENDS 429 (documented here for reference)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PROBLEM: Google blocks automated pytrends requests from the same IP
           with HTTP 429. No amount of sleep() fixes this reliably.

  OPTION 1 — Residential proxy (BEST, ~$10-30/mo)
    providers: BrightData, Oxylabs, Smartproxy
    usage: pass proxies= to TrendReq
    TrendReq(proxies=["https://user:pass@gate.smartproxy.com:10001"])
    → 99.9% success rate, works indefinitely

  OPTION 2 — ScaleSerp / SerpAPI (paid, $50/mo, simplest)
    ScaleSerp has a native Trends endpoint:
    GET https://api.scaleserp.com/trends?api_key=KEY&q=Iran+war&date=today+12-m
    → returns JSON directly, no pytrends needed

  OPTION 3 — Google Trends RSS (free, limited data)
    GET https://trends.google.com/trends/trendingsearches/daily/rss?geo=US
    → only gives real-time trending, no historical data

  OPTION 4 — Delay + retry with VPN rotation (free but fragile)
    Implemented below as the default. Works when IP is not yet blocked.
    If you see 429s, switch to Option 1 or 2.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import datetime
import html
import json
import os
import re
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("future.no_silent_downcasting", True)
load_dotenv()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_TOPIC = "AI replacing software engineers"
OUTPUT_DIR    = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = "llama-3.3-70b-versatile"

# Proxy for Google Trends (optional — see alternatives in header)
# Set TRENDS_PROXY in .env like: TRENDS_PROXY=https://user:pass@host:port
TRENDS_PROXY    = os.getenv("TRENDS_PROXY", "")

YT_SEARCH_URL   = "https://www.googleapis.com/youtube/v3/search"
YT_VIDEOS_URL   = "https://www.googleapis.com/youtube/v3/videos"
YT_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
YT_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
YT_CAPTIONS_URL = "https://www.googleapis.com/youtube/v3/captions"

TREND_COUNTRIES = {
    "india":          "india",
    "united_states":  "united_states",
    "united_kingdom": "united_kingdom",
    "japan":          "japan",
    "australia":      "australia",
    "singapore":      "singapore",
}

# How many top results to fetch before semantic filtering
YT_SEARCH_COUNT         = 25
# How many videos to keep after semantic scoring
YT_KEEP_AFTER_FILTER    = 10
# Minimum relevance score (0-10) to keep a video
YT_MIN_RELEVANCE_SCORE  = 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_duration(iso: str) -> int:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return 0
    h, mn, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mn * 60 + s

def fmt_duration(secs: int) -> str:
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h}h{m}m{s}s" if h else f"{m}m{s}s"

def engagement_pct(stats: dict) -> float:
    try:
        v = int(stats.get("viewCount", "0") or "0")
        if not v:
            return 0.0
        l = int(stats.get("likeCount",   "0") or "0")
        c = int(stats.get("commentCount","0") or "0")
        return round(((l + c) / v) * 100, 4)
    except Exception:
        return 0.0

def decode_html(text: str) -> str:
    return html.unescape(text or "")

def file_slug(text: str, max_len: int = 45) -> str:
    return re.sub(r"[^\w\s-]", "", text).strip().replace(" ", "_")[:max_len]

def safe_df_col(df, preferred: str) -> str | None:
    if df is None or df.empty or len(df.columns) == 0:
        return None
    if preferred in df.columns:
        return preferred
    cols = [c for c in df.columns if c not in ("isPartial", "hasData")]
    return cols[0] if cols else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GROQ CLIENT HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _groq_json(prompt: str, max_tokens: int = 400, temp: float = 0.1) -> dict | None:
    """Single Groq call that always returns a parsed dict or None on failure."""
    if not GROQ_API_KEY:
        return None
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        resp   = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=temp,
            max_tokens=max_tokens,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"   ⚠  Groq call failed: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 0a — LLM KEYWORD EXPANSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def expand_keywords(topic: str) -> dict:
    """
    Produces two search strings from the topic.
      youtube_query   — specific, current-year YouTube search phrase
      trends_keyword  — 1-2 word Google Trends term

    KEY FIXES vs old version:
      • TODAY's real date is injected — model cannot hallucinate 2024
      • Post-processing regex replaces any stale year the model might emit
      • Temperature 0.1 (near-deterministic) — reduces creative drift
      • Filler words ("latest updates", "explained") are explicitly banned
    """
    today        = datetime.date.today()
    current_year = str(today.year)
    fallback     = {
        "youtube_query":  f"{topic} {current_year}",
        "trends_keyword": topic.split()[0][:25],
        "rationale":      "fallback (no Groq key)",
        "current_date":   today.isoformat(),
    }

    if not GROQ_API_KEY:
        return fallback

    prompt = f"""You are a search keyword specialist for a video research tool.

TODAY: {today.isoformat()}
CURRENT YEAR: {current_year}  ← you MUST use this year, NEVER 2024 or any older year.

TOPIC: "{topic}"

Return two search strings:

1. youtube_query (5-8 words)
   • What a real person would type on YouTube to find the MOST RECENT videos about this topic
   • MUST contain the year {current_year}
   • Use exact names/events/technologies from the topic — stay on-topic
   • BANNED words: "latest", "updates", "explained", "analysis", "news", "recap"
   • WRONG: "Israel Iran war 2024 latest updates"
   • RIGHT:  "Israel Iran war {current_year} US airstrikes"

2. trends_keyword (1-2 words MAX — hard limit, Google Trends needs short terms)
   • The highest-volume concrete proper noun or tech name from the topic
   • WRONG: "Iran conflict situation"
   • RIGHT:  "Iran war"

EXAMPLES:
  "Israel Iran war"            → youtube:"Israel Iran war {current_year} US airstrikes"   trends:"Iran war"
  "AI replacing engineers"     → youtube:"AI replacing software engineers {current_year}"  trends:"AI jobs"
  "recession in US"            → youtube:"US recession {current_year} economy"             trends:"US recession"
  "chips technological evolution" → youtube:"AI chip ARM SoC architecture {current_year}" trends:"AI chip"

Respond with a JSON object only — no markdown:
{{"youtube_query":"...","trends_keyword":"...","rationale":"one sentence"}}"""

    result = await _groq_json(prompt, max_tokens=150, temp=0.1)
    if not result:
        return fallback

    yt  = str(result.get("youtube_query",  fallback["youtube_query"])).strip()[:150]
    tr  = str(result.get("trends_keyword", fallback["trends_keyword"])).strip()[:40]
    rat = str(result.get("rationale", "")).strip()

    # ── Post-process: replace ANY stale year with current year ────────────────
    def _fix_year(text: str) -> str:
        # Replace 202x years that are not the current year
        fixed = re.sub(
            r"\b(202\d)\b",
            lambda m: current_year if m.group(1) != current_year else m.group(1),
            text,
        )
        # If no year present, append it
        if not re.search(r"\b202\d\b", fixed):
            fixed = f"{fixed.rstrip()} {current_year}"
        return fixed

    yt = _fix_year(yt)

    # ── Post-process: strip banned filler words from youtube_query ─────────────
    BANNED = ["latest", "updates", "update", "explained", "analysis",
              "recap", "breaking", "news", "top stories"]
    yt_words = yt.split()
    yt_clean  = " ".join(w for w in yt_words if w.lower() not in BANNED)
    if yt_clean.strip():
        yt = yt_clean

    # ── Post-process: enforce trends_keyword ≤ 2 words ───────────────────────
    tr_words = tr.split()
    if len(tr_words) > 2:
        tr = " ".join(tr_words[:2])

    return {
        "youtube_query":  yt,
        "trends_keyword": tr,
        "rationale":      rat,
        "current_date":   today.isoformat(),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 0c — SEMANTIC RELEVANCE SCORING  (LLM layer before details fetch)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def score_and_filter_videos(topic: str, raw_items: list[dict]) -> list[dict]:
    """
    WHY THIS EXISTS:
    YouTube's search API sorts by "relevance" but that means YouTube relevance,
    not *our* topic relevance. A video titled "Middle East Crisis General Overview"
    may rank above "Israel Iran War US Strikes 2026 — what actually happened".
    This LLM layer re-ranks and filters before we spend quota on details.

    INPUT:  raw_items — list of {video_id, title, channel, description_snippet}
                        from the search API (snippet only, no details yet)
    OUTPUT: filtered + re-ranked list, each item gets:
              relevance_score  int 0-10
              relevance_reason str  (≤8 words)
    """
    if not GROQ_API_KEY or not raw_items:
        # No scoring — just return all with null scores
        for item in raw_items:
            item["relevance_score"]  = None
            item["relevance_reason"] = "not scored (no Groq key)"
        return raw_items

    summaries = "\n".join(
        f"{i}: title={v['title']!r}  channel={v['channel']!r}  desc={v['description'][:100]!r}"
        for i, v in enumerate(raw_items)
    )

    prompt = f"""You are a strict content relevance judge for a video research platform.

TOPIC: "{topic}"

Score each YouTube video 0-10 for how directly and substantially it covers this EXACT topic.

Strict scoring:
  9-10  Directly covers the exact topic with specific detail (names, events, dates)
  7-8   Mostly on-topic, may include related context
  5-6   Topic is present but video is too broad or partially off-topic
  3-4   Topic mentioned briefly, video is mainly about something else
  0-2   Clickbait, tangential, unrelated, or compilation/reaction content

Be strict — only give 8+ to videos that are genuinely about "{topic}".

VIDEOS:
{summaries}

Return ONLY valid JSON:
{{"scores":[{{"index":0,"score":7,"reason":"covers topic directly"}},...]}}"""

    result = await _groq_json(prompt, max_tokens=600, temp=0.1)
    if not result:
        for item in raw_items:
            item["relevance_score"]  = None
            item["relevance_reason"] = "scoring failed"
        return raw_items

    score_map = {s["index"]: s for s in result.get("scores", [])}
    for i, item in enumerate(raw_items):
        entry = score_map.get(i, {})
        item["relevance_score"]  = entry.get("score", 5)
        item["relevance_reason"] = entry.get("reason", "")

    # Sort by score desc, then description length as quality proxy
    raw_items.sort(
        key=lambda v: (v.get("relevance_score") or 0, len(v.get("description", ""))),
        reverse=True,
    )

    # Filter out low-relevance videos
    filtered = [v for v in raw_items if (v.get("relevance_score") or 0) >= YT_MIN_RELEVANCE_SCORE]
    if not filtered:
        filtered = raw_items  # safety: keep all if everything scored low

    return filtered[:YT_KEEP_AFTER_FILTER + 5]  # keep a few extra before Shorts filter


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — YOUTUBE DATA API v3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def fetch_transcript(video_id: str) -> tuple[str | None, str | None]:
    """
    Fetch transcript using youtube-transcript-api v1.x (instance-based API).

    v1.0+ BREAKING CHANGES (all static methods removed in v1.2.0):
      OLD (broken): YouTubeTranscriptApi.get_transcript(id)
      OLD (broken): YouTubeTranscriptApi.list_transcripts(id)
      NEW:          ytt = YouTubeTranscriptApi(); ytt.fetch(id)
      NEW:          ytt = YouTubeTranscriptApi(); ytt.list(id)

    fetch() returns a FetchedTranscript object.
    Call .to_raw_data() for list-of-dicts, or iterate snippets directly.

    Returns:
      (text, "manual")         — manually uploaded caption
      (text, "auto-generated") — YouTube ASR caption
      (text, "auto (xx)")      — ASR in language xx
      (None, "disabled")       — creator disabled captions
      (None, "none")           — no transcripts exist
      (None, "error:...")      — unexpected failure
      (None, "not-installed")  — library not installed
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        loop = asyncio.get_running_loop()

        def _snippet_to_text(fetched) -> str:
            """
            FetchedTranscript is iterable yielding FetchedTranscriptSnippet objects.
            Each snippet has a .text attribute. to_raw_data() returns list of dicts.
            Both approaches work — use to_raw_data() for maximum compatibility.
            """
            raw = fetched.to_raw_data()   # list of {"text": ..., "start": ..., "duration": ...}
            return " ".join(d["text"] for d in raw if d.get("text", "").strip())

        def _fetch() -> tuple[str | None, str | None]:
            ytt      = YouTubeTranscriptApi()
            en_langs = ["en", "en-US", "en-GB", "en-IN", "en-AU", "en-CA"]

            # ── Strategy A: fetch() directly — simplest, fastest ─────────────
            # Automatically prefers manual over auto-generated in English.
            try:
                fetched = ytt.fetch(video_id, languages=en_langs)
                text    = _snippet_to_text(fetched)
                if text:
                    kind = "manual" if not fetched.is_generated else "auto-generated"
                    return text[:5000], kind
            except Exception:
                pass

            # ── Strategy B: list() → find specific transcript ─────────────────
            # Use when fetch() fails — gives us more control over language/type.
            try:
                tl = ytt.list(video_id)

                # B1: any manually created English transcript
                try:
                    t    = tl.find_manually_created_transcript(en_langs)
                    text = _snippet_to_text(t.fetch())
                    if text:
                        return text[:5000], "manual"
                except Exception:
                    pass

                # B2: auto-generated English transcript
                try:
                    t    = tl.find_generated_transcript(en_langs)
                    text = _snippet_to_text(t.fetch())
                    if text:
                        return text[:5000], "auto-generated"
                except Exception:
                    pass

                # B3: any language as last resort
                for t in tl:
                    try:
                        text = _snippet_to_text(t.fetch())
                        if text:
                            return text[:5000], f"auto ({t.language_code})"
                    except Exception:
                        continue

                return None, "none"

            except Exception as e:
                err = str(e).lower()
                if "disabled" in err or "no transcript" in err:
                    return None, "disabled"
                if "ipblocked" in err or "requestblocked" in err or "429" in err:
                    return None, "blocked"
                return None, f"error:{str(e)[:120]}"

        return await loop.run_in_executor(None, _fetch)

    except ImportError:
        return None, "not-installed"
    except Exception as e:
        return None, f"error:{str(e)[:80]}"


async def collect_youtube(topic: str, youtube_query: str) -> dict:
    """
    Full YouTube pipeline with semantic filter:
      1. Search (25 results, snippet only)          100 units
      2. LLM semantic scoring + filter              0 units (Groq, free)
      3. videos.list on top results                   1 unit
      4. channels.list                                1 unit
      5. commentThreads for top 5 videos              1 unit each
      6. fetch_transcript for captioned videos        0 units (transcript-api)
         └── fallback to captions.list if needed     50 units each
    """
    out: dict = {
        "topic":                  topic,
        "youtube_query_used":     youtube_query,
        "api_calls":              [],
        "quota_units_used":       0,
        "total_search_results":   0,
        "shorts_filtered_count":  0,
        "semantic_filter_applied": False,
        "videos_before_filter":   0,
        "videos":                 [],
        "error":                  None,
    }

    if not YOUTUBE_API_KEY:
        out["error"] = "YOUTUBE_API_KEY not set in .env"
        return out

    six_months_ago = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=180)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    async with httpx.AsyncClient(timeout=25.0) as http:

        # ── 1. Search — fetch more results upfront for semantic filter ────────
        t0 = time.time()
        try:
            r = await http.get(YT_SEARCH_URL, params={
                "key":            YOUTUBE_API_KEY,
                "q":              youtube_query,
                "part":           "snippet",
                "type":           "video",
                "order":          "relevance",
                "publishedAfter": six_months_ago,
                "maxResults":     YT_SEARCH_COUNT,
            })
            r.raise_for_status()
            search_data = r.json()
        except Exception as e:
            out["error"] = f"search failed: {e}"
            return out

        out["api_calls"].append({"call": "search", "ms": round((time.time()-t0)*1000)})
        out["quota_units_used"] += 100

        # Build raw_items list from search snippets
        raw_items:   list[dict] = []
        snippet_map: dict       = {}
        channel_ids: set[str]   = set()

        for item in search_data.get("items", []):
            vid  = item["id"].get("videoId")
            snip = item.get("snippet", {})
            if not vid:
                continue
            raw_items.append({
                "video_id":    vid,
                "title":       decode_html(snip.get("title", "")),
                "channel":     decode_html(snip.get("channelTitle", "")),
                "channel_id":  snip.get("channelId", ""),
                "description": decode_html(snip.get("description", "")),
                "published_at": snip.get("publishedAt", ""),
            })
            snippet_map[vid] = snip
            channel_ids.add(snip.get("channelId", ""))

        out["total_search_results"]  = len(raw_items)
        out["videos_before_filter"]  = len(raw_items)

        if not raw_items:
            out["error"] = "no videos returned by search"
            return out

        # ── 2. Semantic filter — LLM scores and re-ranks ─────────────────────
        print(f"      ↳ semantic scoring {len(raw_items)} search results...")
        scored_items = await score_and_filter_videos(topic, raw_items)
        out["semantic_filter_applied"] = GROQ_API_KEY != ""

        # Final ordered video_ids list
        video_ids = [item["video_id"] for item in scored_items]

        # ── 3. Video details — stats + tags + contentDetails  (1 unit) ───────
        t0 = time.time()
        try:
            r = await http.get(YT_VIDEOS_URL, params={
                "key":  YOUTUBE_API_KEY,
                "part": "statistics,snippet,contentDetails",
                "id":   ",".join(video_ids),
            })
            r.raise_for_status()
            video_map = {item["id"]: item for item in r.json().get("items", [])}
        except Exception as e:
            out["error"] = f"videos.list failed: {e}"
            return out
        out["api_calls"].append({"call": "videos.list", "ms": round((time.time()-t0)*1000)})
        out["quota_units_used"] += 1

        # ── 4. Channel details — keywords + subscriber count  (1 unit) ───────
        t0 = time.time()
        try:
            r = await http.get(YT_CHANNELS_URL, params={
                "key":  YOUTUBE_API_KEY,
                "part": "statistics,brandingSettings",
                "id":   ",".join(channel_ids),
            })
            r.raise_for_status()
            channel_map = {item["id"]: item for item in r.json().get("items", [])}
        except Exception as e:
            channel_map = {}
        out["api_calls"].append({"call": "channels.list", "ms": round((time.time()-t0)*1000)})
        out["quota_units_used"] += 1

        # ── 5. Comments for top 5 videos  (1 unit each) ──────────────────────
        comment_map: dict = {}
        for vid in video_ids[:5]:
            t0 = time.time()
            try:
                r = await http.get(YT_COMMENTS_URL, params={
                    "key":        YOUTUBE_API_KEY,
                    "part":       "snippet",
                    "videoId":    vid,
                    "order":      "relevance",
                    "maxResults": 60,
                })
                r.raise_for_status()
                items = r.json().get("items", [])
                # Sort by likes — highest liked = most representative sentiment
                items.sort(
                    key=lambda x: int(x["snippet"]["topLevelComment"]["snippet"].get("likeCount", 0)),
                    reverse=True,
                )
                comment_map[vid] = {
                    "total_fetched": len(items),
                    "comments": [
                        {
                            "text":         decode_html(c["snippet"]["topLevelComment"]["snippet"]["textOriginal"]),
                            "like_count":   int(c["snippet"]["topLevelComment"]["snippet"].get("likeCount", 0)),
                            "reply_count":  int(c["snippet"].get("totalReplyCount", 0)),
                            "published_at": c["snippet"]["topLevelComment"]["snippet"].get("publishedAt", ""),
                        }
                        for c in items[:30]
                    ],
                }
                out["api_calls"].append({"call": f"commentThreads({vid[:8]}...)", "ms": round((time.time()-t0)*1000)})
                out["quota_units_used"] += 1
            except Exception as e:
                comment_map[vid] = {"error": str(e), "comments": [], "total_fetched": 0}

        # ── 6. Assemble video objects ─────────────────────────────────────────
        import shlex

        # Build a score_map from scored_items
        score_map = {item["video_id"]: item for item in scored_items}

        for vid in video_ids:
            vdet   = video_map.get(vid, {})
            stats  = vdet.get("statistics", {})
            dsnip  = vdet.get("snippet", {})
            cont   = vdet.get("contentDetails", {})
            ssnip  = snippet_map.get(vid, {})
            cid    = ssnip.get("channelId", "")
            chan   = channel_map.get(cid, {})
            cstat  = chan.get("statistics", {})
            subs   = int(cstat.get("subscriberCount", "0") or "0")
            dur    = parse_duration(cont.get("duration", ""))
            scored = score_map.get(vid, {})

            # Filter Shorts (< 60 seconds)
            if 0 < dur < 60:
                out["shorts_filtered_count"] += 1
                continue

            kw_raw = chan.get("brandingSettings", {}).get("channel", {}).get("keywords", "") or ""
            try:
                chan_kw = shlex.split(kw_raw)
            except Exception:
                chan_kw = kw_raw.split()

            has_caption = cont.get("caption", "false").lower() == "true"
            cm = comment_map.get(vid, {})

            out["videos"].append({
                # identity
                "video_id":              vid,
                "url":                   f"https://www.youtube.com/watch?v={vid}",
                "title":                 decode_html(ssnip.get("title", "") or dsnip.get("title", "")),
                "published_at":          ssnip.get("publishedAt", ""),
                "description_snippet":   decode_html(dsnip.get("description", "")[:300]),
                # semantic score
                "relevance_score":       scored.get("relevance_score"),
                "relevance_reason":      scored.get("relevance_reason", ""),
                # format
                "category_id":           dsnip.get("categoryId", ""),
                "default_language":      dsnip.get("defaultLanguage", "") or dsnip.get("defaultAudioLanguage", ""),
                "duration_seconds":      dur,
                "duration_formatted":    fmt_duration(dur),
                # stats
                "view_count":            int(stats.get("viewCount",    "0") or "0"),
                "like_count":            int(stats.get("likeCount",    "0") or "0"),
                "comment_count":         int(stats.get("commentCount", "0") or "0"),
                "engagement_pct":        engagement_pct(stats),
                # channel
                "channel_name":          decode_html(ssnip.get("channelTitle", "")),
                "channel_id":            cid,
                "channel_subscriber_count": subs,
                "channel_keywords":      chan_kw[:30],
                # seo
                "tags":                  [decode_html(t) for t in dsnip.get("tags", [])],
                # captions — populated in step 7
                "has_caption":           has_caption,
                "caption_language":      None,
                "caption_track_kind":    None,
                "caption_text":          None,
                # comments
                "comments_fetched":      cm.get("total_fetched", 0),
                "comments":              cm.get("comments", []),
            })

        # ── 7. Transcripts ────────────────────────────────────────────────────
        # KEY INSIGHT: YouTube's contentDetails.caption=true only reflects
        # *manually uploaded* caption files. Auto-generated (ASR) captions —
        # which exist on nearly every English video — are NOT reflected there.
        # So we attempt youtube-transcript-api on EVERY video, not just
        # has_caption=True ones.
        all_vids = out["videos"]
        if all_vids:
            transcript_results = await asyncio.gather(
                *[fetch_transcript(v["video_id"]) for v in all_vids],
                return_exceptions=True,
            )
            for v, res in zip(all_vids, transcript_results):
                # Normalise result
                if isinstance(res, Exception):
                    text, kind = None, f"error:{res}"
                elif isinstance(res, tuple) and len(res) == 2:
                    text, kind = res
                else:
                    text, kind = None, "error:unexpected"

                kind = kind or "none"

                if text and not kind.startswith(("error:", "not-installed", "none", "disabled", "blocked")):
                    # ✅ Real transcript text retrieved
                    v["caption_text"]       = text
                    v["caption_track_kind"] = kind
                    v["caption_language"]   = "en"
                    v["has_caption"]        = True
                elif kind == "not-installed":
                    v["caption_text"]       = "[youtube-transcript-api not installed — run: pip install youtube-transcript-api]"
                    v["caption_track_kind"] = "not-installed"
                elif kind == "disabled":
                    v["caption_text"]       = "[Captions disabled by creator]"
                    v["caption_track_kind"] = "disabled"
                elif kind == "blocked":
                    v["caption_text"]       = "[IP blocked by YouTube — transcripts unavailable from this machine]"
                    v["caption_track_kind"] = "blocked"
                elif kind == "none":
                    v["caption_text"]       = "[No transcripts available for this video]"
                    v["caption_track_kind"] = "none"
                elif kind.startswith("error:"):
                    # Unexpected failure — still try captions.list if has_caption=True
                    if v["has_caption"]:
                        t0 = time.time()
                        try:
                            r = await http.get(YT_CAPTIONS_URL, params={
                                "key":     YOUTUBE_API_KEY,
                                "part":    "snippet",
                                "videoId": v["video_id"],
                            })
                            r.raise_for_status()
                            cap_items = r.json().get("items", [])
                            out["api_calls"].append({
                                "call": f"captions.list({v['video_id'][:8]}...)",
                                "ms":   round((time.time()-t0)*1000),
                            })
                            out["quota_units_used"] += 50
                            track = next(
                                (i for i in cap_items if i["snippet"].get("language","").startswith("en")),
                                cap_items[0] if cap_items else None,
                            )
                            if track:
                                v["caption_language"]   = track["snippet"].get("language","")
                                v["caption_track_kind"] = track["snippet"].get("trackKind","")
                                v["caption_text"]       = (
                                    f"[Caption confirmed via API ({v['caption_language']}, "
                                    f"{v['caption_track_kind']}) — transcript-api failed: {kind}]"
                                )
                            else:
                                v["caption_text"] = "[captions.list returned no tracks]"
                        except Exception as e:
                            v["caption_text"] = f"[captions.list error: {e}]"
                    else:
                        v["caption_text"] = None

    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 — GOOGLE TRENDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_pytrends():
    """
    Create TrendReq with:
    • urllib3 v2 shim  (method_whitelist removed in v2)
    • Optional proxy   (set TRENDS_PROXY in .env to avoid 429s)
    • Higher timeout   (Trends is slow from some regions)
    """
    from urllib3.util.retry import Retry
    _orig = Retry.__init__
    def _patched(self, *a, **kw):
        kw.pop("method_whitelist", None)
        _orig(self, *a, **kw)
    Retry.__init__ = _patched

    from pytrends.request import TrendReq
    proxies = {"https": TRENDS_PROXY} if TRENDS_PROXY else {}
    return TrendReq(hl="en-US", tz=0, timeout=(10, 45),
                    retries=2, backoff_factor=1.5, proxies=proxies)


def _trends_call(fn, label: str, errors: list, max_attempts: int = 3):
    """
    Execute a pytrends call with exponential backoff on 429.
    Waits: 5s → 15s → 45s between attempts.

    If you keep getting 429s even with this, you need a proxy.
    See ALTERNATIVES section at the top of this file.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            err = str(e)
            is_429 = "429" in err or "Too Many Requests" in err or "ResponseError" in err
            if is_429 and attempt < max_attempts - 1:
                wait = 5 * (3 ** attempt)   # 5s, 15s, 45s
                print(f"   ⏳ Trends 429 on [{label}] — retry in {wait}s ({attempt+1}/{max_attempts})")
                time.sleep(wait)
            else:
                if is_429:
                    errors.append(
                        f"{label}: 429 rate-limited — set TRENDS_PROXY in .env for reliable access"
                    )
                else:
                    errors.append(f"{label}: {e}")
                return None
    return None


def _df_to_list(df, name_col: str, val_col: str, out_key: str, out_val: str, n: int = 15) -> list[dict]:
    if df is None or df.empty:
        return []
    nc = name_col if name_col in df.columns else (df.columns[0] if len(df.columns) > 0 else None)
    vc = val_col  if val_col  in df.columns else None
    if not nc:
        return []
    rows = []
    for _, row in df.head(n).iterrows():
        rows.append({out_key: row[nc], out_val: row.get(vc, 0) if vc else 0})
    return rows


async def _fetch_trending_rss(geo: str = "US") -> list[str]:
    """
    Fallback: fetch real-time trending searches from Google Trends RSS.
    Returns list of trending terms. Works even when pytrends is blocked.
    """
    url = f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            r = await http.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            root  = ET.fromstring(r.text)
            items = root.findall(".//item/title")
            return [i.text.strip() for i in items if i.text][:20]
    except Exception:
        return []


async def collect_trends(topic: str, trends_keyword: str) -> dict:
    """
    Google Trends — 6 pytrends methods + RSS fallback for trending searches.

    429 STRATEGY:
    • _trends_call() wraps every pytrends call with exponential backoff
    • trending_searches() has a RSS fallback via Google's public feed
    • If all pytrends calls fail, set TRENDS_PROXY=... in .env (see header)

    Methods:
      1. interest_over_time       weekly scores 0-100, trend direction
      2. related_queries          rising + top queries
      3. related_topics           rising + top topics
      4. interest_by_region       global countries + India states
      5. trending_searches        real-time, 6 countries  (+ RSS fallback)
      6. top_charts               prior year US top searches
    """
    out: dict = {
        "topic":                    topic,
        "trends_keyword":           trends_keyword,
        "trend_direction":          "unknown",
        "current_score":            0,
        "peak_score":               0,
        "avg_score_12m":            0,
        "weekly_scores":            [],
        "rising_queries":           [],
        "top_queries":              [],
        "related_topics_rising":    [],
        "related_topics_top":       [],
        "top_regions_global":       [],
        "top_regions_india_states": [],
        "trending_by_country":      {},
        "is_trending_now":          False,
        "top_charts":               [],
        "api_calls":                [],
        "errors":                   [],
    }

    def _run():
        try:
            pt = _make_pytrends()
            kl = [trends_keyword]
            er = out["errors"]

            # ── 1. interest_over_time ─────────────────────────────────────────
            t0 = time.time()
            pt.build_payload(kl, timeframe="today 12-m", geo="")
            iot = _trends_call(pt.interest_over_time, "interest_over_time", er)
            out["api_calls"].append({"call": "interest_over_time", "ms": round((time.time()-t0)*1000)})
            if iot is not None:
                col = safe_df_col(iot, trends_keyword)
                if col:
                    s  = iot[col]
                    sc = [int(v) for v in s.fillna(0).tolist()]
                    out["weekly_scores"] = sc
                    out["current_score"] = sc[-1] if sc else 0
                    out["peak_score"]    = max(sc) if sc else 0
                    out["avg_score_12m"] = round(sum(sc)/len(sc), 1) if sc else 0
                    if len(s) >= 8:
                        ra, ea = s.tail(4).mean(), s.head(4).mean()
                        out["trend_direction"] = (
                            "rising"    if ra > ea * 1.15 else
                            "declining" if ra < ea * 0.85 else
                            "stable"
                        )
            time.sleep(3)   # larger gap to avoid 429 cascade

            # ── 2. related_queries ────────────────────────────────────────────
            t0  = time.time()
            rel = _trends_call(pt.related_queries, "related_queries", er)
            out["api_calls"].append({"call": "related_queries", "ms": round((time.time()-t0)*1000)})
            if rel is not None:
                td = rel.get(trends_keyword, {}) or {}
                out["rising_queries"] = _df_to_list(td.get("rising"), "query", "value", "query", "value")
                out["top_queries"]    = _df_to_list(td.get("top"),    "query", "value", "query", "value")
            time.sleep(3)

            # ── 3. related_topics ─────────────────────────────────────────────
            # Note: "list index out of range" is a pytrends internal parsing bug
            # that fires when Google's response structure is unexpected for the
            # given keyword. We rebuild the payload fresh and catch aggressively.
            t0 = time.time()
            try:
                pt.build_payload(kl, timeframe="today 12-m", geo="")   # fresh payload
                relt = _trends_call(pt.related_topics, "related_topics", er)
            except Exception as e:
                relt = None
                er.append(f"related_topics build: {e}")
            out["api_calls"].append({"call": "related_topics", "ms": round((time.time()-t0)*1000)})
            if relt is not None:
                try:
                    tt = relt.get(trends_keyword, {}) or {}
                    for kind, key in [("rising", "related_topics_rising"), ("top", "related_topics_top")]:
                        try:
                            df = tt.get(kind)
                            if df is None or df.empty or len(df.columns) == 0:
                                continue
                            # pytrends uses 'topic_title'; guard against column name variation
                            tcol = next((c for c in df.columns if "title" in c.lower()), None)
                            if tcol is None:
                                tcol = next(
                                    (c for c in df.columns
                                     if c not in ("value","hasData","isPartial","formattedValue","link","type")),
                                    None,
                                )
                            if tcol and tcol in df.columns:
                                out[key] = [str(v) for v in df[tcol].head(10).tolist() if v is not None]
                        except Exception:
                            pass
                except Exception as e:
                    er.append(f"related_topics parse: {e}")
            time.sleep(3)

            # ── 4a. interest_by_region — global countries ─────────────────────
            t0 = time.time()
            pt.build_payload(kl, timeframe="today 3-m", geo="")
            reg = _trends_call(
                lambda: pt.interest_by_region(resolution="COUNTRY", inc_low_vol=False),
                "interest_by_region(global)", er
            )
            out["api_calls"].append({"call": "interest_by_region(global)", "ms": round((time.time()-t0)*1000)})
            if reg is not None:
                col = safe_df_col(reg, trends_keyword)
                if col:
                    top = reg[col].sort_values(ascending=False).head(20)
                    out["top_regions_global"] = [
                        {"country": c, "score": int(s)} for c, s in top.items() if s > 0
                    ]
            time.sleep(3)

            # ── 4b. interest_by_region — India states ─────────────────────────
            t0 = time.time()
            pt.build_payload(kl, timeframe="today 3-m", geo="IN")
            reg_in = _trends_call(
                lambda: pt.interest_by_region(resolution="REGION", inc_low_vol=True),
                "interest_by_region(India)", er
            )
            out["api_calls"].append({"call": "interest_by_region(India)", "ms": round((time.time()-t0)*1000)})
            if reg_in is not None:
                col = safe_df_col(reg_in, trends_keyword)
                if col:
                    top = reg_in[col].sort_values(ascending=False).head(20)
                    out["top_regions_india_states"] = [
                        {"state": c, "score": int(s)} for c, s in top.items() if s > 0
                    ]
            time.sleep(3)

            # ── 5. trending_searches — 6 countries ────────────────────────────
            t0 = time.time()
            for label, pn in TREND_COUNTRIES.items():
                try:
                    tr = pt.trending_searches(pn=pn)
                    if not tr.empty and len(tr.columns) > 0:
                        col = tr.columns[0]
                        tl  = tr[col].dropna().astype(str).str.lower().tolist()
                        if tl:
                            out["trending_by_country"][label] = tl[:15]
                            if label in ("india", "united_states"):
                                if topic.lower() in tl or trends_keyword.lower() in tl:
                                    out["is_trending_now"] = True
                    time.sleep(1.5)
                except Exception:
                    pass   # handled by RSS fallback below
            out["api_calls"].append({
                "call": f"trending_searches({len(TREND_COUNTRIES)} countries)",
                "ms":   round((time.time()-t0)*1000),
            })
            time.sleep(3)

            # ── 6. top_charts — previous year only ────────────────────────────
            # Google only publishes top_charts in December of the target year.
            # A 404 before then is expected and not an error worth reporting.
            t0        = time.time()
            prev_year = datetime.datetime.now().year - 1
            tc_errors: list = []
            def _top_charts():
                return pt.top_charts(prev_year, hl="en-US", tz=0, geo="US")
            tc = _trends_call(_top_charts, f"top_charts({prev_year})", tc_errors)
            out["api_calls"].append({"call": f"top_charts({prev_year})", "ms": round((time.time()-t0)*1000)})
            if tc is not None and not tc.empty:
                out["top_charts"].append({"year": prev_year, "entries": tc.head(20).to_dict("records")})
            # Only surface non-404 errors (404 = charts not yet published)
            for e in tc_errors:
                if "404" not in e:
                    er.append(e)

        except Exception as e:
            out["errors"].append(f"trends_fatal: {e}")

        return out

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _run)

    # ── RSS fallback for trending searches ────────────────────────────────────
    # If pytrends trending_searches returned nothing (429 or empty),
    # fall back to Google's public RSS feed which is never rate-limited.
    if not out["trending_by_country"]:
        print("   ↳ pytrends trending empty — using RSS fallback...")
        rss_tasks = [
            ("united_states", "US"),
            ("india",         "IN"),
            ("united_kingdom","GB"),
        ]
        rss_results = await asyncio.gather(
            *[_fetch_trending_rss(geo) for _, geo in rss_tasks],
            return_exceptions=True,
        )
        for (label, _), res in zip(rss_tasks, rss_results):
            if isinstance(res, list) and res:
                out["trending_by_country"][label] = res
                tl = [t.lower() for t in res]
                if label in ("india", "united_states"):
                    if topic.lower() in tl or trends_keyword.lower() in tl:
                        out["is_trending_now"] = True
        if out["trending_by_country"]:
            out["api_calls"].append({"call": "trending_rss_fallback", "ms": 0})

    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REPORT BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_report(topic: str, kw: dict, youtube: dict, trends: dict, total_ms: int) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L: list[str] = []

    def h1(t):      L.append(f"\n# {t}\n")
    def h2(t):      L.append(f"\n## {t}\n")
    def h3(t):      L.append(f"\n### {t}\n")
    def row(*cols): L.append("| " + " | ".join(str(c) for c in cols) + " |")
    def div():      L.append("\n---\n")
    def nl():       L.append("")

    # ── Header ────────────────────────────────────────────────────────────────
    h1("StoryBit — Research Data Report")
    L.append(f"**Topic:** {topic}  ")
    L.append(f"**Run at:** {ts}  ")
    L.append(f"**Total time:** {total_ms/1000:.1f}s")
    nl()

    # ── Keyword expansion ─────────────────────────────────────────────────────
    h2("🔍 Step 0 — Keyword Expansion")
    row("Parameter", "Value"); row("---","---")
    row("Date injected",   kw.get("current_date",""))
    row("YouTube query",   f'`{kw.get("youtube_query","")}`')
    row("Trends keyword",  f'`{kw.get("trends_keyword","")}`')
    row("Rationale",       kw.get("rationale","—"))
    nl()

    # ── Summary ───────────────────────────────────────────────────────────────
    h2("📊 Pipeline Summary")
    row("Metric", "Value"); row("---","---")
    row("Search results returned",   youtube.get("total_search_results", 0))
    row("After semantic filter",     youtube.get("videos_before_filter", 0))
    row("Semantic filter applied",   "✅ Yes" if youtube.get("semantic_filter_applied") else "❌ No (no Groq key)")
    row("Shorts filtered out",       youtube.get("shorts_filtered_count", 0))
    row("Final videos collected",    len(youtube.get("videos", [])))
    videos = youtube.get("videos", [])
    transcripts_found = sum(1 for v in videos if v.get("caption_text") and v.get("caption_track_kind") not in ("error","not-installed",None) and not str(v.get("caption_text","")).startswith("["))
    row("Transcripts retrieved",     f"{transcripts_found}/{len(videos)} videos")
    row("YouTube quota used",        f"{youtube.get('quota_units_used', 0)} units")
    row("Trend direction",           trends.get("trend_direction","unknown"))
    row("Trends score (current)",    f"{trends.get('current_score',0)}/100")
    row("Trends score (peak 12m)",   f"{trends.get('peak_score',0)}/100")
    row("Rising queries",            len(trends.get("rising_queries",[])))
    row("Global countries",          len(trends.get("top_regions_global",[])))
    row("India states",              len(trends.get("top_regions_india_states",[])))
    row("Trending now (IN/US)",      "✅ Yes" if trends.get("is_trending_now") else "❌ No")
    nl()
    div()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ① YOUTUBE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    h2(f"① YouTube — {len(youtube.get('videos',[]))} Videos (semantic-filtered + ranked)")
    L.append(f"Query: `{youtube.get('youtube_query_used','')}`")
    nl()

    if youtube.get("error"):
        L.append(f"**⚠ Error:** {youtube['error']}"); nl()

    for i, v in enumerate(youtube.get("videos", []), 1):
        score_badge = f"  🎯 Relevance: {v['relevance_score']}/10" if v.get("relevance_score") is not None else ""
        h3(f"{i}. {v['title']}{score_badge}")

        if v.get("relevance_reason"):
            L.append(f"*{v['relevance_reason']}*"); nl()

        row("Field","Value"); row("---","---")
        row("URL",         v["url"])
        row("Published",   v["published_at"])
        row("Duration",    f"{v['duration_formatted']} ({v['duration_seconds']}s)")
        row("Language",    v["default_language"] or "—")
        row("Views",       f"{v['view_count']:,}")
        row("Likes",       f"{v['like_count']:,}")
        row("Comments",    f"{v['comment_count']:,}")
        row("Engagement",  f"{v['engagement_pct']}%")
        row("Channel",     v["channel_name"])
        row("Subscribers", f"{v['channel_subscriber_count']:,}")
        row("Has Caption", "✅ Yes" if v.get("has_caption") else "❌ No")
        if v.get("caption_language"):    row("Caption Lang", v["caption_language"])
        if v.get("caption_track_kind"):  row("Caption Type", v["caption_track_kind"])
        nl()

        tags = v.get("tags", [])
        L.append(f"**Tags ({len(tags)}):** {', '.join(tags) if tags else '—'}"); nl()

        ckw = v.get("channel_keywords", [])
        L.append(f"**Channel Keywords:** {', '.join(ckw[:20]) if ckw else '—'}"); nl()

        if v.get("description_snippet"):
            L.append(f"**Description:** {v['description_snippet']}"); nl()

        # Transcript
        ct   = v.get("caption_text")
        kind = v.get("caption_track_kind", "")
        if ct and not ct.startswith("[") and not ct.startswith("⚠"):
            h3(f"📝 Transcript  ({kind})")
            L.append(ct[:3000] + ("…" if len(ct) > 3000 else ""))
            nl()
        elif ct:
            h3("📝 Transcript")
            L.append(f"*{ct}*")
            nl()

        # Comments
        coms = v.get("comments", [])
        if coms:
            h3(f"💬 Comments  ({v.get('comments_fetched',0)} fetched · sorted by likes)")
            row("#","👍","↩","Comment"); row("---","---","---","---")
            for j, c in enumerate(coms[:20], 1):
                txt = c["text"].replace("\n"," ").replace("|","/")[:240]
                row(j, c["like_count"], c["reply_count"], txt)
            nl()
        else:
            L.append("*Comments not available*"); nl()

    div()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ② GOOGLE TRENDS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    h2("② Google Trends")
    L.append(f"Keyword: **`{trends.get('trends_keyword','')}`**"); nl()

    h3("1 · Interest Over Time  (12 months)")
    row("Metric","Value"); row("---","---")
    row("Direction",   trends.get("trend_direction","unknown"))
    row("Current",     f"{trends.get('current_score',0)}/100")
    row("Peak",        f"{trends.get('peak_score',0)}/100")
    row("12m Average", f"{trends.get('avg_score_12m',0)}/100")
    nl()
    if trends.get("weekly_scores"):
        L.append(f"**Weekly ({len(trends['weekly_scores'])} weeks):** `{trends['weekly_scores']}`"); nl()

    h3("2 · Related Queries")
    if trends.get("rising_queries"):
        L.append(f"**Rising ({len(trends['rising_queries'])}):**"); nl()
        row("Query","Value"); row("---","---")
        for q in trends["rising_queries"]: row(q["query"], q["value"])
        nl()
    else:
        L.append("*No rising queries*"); nl()

    if trends.get("top_queries"):
        L.append(f"**Top ({len(trends['top_queries'])}):**"); nl()
        row("Query","Value"); row("---","---")
        for q in trends["top_queries"]: row(q["query"], q["value"])
        nl()

    h3("3 · Related Topics")
    if trends.get("related_topics_rising"):
        L.append(f"**Rising:** {', '.join(trends['related_topics_rising'])}"); nl()
    if trends.get("related_topics_top"):
        L.append(f"**Top:** {', '.join(trends['related_topics_top'])}"); nl()
    if not trends.get("related_topics_rising") and not trends.get("related_topics_top"):
        L.append("*None found*"); nl()

    h3("4 · Interest by Region")
    if trends.get("top_regions_global"):
        L.append("**Global — Top Countries (3m):**"); nl()
        row("Country","Score"); row("---","---")
        for r in trends["top_regions_global"]: row(r["country"], f"{r['score']}/100")
        nl()
    if trends.get("top_regions_india_states"):
        L.append("**India — States (3m):**"); nl()
        row("State","Score"); row("---","---")
        for r in trends["top_regions_india_states"]: row(r["state"], f"{r['score']}/100")
        nl()
    if not trends.get("top_regions_global") and not trends.get("top_regions_india_states"):
        L.append("*No regional data*"); nl()

    h3("5 · Trending Searches by Country")
    tbc = trends.get("trending_by_country", {})
    if tbc:
        for country, items in tbc.items():
            L.append(f"**{country.replace('_',' ').title()}:** {', '.join(items[:12])}")
        nl()
    else:
        L.append("*No trending data*"); nl()

    h3("6 · Top Charts")
    if trends.get("top_charts"):
        for chart in trends["top_charts"]:
            L.append(f"**{chart['year']} — US Top Searches:**"); nl()
            row("Rank","Topic"); row("---","---")
            for j, e in enumerate(chart.get("entries",[]), 1):
                row(j, e.get("title") or e.get("query") or str(e))
            nl()
    else:
        L.append("*Not available*"); nl()

    if trends.get("errors"):
        nl()
        L.append("⚠️ **Trends errors:**"); nl()
        for err in trends["errors"]:
            L.append(f"- {err}")
        nl()

    return "\n".join(L)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main(topic: str) -> None:
    today = datetime.date.today()
    print(f"\n{'━'*64}")
    print(f"  StoryBit — Research Pipeline  v4.0")
    print(f"  Topic  : {topic}")
    print(f"  Date   : {today.isoformat()}")
    print(f"{'━'*64}\n")

    t_start = time.time()

    # Step 0a: keyword expansion (date-aware)
    print("▶  Step 0a  LLM keyword expansion...")
    kw = await expand_keywords(topic)
    print(f"   YouTube query  : {kw['youtube_query']}")
    print(f"   Trends keyword : {kw['trends_keyword']}")
    print(f"   Rationale      : {kw['rationale']}\n")

    # Steps 1 & 2: YouTube + Trends in parallel
    print("▶  Steps 1 & 2  YouTube (with semantic filter) + Trends (parallel)...")
    youtube, trends = await asyncio.gather(
        collect_youtube(topic, kw["youtube_query"]),
        collect_trends(topic,  kw["trends_keyword"]),
    )

    total_ms = round((time.time() - t_start) * 1000)

    transcripts_ok = sum(
        1 for v in youtube.get("videos", [])
        if v.get("caption_text")
        and v.get("caption_track_kind") not in ("error", "not-installed", None)
        and not str(v.get("caption_text","")).startswith("[")
    )

    print(f"\n   YouTube : {len(youtube.get('videos',[]))} videos"
          f"  | {youtube.get('shorts_filtered_count',0)} Shorts filtered"
          f"  | semantic={'✅' if youtube.get('semantic_filter_applied') else '❌'}"
          f"  | {youtube.get('quota_units_used',0)} quota units"
          f"  | {transcripts_ok} transcripts")
    print(f"   Trends  : direction={trends.get('trend_direction')}"
          f"  | score={trends.get('current_score')}/100"
          f"  | {len(trends.get('rising_queries',[]))} rising"
          f"  | {len(trends.get('top_regions_global',[]))} countries"
          f"  | {len(trends.get('top_regions_india_states',[]))} IN states")
    print(f"   Total   : {total_ms/1000:.1f}s\n")

    # Save
    output = {
        "meta": {
            "topic":          topic,
            "youtube_query":  kw["youtube_query"],
            "trends_keyword": kw["trends_keyword"],
            "kw_rationale":   kw["rationale"],
            "run_date":       today.isoformat(),
            "timestamp":      datetime.datetime.now().isoformat(),
            "total_ms":       total_ms,
        },
        "youtube": youtube,
        "trends":  trends,
    }

    ts   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{ts}_{file_slug(topic)}"
    jp   = OUTPUT_DIR / f"{base}_raw.json"
    mp   = OUTPUT_DIR / f"{base}_report.md"

    with open(jp, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(build_report(topic, kw, youtube, trends, total_ms))

    print(f"{'━'*64}")
    print(f"  📄 JSON   → {jp}")
    print(f"  📋 Report → {mp}")
    print(f"{'━'*64}\n")

    if youtube.get("error"):
        print(f"⚠  YouTube : {youtube['error']}")
    if trends.get("errors"):
        print("⚠  Trends errors:")
        for e in trends["errors"]:
            print(f"   • {e}")


if __name__ == "__main__":
    _topic = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else DEFAULT_TOPIC
    asyncio.run(main(_topic))