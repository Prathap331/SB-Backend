# ============================================================
# main.py — Migrated to google-genai (new unified SDK)
# pip install google-genai  (replaces google-generativeai)
# ============================================================

from fastapi import Depends, HTTPException, status, Request, Header, BackgroundTasks
from fastapi import FastAPI
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from postgrest.exceptions import APIError
from supabase_auth.types import User
from openai import AsyncOpenAI
from auth_dependencies import get_current_user, login_user, refresh_access_token
from tss_v3 import run_tss
from pipeline_response_adapter import adapt_pipeline_payload
from idea_generation_pipeline import generate_ideas as generate_cags_aligned_ideas, TOPIC_CACHE
from social_market_signals import scan_topic as scan_social_topic
from news_market_signals import scan_topic as scan_news_topic

# ── NEW unified Google GenAI SDK ─────────────────────────────
from google import genai
from google.genai import types as genai_types
# ─────────────────────────────────────────────────────────────

import os
import asyncio
import time
import re
import json
import random
import hmac
import hashlib
import httpx
import nltk
import razorpay
import datetime
from collections import OrderedDict
from typing import Any

from urllib.parse import urlparse
from datetime import datetime as dt
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
from ddgs import DDGS
from readability import Document

load_dotenv()

# ── NLTK data ────────────────────────────────────────────────
project_root = os.path.dirname(os.path.abspath(__file__))
nltk_data_dir = os.path.join(project_root, 'nltk_data')
nltk.data.path.insert(0, nltk_data_dir)

def _ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_path)
        return
    except LookupError:
        pass
    try:
        print(f"Downloading NLTK resource: {download_name}")
        nltk.download(download_name, download_dir=nltk_data_dir, quiet=True)
        nltk.data.find(resource_path)
    except LookupError as e:
        print(f"!!! CRITICAL NLTK DATA ERROR: {e} !!!")


_ensure_nltk_resource("tokenizers/punkt", "punkt")
_ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
print("NLTK 'punkt' and 'punkt_tab' data checked.")

# ── Razorpay ─────────────────────────────────────────────────
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")

if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    print("WARNING: Razorpay API keys not found. Payment endpoints will fail.")
    razorpay_client = None
else:
    razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    print("Razorpay client initialized.")

# ── Groq (AsyncOpenAI-compatible) ────────────────────────────
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found.")
groq_client = AsyncOpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)
GROQ_GENERATION_MODEL = "llama-3.1-8b-instant"
GROQ_SCRIPT_MODEL = os.getenv("GROQ_SCRIPT_MODEL", GROQ_GENERATION_MODEL).strip() or GROQ_GENERATION_MODEL

# ── Google GenAI client(s) (NEW SDK) ─────────────────────────
def _collect_embed_keys() -> list[str]:
    ordered = [
        (os.getenv("GOOGLE_API_KEY1") or "").strip(),
        (os.getenv("GOOGLE_API_KEY2") or "").strip(),
        (os.getenv("GOOGLE_API_KEY") or "").strip(),
    ]
    deduped: list[str] = []
    for key in ordered:
        if key and key not in deduped:
            deduped.append(key)
    return deduped


GOOGLE_EMBED_KEYS = _collect_embed_keys()
if not GOOGLE_EMBED_KEYS:
    raise ValueError("No Google embedding key found. Set GOOGLE_API_KEY1/2 or GOOGLE_API_KEY.")

# Gemini clients — ONLY used for embeddings.
# Order matters: we try KEY1, then KEY2, then GOOGLE_API_KEY.
EMBED_CLIENTS = [genai.Client(api_key=key) for key in GOOGLE_EMBED_KEYS]

# text-embedding-004 was deprecated Jan 14 2026 — migrated to gemini-embedding-001.
# output_dimensionality=768 forces 768-dim output to match existing Supabase DB vectors.
EMBEDDING_MODEL = "gemini-embedding-001"

# ── OpenRouter (LLM generation) ──────────────────────────────
# Primary key + client
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found.")

openrouter_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Secondary key + client (separate quota pool)
openrouter_api_key_2 = os.getenv("OPENROUTER_API_KEY_2")
openrouter_client_2 = AsyncOpenAI(
    api_key=openrouter_api_key_2,
    base_url="https://openrouter.ai/api/v1",
) if openrouter_api_key_2 else None

def _build_groq_key_pool(*raw_values: str | None) -> list[str]:
    keys: list[str] = []
    for raw in raw_values:
        if not raw:
            continue
        for token in str(raw).split(","):
            key = token.strip()
            if key and key not in keys:
                keys.append(key)
    return keys


# Groq idea-generation slots with explicit key2/key3 support.
GROQ_IDEA_KEYS = _build_groq_key_pool(
    os.getenv("GROQ_IDEA_KEYS"),
    os.getenv("GROQ_IDEA_KEY_2"),
    os.getenv("GROQ_IDEA_KEY_3"),
    groq_api_key,
)
GROQ_IDEA_CLIENTS = [AsyncOpenAI(api_key=key, base_url="https://api.groq.com/openai/v1") for key in GROQ_IDEA_KEYS]

# Script-generation can use dedicated key pool; falls back to idea pool.
GROQ_SCRIPT_KEYS = _build_groq_key_pool(
    os.getenv("GROQ_SCRIPT_KEYS"),
    os.getenv("GROQ_SCRIPT_KEY_2"),
    os.getenv("GROQ_SCRIPT_KEY_3"),
)
if not GROQ_SCRIPT_KEYS:
    GROQ_SCRIPT_KEYS = GROQ_IDEA_KEYS
GROQ_SCRIPT_CLIENTS = [AsyncOpenAI(api_key=key, base_url="https://api.groq.com/openai/v1") for key in GROQ_SCRIPT_KEYS]

# Primary model + backup model
GENERATION_MODEL        = "google/gemma-3-27b-it:free"
GENERATION_MODEL_BACKUP = "google/gemma-3n-e4b-it:free"
GENERATION_MODEL_EXTRA  = "deepseek/deepseek-r1-0528-qwen3-8b:free"

print(
    f"Google GenAI (embeddings x{len(EMBED_CLIENTS)}), "
    f"Groq ideas x{len(GROQ_IDEA_CLIENTS)}, Groq scripts x{len(GROQ_SCRIPT_CLIENTS)}, "
    "and OpenRouter clients initialized successfully."
)

PROCESS_DB_MAX_BLOCKS = 3
PROCESS_WEB_MAX_BLOCKS = 3
PROCESS_CONTEXT_MAX_CHARS = 8000
PROCESS_TOPIC_TOKEN_BUDGET = 5200
PROCESS_TOPIC_SUMMARY_MAX_CHARS = 2400
SCRIPT_CONTEXT_MAX_CHARS = 12000
DB_LOOKUP_TIMEOUT_SEC = max(1, int(os.getenv("DB_LOOKUP_TIMEOUT_SEC", "12")))
SOCIAL_SCAN_TIMEOUT_SEC = max(1, int(os.getenv("SOCIAL_SCAN_TIMEOUT_SEC", "10")))
NEWS_SCAN_TIMEOUT_SEC = max(1, int(os.getenv("NEWS_SCAN_TIMEOUT_SEC", "10")))
DEEP_SCRAPE_DISCOVERY_TIMEOUT_SEC = max(1, int(os.getenv("DEEP_SCRAPE_DISCOVERY_TIMEOUT_SEC", "8")))
DEEP_SCRAPE_PER_URL_TIMEOUT_SEC = max(1, int(os.getenv("DEEP_SCRAPE_PER_URL_TIMEOUT_SEC", "10")))
DEEP_SCRAPE_TOTAL_TIMEOUT_SEC = max(1, int(os.getenv("DEEP_SCRAPE_TOTAL_TIMEOUT_SEC", "25")))
DEEP_SCRAPE_MAX_KEYWORDS = max(1, int(os.getenv("DEEP_SCRAPE_MAX_KEYWORDS", "2")))
DEEP_SCRAPE_MAX_RESULTS_PER_KEYWORD = max(1, int(os.getenv("DEEP_SCRAPE_MAX_RESULTS_PER_KEYWORD", "1")))
NEWS_SCRAPE_MAX_RESULTS = max(1, int(os.getenv("NEWS_SCRAPE_MAX_RESULTS", "1")))
HTTPX_SCRAPE_TIMEOUT_SEC = max(1, int(os.getenv("HTTPX_SCRAPE_TIMEOUT_SEC", "8")))
PLAYWRIGHT_GOTO_TIMEOUT_MS = max(1000, int(os.getenv("PLAYWRIGHT_GOTO_TIMEOUT_MS", "10000")))
PLAYWRIGHT_SELECTOR_TIMEOUT_MS = max(1000, int(os.getenv("PLAYWRIGHT_SELECTOR_TIMEOUT_MS", "5000")))
TSS_TIMEOUT_SEC = max(10, int(os.getenv("TSS_TIMEOUT_SEC", "180")))
PIPELINE_MAX_CONCURRENCY = max(1, int(os.getenv("PIPELINE_MAX_CONCURRENCY", "2")))
PROCESS_TOPIC_MAX_CONCURRENCY = max(1, int(os.getenv("PROCESS_TOPIC_MAX_CONCURRENCY", "2")))
PIPELINE_CACHE_TTL_SEC = max(0, int(os.getenv("PIPELINE_CACHE_TTL_SEC", "900")))
PROCESS_TOPIC_CACHE_TTL_SEC = max(0, int(os.getenv("PROCESS_TOPIC_CACHE_TTL_SEC", "1800")))
TOPIC_CACHE_MAX_ITEMS = max(10, int(os.getenv("TOPIC_CACHE_MAX_ITEMS", "300")))
IDEA_CACHE_TTL_HOURS = max(1, int(os.getenv("IDEA_CACHE_TTL_HOURS", "48")))


def _topic_cache_key(topic: str) -> str:
    return " ".join((topic or "").strip().lower().split())


def _cache_get(
    cache: OrderedDict[str, dict],
    key: str,
    ttl_seconds: int,
) -> dict | None:
    row = cache.get(key)
    if not row:
        return None
    if ttl_seconds <= 0:
        cache.pop(key, None)
        return None
    age = time.time() - float(row.get("ts", 0))
    if age > ttl_seconds:
        cache.pop(key, None)
        return None
    cache.move_to_end(key)
    return row.get("data")


def _cache_set(cache: OrderedDict[str, dict], key: str, data: dict) -> None:
    cache[key] = {"ts": time.time(), "data": data}
    cache.move_to_end(key)
    while len(cache) > TOPIC_CACHE_MAX_ITEMS:
        cache.popitem(last=False)


def _parse_utc_datetime(value: Any) -> datetime.datetime | None:
    if not value:
        return None
    if isinstance(value, datetime.datetime):
        candidate = value
    else:
        try:
            candidate = dt.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None
    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=datetime.timezone.utc)
    return candidate.astimezone(datetime.timezone.utc)


def _cache_age_hours(created_at: Any) -> float | None:
    candidate = _parse_utc_datetime(created_at)
    if candidate is None:
        return None
    return max((datetime.datetime.now(datetime.timezone.utc) - candidate).total_seconds() / 3600.0, 0.0)


async def _lookup_topic_cache_db(topic: str) -> dict[str, Any] | None:
    topic_key = _topic_cache_key(topic)
    try:
        response = await asyncio.to_thread(
            lambda: supabase.table("topic_content_cache")
            .select("topic_canonical,payload,created_at,updated_at,expires_at")
            .eq("topic_key", topic_key)
            .gt("expires_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
            .limit(1)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return None
        row = rows[0] or {}
        payload = dict(row.get("payload") or {})
        payload["served_from_cache"] = True
        payload["cache_age_hours"] = round(_cache_age_hours(row.get("created_at")) or 0.0, 3)
        payload["source_of_context"] = payload.get("source_of_context", "CACHE_DB")
        return payload
    except Exception as e:
        print(f"[warn] idea cache DB lookup failed for '{topic}': {e}")
        return None


async def _lookup_topic_cache_db_semantic(topic: str, cache_client: Any | None = None) -> dict[str, Any] | None:
    if cache_client is None:
        return None
    topic_vector = None
    try:
        topic_vector = TOPIC_CACHE._vector_from_client(topic, cache_client)
    except Exception:
        topic_vector = None
    if topic_vector is None:
        return None
    try:
        response = await asyncio.to_thread(
            lambda: supabase.rpc(
                "match_topic_content_cache",
                {
                    "query_embedding": topic_vector,
                    "match_threshold": 0.92,
                    "match_count": 1,
                },
            ).execute()
        )
        rows = response.data or []
        if not rows:
            return None
        row = rows[0] or {}
        payload = dict(row.get("payload") or {})
        payload["served_from_cache"] = True
        payload["cache_age_hours"] = round(_cache_age_hours(row.get("created_at")) or 0.0, 3)
        payload["cache_similarity"] = round(float(row.get("similarity") or 0.0), 3)
        payload["source_of_context"] = payload.get("source_of_context", "CACHE_DB")
        return payload
    except Exception as e:
        print(f"[warn] idea cache semantic DB lookup failed for '{topic}': {e}")
        return None


async def _store_topic_cache_db(topic: str, payload: dict[str, Any], cache_client: Any | None = None) -> None:
    topic_key = _topic_cache_key(topic)
    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=IDEA_CACHE_TTL_HOURS)
    row: dict[str, Any] = {
        "topic_key": topic_key,
        "topic_canonical": topic,
        "payload": payload,
        "expires_at": expires_at.isoformat(),
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        topic_vector = None
        if cache_client is not None:
            try:
                topic_vector = TOPIC_CACHE._vector_from_client(topic, cache_client)
            except Exception:
                topic_vector = None
        if topic_vector is not None:
            row["topic_vec"] = topic_vector
        await asyncio.to_thread(
            lambda: supabase.table("topic_content_cache")
            .upsert(row, on_conflict="topic_key")
            .execute()
        )
    except Exception as e:
        print(f"[warn] idea cache DB store failed for '{topic}': {e}")


async def _lookup_topic_cache(topic: str, cache_client: Any | None = None) -> dict[str, Any] | None:
    cached = TOPIC_CACHE.lookup(topic, cache_client)
    if cached:
        return cached
    db_semantic = await _lookup_topic_cache_db_semantic(topic, cache_client)
    if db_semantic:
        TOPIC_CACHE.store(topic, db_semantic, cache_client)
        return db_semantic
    db_cached = await _lookup_topic_cache_db(topic)
    if db_cached:
        TOPIC_CACHE.store(topic, db_cached, cache_client)
    return db_cached


async def _run_singleflight_cached(
    *,
    group: str,
    topic: str,
    ttl_seconds: int,
    compute_coro,
) -> tuple[dict, bool]:
    key = _topic_cache_key(topic)
    async with _request_state_lock:
        cache = _response_cache[group]
        cached = _cache_get(cache, key, ttl_seconds)
        if cached is not None:
            return cached, True
        task = _inflight_requests[group].get(key)
        if task is None:
            task = asyncio.create_task(compute_coro())
            _inflight_requests[group][key] = task
            task_owner = True
        else:
            task_owner = False

    try:
        result = await task
        if task_owner and ttl_seconds > 0:
            async with _request_state_lock:
                _cache_set(_response_cache[group], key, result)
        return result, False
    finally:
        async with _request_state_lock:
            current = _inflight_requests[group].get(key)
            if current is task:
                _inflight_requests[group].pop(key, None)


def _estimate_tokens(text: str) -> int:
    # Rough heuristic for English prompts; good enough for routing decisions.
    return max(1, len(text) // 4)


def _cap_blocks(blocks: list[str], max_blocks: int, max_chars: int) -> str:
    selected = [b.strip() for b in blocks if b and b.strip()][:max_blocks]
    merged = "\n\n".join(selected)
    if len(merged) > max_chars:
        merged = merged[:max_chars]
    return merged


async def _summarize_context_for_ideas(topic: str, db_context: str, web_context: str) -> tuple[str, str]:
    context_blob = (
        f"FOUNDATIONAL KNOWLEDGE:\n{db_context}\n\n"
        f"LATEST NEWS:\n{web_context}\n"
    )
    summarize_prompt = f"""
    Summarize the research context for YouTube ideation on "{topic}".
    Keep only the most important facts, conflicts, timelines, and hooks.
    Return plain text with two sections:
    1) FOUNDATIONAL KNOWLEDGE
    2) LATEST NEWS
    Hard limit: 1200 words total.
    """
    try:
        summary = await openrouter_generate(
            [
                {"role": "user", "content": summarize_prompt},
                {"role": "user", "content": context_blob[:16000]},
            ]
        )
    except Exception as exc:
        print(f"Context summarization failed, using hard-trim fallback: {exc}")
        return (
            db_context[:PROCESS_TOPIC_SUMMARY_MAX_CHARS // 2],
            web_context[:PROCESS_TOPIC_SUMMARY_MAX_CHARS // 2],
        )

    text = summary.strip()
    if not text:
        return (
            db_context[:PROCESS_TOPIC_SUMMARY_MAX_CHARS // 2],
            web_context[:PROCESS_TOPIC_SUMMARY_MAX_CHARS // 2],
        )

    # Keep a single compact summary in web section if structure is not parseable.
    if "LATEST NEWS" not in text and "FOUNDATIONAL KNOWLEDGE" not in text:
        return ("", text[:PROCESS_TOPIC_SUMMARY_MAX_CHARS])
    return ("", text[:PROCESS_TOPIC_SUMMARY_MAX_CHARS])


def _extract_retry_delay_seconds(error_text: str) -> float | None:
    if not error_text:
        return None
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", error_text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _is_daily_quota_error(error_text: str) -> bool:
    t = (error_text or "").lower()
    return "resource_exhausted" in t and "perday" in t


def _embed_with_failover(
    *,
    contents: str | list[str],
    task_type: str,
    output_dimensionality: int = 768,
):
    last_exc: Exception | None = None
    for idx, client in enumerate(EMBED_CLIENTS, start=1):
        try:
            return client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config=genai_types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=output_dimensionality,
                ),
            )
        except Exception as exc:
            last_exc = exc
            print(f"Embedding client slot {idx} failed: {exc}")
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("No embedding client available.")


async def _embed_chunks_with_backoff(chunks: list[str]) -> list[list[float]] | None:
    if not chunks:
        return []
    batch_size = 20
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        attempt = 0
        while attempt < 4:
            attempt += 1
            try:
                embed_response = _embed_with_failover(
                    contents=batch,
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768,
                )
                vectors.extend([e.values for e in embed_response.embeddings])
                break
            except Exception as exc:
                err = str(exc)
                if _is_daily_quota_error(err):
                    print("BACKGROUND TASK: Gemini daily embedding quota exhausted; skipping ingest.")
                    return None
                delay = _extract_retry_delay_seconds(err)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    wait = delay if delay is not None else min(2 ** attempt + random.random(), 30)
                    print(f"BACKGROUND TASK: Embedding throttled. Retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)
                    continue
                print(f"BACKGROUND TASK: Embedding failed with non-retryable error: {exc}")
                return None
        else:
            print("BACKGROUND TASK: Embedding retries exhausted for one batch.")
            return None
    return vectors


async def openrouter_generate(messages: list) -> str:
    """
    6-slot fallback chain across 2 keys × 3 models.
    Order: primary key (3 models) → secondary key (3 models).
    Each slot gets 1 retry on 429 before moving to next slot.
    Skip 404 slots immediately — no retry on missing models.
    """
    slots = [
        (openrouter_client, GENERATION_MODEL),
        (openrouter_client, GENERATION_MODEL_BACKUP),
        (openrouter_client, GENERATION_MODEL_EXTRA),
    ]
    if openrouter_client_2:
        slots += [
            (openrouter_client_2, GENERATION_MODEL),
            (openrouter_client_2, GENERATION_MODEL_BACKUP),
            (openrouter_client_2, GENERATION_MODEL_EXTRA),
        ]

    last_error = None
    for client, model in slots:
        for attempt in range(2):
            try:
                completion = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return completion.choices[0].message.content
            except Exception as e:
                last_error = e
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate" in err_str.lower()
                is_not_found  = "404" in err_str or "No endpoints" in err_str or "No allowed providers" in err_str
                if is_not_found:
                    print(f"OpenRouter 404 on {model} — skipping slot...")
                    break  # don't retry 404s
                elif is_rate_limit and attempt == 0:
                    wait = 1
                    print(f"OpenRouter 429 on {model} — retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"OpenRouter failed on {model}: {e} — trying next slot...")
                    break

    raise Exception(f"All OpenRouter slots exhausted. Last error: {last_error}")

async def _groq_generate_with_slots(
    messages: list,
    clients: list[AsyncOpenAI],
    model: str,
    label: str,
) -> str:
    last_error = None
    for slot_idx, client in enumerate(clients, start=1):
        for attempt in range(2):
            try:
                completion = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return completion.choices[0].message.content
            except Exception as exc:
                last_error = exc
                err_text = str(exc).lower()
                is_rate_limit = "429" in err_text or "rate" in err_text
                if is_rate_limit and attempt == 0:
                    await asyncio.sleep(1)
                    continue
                print(f"Groq {label} slot {slot_idx} failed on attempt {attempt + 1}: {exc}")
                break
    raise Exception(f"All Groq {label} slots exhausted. Last error: {last_error}")


async def groq_idea_generate(messages: list, model: str = GROQ_GENERATION_MODEL) -> str:
    return await _groq_generate_with_slots(messages, GROQ_IDEA_CLIENTS, model, "idea")


async def groq_script_generate(messages: list, model: str = GROQ_SCRIPT_MODEL) -> str:
    return await _groq_generate_with_slots(messages, GROQ_SCRIPT_CLIENTS, model, "script")


async def generate_script_content(messages: list) -> str:
    try:
        return await groq_script_generate(messages)
    except Exception as exc:
        print(f"Groq script generation failed, falling back to OpenRouter: {exc}")
        return await openrouter_generate(messages)

# ── Supabase ─────────────────────────────────────────────────
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("Supabase credentials not found in .env file")
supabase: Client = create_client(supabase_url, supabase_key)
print("Supabase client initialized.")


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = 250, chunk_overlap: int = 50) -> list[str]:
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        # Fallback for cold-start environments where NLTK data is still being provisioned.
        sentences = re.split(r'(?<=[.!?])\s+', text or "")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            overlap_words = current_chunk.split()[-chunk_overlap:]
            current_chunk = " ".join(overlap_words) + " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def add_scraped_data_to_db(
    article_title: str,
    article_text: str,
    article_url: str,
    category: str = "",
    topic: str = "",
    tags: list | None = None,
):
    """
    Background task: chunk → embed → insert to Supabase.

    Metadata shape (web_scrape):
      {
        "category":   str   — from the user's topic request context (empty if unknown)
        "topic":      str   — the user's searched topic
        "tags":       list  — tags inferred from topic context
        "domain":     str   — netloc extracted from article_url
        "scraped_at": str   — ISO timestamp of when this was scraped
        "author":     {
          "has_credentials": bool,
          "name":            str or null,   — publication/author name from URL/title
          "description":     str or null,   — short label e.g. "News publication"
        }
      }
    """
    if tags is None:
        tags = []

    print(f"BACKGROUND TASK: Starting upload for '{article_title[:30]}...'")
    try:
        raw_chunks = chunk_text(article_text)
        chunks = [c for c in raw_chunks if c and not c.isspace()]
        if not chunks:
            print("BACKGROUND TASK: No valid chunks.")
            return

        embeddings = asyncio.run(_embed_chunks_with_backoff(chunks))
        if embeddings is None:
            return

        domain = urlparse(article_url).netloc.lstrip('www.') if article_url else ""
        scraped_at = dt.now().isoformat()

        # Author/publication credentials — derived from domain
        has_credentials = bool(domain)
        author_info = {
            "has_credentials": has_credentials,
            "name": domain if has_credentials else None,
            "description": "Web publication" if has_credentials else None,
        }

        documents_to_insert = [
            {
                "content":      chunk,
                "embedding":    embeddings[i],
                "source_title": article_title,
                "source_url":   article_url,
                "source_type":  "web_scrape",
                "metadata": {
                    "category":   category,
                    "topic":      topic,
                    "tags":       tags,
                    "domain":     domain,
                    "scraped_at": scraped_at,
                    "author":     author_info,
                },
            }
            for i, chunk in enumerate(chunks)
        ]
        supabase.table('documents').insert(documents_to_insert).execute()
        print(f"BACKGROUND TASK: Uploaded {len(documents_to_insert)} chunks.")
    except Exception as e:
        print(f"BACKGROUND TASK: Failed. Error: {e}")


# Max 3 concurrent Playwright browsers — each uses ~300MB RAM
# Prevents memory exhaustion when multiple requests arrive simultaneously
_playwright_semaphore = asyncio.Semaphore(3)
_pipeline_request_semaphore = asyncio.Semaphore(PIPELINE_MAX_CONCURRENCY)
_process_topic_request_semaphore = asyncio.Semaphore(PROCESS_TOPIC_MAX_CONCURRENCY)
_request_state_lock = asyncio.Lock()
_response_cache = {
    "pipeline_metrics": OrderedDict(),
    "process_topic": OrderedDict(),
}
_inflight_requests = {
    "pipeline_metrics": {},
    "process_topic": {},
}


def _extract_text_from_html(html: str) -> tuple[str, str]:
    """Extract title and clean text from raw HTML using readability + BeautifulSoup."""
    doc = Document(html)
    title = doc.title()
    soup = BeautifulSoup(doc.summary(), 'html.parser')
    text = soup.get_text(separator='\n', strip=True)
    return title, text


async def _scrape_with_httpx(url: str) -> tuple[str, str] | None:
    """
    Tier 1: httpx with full browser headers.
    Fast. Works on most sites. Fails on Cloudflare, Wikipedia, paywalled sites.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=HTTPX_SCRAPE_TIMEOUT_SEC) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return _extract_text_from_html(response.text)


async def _scrape_with_playwright(url: str) -> tuple[str, str] | None:
    """
    Tier 2: Real headless Chromium via Playwright.
    Defeats TLS fingerprinting, JS challenges, and most bot detection.
    Runs in a thread pool executor so it never blocks the async event loop.
    Requires: pip install playwright && playwright install chromium
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("  [Playwright] Not installed. Run: pip install playwright && playwright install chromium")
        return None

    async def _run():
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
                locale="en-US",
                timezone_id="America/New_York",
                java_script_enabled=True,
            )
            await context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_GOTO_TIMEOUT_MS)
                await page.wait_for_selector("body", timeout=PLAYWRIGHT_SELECTOR_TIMEOUT_MS)
                html = await page.content()
                return _extract_text_from_html(html)
            finally:
                await browser.close()

    # Semaphore limits concurrent Playwright instances — prevents RAM exhaustion in prod
    async with _playwright_semaphore:
        loop = asyncio.get_running_loop()
        # Run in thread pool so heavy browser process never blocks the event loop
        return await loop.run_in_executor(None, lambda: asyncio.run(_run()))


# Domains that consistently return junk, non-English content, or block scrapers
_SCRAPE_BLOCKLIST = {
    'zhidao.baidu.com', 'baidu.com', 'en.cppreference.com', 'cppreference.com',
    'stackoverflow.com', 'github.com', 'reddit.com', 'twitter.com', 'x.com',
    'instagram.com', 'facebook.com', 'linkedin.com', 'pinterest.com',
    'researchgate.net', 'academia.edu', 'jstor.org',
}


async def scrape_url(
    client: httpx.AsyncClient,   # kept for signature compatibility
    url: str,
    scraped_urls: set,
    snippet: str = "",           # Tier 3 fallback text from search result
) -> dict | None:
    """
    3-tier scraping with automatic fallback:
      Tier 1 → httpx (fast)
      Tier 2 → Playwright headless Chrome (robust, defeats bot detection)
      Tier 3 → Use DDGS snippet directly (always works, less text)
    """
    if url in scraped_urls:
        return None
    # Skip known junk/non-English/blocked domains immediately
    domain = urlparse(url).netloc.lstrip('www.')
    if any(domain == b or domain.endswith('.' + b) for b in _SCRAPE_BLOCKLIST):
        print(f"  ⊘ Skipped blocklisted domain: {domain}")
        return None
    print(f"Scraping: {url}")

    # ── Tier 1: httpx ────────────────────────────────────────
    try:
        title, text = await _scrape_with_httpx(url)
        if text and len(text) > 200:
            scraped_urls.add(url)
            print(f"  ✓ Tier 1 (httpx) succeeded: {url[:60]}")
            return {"url": url, "title": title, "text": text}
    except Exception as e:
        print(f"  ✗ Tier 1 (httpx) failed: {e} — trying Playwright...")

    # ── Tier 2: Playwright ───────────────────────────────────
    try:
        result = await _scrape_with_playwright(url)
        if result:
            title, text = result
            if text and len(text) > 200:
                scraped_urls.add(url)
                print(f"  ✓ Tier 2 (Playwright) succeeded: {url[:60]}")
                return {"url": url, "title": title, "text": text}
    except Exception as e:
        print(f"  ✗ Tier 2 (Playwright) failed: {e} — using snippet fallback...")

    # ── Tier 3: Search snippet fallback ─────────────────────
    if snippet and len(snippet) > 50:
        print(f"  ✓ Tier 3 (snippet fallback) used for: {url[:60]}")
        scraped_urls.add(url)
        return {"url": url, "title": url, "text": snippet}

    print(f"  ✗ All tiers failed for: {url[:60]}")
    return None


async def deep_search_and_scrape(keywords: list[str], scraped_urls: set) -> list[dict]:
    print("--- DEEP WEB SCRAPE: Starting full search... ---")
    urls_to_scrape = []   # list of (url, snippet) tuples

    def _discover_candidates() -> list[tuple[str, str]]:
        discovered: list[tuple[str, str]] = []
        with DDGS(timeout=8) as ddgs:
            for keyword in keywords[:DEEP_SCRAPE_MAX_KEYWORDS]:
                results = list(ddgs.text(keyword, region='wt-wt', max_results=DEEP_SCRAPE_MAX_RESULTS_PER_KEYWORD))
                if results:
                    # Take top result; keep snippet for Tier 3 fallback
                    top = results[0]
                    discovered.append((top['href'], top.get('body', '')))
        return discovered

    try:
        urls_to_scrape = await asyncio.wait_for(
            asyncio.to_thread(_discover_candidates),
            timeout=DEEP_SCRAPE_DISCOVERY_TIMEOUT_SEC,
        )
    except Exception as e:
        print(f"--- DEEP WEB SCRAPE: Discovery failed fast: {e} ---")
        return []

    # Deduplicate by URL
    seen = set()
    unique = []
    for url, snippet in urls_to_scrape:
        if url not in seen:
            seen.add(url)
            unique.append((url, snippet))

    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.wait_for(
                scrape_url(client, url, scraped_urls, snippet),
                timeout=DEEP_SCRAPE_PER_URL_TIMEOUT_SEC,
            )
            for url, snippet in unique
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        cleaned = []
        for result in results:
            if isinstance(result, Exception):
                continue
            if result and result.get("text"):
                cleaned.append(result)
        return cleaned


async def _generate_search_keywords(topic: str) -> list[str]:
    """Generate 3 focused, English-language news search keywords using Groq. ~0.5s."""
    keyword_prompt = f"""
    Your ONLY task is to generate 3 diverse search engine keyword phrases for: '{topic}'.
    Rules:
    1. Return ONLY the 3 phrases.
    2. NO numbers, markdown, or introductory text.
    3. Each phrase on a new line.

    EXAMPLE INPUT: Is coding dead?
    EXAMPLE OUTPUT:
    future of programming jobs automation
    AI replacing software developers
    demand for software engineers 2025
    """
    chat_completion = await groq_client.chat.completions.create(
        messages=[{"role": "user", "content": keyword_prompt}],
        model=GROQ_GENERATION_MODEL,
    )
    raw_text = chat_completion.choices[0].message.content
    keywords_in_quotes = re.findall(r'"(.*?)"', raw_text)
    keywords = keywords_in_quotes if keywords_in_quotes else [
        kw.strip() for kw in raw_text.strip().split('\n') if kw.strip()
    ]
    return keywords[:3]


async def get_latest_news_context(topic: str, scraped_urls: set) -> list[dict]:
    print("--- LIGHT WEB SCRAPE: Starting lightweight news search... ---")
    try:
        keyword = f"{topic} latest news today"
        url_snippet_pairs = []
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.text(keyword, region='wt-wt', max_results=NEWS_SCRAPE_MAX_RESULTS))
            for r in results:
                url_snippet_pairs.append((r['href'], r.get('body', '')))
        async with httpx.AsyncClient() as client:
            tasks = [
                asyncio.wait_for(
                    scrape_url(client, url, scraped_urls, snippet),
                    timeout=DEEP_SCRAPE_PER_URL_TIMEOUT_SEC,
                )
                for url, snippet in url_snippet_pairs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [
                r for r in results
                if not isinstance(r, Exception) and r and r.get("text")
            ]
    except Exception as e:
        print(f"--- WEB TASK: Error during news scraping: {e} ---")
        return []


async def get_db_context(topic: str) -> list[dict]:
    """
    Two-stage DB lookup for a topic:

    Stage 1 — HyDE vector search (semantic):
      Groq generates a hypothetical encyclopedia paragraph for the topic,
      we embed it and run cosine similarity against all stored chunks.
      Threshold: 0.55 (relaxed so near-miss topics like "Israel Iran war"
      still match chunks stored as "Iran-Israel conflict" etc.)

    Stage 2 — Full-text keyword fallback:
      If vector search returns fewer than 3 results, we also run
      search_documents() (the tsvector RPC) using the raw topic string.
      Results are merged and de-duplicated by id.

    Returns up to 8 combined results.
    """
    print("--- DB TASK: Starting two-stage DB search... ---")
    combined: dict[int, dict] = {}  # id → row, de-duplicated

    try:
        loop = asyncio.get_running_loop()

        # ── Stage 1: HyDE vector search ──────────────────────
        hyde_prompt = f"""
        Write a short, factual, encyclopedia-style paragraph that provides a direct answer
        to the following topic. Be concise and include key terms.

        Topic: "{topic}"
        """
        chat_completion = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": hyde_prompt}],
            model=GROQ_GENERATION_MODEL,
        )
        hypothetical_document = chat_completion.choices[0].message.content
        print(f"--- DB TASK: HyDE doc: {hypothetical_document[:100]}...")

        embed_response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: _embed_with_failover(
                    contents=hypothetical_document,
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768,
                ),
            ),
            timeout=DB_LOOKUP_TIMEOUT_SEC,
        )
        query_embedding = embed_response.embeddings[0].values

        try:
            vector_response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: supabase.rpc(
                        'match_documents',
                        {
                            'query_embedding':  query_embedding,
                            'match_threshold':  0.55,   # relaxed: was 0.65
                            'match_count':      8,
                        }
                    ).execute()
                ),
                timeout=DB_LOOKUP_TIMEOUT_SEC,
            )
            vector_results = vector_response.data or []
            for row in vector_results:
                combined[row['id']] = row
            print(f"--- DB TASK: Vector search → {len(vector_results)} results ---")
        except asyncio.TimeoutError:
            print(f"--- DB TASK: Vector search timed out after {DB_LOOKUP_TIMEOUT_SEC}s ---")
            vector_results = []

        # ── Stage 2: Keyword fallback if vector was thin ──────
        if len(combined) < 3:
            print(f"--- DB TASK: Thin vector results ({len(combined)}), running keyword search... ---")
            try:
                keyword_response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: supabase.rpc(
                            'search_documents',
                            {
                                'search_query': topic,
                                'match_count':  8,
                            }
                        ).execute()
                    ),
                    timeout=DB_LOOKUP_TIMEOUT_SEC,
                )
                keyword_results = keyword_response.data or []
                for row in keyword_results:
                    if row['id'] not in combined:
                        combined[row['id']] = row
                print(f"--- DB TASK: Keyword search → {len(keyword_results)} results, total unique: {len(combined)} ---")
            except asyncio.TimeoutError:
                print(f"--- DB TASK: Keyword search timed out after {DB_LOOKUP_TIMEOUT_SEC}s ---")

    except Exception as e:
        print(f"--- DB TASK: Error: {e} ---")
        return []

    results = list(combined.values())
    print(f"--- DB TASK: Returning {len(results)} total DB docs ---")
    return results


# ════════════════════════════════════════════════════════════
# SCRIPT STRUCTURE OPTIONS
# ════════════════════════════════════════════════════════════

STRUCTURE_GUIDANCE = {
    "problem_solution": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (~10%)
    - Problem / Conflict (~15%)
    - Evidence & Data (~20%)
    - Real-world Examples (~25%)
    - Potential Solutions / Insights (~25%)
    - Call to Action (~5%)
    """,
    "storytelling": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (Introduce Ordinary World) (~10%)
    - Call to Adventure / Inciting Incident (~10%)
    - Trials & Tribulations (Rising Action, using examples/data) (~50%)
    - Climax / Resolution (~20%)
    - Reflection & Takeaway (Call to Action) (~10%)
    """,
    "listicle": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (State the list topic & number) (~10%)
    - Item 1 (~15-20%) / Item 2 (~15-20%) / Item 3 (~15-20%) / Item X (~15-20%)
    - (Optional) Bonus Item / Honorable Mentions (~10%)
    - Conclusion & Call to Action (~10%)
    """,
    "chronological": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (~10%)
    - Early Beginnings / Origins (~20%)
    - Key Developments / Turning Points (~40%)
    - Later Stages / Modern Impact (~20%)
    - Conclusion & Reflection (~10%)
    """,
    "myth_debunking": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (Introduce common misconception) (~10%)
    - Myth 1 & Fact 1 (~25%) / Myth 2 & Fact 2 (~25%) / Myth 3 & Fact 3 (~25%)
    - Conclusion & Call to Action (~15%)
    """,
    "tech_review": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (~10%)
    - Design & Build Quality (~15%)
    - Key Features & Specs (~20%)
    - Performance & User Experience (~30%)
    - Pros & Cons (~10%)
    - Verdict & Recommendation (~15%)
    """,
}


# ════════════════════════════════════════════════════════════
# FASTAPI APP
# ════════════════════════════════════════════════════════════

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://www.storybit.tech",
    "https://storybit.tech",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ──────────────────────────────────────────

class PromptRequest(BaseModel):
    topic: str


class ScriptRequest(BaseModel):
    topic: str
    emotional_tone: str | None = "engaging"
    creator_type: str | None = "educator"
    audience_description: str | None = "a general audience interested in learning"
    accent: str | None = "neutral"
    duration_minutes: int | None = 10
    script_structure: str | None = "problem_solution"


class CreateOrderRequest(BaseModel):
    amount: int
    currency: str = "INR"
    receipt: str | None = None
    target_tier: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class GenerateIdeasRequest(BaseModel):
    topic: str
    max_angles: int | None = 5
    ideas_per_angle: int | None = 3
    used_angle_ids: list[str] | None = None
    force_refresh: bool | None = False


async def _safe_scan_topic_signals(
    *,
    label: str,
    scanner: Any,
    topic: str,
    timeout_sec: int,
    fallback_key: str,
) -> dict[str, Any]:
    try:
        payload = await asyncio.wait_for(asyncio.to_thread(scanner, topic), timeout=timeout_sec)
        return payload if isinstance(payload, dict) else {fallback_key: []}
    except Exception as exc:
        print(f"[warn] {label} scan failed ({exc}); falling back to empty signal")
        return {fallback_key: []}


# ════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════

@app.get("/")
async def read_root():
    return {"status": "Welcome"}

@app.post("/token")
async def token(form_data: OAuth2PasswordRequestForm = Depends()):
    return await login_user(form_data)


@app.post("/refresh-token")
async def refresh_token(request: RefreshTokenRequest):
    return await refresh_access_token(request.refresh_token)


@app.post("/pipeline-metrics")
async def pipeline_metrics(request: PromptRequest):
    """
    Frontend integration endpoint for the new scoring stack.
    Returns the full run_tss payload:
      TSS + CSI + CAGS + verdict
    """
    try:
        async def _compute() -> dict:
            async with _pipeline_request_semaphore:
                result = await asyncio.wait_for(run_tss(request.topic), timeout=TSS_TIMEOUT_SEC)
                return adapt_pipeline_payload(result)

        payload, cache_hit = await _run_singleflight_cached(
            group="pipeline_metrics",
            topic=request.topic,
            ttl_seconds=PIPELINE_CACHE_TTL_SEC,
            compute_coro=_compute,
        )
        if cache_hit:
            print(f"pipeline-metrics cache hit for topic: {request.topic}")
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline metrics failed: {e}")


# ── /process-topic ───────────────────────────────────────────

@app.post("/process-topic")
async def process_topic(request: PromptRequest, background_tasks: BackgroundTasks):
    total_start_time = time.time()
    print(f"Received topic: {request.topic}")

    async def _compute() -> dict:
        async with _process_topic_request_semaphore:
            scraped_urls: set = set()
            source_of_context = ""
            base_keywords: list[str] = []

            print("--- Phase 1: DB lookup + keyword gen in parallel ---")
            db_results, base_keywords = await asyncio.gather(
                get_db_context(request.topic),
                _generate_search_keywords(request.topic),
            )
            print(f"--- Phase 1 done: {len(db_results)} DB docs, keywords: {base_keywords} ---")

            db_count = len(db_results)
            if db_count >= 5:
                source_of_context = "DATABASE_RICH"
                print(f"--- DB RICH ({db_count} docs): Light news scrape for freshness. ---")
                new_articles = await get_latest_news_context(request.topic, scraped_urls)
            elif db_count >= 1:
                source_of_context = "DATABASE_PARTIAL"
                print(f"--- DB PARTIAL ({db_count} docs): Deep scrape to supplement. ---")
                new_articles = await deep_search_and_scrape(base_keywords, scraped_urls)
            else:
                source_of_context = "DEEP_SCRAPE"
                print(f"--- DB MISS: Full deep scrape with keywords: {base_keywords} ---")
                new_articles = await deep_search_and_scrape(base_keywords, scraped_urls)

            db_context, web_context = "", ""
            source_urls = []

            if db_results:
                db_blocks = [item.get('content', '') for item in db_results]
                db_context = _cap_blocks(db_blocks, PROCESS_DB_MAX_BLOCKS, PROCESS_CONTEXT_MAX_CHARS // 2)
                source_urls.extend(list(set([
                    item['source_url'] for item in db_results if item.get('source_url')
                ])))

            if new_articles:
                web_blocks = [
                    f"Source: {art['title']}\n{art['text']}" for art in new_articles
                ]
                web_context = _cap_blocks(web_blocks, PROCESS_WEB_MAX_BLOCKS, PROCESS_CONTEXT_MAX_CHARS // 2)
                source_urls.extend([art['url'] for art in new_articles])
                for article in new_articles:
                    background_tasks.add_task(
                        add_scraped_data_to_db,
                        article['title'], article['text'], article['url'],
                        "",
                        request.topic,
                        base_keywords,
                    )

            if not db_context and not web_context:
                return {"error": "Could not find any information."}

            merged_context = f"{db_context}\n\n{web_context}"
            if _estimate_tokens(merged_context) > PROCESS_TOPIC_TOKEN_BUDGET:
                print("Process-topic context is large; summarizing before final idea generation.")
                db_context, web_context = await _summarize_context_for_ideas(
                    request.topic, db_context, web_context
                )
            print("--- Step 3 (Final Idea Gen) using CAGS pipeline ---")
            tss_payload = await asyncio.wait_for(run_tss(request.topic), timeout=TSS_TIMEOUT_SEC)
            cags_payload = tss_payload.get("cags") or {}
            gap_angles = cags_payload.get("gap_angles") or []
            briefs = cags_payload.get("briefs") or []
            perspective_tree = cags_payload.get("perspective_tree") or []
            if not gap_angles or not perspective_tree:
                return {"error": "Could not produce CAGS-aligned ideas."}

            social_payload, news_payload = await asyncio.gather(
                _safe_scan_topic_signals(
                    label="social",
                    scanner=scan_social_topic,
                    topic=request.topic,
                    timeout_sec=SOCIAL_SCAN_TIMEOUT_SEC,
                    fallback_key="sample_posts",
                ),
                _safe_scan_topic_signals(
                    label="news",
                    scanner=scan_news_topic,
                    topic=request.topic,
                    timeout_sec=NEWS_SCAN_TIMEOUT_SEC,
                    fallback_key="sample_articles",
                ),
            )
            social_data = social_payload.get("sample_posts") or []
            news_data = news_payload.get("sample_articles") or []

            idea_payload = await generate_cags_aligned_ideas(
                topic=request.topic,
                gap_angles=gap_angles,
                briefs=briefs,
                perspective_tree=perspective_tree,
                social_data=social_data,
                news_data=news_data,
                db_context=db_context,
                web_context=web_context,
                max_angles=4,
                ideas_per_angle=3,
                used_angle_ids=[],
                groq_client=groq_client,
                gemini_client=EMBED_CLIENTS[0] if EMBED_CLIENTS else None,
                cache_lookup=TOPIC_CACHE.lookup,
                cache_store=None,
            )

            final_ideas = []
            final_descriptions = []
            for cluster in idea_payload.get("idea_clusters") or []:
                for variant in cluster.get("idea_variants") or []:
                    title = str(variant.get("title") or "").strip()
                    description = str(variant.get("description") or "").strip()
                    if title and description:
                        final_ideas.append(title)
                        final_descriptions.append(description)

            idea_payload["source_of_context"] = source_of_context
            idea_payload["generated_keywords"] = base_keywords
            idea_payload["source_urls"] = list(set(source_urls))
            idea_payload["scraped_text_context"] = f"DB CONTEXT:\n{db_context}\n\nWEB CONTEXT:\n{web_context}"
            idea_payload["ideas"] = final_ideas
            idea_payload["descriptions"] = final_descriptions
            idea_payload["total_request_time_sec"] = round(time.time() - total_start_time, 2)
            return idea_payload

    try:
        payload, cache_hit = await _run_singleflight_cached(
            group="process_topic",
            topic=request.topic,
            ttl_seconds=PROCESS_TOPIC_CACHE_TTL_SEC,
            compute_coro=_compute,
        )
        if cache_hit:
            print(f"process-topic cache hit for topic: {request.topic}")
        print(f"Total request time: {time.time() - total_start_time:.2f}s")
        return payload
    except Exception as e:
        print(f"Error in /process-topic: {e}")
        return {"error": "An error occurred in the processing pipeline."}


@app.post("/generate-ideas")
async def generate_ideas_endpoint(
    request: GenerateIdeasRequest,
    background_tasks: BackgroundTasks,
):
    """
    CAGS-aligned idea generation endpoint.
    Uses the TSS/CAGS output as the gap source and applies the new
    idea-cluster pipeline with diversity, depth checks, and cache support.
    """
    total_start_time = time.time()
    topic = request.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic must be a non-empty string")

    cache_client = EMBED_CLIENTS[0] if EMBED_CLIENTS else None
    if not request.force_refresh:
        cached = await _lookup_topic_cache(topic, cache_client)
        if cached:
            cached["source_of_context"] = cached.get("source_of_context", "CACHE")
            cached["served_from_cache"] = True
            return cached

    try:
        print(f"GENERATE IDEAS for topic: {topic}")

        # Use current TSS/CAGS as the authoritative gap source.
        tss_payload = await asyncio.wait_for(run_tss(topic), timeout=TSS_TIMEOUT_SEC)
        cags_payload = tss_payload.get("cags") or {}
        gap_angles = cags_payload.get("gap_angles") or []
        briefs = cags_payload.get("briefs") or []
        perspective_tree = cags_payload.get("perspective_tree") or []
        if not gap_angles or not perspective_tree:
            raise HTTPException(status_code=422, detail="No viable CAGS angles were produced.")

        # Research context used by the depth check and prompt seeding.
        print("--- Idea Gen: DB lookup + keyword gen in parallel ---")
        db_results, base_keywords = await asyncio.gather(
            get_db_context(topic),
            _generate_search_keywords(topic),
        )
        scraped_urls: set = set()
        db_count = len(db_results)
        if db_count >= 5:
            source_of_context = "DATABASE_RICH"
            new_articles = await get_latest_news_context(topic, scraped_urls)
        elif db_count >= 1:
            source_of_context = "DATABASE_PARTIAL"
            new_articles = await deep_search_and_scrape(base_keywords, scraped_urls)
        else:
            source_of_context = "DEEP_SCRAPE"
            new_articles = await deep_search_and_scrape(base_keywords, scraped_urls)

        db_context, web_context = "", ""
        source_urls: list[str] = []
        if db_results:
            db_blocks = [item.get("content", "") for item in db_results]
            db_context = _cap_blocks(db_blocks, PROCESS_DB_MAX_BLOCKS, PROCESS_CONTEXT_MAX_CHARS // 2)
            source_urls.extend(list(set([item["source_url"] for item in db_results if item.get("source_url")])))
        if new_articles:
            web_blocks = [f"Source: {art['title']}\n{art['text']}" for art in new_articles]
            web_context = _cap_blocks(web_blocks, PROCESS_WEB_MAX_BLOCKS, PROCESS_CONTEXT_MAX_CHARS // 2)
            source_urls.extend([art["url"] for art in new_articles])
            for article in new_articles:
                background_tasks.add_task(
                    add_scraped_data_to_db,
                    article["title"],
                    article["text"],
                    article["url"],
                    "",
                    topic,
                    base_keywords,
                )

        social_payload, news_payload = await asyncio.gather(
            _safe_scan_topic_signals(
                label="social",
                scanner=scan_social_topic,
                topic=topic,
                timeout_sec=SOCIAL_SCAN_TIMEOUT_SEC,
                fallback_key="sample_posts",
            ),
            _safe_scan_topic_signals(
                label="news",
                scanner=scan_news_topic,
                topic=topic,
                timeout_sec=NEWS_SCAN_TIMEOUT_SEC,
                fallback_key="sample_articles",
            ),
        )
        social_data = social_payload.get("sample_posts") or []
        news_data = news_payload.get("sample_articles") or []

        idea_clusters = await generate_cags_aligned_ideas(
            topic=topic,
            gap_angles=gap_angles,
            briefs=briefs,
            perspective_tree=perspective_tree,
            social_data=social_data,
            news_data=news_data,
            db_context=db_context,
            web_context=web_context,
            max_angles=int(request.max_angles or 5),
            ideas_per_angle=int(request.ideas_per_angle or 3),
            used_angle_ids=request.used_angle_ids or [],
            groq_client=groq_client,
            gemini_client=EMBED_CLIENTS[0] if EMBED_CLIENTS else None,
            cache_lookup=TOPIC_CACHE.lookup,
            cache_store=None,
        )

        # Enrich response with the core research signals that produced it.
        idea_clusters["source_of_context"] = source_of_context
        idea_clusters["generated_keywords"] = base_keywords
        idea_clusters["source_urls"] = list(set(source_urls))
        idea_clusters["cags"] = {
            "tss": tss_payload.get("tss"),
            "csi": (tss_payload.get("csi") or {}).get("csi"),
            "total_angles": len(gap_angles),
        }
        idea_clusters["total_request_time_sec"] = round(time.time() - total_start_time, 2)
        TOPIC_CACHE.store(topic, idea_clusters, cache_client)
        background_tasks.add_task(_store_topic_cache_db, topic, idea_clusters, cache_client)

        print(f"Total /generate-ideas time: {idea_clusters['total_request_time_sec']:.2f}s")
        return idea_clusters
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /generate-ideas: {e}")
        return {"error": "An error occurred in the idea generation pipeline."}


# ── /generate-script ─────────────────────────────────────────

@app.post("/generate-script")
async def generate_script(
    request: ScriptRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    total_start_time = time.time()
    user_id = current_user.id
    print(f"SCRIPT GENERATION from user ({user_id}): '{request.topic}'")

    # ── Credit check ─────────────────────────────────────────
    IDEA_COST = 3
    try:
        profile_response = (
            supabase.table('profiles')
            .select('credits_remaining, user_tier')
            .eq('id', user_id)
            .single()
            .execute()
        )
        profile = profile_response.data
        if not profile:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="User profile not found.")

        credits = profile.get('credits_remaining', 0)
        user_tier = profile.get('user_tier', 'free')

        if user_tier != 'admin' and credits < IDEA_COST:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=(
                    f"Insufficient credits. This action requires {IDEA_COST} credit(s). "
                    f"You have {credits}."
                ),
            )
        print(f"User {user_id} (Tier: {user_tier}) has {credits} credits.")
    except APIError as e:
        raise HTTPException(status_code=500, detail=f"DB error checking profile: {e.message}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error checking credits: {e}")
        raise HTTPException(status_code=500, detail="Error checking profile.")
    # ─────────────────────────────────────────────────────────

    print(
        f"Personalization — Duration: {request.duration_minutes}min, "
        f"Tone: {request.emotional_tone}, Type: {request.creator_type}, "
        f"Audience: {request.audience_description}, Accent: {request.accent}"
    )

    try:
        # ── Phase 1: DB + keyword gen in parallel (no sleep) ─
        print("--- Phase 1: DB lookup + keyword gen in parallel ---")
        db_results, base_keywords = await asyncio.gather(
            get_db_context(request.topic),
            _generate_search_keywords(request.topic),
        )
        print(f"--- Phase 1 done: {len(db_results)} DB docs ---")

        scraped_urls: set = set()

        # ── Phase 2: Web scrape strategy based on DB quality ──
        db_count = len(db_results)
        if db_count >= 5:
            print(f"--- DB RICH ({db_count} docs): Light news scrape for freshness. ---")
            new_articles = await get_latest_news_context(request.topic, scraped_urls)
        elif db_count >= 1:
            print(f"--- DB PARTIAL ({db_count} docs): Deep scrape to supplement. ---")
            new_articles = await deep_search_and_scrape(base_keywords, scraped_urls)
        else:
            print(f"--- DB MISS: Full deep scrape with keywords: {base_keywords} ---")
            new_articles = await deep_search_and_scrape(base_keywords, scraped_urls)

        # ── Step 2: Merge context ─────────────────────────────
        db_context, web_context = "", ""
        if db_results:
            db_blocks = [item.get('content', '') for item in db_results]
            db_context = _cap_blocks(db_blocks, PROCESS_DB_MAX_BLOCKS, SCRIPT_CONTEXT_MAX_CHARS // 2)
        if new_articles:
            web_blocks = [
                f"Source: {art['title']}\n{art['text']}" for art in new_articles
            ]
            web_context = _cap_blocks(web_blocks, PROCESS_WEB_MAX_BLOCKS, SCRIPT_CONTEXT_MAX_CHARS // 2)
            for article in new_articles:
                background_tasks.add_task(
                    add_scraped_data_to_db,
                    article['title'], article['text'], article['url'],
                    "",               # category unknown at live-scrape time
                    request.topic,    # topic from user request
                    base_keywords,    # use generated keywords as tags proxy
                )

        if not db_context and not web_context:
            return {"error": "Could not find any research material to write the script."}

        # ── Step 3: Build & run script prompt ─────────────────
        WORDS_PER_MINUTE = 130
        target_duration = request.duration_minutes or 10
        target_word_count = target_duration * WORDS_PER_MINUTE

        requested_structure = request.script_structure or "problem_solution"
        structure_guidance_text = STRUCTURE_GUIDANCE.get(
            requested_structure, STRUCTURE_GUIDANCE["problem_solution"]
        )
        script_prompt = f"""
        You are a professional YouTube scriptwriter creating natural, engaging, conversational scripts.

        **Creator Profile:**
        * Creator Type: {request.creator_type}
        * Target Audience: {request.audience_description}
        * Desired Emotional Tone: {request.emotional_tone}
        * Accent/Dialect: {request.accent}

        **Task:**
        Generate a complete YouTube script of approximately **{target_duration} minutes**
        (~{target_word_count} words) based on the topic below, using the research context.

        **Script Style:**
        - Output only spoken dialogue — no section titles, stage directions, or metadata.
        - Speak directly to the viewer — friendly, confident, slightly spontaneous.
        - Short and medium sentences, natural pauses (…) or dashes, occasional repetition.
        - Include interjections, rhetorical questions, humor, brief asides.
        - Personal anecdotes or opinions ("I remember…", "When I tried this…").
        - Visual and emotional imagery ("Imagine this…", "Picture it like…").
        - Hook viewers emotionally in first 15-30 seconds.
        - Alternate between facts, insights, reactions, short reflections.
        - Inclusive language: "you guys", "we all", "my friends".
        - Natural pacing as if recording live.
        - Stay close to **{target_word_count} words** (±50).

        {structure_guidance_text}

        **Main Topic:** "{request.topic}"

        **Research:**
        FOUNDATIONAL KNOWLEDGE: {db_context}
        LATEST NEWS: {web_context}

        **Notes:**
        - Opening: curiosity-driven hook, pulls viewer in within 15-30 seconds.
        - Use storytelling: tension, suspense, surprise, moral dilemmas.
        - Make historical/technical details immersive, not lecture-like.
        - Narrative arc: build curiosity → climax → reflection.
        """

        # Groq-first script generation with OpenRouter fallback.
        script_text = await generate_script_content([{"role": "user", "content": script_prompt}])

        print(f"--- Script generation took {time.time() - total_start_time:.2f}s ---")

        # ── Step 4: Analyse the script with Groq ─────────────
        ANALYSIS_PROMPT = f"""
        You are an expert script analyzer. Analyze the provided YouTube script:

        1. **Real-world Examples:** Count distinct real-world examples/case studies/stories.
        2. **Research Facts/Stats:** Count distinct research findings, statistics, data points.
        3. **Proverbs/Sayings:** Count common proverbs, idioms, or well-known sayings.
        4. **Emotional Depth:** Rate overall emotional engagement: Low, Medium, or High.

        Return ONLY a JSON object — no explanation, no other text.

        EXAMPLE OUTPUT:
        {{
          "examples_count": 3,
          "research_facts_count": 5,
          "proverbs_count": 1,
          "emotional_depth": "Medium"
        }}

        --- SCRIPT ---
        {script_text}
        --- END ---
        """

        analysis_start = time.time()
        analysis_completion = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": ANALYSIS_PROMPT}],
            model=GROQ_GENERATION_MODEL,
        )
        analysis_raw = analysis_completion.choices[0].message.content
        print(f"--- Script analysis took {time.time() - analysis_start:.2f}s ---")

        analysis_results = {
            "examples_count": 0,
            "research_facts_count": 0,
            "proverbs_count": 0,
            "emotional_depth": "Unknown",
        }
        try:
            clean = analysis_raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            analysis_results.update({
                "examples_count": parsed.get("examples_count", 0),
                "research_facts_count": parsed.get("research_facts_count", 0),
                "proverbs_count": parsed.get("proverbs_count", 0),
                "emotional_depth": parsed.get("emotional_depth", "Unknown"),
            })
        except (json.JSONDecodeError, Exception) as e:
            print(f"Analysis parse error: {e}")

        generated_word_count = len(script_text.split())
        print(f"Generated word count: {generated_word_count}")
        print(f"Total /generate-script time: {time.time() - total_start_time:.2f}s")

        # ── Decrement credits ─────────────────────────────────
        if user_tier != 'admin':
            try:
                new_balance = max(0, credits - IDEA_COST)
                update_result = (
                    supabase.table('profiles')
                    .update({'credits_remaining': new_balance})
                    .eq('id', user_id)
                    .execute()
                )
                if update_result.data:
                    print(f"Decremented {IDEA_COST} credit(s) for {user_id}. Balance: {new_balance}")
                else:
                    print(f"WARN: Credit decrement returned no rows for {user_id}.")
            except APIError as e:
                print(f"ERROR: Credit decrement DB error for {user_id}: {e.message}")
            except Exception as e:
                print(f"ERROR: Unexpected credit decrement error for {user_id}: {e}")
        else:
            print(f"Admin user {user_id} — no credits decremented.")

        return {
            "script": script_text,
            "estimated_word_count": generated_word_count,
            "source_urls": list(scraped_urls),
            "analysis": analysis_results,
        }

    except Exception as e:
        print(f"SCRIPT GENERATION error: {e}")
        return {"error": "An error occurred during the script generation pipeline."}


# ── /payments/create-order ───────────────────────────────────

@app.post("/payments/create-order")
async def create_razorpay_order(
    request_data: CreateOrderRequest,
    current_user: User = Depends(get_current_user),
):
    if not razorpay_client:
        raise HTTPException(status_code=503, detail="Payment service unavailable.")

    user_id = current_user.id
    amount = request_data.amount
    currency = request_data.currency

    if amount <= 0:
        raise HTTPException(status_code=400, detail="Invalid amount.")
    if request_data.target_tier not in ['basic', 'pro']:
        raise HTTPException(status_code=400, detail="Invalid target tier.")

    order_data = {
        "amount": amount,
        "currency": currency,
        "receipt": request_data.receipt or f"rec_{int(time.time())}",
        "notes": {
            "user_id": str(user_id),
            "target_tier": request_data.target_tier,
        },
    }
    try:
        order = razorpay_client.order.create(data=order_data)
        print(f"Created Razorpay order {order['id']} for user {user_id}")
        return {
            "order_id": order['id'],
            "key_id": RAZORPAY_KEY_ID,
            "amount": amount,
            "currency": currency,
        }
    except Exception as e:
        print(f"Error creating Razorpay order: {e}")
        raise HTTPException(status_code=500, detail="Could not create payment order.")


# ── /payments/webhook ────────────────────────────────────────

@app.post("/payments/webhook")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str | None = Header(None),
):
    if not RAZORPAY_WEBHOOK_SECRET or not razorpay_client:
        print("Webhook received but service not configured.")
        return {"status": "Webhook ignored"}

    body = await request.body()

    # Verify signature
    try:
        razorpay_client.utility.verify_webhook_signature(
            body.decode('utf-8'),
            x_razorpay_signature,
            RAZORPAY_WEBHOOK_SECRET,
        )
        print("Webhook signature verified.")
    except razorpay.errors.SignatureVerificationError as e:
        print(f"Webhook signature failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")
    except Exception as e:
        print(f"Webhook verification error: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing error.")

    # Process event
    try:
        event_data = json.loads(body)
        event_type = event_data.get('event')
        print(f"Received webhook event: {event_type}")

        if event_type == 'order.paid':
            order_entity = event_data['payload']['order']['entity']
            order_id   = order_entity.get('id', 'unknown')
            payment_id = event_data['payload']['payment']['entity'].get('id', 'unknown')
            notes = order_entity.get('notes', {})
            user_id = notes.get('user_id')
            target_tier = notes.get('target_tier')

            if not user_id or not target_tier:
                print(f"ERROR: Missing notes in order {order_id}.")
                return {"status": "error", "message": "Missing required order notes."}

            credits_to_add = 0
            if target_tier.lower() == 'basic':
                credits_to_add = 50
            elif target_tier.lower() == 'pro':
                credits_to_add = 200

            try:
                profile_resp = (
                    supabase.table('profiles')
                    .select('credits_remaining')
                    .eq('id', user_id)
                    .single()
                    .execute()
                )
                current_credits = profile_resp.data.get('credits_remaining', 0) if profile_resp.data else 0
                new_credits = current_credits + credits_to_add

                update_result = (
                    supabase.table('profiles')
                    .update({'user_tier': target_tier, 'credits_remaining': new_credits})
                    .eq('id', user_id)
                    .execute()
                )
                if update_result.data:
                    print(f"Updated user {user_id} → tier '{target_tier}', credits {new_credits}.")
                else:
                    print(f"WARN: Failed to update profile for {user_id} after payment {payment_id}.")
            except APIError as e:
                print(f"ERROR: Supabase error for {user_id}: {e}")
            except Exception as e:
                print(f"ERROR: Unexpected error for {user_id}: {e}")

        elif event_type == 'payment.captured':
            print("Ignoring 'payment.captured' (handled by 'order.paid').")

        elif event_type == 'payment.failed':
            payment_entity = event_data['payload']['payment']['entity']
            print(
                f"Payment failed for order {payment_entity.get('order_id')}. "
                f"Reason: {payment_entity.get('error_description')}"
            )
        else:
            print(f"Ignoring unhandled event: {event_type}")

        return {"status": "Webhook processed successfully"}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    except Exception as e:
        print(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
