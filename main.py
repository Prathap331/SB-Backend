from fastapi import Depends, HTTPException, Request, Header, BackgroundTasks,UploadFile, File,Form
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
from researchAgent.tss_v3 import run_tss
from pipeline.pipeline_response_adapter import adapt_pipeline_payload
from pipeline.idea_generation_pipeline import generate_ideas as generate_cags_aligned_ideas, TOPIC_CACHE
from signals.social_market_signals import scan_topic as scan_social_topic
from signals.news_market_signals import scan_topic as scan_news_topic
import os
from openai import OpenAI
from channelMemory.aiIntel import get_intelligence
from researchAgent.tss_v4 import get_trends_serpapi,build_trend_dashboard , build_youtube_summary , scan_topic , build_news_summary
from researchAgent.eci import get_google_trends_serpapi,get_youtube_data

from shared.schemas.pipeline_context import (
    AgentPipelineContext,
    extract_angle_for_prompt,
    staleness_hours,
)
from script_templates.registry import TEMPLATE_REGISTRY
from script_templates.selector import select_template_key
from script_templates.injector import assemble_structure_section, assemble_chapter_scaffold

from google import genai
from seoAgent.seo import seo_agent
from ddgs import DDGS
import os
import asyncio
import time
import re
import json
import random
import httpx
import nltk
import razorpay
import datetime
from typing import Any
from urllib.parse import urlparse
from datetime import datetime as dt
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
from ddgs import DDGS
from readability import Document
from pytrends.request import TrendReq
from sentence_transformers import SentenceTransformer
from channelMemory.channelMemory import process_pdf
from channelMemory.aiIntel import get_chunks_from_db

load_dotenv()

project_root = os.path.dirname(os.path.abspath(__file__))
nltk_data_dir = os.path.join(project_root, 'nltk_data')
nltk.data.path.insert(0, nltk_data_dir)

print(os.getenv("RAZORPAY_WEBHOOK_SECRET"))

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

api_key = os.getenv("apiKey")
gnews_key = os.getenv("GnewsApi")
google_api_key = os.getenv("GOOGLE_API_KEY")

url= os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

Hf_token = os.getenv("Hf_token")

print(Hf_token)

url = "https://router.huggingface.co/v1/chat/completions"


headers = {
        "Authorization": f"Bearer {Hf_token}",
        "Content-Type": "application/json"
    }


pytrends = TrendReq(hl='en-US', tz=360)

supabase = create_client(url, key)

client = genai.Client(api_key=google_api_key)

print(google_api_key)

model = SentenceTransformer('all-MiniLM-L6-v2')

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
SCRIPT_FRAMECHECK_PROVIDER = (os.getenv("SCRIPT_FRAMECHECK_PROVIDER") or "groq").strip().lower()

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

GENERATION_MODEL        = "google/gemma-3-27b-it:free"
GENERATION_MODEL_BACKUP = "google/gemma-3n-e4b-it:free"
GENERATION_MODEL_EXTRA  = "deepseek/deepseek-r1-0528-qwen3-8b:free"

print(
    f"Google GenAI (embeddings x{len(EMBED_CLIENTS)}), "
    f"Groq ideas x{len(GROQ_IDEA_CLIENTS)}, Groq scripts x{len(GROQ_SCRIPT_CLIENTS)}, "
    "and OpenRouter clients initialized successfully."
)

PROCESS_DB_MAX_BLOCKS = 5
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


deepseek_client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")



print("deepseek", os.environ.get("DEEPSEEK_API_KEY"))

# def _topic_cache_key(topic: str) -> str:
#     return " ".join((topic or "").strip().lower().split())


# def _parse_utc_datetime(value: Any) -> datetime.datetime | None:
#     if not value:
#         return None
#     if isinstance(value, datetime.datetime):
#         candidate = value
#     else:
#         try:
#             candidate = dt.fromisoformat(str(value).replace("Z", "+00:00"))
#         except Exception:
#             return None
#     if candidate.tzinfo is None:
#         candidate = candidate.replace(tzinfo=datetime.timezone.utc)
#     return candidate.astimezone(datetime.timezone.utc)


# def _cache_age_hours(created_at: Any) -> float | None:
#     candidate = _parse_utc_datetime(created_at)
#     if candidate is None:
#         return None
#     return max((datetime.datetime.now(datetime.timezone.utc) - candidate).total_seconds() / 3600.0, 0.0)


# async def _lookup_topic_cache_db(topic: str) -> dict[str, Any] | None:
#     topic_key = _topic_cache_key(topic)
#     try:
#         response = await asyncio.to_thread(
#             lambda: supabase.table("topic_content_cache")
#             .select("topic_canonical,payload,created_at,updated_at,expires_at")
#             .eq("topic_key", topic_key)
#             .gt("expires_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
#             .limit(1)
#             .execute()
#         )
#         rows = response.data or []
#         if not rows:
#             return None
#         row = rows[0] or {}
#         payload = dict(row.get("payload") or {})
#         if not _payload_has_ideas(payload):
#             return None
#         payload.pop("cags", None)
#         payload["served_from_cache"] = True
#         payload["cache_age_hours"] = round(_cache_age_hours(row.get("created_at")) or 0.0, 3)
#         payload["source_of_context"] = payload.get("source_of_context", "CACHE_DB")
#         return payload
#     except Exception as e:
#         print(f"[warn] idea cache DB lookup failed for '{topic}': {e}")
#         return None


# async def _lookup_topic_cache_db_semantic(topic: str, cache_client: Any | None = None) -> dict[str, Any] | None:
#     if cache_client is None:
#         return None
#     topic_vector = None
#     try:
#         topic_vector = TOPIC_CACHE._vector_from_client(topic, cache_client)
#     except Exception:
#         topic_vector = None
#     if topic_vector is None:
#         return None
#     try:
#         response = await asyncio.to_thread(
#             lambda: supabase.rpc(
#                 "match_topic_content_cache",
#                 {
#                     "query_embedding": topic_vector,
#                     "match_threshold": 0.92,
#                     "match_count": 1,
#                 },
#             ).execute()
#         )
#         rows = response.data or []
#         if not rows:
#             return None
#         row = rows[0] or {}
#         payload = dict(row.get("payload") or {})
#         if not _payload_has_ideas(payload):
#             return None
#         payload.pop("cags", None)
#         payload["served_from_cache"] = True
#         payload["cache_age_hours"] = round(_cache_age_hours(row.get("created_at")) or 0.0, 3)
#         payload["cache_similarity"] = round(float(row.get("similarity") or 0.0), 3)
#         payload["source_of_context"] = payload.get("source_of_context", "CACHE_DB")
#         return payload
#     except Exception as e:
#         print(f"[warn] idea cache semantic DB lookup failed for '{topic}': {e}")
#         return None


# async def _store_topic_cache_db(topic: str, payload: dict[str, Any], cache_client: Any | None = None) -> None:
#     topic_key = _topic_cache_key(topic)
#     expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=IDEA_CACHE_TTL_HOURS)
#     row: dict[str, Any] = {
#         "topic_key": topic_key,
#         "topic_canonical": topic,
#         "payload": payload,
#         "expires_at": expires_at.isoformat(),
#         "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
#     }
#     try:
#         topic_vector = None
#         if cache_client is not None:
#             try:
#                 topic_vector = TOPIC_CACHE._vector_from_client(topic, cache_client)
#             except Exception:
#                 topic_vector = None
#         if topic_vector is not None:
#             row["topic_vec"] = topic_vector
#         await asyncio.to_thread(
#             lambda: supabase.table("topic_content_cache")
#             .upsert(row, on_conflict="topic_key")
#             .execute()
#         )
#     except Exception as e:
#         print(f"[warn] idea cache DB store failed for '{topic}': {e}")


# def _build_cache_payload(payload: dict[str, Any]) -> dict[str, Any]:
#     """
#     Keep idea cache stable: do not persist volatile trend metrics (TSS/CSI/CAGS)
#     because they are time-sensitive and should be computed fresh.
#     """
#     cached = dict(payload or {})
#     cached.pop("cags", None)
#     return cached


# async def _lookup_topic_cache(topic: str, cache_client: Any | None = None) -> dict[str, Any] | None:
#     cached = TOPIC_CACHE.lookup(topic, cache_client)
#     if cached:
#         cached.pop("cags", None)
#         return cached
#     db_semantic = await _lookup_topic_cache_db_semantic(topic, cache_client)
#     if db_semantic:
#         TOPIC_CACHE.store(topic, db_semantic, cache_client)
#         return db_semantic
#     db_cached = await _lookup_topic_cache_db(topic)
#     if db_cached:
#         TOPIC_CACHE.store(topic, db_cached, cache_client)
#     return db_cached




def _cap_blocks(blocks: list[str], max_blocks: int, max_chars: int) -> str:
    selected = [b.strip() for b in blocks if b and b.strip()][:max_blocks]
    merged = "\n\n".join(selected)
    if len(merged) > max_chars:
        merged = merged[:max_chars]
    return merged


# def _extract_retry_delay_seconds(error_text: str) -> float | None:
#     if not error_text:
#         return None
#     match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", error_text, flags=re.IGNORECASE)
#     if not match:
#         return None
#     try:
#         return float(match.group(1))
#     except Exception:
#         return None


# def _is_daily_quota_error(error_text: str) -> bool:
#     t = (error_text or "").lower()
#     return "resource_exhausted" in t and "perday" in t


# def _is_embedding_quota_error(error_text: str) -> bool:
#     t = (error_text or "").lower()
#     return "resource_exhausted" in t or "quota" in t


def _is_groq_rate_limit_error(error_text: str) -> bool:
    t = (error_text or "").lower()
    return "rate limit" in t or "rate_limit_exceeded" in t or "too many requests" in t


# def _payload_has_ideas(payload: dict[str, Any] | None) -> bool:
#     if not isinstance(payload, dict):
#         return False
#     if _payload_uses_fallback_variants(payload):
#         return False
#     ideas = payload.get("ideas")
#     if isinstance(ideas, list) and any(str(item).strip() for item in ideas):
#         return True
#     clusters = payload.get("idea_clusters")
#     if not isinstance(clusters, list) or not clusters:
#         return False
#     for cluster in clusters:
#         variants = (cluster or {}).get("idea_variants")
#         if not isinstance(variants, list):
#             continue
#         for variant in variants:
#             title = str((variant or {}).get("title") or "").strip()
#             desc = str((variant or {}).get("description") or "").strip()
#             if title and desc:
#                 return True
#     return False


def _payload_uses_fallback_variants(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    clusters = payload.get("idea_clusters")
    if not isinstance(clusters, list):
        return False
    for cluster in clusters:
        variants = (cluster or {}).get("idea_variants")
        if not isinstance(variants, list):
            continue
        for variant in variants:
            reason = str((variant or {}).get("gap_reason") or "").lower()
            if "fallback expansion" in reason:
                return True
    return False


GEMINI_EMBED_BLOCKED_UNTIL_TS = 0.0




def _embed_with_failover(
    *,
    contents: str | list[str],
    task_type: str = None,  # kept for compatibility (not used)
    output_dimensionality: int = 384,  # MiniLM outputs 384
):
    try:
        # Normalize input
        if isinstance(contents, str):
            contents = [contents]

        embeddings = model.encode(
            contents,
            convert_to_numpy=True,
            normalize_embeddings=True  # good for cosine similarity
        )

        # Ensure correct shape/output
        if len(embeddings) == 1:
            return embeddings[0].tolist()

        return embeddings.tolist()

    except Exception as e:
        print(f"Local embedding failed: {e}")
        raise



_st_model: SentenceTransformer | None = None

def _get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer("BAAI/bge-base-en-v1.5")  
    return _st_model


async def _embed_chunks_with_backoff(chunks: list[str]) -> list[list[float]] | None:
    if not chunks:
        return []

    batch_size = 20
    vectors: list[list[float]] = []
    model = _get_st_model()

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda b=batch: model.encode(b, normalize_embeddings=True).tolist(),
            )
            vectors.extend(embeddings)
        except Exception as exc:
            print(f"BACKGROUND TASK: Embedding failed: {exc}")
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

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
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
    if tags is None:
        tags = []

    print(f"BACKGROUND TASK: Starting upload for '{article_title[:50]}...'")
    try:
        # Chunk the article text properly
        chunks = chunk_text(article_text, chunk_size=500, chunk_overlap=50)
        if not chunks:
            print(f"BACKGROUND TASK: No chunks generated for '{article_title[:50]}', skipping.")
            return

        print(f"BACKGROUND TASK: '{article_title[:50]}' → {len(chunks)} chunks to upload.")

        embeddings = asyncio.run(_embed_chunks_with_backoff(chunks))
        if embeddings is None or len(embeddings) != len(chunks):
            print(f"BACKGROUND TASK: Embedding failed or mismatch for '{article_title[:50]}', skipping.")
            return

        domain = urlparse(article_url).netloc.lstrip('www.') if article_url else ""
        scraped_at = dt.now().isoformat()

        author_info = {
            "has_credentials": bool(domain),
            "name": domain if domain else None,
            "description": "Web publication" if domain else None,
        }

        rows_to_insert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            rows_to_insert.append({
                "content":      chunk,
                "embedding":    embedding,
                "source_title": article_title,
                "source_url":   article_url,
                "source_type":  "web_scrape",
                "topic":        topic,
                "category":     category,
                "metadata": {
                    "tags":        tags,
                    "domain":      domain,
                    "scraped_at":  scraped_at,
                    "author":      author_info,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            })

        supabase.table('RAG_web_scraped').insert(rows_to_insert).execute()
        print(f"BACKGROUND TASK: Successfully uploaded {len(rows_to_insert)} chunks for '{article_title[:50]}'")

    except Exception as e:
        print(f"BACKGROUND TASK: Failed for '{article_title[:50]}'. Error: {e}")


# Max 3 concurrent Playwright browsers — each uses ~300MB RAM
# Prevents memory exhaustion when multiple requests arrive simultaneously
_playwright_semaphore = asyncio.Semaphore(3)
_pipeline_request_semaphore = asyncio.Semaphore(PIPELINE_MAX_CONCURRENCY)
# _process_topic_request_semaphore = asyncio.Semaphore(PROCESS_TOPIC_MAX_CONCURRENCY)
# _request_state_lock = asyncio.Lock()
# _response_cache = {
#     "pipeline_metrics": OrderedDict(),
#     "process_topic": OrderedDict(),
# }
# _inflight_requests = {
#     "pipeline_metrics": {},
#     "process_topic": {},
# }


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


_SCRAPE_BLOCKLIST = {
    'zhidao.baidu.com', 'baidu.com', 'en.cppreference.com', 'cppreference.com',
    'stackoverflow.com', 'github.com', 'reddit.com', 'twitter.com', 'x.com',
    'instagram.com', 'facebook.com', 'linkedin.com', 'pinterest.com',
    'researchgate.net', 'academia.edu', 'jstor.org',
}


async def scrape_url(
    url: str,
    scraped_urls: set,
    snippet: str = "",           
) -> dict | None:
    """
    3-tier scraping with automatic fallback:
      Tier 1 → httpx (fast)
      Tier 2 → Playwright headless Chrome (robust, defeats bot detection)
      Tier 3 → Use DDGS snippet directly (always works, less text)
    """
    if url in scraped_urls:
        return None
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
    print(f"--- DEEP WEB SCRAPE: Starting with {len(keywords)} keywords... ---")

    # Discover up to 10 unique URLs across all keywords + both search engines
    try:
        url_snippet_pairs = await asyncio.wait_for(
            asyncio.to_thread(_discover_urls, keywords, DEEP_SCRAPE_MAX_RESULTS_PER_KEYWORD),
            timeout=DEEP_SCRAPE_DISCOVERY_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        print("--- DEEP WEB SCRAPE: Discovery timed out ---")
        return []
    except Exception as e:
        print(f"--- DEEP WEB SCRAPE: Discovery failed: {e} ---")
        return []

    if not url_snippet_pairs:
        print("--- DEEP WEB SCRAPE: No URLs discovered ---")
        return []

    print(f"--- DEEP WEB SCRAPE: Scraping {len(url_snippet_pairs)} URLs... ---")

    tasks = [
        asyncio.wait_for(
            scrape_url(url, scraped_urls, snippet),
            timeout=DEEP_SCRAPE_PER_URL_TIMEOUT_SEC,
        )
        for url, snippet in url_snippet_pairs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    cleaned = []
    for r in results:
        if isinstance(r, Exception):
            continue
        if r and r.get("text"):
            cleaned.append(r)

    print(f"--- DEEP WEB SCRAPE: Got {len(cleaned)} valid articles ---")
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


NEWS_SCRAPE_MAX_RESULTS         = 10
DEEP_SCRAPE_MAX_KEYWORDS        = 6
DEEP_SCRAPE_MAX_RESULTS_PER_KEYWORD = 10  
DEEP_SCRAPE_PER_URL_TIMEOUT_SEC = 12
DEEP_SCRAPE_DISCOVERY_TIMEOUT_SEC = 30
MIN_ARTICLES_THRESHOLD          = 5       


def _ddgs_search(keyword: str, max_results: int) -> list[tuple[str, str]]:
    """Run a single DDGS search, return (url, snippet) pairs."""
    found = []
    try:
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.text(keyword, region='wt-wt', max_results=max_results))
            for r in results:
                href = r.get('href', '')
                snippet = r.get('body', '')
                if href:
                    found.append((href, snippet))
    except Exception as e:
        print(f"    [DDGS] Search failed for '{keyword}': {e}")
    return found


def _google_search(keyword: str, max_results: int) -> list[tuple[str, str]]:
    """
    Google search via DDGS backend (uses Google under the hood).
    Falls back gracefully if unavailable.
    """
    found = []
    try:
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.text(keyword, region='wt-wt', max_results=max_results, backend='google'))
            for r in results:
                href = r.get('href', '')
                snippet = r.get('body', '')
                if href:
                    found.append((href, snippet))
    except Exception as e:
        print(f"    [Google] Search failed for '{keyword}', falling back to DDGS only: {e}")
    return found


def _discover_urls(keywords: list[str], max_results_per_keyword: int) -> list[tuple[str, str]]:
    """
    Search both DDGS and Google for each keyword.
    Returns up to 10 unique (url, snippet) pairs total across all keywords.
    """
    seen_urls: set[str] = set()
    all_results: list[tuple[str, str]] = []

    for keyword in keywords[:DEEP_SCRAPE_MAX_KEYWORDS]:
        print(f"    [DISCOVER] Searching: '{keyword}'")

        # Search DDGS
        ddgs_results = _ddgs_search(keyword, max_results_per_keyword)
        print(f"        DDGS → {len(ddgs_results)} results")

        # Search Google
        google_results = _google_search(keyword, max_results_per_keyword)
        print(f"        Google → {len(google_results)} results")

        # Merge, deduplicate
        for url, snippet in ddgs_results + google_results:
            if url not in seen_urls:
                seen_urls.add(url)
                all_results.append((url, snippet))

        if len(all_results) >= 10:
            break  

    print(f"    [DISCOVER] Total unique URLs found: {len(all_results)}")
    return all_results[:10] 


async def get_latest_news_context(topic: str, scraped_urls: set) -> list[dict]:
    print("--- LIGHT WEB SCRAPE: Starting lightweight news search... ---")
    try:
        keywords = [
            f"{topic} latest news today",
            f"{topic} 2025 update",
        ]

        try:
            url_snippet_pairs = await asyncio.wait_for(
                asyncio.to_thread(_discover_urls, keywords, NEWS_SCRAPE_MAX_RESULTS),
                timeout=DEEP_SCRAPE_DISCOVERY_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            print("--- LIGHT WEB SCRAPE: Discovery timed out ---")
            return []

        print(f"--- LIGHT WEB SCRAPE: Scraping {len(url_snippet_pairs)} URLs... ---")

        tasks = [
            asyncio.wait_for(
                scrape_url(url, scraped_urls, snippet),
                timeout=DEEP_SCRAPE_PER_URL_TIMEOUT_SEC,
            )
            for url, snippet in url_snippet_pairs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        cleaned = [
            r for r in results
            if not isinstance(r, Exception) and r and r.get("text")
        ]
        print(f"--- LIGHT WEB SCRAPE: Got {len(cleaned)} valid articles ---")
        return cleaned

    except Exception as e:
        print(f"--- LIGHT WEB SCRAPE: Error: {e} ---")
        return []


async def get_db_context(topic: str, hypothetical_document: str = None) -> list[dict]:
    print("--- DB TASK: Starting two-stage DB search... ---")
    combined: dict[int, dict] = {}

    try:
        loop = asyncio.get_running_loop()

        # Use pre-generated HyDE doc if provided, skip regeneration
        if hypothetical_document is None:
            hypothetical_document = topic
            try:
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
                print(f"--- DB TASK: HyDE doc generated: {hypothetical_document[:100]}...")
            except Exception as exc:
                print(f"--- DB TASK: HyDE failed, falling back to raw topic: {exc} ---")
                hypothetical_document = topic

        print(f"--- DB TASK: Embedding HyDE document for semantic search ---")

        # Embed the HyDE document 
        try:
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
            print(f"--- DB TASK: Embedding generated, dimension: {len(query_embedding)} ---")

            # Vector search against DB chunks 
            try:
                vector_response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: supabase.rpc(
                            'match_documents',
                            {
                                'query_embedding': query_embedding,
                                'match_threshold': 0.55,
                                'match_count':     8,
                            }
                        ).execute()
                    ),
                    timeout=DB_LOOKUP_TIMEOUT_SEC,
                )
                vector_results = vector_response.data or []
                for row in vector_results:
                    combined[row['id']] = row
                print(f"--- DB TASK: Semantic search → {len(vector_results)} chunks retrieved ---")
                for i, row in enumerate(vector_results):
                    print(f"    [{i+1}] id={row['id']} | similarity={row.get('similarity', 'N/A'):.3f} | {row.get('content', '')[:80]}...")

            except asyncio.TimeoutError:
                print(f"--- DB TASK: Vector search timed out after {DB_LOOKUP_TIMEOUT_SEC}s ---")

        except Exception as exc:
            print(f"--- DB TASK: Embedding failed: {exc} ---")

        if len(combined) < 3:
            print(f"--- DB TASK: Only {len(combined)} semantic results, running keyword fallback... ---")
            topic_terms = [term for term in re.findall(r"[A-Za-z0-9']+", topic.lower()) if len(term) > 2][:5]
            if not topic_terms:
                topic_terms = [topic.lower()]

            seen_ids: set = set(combined.keys())
            keyword_rows: list[dict] = []

            for term in topic_terms:
                try:
                    response = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda term=term: supabase.table("RAG_web_scraped")
                            .select("id, content, source_title, source_url, metadata, created_at")
                            .or_(f"content.ilike.%{term}%,source_title.ilike.%{term}%")
                            .limit(8)
                            .execute()
                        ),
                        timeout=DB_LOOKUP_TIMEOUT_SEC,
                    )
                except Exception as exc:
                    print(f"--- DB TASK: Keyword fallback failed for term '{term}': {exc} ---")
                    continue

                for row in (response.data or []):
                    row_id = row.get("id")
                    if row_id is None or row_id in seen_ids:
                        continue
                    seen_ids.add(row_id)
                    combined[row_id] = row
                    keyword_rows.append(row)

            print(f"--- DB TASK: Keyword fallback → {len(keyword_rows)} extra chunks, total unique: {len(combined)} ---")

    except Exception as e:
        print(f"--- DB TASK: Error: {e} ---")
        return []

    results = list(combined.values())
    print(f"--- DB TASK: Returning {len(results)} total chunks ---")
    return results



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
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Lifespan] Server started.")
    yield 
    print("[Lifespan] Shutting down.")

app = FastAPI(lifespan=lifespan)

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
    topic: str | None = None
    userId : str | None = None
    context: AgentPipelineContext | None = None
    duration_minutes: int | None = 10
    # template_key_override: str | None = None
    # user_wpm: int | None = None


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


class ChannelContextInput(BaseModel):
    channel_id: str | None = None
    channel_niche: str | None = None
    subscriber_count: int | None = None
    top_video_titles: list[str] | None = None
    existing_hashtags: list[str] | None = None
    avg_ctr_pct: float | None = None


class SEOAgentRequest(BaseModel):
    context: AgentPipelineContext
    channel_context: ChannelContextInput | None = None


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


BLOCKED_TITLE_TYPES = {
    "CAT-03": ["controversy"],
    "CAT-04": ["controversy"],
}

CAT_FACE_DEFAULTS = {
    "CAT-01": False,
    "CAT-02": True,
    "CAT-03": False,
    "CAT-04": False,
    "CAT-05": True,
    "CAT-06": True,
    "CAT-07": False,
    "CAT-08": True,
}

SEO_SYNTHESIS_PROMPT = """
You are an expert YouTube SEO Analyst and Title Strategist.

VIDEO ANGLE: "{angle_string}"
STAKEHOLDER: {who} | LENS: {what} | FRAME: {story_frame}
AUDIENCE PROFILE: {audience_profile}
CATEGORY: {cat_id} — {cat_label}

COMPETITIVE DATA:
Top 5 YouTube Titles: {competing_titles}
Top 5 PAA Questions: {paa_questions}

CTR SIGNAL (pre-computed):
ctr_potential: {ctr_label} (score: {ctr_score})
{degraded_note}

TITLE SAFETY:
- category blocked title types: {blocked_title_types}
- max title length: 70 chars
- no fabricated quotes / factual claims

Return ONLY valid JSON:
{{
  "search_intent_type": "educational|entertainment|comparative|news_driven|problem_solving|inspirational",
  "recommended_structure": "problem_solution|storytelling|listicle|chronological|myth_debunking|tech_review",
  "ctr_potential": "{ctr_label}",
  "ctr_signal_degraded": {ctr_signal_degraded},
  "justification": "2 sentence rationale",
  "recommended_titles": [
    {{"type":"curiosity_gap|data_led|how_to|controversy|narrative","title":"...","rationale":"..."}}
  ],
  "keyword_clusters": {{
    "primary": [],
    "secondary": [],
    "longtail": [],
    "question_based": []
  }},
  "description_template": {{
    "hook": "",
    "body_bullets": [],
    "outro": ""
  }},
  "thumbnail_brief": [
    {{"concept_type":"curiosity_gap|data_driven|face_reaction|before_after","text_overlay":"","visual_theme":"","colour_temperature":"warm|cool|high_contrast","face_recommended":true,"rationale":""}}
  ],
  "hashtags": [],
  "chapter_structure": [
    {{"index":1,"title":"","covers":"","section_pct":0.2}}
  ],
  "key_questions_to_answer": []
}}
"""


def _strip_json_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = _strip_json_fences(raw)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return {}




SEO_INTENT_TYPES = {
    "educational",
    "entertainment",
    "comparative",
    "news_driven",
    "problem_solving",
    "inspirational",
}

SEO_STRUCTURES = {
    "problem_solution",
    "storytelling",
    "listicle",
    "chronological",
    "myth_debunking",
    "tech_review",
}


WPM_BY_CREATOR_TYPE = {
    "storyteller": 120,
    "educator": 145,
    "entertainer": 160,
    "journalist": 135,
    "commentator": 140,
}
WPM_DEFAULT = 130
SCRIPT_CREDIT_COST = 3
PROMPT_TOKEN_BUDGET = 6500
FIXED_PROMPT_OVERHEAD = 800


def get_wpm(creator_type: str, user_wpm: int | None) -> int:
    if user_wpm is not None and 80 <= int(user_wpm) <= 200:
        return int(user_wpm)
    return int(WPM_BY_CREATOR_TYPE.get((creator_type or "").strip().lower(), WPM_DEFAULT))


def assess_context_quality(db_ctx: str, web_ctx: str) -> tuple[bool, int]:
    combined = f"{db_ctx} {web_ctx}"
    word_count = len([w for w in combined.split() if len(w) > 3])
    return word_count >= 100, word_count


def estimate_tokens(text: str) -> int:
    return int(max(len((text or "").split()) * 1.35, 0))


def trim_to_budget(
    db_ctx: str,
    web_ctx: str,
    social: list[dict[str, Any]],
    news: list[dict[str, Any]],
    angle_spec_tokens: int,
    seo_section_tokens: int,
) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]], bool]:
    budget = PROMPT_TOKEN_BUDGET - FIXED_PROMPT_OVERHEAD
    budget -= angle_spec_tokens + seo_section_tokens
    budget -= estimate_tokens(db_ctx)
    truncated = False
    wc = web_ctx
    s = list(social or [])
    n = list(news or [])
    if estimate_tokens(wc) > budget:
        wc = wc[: int(len(wc) * 0.6)]
        truncated = True
    if estimate_tokens(wc) > budget:
        s = s[:3]
        n = n[:3]
        truncated = True
    return db_ctx, wc, s, n, truncated


def check_depth_alignment(target_wc: int, depth_check_target: int) -> dict[str, Any] | None:
    if target_wc > int(depth_check_target * 1.15):
        return {
            "type": "content_depth_warning",
            "target_words": target_wc,
            "depth_checked_words": depth_check_target,
            "message": (
                f"Target {target_wc}w exceeds research depth validated for this idea "
                f"({depth_check_target}w). Later sections may be less factual."
            ),
            "recommendation": "Reduce duration_minutes or lower user_wpm.",
        }
    return None


async def check_and_deduct_credits(
    user_id: str,
    async_mode: bool,
    job_id: str | None = None,
) -> dict[str, Any]:
    profile = (
        supabase.table("profiles")
        .select("credits_remaining, user_tier")
        .eq("id", user_id)
        .single()
        .execute()
    )
    if not profile.data:
        raise HTTPException(status_code=404, detail="User profile not found")
    tier = profile.data.get("user_tier", "free")
    credits = int(profile.data.get("credits_remaining", 0) or 0)
    if tier == "admin":
        return {"admin": True, "deducted": False}
    if credits < SCRIPT_CREDIT_COST:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "insufficient_credits",
                "balance": credits,
                "required": SCRIPT_CREDIT_COST,
            },
        )
    if async_mode:
        new_balance = max(0, credits - SCRIPT_CREDIT_COST)
        supabase.table("profiles").update({"credits_remaining": new_balance}).eq("id", user_id).execute()
        return {"admin": False, "deducted": True, "new_balance": new_balance}
    return {"admin": False, "deducted": False, "credits": credits}


async def deduct_after_success(user_id: str, credits: int) -> None:
    new_balance = max(0, int(credits) - SCRIPT_CREDIT_COST)
    supabase.table("profiles").update({"credits_remaining": new_balance}).eq("id", user_id).execute()


async def issue_refund(user_id: str, job_id: str) -> None:
    profile = (
        supabase.table("profiles")
        .select("credits_remaining")
        .eq("id", user_id)
        .single()
        .execute()
    )
    if not profile.data:
        return
    new_balance = int(profile.data.get("credits_remaining", 0) or 0) + SCRIPT_CREDIT_COST
    supabase.table("profiles").update({"credits_remaining": new_balance}).eq("id", user_id).execute()
    supabase.table("script_jobs").update({"refund_issued": True}).eq("id", job_id).execute()


def compute_chapter_timestamps(script: str, chapter_structure: list[dict[str, Any]], wpm: int) -> list[dict[str, Any]]:
    words = (script or "").split()
    total_words = max(len(words), 1)
    result: list[dict[str, Any]] = []
    cumulative_pct = 0.0
    for idx, ch in enumerate(chapter_structure):
        word_pos = int(cumulative_pct * total_words)
        seconds = int((word_pos / max(wpm, 1)) * 60)
        pct = float(ch.get("section_pct", 0.0) or 0.0)
        result.append(
            {
                **ch,
                "index": idx + 1,
                "timestamp_seconds": seconds,
                "timestamp_fmt": f"{seconds//60}:{seconds%60:02d}",
                "section_pct": pct,
            }
        )
        cumulative_pct += pct
    return result


def _to_section_label(name: str) -> str:
    raw = re.sub(r"[^a-zA-Z0-9]+", "_", str(name or "").strip()).strip("_")
    return raw.upper() or "SECTION"


def build_script_sections(
    script: str,
    chapter_structure: list[dict[str, Any]] | None = None,
    template_segments: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    text = (script or "").strip()
    if not text:
        return []

    words = text.split()
    total_words = max(len(words), 1)
    segs = list(template_segments or [])
    sections: list[dict[str, str]] = []

    if segs:
        cumulative_pct = 0.0
        for idx, seg in enumerate(segs):
            pct = float(seg.get("pct", 0.0) or 0.0)
            start = int(cumulative_pct * total_words)
            if idx == len(segs) - 1:
                end = total_words
            else:
                end = int((cumulative_pct + pct) * total_words)
            cumulative_pct += pct

            content = " ".join(words[start:end]).strip()
            if not content:
                continue
            segment_name = str(seg.get("name") or f"Segment {idx + 1}").strip()
            sections.append(
                {
                    "section_label": _to_section_label(segment_name),
                    "heading": segment_name,
                    "content": content,
                }
            )
        if sections:
            return sections

    chapters = list(chapter_structure or [])

    if chapters:
        cumulative_pct = 0.0
        for idx, ch in enumerate(chapters):
            pct = float(ch.get("section_pct", 0.0) or 0.0)
            start = int(cumulative_pct * total_words)
            if idx == len(chapters) - 1:
                end = total_words
            else:
                end = int((cumulative_pct + pct) * total_words)
            cumulative_pct += pct

            content = " ".join(words[start:end]).strip()
            if not content:
                continue
            heading = str(ch.get("title") or f"Section {idx + 1}").strip()
            sections.append(
                {
                    "section_label": _to_section_label(heading),
                    "heading": heading,
                    "content": content,
                }
            )
        if sections:
            return sections

    cut_1 = int(0.2 * total_words)
    cut_2 = int(0.8 * total_words)
    slices = [
        ("Introduction", " ".join(words[:cut_1]).strip()),
        ("Main Analysis", " ".join(words[cut_1:cut_2]).strip()),
        ("Conclusion", " ".join(words[cut_2:]).strip()),
    ]
    for heading, content in slices:
        if content:
            sections.append(
                {
                    "section_label": _to_section_label(heading),
                    "heading": heading,
                    "content": content,
                }
            )
    return sections


def render_labeled_script(script_sections: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for sec in script_sections:
        label = str(sec.get("section_label") or "BODY").strip().upper()
        heading = str(sec.get("heading") or "Section").strip()
        content = str(sec.get("content") or "").strip()
        if not content:
            continue
        blocks.append(
            f"[SECTION: {label}]\n"
            f"HEADING: {heading}\n"
            f"CONTENT:\n{content}"
        )
    return "\n\n".join(blocks)


def assemble_sources(
    db_context: str,
    social_data: list[dict[str, Any]],
    news_data: list[dict[str, Any]],
    scraped_urls: list[str],
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    if (db_context or "").strip():
        sources.append(
            {
                "type": "database",
                "title": "Knowledge base (Supabase)",
                "url": None,
                "snippet": db_context[:200],
            }
        )
    for url in scraped_urls:
        sources.append({"type": "web_article", "title": url, "url": url, "snippet": ""})
    for s in (social_data or [])[:5]:
        sources.append(
            {
                "type": "social",
                "title": s.get("title", ""),
                "url": s.get("url"),
                "snippet": str(s.get("body", ""))[:200],
            }
        )
    for n in (news_data or [])[:5]:
        sources.append(
            {
                "type": "news",
                "title": n.get("title", ""),
                "url": n.get("url"),
                "snippet": str(n.get("body", ""))[:200],
            }
        )
    return sources


ANALYSIS_PROMPT_V2 = """
Analyse this YouTube script and return ONLY valid JSON.

Script:
{script}

Angle specification:
  story_frame target: {story_frame}
  system_dynamic target: {system_dynamic}

Return:
{{
  "examples_count": 0,
  "research_facts_count": 0,
  "proverbs_count": 0,
  
  "emotional_depth": "Low|Medium|High",
  "frame_executed": {{
    "story_frame_target": "{story_frame}",
    "is_executed": true,
    "confidence": "Low|Med|High",
    "evidence": ""
  }},
  "dynamic_revealed": {{
    "system_dynamic_target": "{system_dynamic}",
    "is_revealed": true,
    "confidence": "Low|Med|High",
    "evidence": ""
  }}
}}
"""


async def analyse_script_v2(script: str, angle: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = await groq_idea_generate(
            [
                {
                    "role": "user",
                    "content": ANALYSIS_PROMPT_V2.format(
                        script=(script or "")[:8000],
                        story_frame=str(angle.get("story_frame", "")),
                        system_dynamic=str(angle.get("system_dynamic", "")),
                    ),
                }
            ],
            model=GROQ_GENERATION_MODEL,
        )
        parsed = _parse_json_object(raw)
        if parsed:
            return parsed
    except Exception:
        pass
    return {
        "examples_count": 0,
        "research_facts_count": 0,
        "proverbs_count": 0,
        "emotional_depth": "Unknown",
        "frame_executed": {
            "story_frame_target": str(angle.get("story_frame", "")),
            "is_executed": None,
            "confidence": "Low",
            "evidence": "Analysis failed",
        },
        "dynamic_revealed": {
            "system_dynamic_target": str(angle.get("system_dynamic", "")),
            "is_revealed": None,
            "confidence": "Low",
            "evidence": "Analysis failed",
        },
    }


async def framecheck_generate_text(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    provider = SCRIPT_FRAMECHECK_PROVIDER
    if provider == "openrouter":
        try:
            return await openrouter_generate(messages)
        except Exception as exc:
            print(f"Framecheck OpenRouter failed, falling back to Groq: {exc}")
            return await groq_script_generate(messages)
    if provider == "auto":
        return await generate_script_content(messages)
    try:
        return await groq_script_generate(messages)
    except Exception as exc:
        print(f"Framecheck Groq failed, falling back to OpenRouter: {exc}")
        return await openrouter_generate(messages)


async def generate_with_frame_check(prompt: str, angle: dict[str, Any], story_frame: str) -> tuple[str, dict[str, Any], bool]:
    script_v1 = await framecheck_generate_text(prompt)
    analysis_v1 = await analyse_script_v2(script_v1, angle)
    frame_check = analysis_v1.get("frame_executed", {}) or {}
    needs_regen = bool(frame_check.get("is_executed") is False and str(frame_check.get("confidence")) == "High")
    if not needs_regen:
        return script_v1, analysis_v1, False
    corrective = (
        f"\n\nCORRECTION: Previous draft did not execute '{story_frame}' frame strongly enough. "
        f"Rewrite with '{story_frame}' as the primary structural device in each major section."
    )
    script_v2 = await framecheck_generate_text(prompt + corrective)
    analysis_v2 = await analyse_script_v2(script_v2, angle)
    if bool((analysis_v2.get("frame_executed", {}) or {}).get("is_executed")):
        return script_v2, analysis_v2, True
    return script_v1, analysis_v1, True


def _validate_script_entry(ctx: AgentPipelineContext, template_key_override: str | None = None) -> list[Any]:
    stale_h = staleness_hours(ctx.pipeline_assembled_at)
    if stale_h > 2.0:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "pipeline_context_stale",
                "staleness_hours": round(stale_h, 1),
                "message": "Re-run from TSS to refresh trend signals.",
            },
        )
    if (ctx.selected_idea or {}).get("idea_id") != ctx.selected_idea_id:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "selected_idea_id_mismatch",
                "provided": ctx.selected_idea_id,
                "cluster_id": (ctx.selected_idea or {}).get("idea_id"),
            },
        )
    if ctx.seo_output is None:
        raise HTTPException(status_code=400, detail="seo_output is required. Run /seo-agent before /generate-script.")
    if template_key_override and template_key_override not in TEMPLATE_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_template_key", "valid_keys": sorted(TEMPLATE_REGISTRY.keys())},
        )
    warnings: list[Any] = []
    if stale_h > 1.0:
        warnings.append("context_stale_warning")
    return warnings


async def run_script_sync_context(
    request: ScriptRequest,
    wpm: int,
) -> dict[str, Any]:
    ctx = request.context
    assert ctx is not None
    warnings = _validate_script_entry(ctx, request.template_key_override)
    target_wc = int((request.duration_minutes or 10) * wpm)
    depth_target = int((((ctx.selected_idea or {}).get("content_depth") or {}).get("required_words") or 2600))
    depth_warn = check_depth_alignment(target_wc, depth_target)
    if depth_warn:
        warnings.append(depth_warn)

    angle_for_prompt = extract_angle_for_prompt(ctx.gap_context or {})
    template_key, selection_method = select_template_key(
        scored_angle=ctx.gap_context or {},
        tss_scores=ctx.tss_scores or {},
        seo_output=ctx.seo_output or {},
        template_key_override=request.template_key_override,
    )
    template = TEMPLATE_REGISTRY[template_key]
    seo = ctx.seo_output or {}
    primary_keyword = ((seo.get("keyword_clusters") or {}).get("primary") or [""])[0]
    rec_title = str(((seo.get("recommended_titles") or [{}])[0]).get("title", "") or "")
    chapters = list(seo.get("chapter_structure") or [])
    chapter_scaffold = assemble_chapter_scaffold(chapters, int(request.duration_minutes or 10), wpm)

    social_data = list(ctx.social_data or [])
    news_data = list(ctx.news_data or [])
    db_ctx, web_ctx, social_data, news_data, truncation_applied = trim_to_budget(
        ctx.db_context or "",
        ctx.web_context or "",
        social_data,
        news_data,
        angle_spec_tokens=estimate_tokens(json.dumps(angle_for_prompt)),
        seo_section_tokens=estimate_tokens(json.dumps(seo)),
    )

    structure_section = assemble_structure_section(template_key, target_wc)
    social_summary = "\n".join(
        [f"- {s.get('title','')}: {str(s.get('body',''))[:100]}" for s in social_data[:5]]
    )
    news_summary = "\n".join([f"- {n.get('title','')}" for n in news_data[:5]])

    prompt = f"""
ROLE: You are a professional YouTube scriptwriter who writes engaging, research-backed, spoken scripts.

Creator type: {request.creator_type}
Emotional tone: {request.emotional_tone}
Audience: {request.audience_description}
Accent/dialect: {request.accent}

ANGLE SPECIFICATION:
- Stakeholder perspective: {angle_for_prompt.get('who')}
- Disciplinary lens: {angle_for_prompt.get('what')}
- Time/scale: {angle_for_prompt.get('when')}, {angle_for_prompt.get('scale')}
- System dynamic: {angle_for_prompt.get('system_dynamic')}
- Power layer: {angle_for_prompt.get('power_layer')}
- Narrative frame: {angle_for_prompt.get('story_frame')}
- Full angle: "{angle_for_prompt.get('angle_string')}"
- Opening hook seed: "{angle_for_prompt.get('hook_sentence')}"

SEO ALIGNMENT:
- Recommended title seed: "{rec_title}"
- Primary keyword: "{primary_keyword}"
- Chapter scaffold:
{chapter_scaffold}

TASK:
Write a spoken YouTube script of exactly {target_wc} words (±50). Duration: {request.duration_minutes} minutes.

STYLE RULES:
- Output only spoken dialogue — no section titles, stage directions, or metadata.
- Speak directly to the viewer — friendly, confident, slightly spontaneous.
- Hook viewers emotionally in the first 15–30 seconds.
- Keep script factual and grounded in provided research context.

{structure_section}

RESEARCH MATERIAL:
Knowledge base: {db_ctx}
Web sources: {web_ctx}
Social signals: {social_summary}
News context: {news_summary}
"""
    script_text, analysis, regeneration_attempted = await generate_with_frame_check(
        prompt,
        angle_for_prompt,
        str(angle_for_prompt.get("story_frame") or ""),
    )

    timestamps = compute_chapter_timestamps(script_text, chapters, wpm)
    template_segments = list(template.get("segments") or [])
    script_sections = build_script_sections(script_text, chapters, template_segments)
    script_labeled = render_labeled_script(script_sections)
    sources = assemble_sources(db_ctx, web_ctx, social_data, news_data, [])
    quality_gate_passed = bool((analysis.get("frame_executed", {}) or {}).get("is_executed"))

    return {
        "script": script_text,
        "estimated_word_count": len((script_text or "").split()),
        "script_sections": script_sections,
        "script_labeled": script_labeled,
        "sources": sources,
        "corrected_chapter_timestamps": timestamps,
        "analysis": {
            **analysis,
            "quality_gate_passed": quality_gate_passed,
        },
        "regeneration_attempted": regeneration_attempted,
        "truncation_applied": truncation_applied,
        "selected_template_key": template_key,
        "selected_template_name": template.get("name"),
        "template_selection_method": selection_method,
        "warnings": warnings,
    }


async def run_script_worker(job_id: str, request_body: dict[str, Any], user_id: str, wpm: int) -> None:
    try:
        supabase.table("script_jobs").update({"status": "running", "progress_pct": 20}).eq("id", job_id).execute()
        req = ScriptRequest.model_validate(request_body)
        result = await run_script_sync_context(req, wpm=wpm)
        result_analysis = dict(result.get("analysis") or {})
        if result.get("script_sections"):
            result_analysis["script_sections"] = result.get("script_sections")
        if result.get("script_labeled"):
            result_analysis["script_labeled"] = result.get("script_labeled")
        supabase.table("script_jobs").update(
            {
                "status": "completed",
                "progress_pct": 100,
                "result_script": result.get("script"),
                "result_analysis": result_analysis,
                "result_sources": result.get("sources"),
                "result_timestamps": result.get("corrected_chapter_timestamps"),
                "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        ).eq("id", job_id).execute()
    except Exception as exc:
        await issue_refund(user_id, job_id)
        supabase.table("script_jobs").update(
            {
                "status": "failed",
                "progress_pct": 100,
                "error_message": str(exc),
                "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        ).eq("id", job_id).execute()


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


@app.post("/analyze")
async def analyze(request: PromptRequest):
    try:
        youtube_result = await asyncio.to_thread(
            build_youtube_summary, request.topic
        )
        score = youtube_result.get("score") or youtube_result.get("youtube", {}).get("score", 0)
        if score == 100:
            tss_result = await pipeline_metrics(request)
            return tss_result
        else:
            eci_result = await eci(request)
            return eci_result
    except Exception as e:
        return {"error": str(e)}


@app.post("/pipeline-metrics")
async def pipeline_metrics(request: PromptRequest):
    try:
        trends_data = await asyncio.to_thread(get_trends_serpapi, request.topic)
        trend_dashboard = build_trend_dashboard(trends_data)
        social_result = await asyncio.to_thread(scan_topic, request.topic)
        youtube_result = await asyncio.to_thread(build_youtube_summary, request.topic)
        news_result = await asyncio.to_thread(build_news_summary,request.topic)
        return {
            "topic":   request.topic,
            "trends":  trend_dashboard,
            "youtube": youtube_result,
            "social":  social_result["dashboard"],
            "news_result" : news_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline metrics failed: {e}")

@app.post("/eci")
async def eci(request: PromptRequest):
    try:
        google_data = await asyncio.to_thread(get_google_trends_serpapi,request.topic)
        youtube_data = await asyncio.to_thread(get_youtube_data , request.topic)
        return {
         "google_data" : google_data,
         "youtube_data" : youtube_data  
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline metrics failed: {e}")

# ── /process-topic ───────────────────────────────────────────
@app.post("/process-topic")
async def process_topic(request: PromptRequest, background_tasks: BackgroundTasks):
    topic = (request.topic or "").strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic must be a non-empty string")

    print("Received /process-topic request; forwarding to /generate-ideas pipeline.")
    gen_request = GenerateIdeasRequest(
        topic=topic,
        max_angles=4,
        ideas_per_angle=3,
        used_angle_ids=[],
        force_refresh=False,
    )
    try:
        payload = await generate_ideas_endpoint(gen_request, background_tasks)
        if isinstance(payload, dict):
            payload["legacy_route"] = "process-topic"
        return payload
    except HTTPException:
        raise
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


    try:
        print(f"GENERATE IDEAS for topic: {topic}")

        tss_payload = await asyncio.wait_for(run_tss(topic), timeout=TSS_TIMEOUT_SEC)
        cags_payload = tss_payload.get("cags") or {}
        gap_angles = cags_payload.get("gap_angles") or []
        briefs = cags_payload.get("briefs") or []
        perspective_tree = cags_payload.get("perspective_tree") or []
        if not gap_angles or not perspective_tree:
            raise HTTPException(status_code=422, detail="No viable CAGS angles were produced.")

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
            deepseek_client=deepseek_client,
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
        final_ideas = []
        final_descriptions = []
        for cluster in idea_clusters.get("idea_clusters") or []:
            for variant in cluster.get("idea_variants") or []:
                title = str(variant.get("title") or "").strip()
                description = str(variant.get("description") or "").strip()
                if title and description:
                    final_ideas.append(title)
                    final_descriptions.append(description)
        idea_clusters["ideas"] = final_ideas
        idea_clusters["descriptions"] = final_descriptions
        idea_clusters["scraped_text_context"] = f"DB CONTEXT:\n{db_context}\n\nWEB CONTEXT:\n{web_context}"
        idea_clusters["total_request_time_sec"] = round(time.time() - total_start_time, 2)
        if len(final_ideas) > 0:
            if _payload_uses_fallback_variants(idea_clusters):
                print(f"Skipping cache write for '{topic}' because fallback variants were used.")
                idea_clusters["cache_write_skipped"] = "fallback_variants"
                print(f"Total /generate-ideas time: {idea_clusters['total_request_time_sec']:.2f}s")
                return idea_clusters
        else:
            print(f"Skipping cache write for '{topic}' because no ideas were generated.")

        print(f"Total /generate-ideas time: {idea_clusters['total_request_time_sec']:.2f}s")
        return idea_clusters
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /generate-ideas: {e}")
        return {"error": "An error occurred in the idea generation pipeline."}


async def pick_best_template(topic: str, templates: list) -> dict:
    """Uses LLM to pick the best matching template based on topic + each template's 'about' field."""
    try:
        options = "\n\n".join([
            f"Key: {t['key']}\nTitle: {t['tittle']}\nAbout: {t['about']}"
            for t in templates
        ])

        prompt = f"""
        You are a content structure selector.

        Given a video topic, pick the SINGLE best template from the options below.
        Match based on the topic's nature and the template's 'about' description.

        Return ONLY the key value (e.g. "1" or "2"). Nothing else.

        Topic: \"\"\"{topic}\"\"\"

        Templates:
        {options}
                """

        response = deepseek_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "Return only the key value of the best matching template."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        best_key = response.choices[0].message.content.strip()
        matched = next((t for t in templates if str(t["key"]) == best_key), templates[0])
        return matched

    except Exception as e:
        print(f"Template selection failed: {e}")
        return templates[0]  



async def get_structure(content: str) -> dict:
    try:
        prompt = f"""
        You are a strict content classifier.

        Classify the given content into exactly ONE category.

        Return ONLY the category name.

        Categories:
        - PHILOSOPHY & IDEAS
        - PSYCHOLOGY & BEHAVIOUR
        - HISTORY & CIVILISATION
        - BIOGRAPHY & LEGACY
        - SCIENCE & TECHNOLOGY
        - ECONOMICS & SOCIETY
        - ANALYSIS & BREAKDOWNS
        - NEWS & CONTEMPORARY EVENTS
        - THOUGHT LEADERSHIP & DISCUSSION
        - MOTIVATIONAL & INSPIRATIONAL

        Content:
        \"\"\"{content}\"\"\"
        """

        response = deepseek_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "Return only the category name."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        category = response.choices[0].message.content.strip()

        return {"category": category}

    except Exception as e:
        return {"category": "UNKNOWN", "error": str(e)}




async def get_channel_profile(userId: str):
    print(userId)
    try:
        channel_profile = (
            supabase
            .table("user_channel_memory_input")
            .select("Summary")
            .eq("userId", userId)
            .execute()
        )

        return channel_profile.data

    except Exception as e:
        print(e)



class UnlockRequest(BaseModel):
    userId: str
    duration: int



@app.post("/unlock")
async def cut_credits(request: UnlockRequest):
    try:
        res = supabase.table('user_profiles') \
            .select('credits_remaining, user_tier') \
            .eq('id', request.userId) \
            .single() \
            .execute()

        old_credits = res.data["credits_remaining"]
        user_tier = res.data["user_tier"]

        if old_credits <= 0:
            return {"message": "credits not sufficient"}

        if old_credits < request.duration:
            return {"message": "credits not sufficient"}

        new_credits = old_credits - request.duration

        supabase.table('user_profiles') \
            .update({'credits_remaining': new_credits}) \
            .eq('id', request.userId) \
            .execute()

        if user_tier != "free":
            res2 = supabase.table('subscriptions') \
                .select('credits') \
                .eq('userId', request.userId) \
                .single() \
                .execute()

            subscription_credits = res2.data["credits"]

            if subscription_credits <= 0 or subscription_credits < request.duration:
                return {"message": "credits not sufficient"}

            new_subscription_credits = subscription_credits - request.duration

            supabase.table('subscriptions') \
                .update({'credits': new_subscription_credits}) \
                .eq('userId', request.userId) \
                .execute()

            print("subscription update done")

        print("profile update done, tier:", user_tier)
        return {"message": "success"}

    except Exception as e:
        print("error:", e)
        return {"message": "error"}
    

@app.post("/generate-script") 
async def generate_script(request: ScriptRequest, background_tasks: BackgroundTasks):
    total_start_time = time.time()
    print(f"SCRIPT GENERATION: Received request for topic: '{request.topic}'")

    try:
        channel_profile = await get_channel_profile(request.userId)
        summary = channel_profile[0]["Summary"] if channel_profile else None

        # Generate HyDE document FIRST before anything else 
        hyde_document = request.topic  
        try:
            hyde_prompt = f"""
            Write a short, factual, encyclopedia-style paragraph that provides a direct answer
            to the following topic. Be concise and include key terms.

            Topic: "{request.topic}"
            """
            hyde_completion = await groq_client.chat.completions.create(
                messages=[{"role": "user", "content": hyde_prompt}],
                model=GROQ_GENERATION_MODEL,
            )
            hyde_document = hyde_completion.choices[0].message.content
            print(f"--- HyDE DOCUMENT GENERATED ---\n{hyde_document}\n--- END HyDE DOCUMENT ---")
        except Exception as exc:
            print(f"--- HyDE generation failed, using raw topic as fallback: {exc} ---")


        content_category = await get_structure(request.topic)
        category = content_category["category"]
        print(f"Category: {category}")

        res = supabase.table("script_structures").select("*").eq("catergory name", category).execute()
        all_templates_raw = res.data

        all_templates = []
        for row in all_templates_raw:
            for template in row["Structure"]:
                all_templates.append(template)

        best_template = await pick_best_template(request.topic, all_templates)
        print(f"Selected template: {best_template['tittle']}")

        structure = best_template.get("segments", [])

        filtered_structure = structure

        template_meta = {
        "title": best_template.get("tittle"),
        "about": best_template.get("about"),
        }

        selected_idea_id = random.randint(1, 1000)  
        selected_angle_id = random.randint(1, 1000)

        json_generation_prompt = f"""
        You are an expert YouTube SEO strategist and content ideation assistant.

        Return ONLY valid JSON.

        OUTPUT FORMAT:

        {{
        "context": {{
            "topic": "",
            "keywords": [],
            "selected_idea": {{
            "title": "",
            "idea_id": "{selected_idea_id}"
            }},
            "selected_idea_id": "{selected_idea_id}",
            "selected_angle_id": "{selected_angle_id}",
            "gap_context": {{
            "problem": "",
            "insight": "",
            "angle_string": ""
            }},
            "pipeline_assembled_at": "2026-04-10T16:00:00"
        }}
        }}

        RULES:
        - Return ONLY valid JSON
        - No markdown, no explanation, no comments
        - keywords: 8–15 items
        - selected_idea.idea_id MUST match selected_idea_id

        INPUT:
        Topic: {request.topic}
        """

        response1 = deepseek_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "You must return only valid JSON"},
                {"role": "user", "content": json_generation_prompt},
            ],
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}}
        )

        text = response1.choices[0].message.content
        data = json.loads(text) 

        request_obj = SEOAgentRequest.model_validate(data)
        res = await seo_agent(request_obj)
        print(res)

        db_task = asyncio.create_task(get_db_context(request.topic, hyde_document))  
        await asyncio.sleep(11)

        db_results = []
        new_articles = []
        scraped_urls = set()
        base_keywords = []

        if db_task.done():
            db_results = db_task.result()
            print(f"--- DB task finished early. Found {len(db_results)} documents. ---")

        if len(db_results) >= 3:
            print("--- DB HIT: Performing LIGHT web scrape for latest news. ---")
            new_articles = await get_latest_news_context(request.topic, scraped_urls)

        else:
            print("--- DB MISS or SLOW: Initiating DEEP web scrape. ---")

            base_keywords = [request.topic]  
            try:
                keyword_prompt = f"""
                Your ONLY task is to generate 5 diverse search engine keyword phrases for the topic: '{request.topic}'.
                Follow these rules STRICTLY:
                1. Return ONLY the 5 phrases, nothing else.
                2. DO NOT add numbers, markdown, bullet points, explanations, or any introductory text.
                3. Each phrase must be on a new line.
                4. Make them diverse — cover different angles, audiences, and search intents.

                EXAMPLE INPUT: Is coding dead?
                EXAMPLE OUTPUT:
                future of programming jobs automation
                AI replacing software developers
                demand for software engineers 2025
                will programmers become obsolete
                coding careers vs AI tools
                """

                kw_completion = await groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": keyword_prompt}],
                    model=GROQ_GENERATION_MODEL,
                )
                raw_text = kw_completion.choices[0].message.content.strip()
                print(f"--- DEEP SCRAPE: Raw keywords from Groq:\n{raw_text} ---")

                keywords_in_quotes = re.findall(r'"(.*?)"', raw_text)
                if keywords_in_quotes:
                    base_keywords = keywords_in_quotes
                else:
                    base_keywords = [kw.strip() for kw in raw_text.split('\n') if kw.strip()]

            except Exception as e:
                print(f"--- DEEP SCRAPE: Keyword generation failed, using topic as fallback: {e} ---")
                base_keywords = [request.topic]

            targeted_keywords = (
                base_keywords +
                [f"{request.topic} 2025"] +
                [f"{kw} site:reddit.com" for kw in base_keywords[:3]]
            )
            targeted_keywords = list(dict.fromkeys(targeted_keywords))  
            print(f"--- DEEP SCRAPE: Searching with {len(targeted_keywords)} keywords: {targeted_keywords} ---")
            new_articles = await deep_search_and_scrape(targeted_keywords, scraped_urls)


        if not db_task.done():
            print("--- Waiting for DB task to complete... ---")
            db_results = await db_task
            print(f"--- DB task finished. Found {len(db_results)} documents. ---")

        # ── Build contexts ──────────────────────────────────────────────────────
        db_context, web_context = "", ""
        if db_results:
            db_context = "\n\n".join([item['content'] for item in db_results])
        if new_articles:
            web_context = "\n\n".join([f"Source: {art['title']}\n{art['text']}" for art in new_articles])

        # ── Generate script ─────────────────────────────────────────────────────
        print("SCRIPT GENERATION: Generating personalized script...")

        WORDS_PER_MINUTE = 130
        target_duration = request.duration_minutes if request.duration_minutes else 10
        target_word_count = target_duration * WORDS_PER_MINUTE
        print(f"Targeting {target_duration} minutes / approx. {target_word_count} words.")

        channel_profile = ""

        script_prompt = f"""
        You are a professional YouTube scriptwriter who creates natural, engaging, and conversational scripts that feel like a real YouTuber speaking directly to the camera.

        **Your Task:**
        You must write the script EXACTLY in this voice, tone, and structure:
        {summary}
        
        Interpret this as the creator's permanent speaking identity. Every line of the script must reflect this style. Do NOT ignore or average it out.

        Generate a complete YouTube video script of approximately **{target_duration} minutes** (~{target_word_count} words) based on the **main topic** below, using the provided **research context**.

        **Script Style & Flow:**
        - Output only the spoken dialogue — what the YouTuber would actually say aloud.
        - **Do NOT include** section titles, notes, stage directions, or metadata.
        - Speak directly to the viewer — friendly, confident, slightly spontaneous, and off-the-cuff.
        - Use **short and medium-length sentences**, natural pauses (…) or dashes, and occasional repetition for emphasis.
        - Include interjections, rhetorical questions, playful digressions, humor, and brief asides ("Wait, actually…", "Can you believe that…?", "By the way…").
        - Include personal anecdotes or opinions ("I remember…", "When I tried this…").
        - Use **visual and emotional imagery** to make scenes vivid ("Imagine this…", "Picture it like…").
        - Hook viewers emotionally in the first 15–30 seconds.
        - Alternate between facts, insights, reactions, and short reflections to keep pacing dynamic.
        - Treat the script as a conversation with the audience — inclusive language like "you guys", "we all", "my friends".
        - Build suspense naturally with rhetorical questions, mini cliffhangers, or curiosity hooks.
        - Use relatable analogies or humor when explaining complex topics.
        - Occasionally reference the creator's regional or cultural context for relatability.
        - Maintain natural pacing as if recording live — mix excitement, storytelling, and factual explanation.
        - Stay close to **{target_word_count} words** (±50).

        **Main Topic/Idea:** "{request.topic}"

        **Research Context:**
        FOUNDATIONAL KNOWLEDGE (from database): {db_context}
        LATEST NEWS (from web): {web_context}

        **Additional Notes:**
        - Make the opening a curiosity-driven hook that emotionally pulls the viewer in within 15–30 seconds.
        - Use storytelling techniques: tension, suspense, surprise, and moral dilemmas when relevant.
        - Make historical or technical details feel immersive and personal, not like a lecture.
        - Emphasize the narrative arc: build curiosity, climax, and reflection for the audience.
        - Ensure adaptability: script should feel natural regardless of topic, duration, or target audience.
        """

        response3 = deepseek_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "You must return only valid JSON"},
                {"role": "user", "content": script_prompt},
            ],
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}}
        )

        text3 = response3.choices[0].message.content

        total_end_time = time.time()
        print(f"--- PROFILING: Script generation took {total_end_time - total_start_time:.2f} seconds ---")

        # ── Analyze script ──────────────────────────────────────────────────────
        ANALYSIS_PROMPT_TEMPLATE = """
        You are an expert script analyzer.

        Your job is to carefully analyze the YouTube script and IDENTIFY + COUNT specific elements.

        IMPORTANT: Do NOT assume zero unless you are absolutely certain none exist.

        ----------------------
        DEFINITIONS (STRICT)
        ----------------------

        1. Real-world Examples:
        - Any specific story, scenario, case study, or real-life situation
        - Includes hypothetical but realistic situations
        - Example: "A student who studies daily will succeed"

        2. Research Facts / Stats:
        - Any number, percentage, study, data point, or measurable claim
        - Even approximate values count
        - Example: "90% of startups fail", "Studies show..."

        3. Proverbs / Sayings:
        - Common traditional proverbs, idioms, or widely recognized sayings
        - Must be culturally established phrases, not personal quotes or random sentences
        - Typically short, fixed expressions used to convey general life wisdom

        4. Emotional Depth:
        - LOW → Informational, dry, no emotional hooks
        - MEDIUM → Some engagement, mild storytelling or relatability
        - HIGH → Strong emotional storytelling, persuasive, engaging

        5. history Facts:
        - Verified historical events, timelines, or occurrences from the past
        - Must be factual and time-specific

        ----------------------
        PROCESS (MANDATORY)
        ----------------------

        Step 1: Extract all matches for each category
        Step 2: Count them
        Step 3: Return result

        If unsure → COUNT it (be slightly generous, not strict)

        ----------------------
        OUTPUT FORMAT (STRICT JSON ONLY)
        ----------------------
        {{
        "examples_count": <number>,
        "research_facts_count": <number>,
        "proverbs_count": <number>,
        "history_facts":<number>,
        "emotional_depth": "Low | Medium | High"
        }}
        ----------------------
        SCRIPT
        ----------------------
        {script_text}
        ----------------------
        """

        print("SCRIPT ANALYSIS: Analyzing generated script...")
        analysis_start_time = time.time()
        analysis_prompt_filled = ANALYSIS_PROMPT_TEMPLATE.format(script_text=text3)

        response4 = deepseek_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "You must return only valid JSON"},
                {"role": "user", "content": analysis_prompt_filled},
            ],
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}}
        )

        text4 = response4.choices[0].message.content

        analysis_end_time = time.time()
        print(f"--- PROFILING: Script analysis took {analysis_end_time - analysis_start_time:.2f} seconds ---")

        analysis_results = {
            "examples_count": 0,
            "research_facts_count": 0,
            "proverbs_count": 0,
            "emotional_depth": "Unknown",
            "history": 0
        }
        try:
            analysis_data = json.loads(text4)
            analysis_results["examples_count"] = analysis_data.get("examples_count", 0)
            analysis_results["research_facts_count"] = analysis_data.get("research_facts_count", 0)
            analysis_results["proverbs_count"] = analysis_data.get("proverbs_count", 0)
            analysis_results["emotional_depth"] = analysis_data.get("emotional_depth", "Unknown")
            analysis_results["history"] = analysis_data.get("history_facts", 0)
            print(f"Script Analysis Results: {analysis_results}")
        except json.JSONDecodeError:
            print("SCRIPT ANALYSIS: Failed to parse analysis JSON response from AI.")
        except Exception as e:
            print(f"SCRIPT ANALYSIS: Error during analysis parsing: {e}")

        total_end_time = time.time()
        print(f"--- PROFILING: Total /generate-script analysis request time was {total_end_time - total_start_time:.2f} seconds ---")

        generated_word_count = len(text3.split())
        print(f"Generated script word count: approx. {generated_word_count}")

        # ── Extract category ────────────────────────────────────────────────────
        category_prompt = f"""
        You are a content categorization expert.

        Given the topic and script below, return ONLY valid JSON with the main category and up to 2 subcategories.

        OUTPUT FORMAT:
        {{
            "category": "<main category>",
            "subcategories": ["<subcategory 1>", "<subcategory 2>"]
        }}

        RULES:
        - Return ONLY valid JSON, no markdown, no explanation
        - category: broad genre (e.g. "Technology", "Finance", "Health", "Education", "Entertainment")
        - subcategories: more specific niches, max 2 (e.g. ["Artificial Intelligence", "Future of Work"])
        - If only 1 subcategory fits, return a list with 1 item

        TOPIC: {request.topic}
        SCRIPT (first 500 words): {" ".join(text3.split()[:500])}
        """

        category_response = deepseek_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "You must return only valid JSON"},
                {"role": "user", "content": category_prompt},
            ],
            stream=False
        )

        script_categories = {"category": "Unknown", "subcategories": []}
        try:
            raw_cat = category_response.choices[0].message.content
            script_categories = json.loads(raw_cat)
            script_categories["subcategories"] = script_categories.get("subcategories", [])[:2]
            print(f"Script categories: {script_categories}")
        except (json.JSONDecodeError, Exception) as e:
            print(f"Category parsing failed: {e}")

        if new_articles:
            for article in new_articles:
                background_tasks.add_task(
                    add_scraped_data_to_db,
                    article['title'],
                    article['text'],
                    article['url'],
                    script_categories.get("category", ""),
                    request.topic,
                    script_categories.get("subcategories", []),
                )
            print(f"BACKGROUND TASKS: Scheduled {len(new_articles)} articles for DB upload.")

        # ── Return ──────────────────────────────────────────────────────────────
        return {
            "script": text3,
            "estimated_word_count": generated_word_count,
            "source_urls": list(scraped_urls),
            "analysis": analysis_results,
            "structure": filtered_structure,
            "template_meta" : template_meta,
            "seo": res,
            "category": script_categories["category"],
            "subcategories": script_categories["subcategories"],
        }

    except Exception as e:
        print(f"SCRIPT GENERATION: An error occurred: {e}")
        return {"error": "An error occurred during the script generation pipeline."}



import os
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
 
 
def generate_invoice_pdf(
    invoice_no,
    customer_name,
    customer_address,
    customer_phone,
    item_name,
    amount,
    plan,
    due_date=None,
    output_dir="invoices",
):
    styles = getSampleStyleSheet()
 
    brand_style         = ParagraphStyle('Brand',        parent=styles['Normal'], fontSize=24,  fontName='Helvetica-Bold', textColor=colors.HexColor('#1a1a2e'), alignment=TA_LEFT)
    company_info_style  = ParagraphStyle('CompanyInfo',  parent=styles['Normal'], fontSize=8.5, fontName='Helvetica',      textColor=colors.HexColor('#444444'), alignment=TA_LEFT, leading=14)
    invoice_label_style = ParagraphStyle('InvoiceLabel', parent=styles['Normal'], fontSize=12,  fontName='Helvetica-Bold', textColor=colors.HexColor('#1a1a2e'), alignment=TA_RIGHT)
    section_header_style= ParagraphStyle('SectionHdr',   parent=styles['Normal'], fontSize=7.5, fontName='Helvetica-Bold', textColor=colors.HexColor('#888888'), spaceAfter=2)
    body_style          = ParagraphStyle('Body',         parent=styles['Normal'], fontSize=10,  fontName='Helvetica',      textColor=colors.HexColor('#1a1a2e'), leading=15)
    body_bold_style     = ParagraphStyle('BodyBold',     parent=styles['Normal'], fontSize=10,  fontName='Helvetica-Bold', textColor=colors.HexColor('#1a1a2e'), leading=15)
 
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{invoice_no}.pdf")
 
    PAGE_W, PAGE_H = A4
    LM = RM = 20 * mm
    TM = 18 * mm
    BM = 18 * mm
    W  = PAGE_W - LM - RM
 
    FOOTER_H = 24 * mm
    FOOTER_Y = BM
 
    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#cccccc'))
        canvas.setLineWidth(0.8)
        canvas.roundRect(LM, FOOTER_Y, W, FOOTER_H, 3, stroke=1, fill=0)
        cx = PAGE_W / 2
        y1 = FOOTER_Y + FOOTER_H - 6    * mm
        y2 = FOOTER_Y + FOOTER_H - 11   * mm
        y3 = FOOTER_Y + FOOTER_H - 15.5 * mm
        y4 = FOOTER_Y + FOOTER_H - 19.5 * mm
        canvas.setFont('Helvetica-Bold', 8.5)
        canvas.setFillColor(colors.HexColor('#1a1a2e'))
        canvas.drawCentredString(cx, y1, "Details Under GST")
        canvas.drawCentredString(cx, y2, "Morpho Technologies Pvt. Ltd.")
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#333333'))
        canvas.drawCentredString(cx, y3, "Flat no: 502, Plot no. MIG 891, KPHB Phase 3, Kukatpally, Hyderabad, Telangana, India - 500072")
        canvas.drawCentredString(cx, y4, "GSTIN: 36AAQCM4860P1ZK")
        canvas.setFont('Helvetica', 7.5)
        canvas.setFillColor(colors.HexColor('#999999'))
        canvas.drawCentredString(cx, FOOTER_Y - 5 * mm, "This is a computer generated invoice.")
        canvas.restoreState()
 
    doc = SimpleDocTemplate(
        file_path, pagesize=A4,
        rightMargin=RM, leftMargin=LM,
        topMargin=TM, bottomMargin=BM + FOOTER_H + 12 * mm,
    )
 
    elements = []
 
    # ── TITLE ROW ──
    elements.append(Table(
        [[Paragraph("<b>StoryBit</b>", brand_style), Paragraph("TAX INVOICE", invoice_label_style)]],
        colWidths=[W*0.55, W*0.45],
        style=TableStyle([
            ('VALIGN',        (0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',   (0,0),(-1,-1),0),
            ('RIGHTPADDING',  (0,0),(-1,-1),0),
            ('TOPPADDING',    (0,0),(-1,-1),0),
            ('BOTTOMPADDING', (0,0),(-1,-1),0),
        ])
    ))
    elements.append(Spacer(1, 9*mm))
 
    # ── COMPANY INFO ──
    for line in [
        "Flat no. 502, Meenakshi enclave MIG 891",
        "KPHB phase 3, Kukatpally, Hyderabad, 500072",
        "GSTIN: 36AAQCM4860P1ZK",
        "support@storybit.tech",
    ]:
        elements.append(Paragraph(line, company_info_style))
 
    elements.append(Spacer(1, 5*mm))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1a1a2e')))
    elements.append(Spacer(1, 6*mm))
 
    # ── INVOICE META ──
    if due_date is None:
        due_date = datetime.datetime.now() + datetime.timedelta(days=7)
    due_date_str = (
        datetime.datetime.fromisoformat(due_date[:19]).strftime('%d %b %Y')
        if isinstance(due_date, str) else due_date.strftime('%d %b %Y')
    )
    meta_table = Table([
        [Paragraph("INVOICE NO.", section_header_style),  Paragraph("INVOICE DATE", section_header_style), Paragraph("DUE DATE", section_header_style)],
        [Paragraph(f"<b>{invoice_no}</b>", body_bold_style), Paragraph(f"<b>{datetime.datetime.now().strftime('%d %b %Y')}</b>", body_bold_style), Paragraph(f"<b>{due_date_str}</b>", body_bold_style)],
    ], colWidths=[W*0.34, W*0.33, W*0.33])
    meta_table.setStyle(TableStyle([
        ('LEFTPADDING',   (0,0),(-1,-1), 0),
        ('RIGHTPADDING',  (0,0),(-1,-1), 0),
        ('BOTTOMPADDING', (0,0),(-1,-1), 2),
        ('TOPPADDING',    (0,0),(-1,-1), 2),
        ('ALIGN', (1,0),(1,-1),'CENTER'),
        ('ALIGN', (2,0),(2,-1),'RIGHT'),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 6*mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#dddddd')))
    elements.append(Spacer(1, 6*mm))
 
    # ── BILL TO ──
    elements.append(Paragraph("BILL TO", section_header_style))
    elements.append(Spacer(1, 1*mm))
    elements.append(Paragraph(f"<b>{customer_name}</b>", body_bold_style))
    elements.append(Paragraph(customer_address, body_style))
    elements.append(Paragraph(f"Phone: {customer_phone}", body_style))
    elements.append(Spacer(1, 7*mm))
 
    # ── AMOUNTS ──
    subtotal    = round(amount, 2)
    cgst        = round(amount * 0.09, 2)
    sgst        = round(amount * 0.09, 2)
    gst_total   = round(cgst + sgst, 2)
    grand_total = round(subtotal + gst_total, 2)
 
    # ── COLUMN WIDTHS — shared by item row AND gst summary rows ──
    # ITEM(0) | PLAN(1) | RATE(2) | QTY(3) | TOTAL(4)
    CW = [W*0.34, W*0.12, W*0.18, W*0.12, W*0.24]
 
    def lp(bold=False, tc=colors.HexColor('#1a1a2e')):
        return ParagraphStyle('_l', parent=styles['Normal'], fontSize=10,
                              fontName='Helvetica-Bold' if bold else 'Helvetica',
                              textColor=tc, alignment=TA_LEFT)
 
    def rp(bold=False, tc=colors.HexColor('#1a1a2e')):
        return ParagraphStyle('_r', parent=styles['Normal'], fontSize=10,
                              fontName='Helvetica-Bold' if bold else 'Helvetica',
                              textColor=tc, alignment=TA_RIGHT)
 
    # GST summary rows — cols 0-2 are blank, col 3 = label, col 4 = value
    gst_defs = [
        ("Subtotal",        f"Rs. {subtotal:.2f}",    False, colors.HexColor('#f8f8fb'), colors.HexColor('#1a1a2e')),
        ("CGST (9%)",       f"Rs. {cgst:.2f}",        False, colors.HexColor('#f8f8fb'), colors.HexColor('#1a1a2e')),
        ("SGST (9%)",       f"Rs. {sgst:.2f}",        False, colors.HexColor('#f8f8fb'), colors.HexColor('#1a1a2e')),
        ("Total GST (18%)", f"Rs. {gst_total:.2f}",   False, colors.HexColor('#eef0f8'), colors.HexColor('#1a1a2e')),
        ("Grand Total",     f"Rs. {grand_total:.2f}", True,  colors.HexColor('#1a1a2e'), colors.white),
    ]
 
    table_data = [
        ['ITEM', 'PLAN', 'RATE', 'QTY', 'TOTAL'],                                   # row 0
        [item_name, plan.title(), f"Rs. {amount:.2f}", "1", f"Rs. {amount:.2f}"],   # row 1
    ]
    for label, value, bold, bg, tc in gst_defs:
        table_data.append(["", "", "", Paragraph(label, lp(bold, tc)), Paragraph(value, rp(bold, tc))])
 
    WHITE = colors.white
 
    ts = TableStyle([
        # header
        ('BACKGROUND',    (0,0),(-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',     (0,0),(-1,0), WHITE),
        ('FONTNAME',      (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,0), 9),
        ('TOPPADDING',    (0,0),(-1,0), 9),
        ('BOTTOMPADDING', (0,0),(-1,0), 9),
        ('ALIGN',         (0,0),(0,0),  'LEFT'),
        ('ALIGN',         (1,0),(1,0),  'CENTER'),
        ('ALIGN',         (2,0),(-1,0), 'RIGHT'),
        ('ALIGN',         (3,0),(3,0),  'CENTER'),
        # item row
        ('BACKGROUND',    (0,1),(-1,1), colors.HexColor('#f8f8fb')),
        ('FONTNAME',      (0,1),(-1,1), 'Helvetica'),
        ('FONTSIZE',      (0,1),(-1,1), 10),
        ('TOPPADDING',    (0,1),(-1,1), 10),
        ('BOTTOMPADDING', (0,1),(-1,1), 10),
        ('ALIGN',         (0,1),(0,1),  'LEFT'),
        ('ALIGN',         (1,1),(1,1),  'CENTER'),
        ('ALIGN',         (2,1),(-1,1), 'RIGHT'),
        ('ALIGN',         (3,1),(3,1),  'CENTER'),
        # grid only for header + item rows
        ('GRID',          (0,0),(-1,1), 0.5, colors.HexColor('#dddddd')),
        # global padding
        ('LEFTPADDING',   (0,0),(-1,-1), 8),
        ('RIGHTPADDING',  (0,0),(-1,-1), 8),
        # suppress ALL borders on left 3 cols of gst rows (rows 2+)
        ('BACKGROUND',    (0,2),(2,-1), WHITE),
        ('LINEABOVE',     (0,2),(2,-1), 0, WHITE),
        ('LINEBELOW',     (0,2),(2,-1), 0, WHITE),
        ('LINEBEFORE',    (0,2),(0,-1), 0, WHITE),
        ('LINEAFTER',     (2,2),(2,-1), 0, WHITE),
        ('TOPPADDING',    (0,2),(2,-1), 0),
        ('BOTTOMPADDING', (0,2),(2,-1), 0),
    ])
 
    for i, (label, value, bold, bg, tc) in enumerate(gst_defs):
        r = 2 + i
        ts.add('BACKGROUND',    (3,r),(4,r), bg)
        ts.add('GRID',          (3,r),(4,r), 0.5, colors.HexColor('#dddddd'))
        ts.add('TOPPADDING',    (3,r),(4,r), 6)
        ts.add('BOTTOMPADDING', (3,r),(4,r), 6)
 
    combined = Table(table_data, colWidths=CW)
    combined.setStyle(ts)
    elements.append(combined)
    elements.append(Spacer(1, 8*mm))
 
    doc.build(elements, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return file_path



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



@app.post("/payments/webhook")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str | None = Header(None),
):
    body = await request.body()

    if not x_razorpay_signature:
        raise HTTPException(status_code=400, detail="Missing signature header.")

    if not RAZORPAY_WEBHOOK_SECRET or not razorpay_client:
        print("Webhook received but service not configured.")
        return {"status": "Webhook ignored"}
    
    invoice_url = None

    try:
        razorpay_client.utility.verify_webhook_signature(
            body.decode('utf-8'),
            x_razorpay_signature,
            RAZORPAY_WEBHOOK_SECRET,
        )
    except razorpay.errors.SignatureVerificationError as e:
        print(f"Webhook signature failed: {e}")
        print(f"DEBUG secret used: '{RAZORPAY_WEBHOOK_SECRET}'")
        print(f"DEBUG signature header: '{x_razorpay_signature}'")
        print(f"DEBUG body preview: {body[:200]}")
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")
    except Exception as e:
        print(f"Webhook verification error: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing error.")

    try:
        event_data = json.loads(body)
        event_type = event_data.get('event')
        print(f"Received webhook event: {event_type}")

        # ORDER PAID
        if event_type == 'order.paid':
            order_entity   = event_data['payload']['order']['entity']
            payment_entity = event_data['payload']['payment']['entity']

            order_id    = order_entity.get('id', 'unknown')
            payment_id  = payment_entity.get('id', 'unknown')
            amount_paid = order_entity.get('amount', 0) / 100

            notes       = order_entity.get('notes', {})
            user_id     = notes.get('user_id')
            target_tier = notes.get('target_tier')

            if not user_id or not target_tier:
                print(f"ERROR: Missing notes in order {order_id}.")
                return {"status": "error", "message": "Missing required order notes."}

            plan_config = {
                'basic': {'credits': 100, 'validity_days': 30},
                'pro':   {'credits': 200, 'validity_days': 30},
            }
            config = plan_config.get(target_tier.lower())
            if not config:
                print(f"ERROR: Unknown tier '{target_tier}' in order {order_id}.")
                return {"status": "error", "message": "Unknown plan tier."}

            credits_to_add = config['credits']
            validity_days  = config['validity_days']
            now            = datetime.datetime.now(datetime.timezone.utc)
            validity_date  = now + datetime.timedelta(days=validity_days)

            # --- Update user profile ---
            try:
                profile_resp = (
                    supabase.table('user_profiles')
                    .select('credits_remaining')
                    .eq('id', user_id)
                    .single()
                    .execute()
                )
                current_credits = (
                    profile_resp.data.get('credits_remaining', 0)
                    if profile_resp.data else 0
                )
                new_credits = current_credits + credits_to_add

                update_result = (
                    supabase.table('user_profiles')
                    .update({'user_tier': target_tier, 'credits_remaining': new_credits})
                    .eq('id', user_id)
                    .execute()
                )
                if update_result.data:
                    print(f"Updated user {user_id} → tier '{target_tier}', credits {new_credits}.")
                else:
                    print(f"WARN: Failed to update profile for {user_id} after payment {payment_id}.")

            except APIError as e:
                print(f"ERROR: Supabase profiles error for {user_id}: {e}")
            except Exception as e:
                print(f"ERROR: Unexpected profiles error for {user_id}: {e}")

            # --- Insert subscription row ---
            try:
                subscription_row = {
                    "userId":               user_id,
                    "amount":               amount_paid,
                    "plan":                 target_tier.lower(),
                    "purchased_date":       now.isoformat(),
                    "validity":             validity_date.isoformat(),
                    "credits":              credits_to_add,
                    "payment_status":       "paid",
                    "rayzorpay_payment_id": payment_id,
                    "razorpay_order_id":    order_id,
                }
                sub_result = (
                    supabase.table('subscriptions')
                    .upsert(subscription_row, on_conflict='userId')
                    .execute()
                )

                if sub_result.data:
                    print(f"Inserted subscription row for user {user_id}, order {order_id}.")

                    try:
                        profile_data = (
                            supabase.table("user_profiles")
                            .select("full_name, phone, billing_address")
                            .eq("id", user_id)
                            .single()
                            .execute()
                        )

                        profile          = profile_data.data or {}
                        customer_name    = profile.get("full_name", "Customer")
                        customer_phone   = profile.get("phone", "")
                        customer_address = profile.get("billing_address", "")


                        invoice_path = generate_invoice_pdf(
                            invoice_no=generate_invoice_number(),
                            customer_name=customer_name,
                            customer_address=customer_address,
                            customer_phone=customer_phone,
                            item_name=f"StoryBit {target_tier.title()} Plan",
                            amount=amount_paid,
                            plan=target_tier,
                            due_date=validity_date
                        )

                        storage_path = f"{user_id}/INV-{order_id}.pdf"

                        with open(invoice_path, "rb") as f:
                            supabase.storage.from_("invoices").upload(
                                path=storage_path,
                                file=f,
                                file_options={"content-type": "application/pdf"},
                            )

                        signed = supabase.storage.from_("invoices").create_signed_url(
                            path=storage_path,
                            expires_in=60 * 60 * 24 * 365,
                        )
                        invoice_url = signed["signedURL"]

                        supabase.table("subscriptions").update(
                            {"invoice_url": invoice_url}
                        ).eq("razorpay_order_id", order_id).execute()

                        os.remove(invoice_path)
                        print(f"Invoice uploaded: {invoice_url}")


                    except Exception as e:
                        print(f"ERROR generating/uploading invoice for {user_id}: {e}")

                else:
                    print(f"WARN: Subscription insert returned no data for order {order_id}.")

            except APIError as e:
                print(f"ERROR: Supabase subscriptions error for {user_id}: {e}")
            except Exception as e:
                print(f"ERROR: Unexpected subscriptions error for {user_id}: {e}")

        # PAYMENT CAPTURED (ignored)
        elif event_type == 'payment.captured':
            print("Ignoring 'payment.captured' (handled by 'order.paid').")

        # PAYMENT FAILED
        elif event_type == 'payment.failed':
            payment_entity   = event_data['payload']['payment']['entity']
            failed_order_id  = payment_entity.get('order_id', 'unknown')
            failed_payment_id = payment_entity.get('id', 'unknown')
            error_desc       = payment_entity.get('error_description', 'No description')

            print(f"Payment failed for order {failed_order_id}. Reason: {error_desc}")

            notes       = payment_entity.get('notes', {})
            user_id     = notes.get('user_id')
            target_tier = notes.get('target_tier')
            amount_paid = payment_entity.get('amount', 0) / 100

            if user_id:
                try:
                    failed_row = {
                        "userId":               user_id,
                        "amount":               amount_paid,
                        "plan":                 (target_tier or 'unknown').lower(),
                        "purchased_date":       datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "validity":             None,
                        "credits":              0,
                        "payment_status":       "failed",
                        "rayzorpay_payment_id": failed_payment_id,
                        "razorpay_order_id":    failed_order_id,
                    }
                    supabase.table('subscriptions').insert(failed_row).execute()
                    print(f"Inserted failed subscription record for user {user_id}.")
                except Exception as e:
                    print(f"ERROR: Could not log failed payment for user {user_id}: {e}")

        # UNHANDLED
        else:
            print(f"Ignoring unhandled event: {event_type}")

        return {"invoice_url" : invoice_url}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    except Exception as e:
        print(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")



@app.get("/payments/invoice/{order_id}")
async def get_invoice(order_id: str, current_user: User = Depends(get_current_user)):
    result = (
        supabase.table("subscriptions")
        .select("invoice_url")
        .eq("razorpay_order_id", order_id)
        .eq("userId", str(current_user.id))
        .single()
        .execute()
    )
    if not result.data or not result.data.get("invoice_url"):
        raise HTTPException(status_code=404, detail="Invoice not found.")
    return {"invoice_url": result.data["invoice_url"]}




@app.get('/trending-data')
def content_radar():
    res = supabase.table("content_radar").select("*").execute()
    return {"message": res.data}


async def run_intelligence_for_user(userId):
    response = (
        supabase
        .table("user_channel_memory")
        .select("text")
        .eq("userId", userId)         
        .execute()
    )
    data = response.data

    if not data:
        print(f"No chunks found for user {userId}")
        return

    combined_text = "\n\n".join(item["text"] for item in data)

    await get_intelligence(combined_text, userId)


@app.post("/upload")
async def upload(file: UploadFile = File(...), userId: str = Form(...)):
    file_bytes = await file.read()

    chunks = process_pdf(file_bytes, userId)

    supabase.table('user_channel_memory').upsert(
        chunks,
        on_conflict="chunk_id"
    ).execute()

    asyncio.create_task(run_intelligence_for_user(userId))  

    return {"message": "Uploaded and processed"}


