from fastapi import Depends, HTTPException, Request, Header,UploadFile, File,Form
from fastapi import FastAPI
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client
from postgrest.exceptions import APIError
from supabase_auth.types import User
from auth_dependencies import get_current_user, login_user, refresh_access_token
import os
from openai import OpenAI
from channelMemory.aiIntel import get_intelligence
from researchAgent.tss_v4 import get_trends_serpapi,build_trend_dashboard , build_youtube_summary , scan_topic , build_news_summary
from researchAgent.eci import get_google_trends_serpapi,get_youtube_data

from shared.schemas.pipeline_context import (
    AgentPipelineContext,
)
from ddgs import DDGS
import asyncio
import time
import re
import json
import random
import nltk
import razorpay
import datetime
from urllib.parse import urlparse
from ddgs import DDGS
from pytrends.request import TrendReq
from channelMemory.channelMemory import process_pdf
from typing import List


load_dotenv()

project_root = os.path.dirname(os.path.abspath(__file__))
nltk_data_dir = os.path.join(project_root, 'nltk_data')
nltk.data.path.insert(0, nltk_data_dir)

print(os.getenv("RAZORPAY_WEBHOOK_SECRET"))

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")

api_key = os.getenv("apiKey")
gnews_key = os.getenv("GnewsApi")
google_api_key = os.getenv("GOOGLE_API_KEY")

supabase_url_env = os.getenv("SUPABASE_URL")
supabase_key_env = os.getenv("SUPABASE_KEY")

Hf_token = os.getenv("Hf_token")

print(Hf_token)

hf_url = "https://router.huggingface.co/v1/chat/completions"

hf_headers = {
    "Authorization": f"Bearer {Hf_token}",
    "Content-Type": "application/json"
}

pytrends = TrendReq(hl='en-US', tz=360)

supabase = create_client(supabase_url_env, supabase_key_env)


_st_model = None

def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        print("--- EMBEDDING: Loading SentenceTransformer model (first use) ---")
        _st_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("--- EMBEDDING: Model loaded ---")
    return _st_model


if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    print("WARNING: Razorpay API keys not found. Payment endpoints will fail.")
    razorpay_client = None
else:
    razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    print("Razorpay client initialized.")

print(os.getenv("STABILITY_API_KEY"))

SCRIPT_FRAMECHECK_PROVIDER = (os.getenv("SCRIPT_FRAMECHECK_PROVIDER") or "groq").strip().lower()

deepseek_client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

print("deepseek", os.environ.get("DEEPSEEK_API_KEY"))

if not supabase_url_env or not supabase_key_env:
    raise ValueError("Supabase credentials not found in .env file")
print("Supabase client initialized.")


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Lifespan] Server started.")
    yield
    print("[Lifespan] Shutting down.")

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "https://www.storio.tech",
    "https://storio.tech",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    topic: str


class CreateOrderRequest(BaseModel):
    amount: float
    currency: str = "INR"
    receipt: str | None = None
    target_tier: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class GenerateIdeasRequest(BaseModel):
    topic: str

class ChannelContextInput(BaseModel):
    userId: str
    channel_id: str | None = None
    channel_niche: str | None = None
    subscriber_count: int | None = None
    top_video_titles: list[str] | None = None
    existing_hashtags: list[str] | None = None
    avg_ctr_pct: float | None = None




# ROUTES

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
        news_result = await asyncio.to_thread(build_news_summary, request.topic)
        return {
            "topic":      request.topic,
            "trends":     trend_dashboard,
            "youtube":    youtube_result,
            "social":     social_result["dashboard"],
            "news_result": news_result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline metrics failed: {e}")


@app.post("/eci")
async def eci(request: PromptRequest):
    try:
        google_data = await asyncio.to_thread(get_google_trends_serpapi, request.topic)
        youtube_data = await asyncio.to_thread(get_youtube_data, request.topic)
        return {
            "google_data": google_data,
            "youtube_data": youtube_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline metrics failed: {e}")

















import io
import os
import re
import json
import time
import math
import base64
import uuid
import hashlib
import asyncio
import contextvars
import concurrent.futures
from urllib.parse import urlparse

import requests
import numpy as np
from sqlalchemy import create_engine, text, bindparam
from sklearn.feature_extraction.text import HashingVectorizer
from fastapi import HTTPException
from openai import OpenAI
import trafilatura

try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("[TOKENS] tiktoken not installed — falling back to word-based token "
          "approximation. Install with: pip install tiktoken")

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GPT_IMAGE_MODEL = os.getenv("GPT_IMAGE_MODEL", "gpt-image-2")
GPT_IMAGE_SIZE = os.getenv("GPT_IMAGE_SIZE", "1536x1024")
GPT_IMAGE_QUALITY = os.getenv("GPT_IMAGE_QUALITY", "high")


#
_ENCODE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("ENCODE_EXECUTOR_WORKERS", "4")),
    thread_name_prefix="encode",
)

_http_session = requests.Session()
_http_adapter = requests.adapters.HTTPAdapter(
    pool_connections=20, pool_maxsize=20, max_retries=1
)
_http_session.mount("https://", _http_adapter)
_http_session.mount("http://", _http_adapter)

_MAX_CONCURRENT_PIPELINES = int(os.getenv("MAX_CONCURRENT_PIPELINES", "20"))
_pipeline_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_PIPELINES)

_MAX_CONCURRENT_ENCODES = int(os.getenv("MAX_CONCURRENT_ENCODES", "4"))
_encode_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_ENCODES)
_MAX_CONCURRENT_SCRAPES = int(os.getenv("MAX_CONCURRENT_SCRAPES", "8"))
_scrape_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SCRAPES)

OPENAI_CALL_TIMEOUT = float(os.getenv("OPENAI_CALL_TIMEOUT", "45"))


async def _run_encode(fn):
    """Run a CPU-bound model.encode(...) call on the dedicated encode
    executor, gated by a semaphore."""
    async with _encode_semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_ENCODE_EXECUTOR, fn)


async def _run_scrape(fn, *args, **kwargs):
    """Run a blocking network call (DDGS search, trafilatura fetch,
    scrapetube search) gated by a semaphore to cap total concurrent
    outbound connections."""
    async with _scrape_semaphore:
        return await asyncio.to_thread(fn, *args, **kwargs)


async def _openai_create_with_timeout(call_fn, timeout: float = OPENAI_CALL_TIMEOUT):
    """Run a blocking openai_client.chat.completions.create(...) call with
    a hard timeout so a hung API call can't hold request memory forever."""
    return await asyncio.wait_for(asyncio.to_thread(call_fn), timeout=timeout)


USER_PROFILES_TABLE = "user_profiles"
USER_PROFILES_ID_COLUMN = "id"


async def _user_exists_in_profiles(user_id: str | None) -> bool:
    if not user_id or not str(user_id).strip():
        return False

    user_id = str(user_id).strip()

    try:
        result = await asyncio.to_thread(
            lambda: supabase.table(USER_PROFILES_TABLE)
            .select(USER_PROFILES_ID_COLUMN)
            .eq(USER_PROFILES_ID_COLUMN, user_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        print(f"[AUTH] user_profiles lookup failed for userId={user_id}: {e}")
        return False

    rows = result.data or []
    exists = len(rows) > 0

    if exists:
        print(f"[AUTH] userId={user_id} verified against '{USER_PROFILES_TABLE}.{USER_PROFILES_ID_COLUMN}'")
    else:
        print(f"[AUTH] REJECTED — userId={user_id} not found in '{USER_PROFILES_TABLE}.{USER_PROFILES_ID_COLUMN}'")

    return exists


async def require_valid_user(user_id: str | None) -> None:
    if not user_id or not str(user_id).strip():
        raise HTTPException(status_code=401, detail="userId is required")

    if not await _user_exists_in_profiles(user_id):
        raise HTTPException(
            status_code=403,
            detail="Access denied: userId not found in user_profiles",
        )


HASH_FEATURES = 2**18
MAX_WEB_SOURCES = 10
MAX_YOUTUBE_SOURCES = 7
MAX_DB_CHUNKS = 7
MAX_SCRIPT_CONTEXT_CHUNKS = 20

MAX_BOOKS = 7

WEB_CONTENT_SIMILARITY_THRESHOLD = 0.4
DB_SIMILARITY_THRESHOLD = 0.5

WORDS_PER_MINUTE = 140

TABLES = [
    "duplicate_RAG_Entrepreneurship",
    "duplicate_RAG_Anthropology",
    "duplicate_RAG_Biography",
]

BOOKS_TABLE_NAME = "english_books"
THUMBNAILS_BUCKET = "generated-thumbnails"


def to_pgvector(embedding) -> str:
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"



_bge_model = None


def _get_st_model():
    global _bge_model
    if _bge_model is None:
        from sentence_transformers import SentenceTransformer
        print("[MODEL] Loading BAAI/bge-m3")
        _bge_model = SentenceTransformer("BAAI/bge-m3")
        print("[MODEL] BAAI/bge-m3 loaded")
    return _bge_model



class Idea(BaseModel):
    title: str
    description: str


class SaveIdeasRequest(BaseModel):
    userId: str
    topic: str
    topic_summary: str
    ideas: List[Idea]


@app.post("/save-ideas")
async def save_ideas(data: SaveIdeasRequest):
    print("User ID:", data.userId)
    print("Topic:", data.topic)
    print("Topic Summary:", data.topic_summary)

    print("\nIdeas:")
    for i, idea in enumerate(data.ideas, start=1):
        print(f"\n{i}. {idea.title}")
        print(idea.description)

    model = _get_st_model()

    topic_embedding, summary_embedding = await _run_encode(
        lambda: model.encode(
            [data.topic, data.topic_summary],
            normalize_embeddings=True,
        )
    )

    ideas_payload = [idea.model_dump() for idea in data.ideas]

    row = {
        "userId": data.userId,
        "topic": data.topic,
        "ideas": ideas_payload,
        "topic_embeddings": to_pgvector(topic_embedding),
        "summary_embeddings": to_pgvector(summary_embedding),
    }

    try:
        result = supabase.table("saved_ideas").insert(row).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase insert failed: {e}")

    return {
        "message": "Ideas received successfully",
        "total_ideas": len(data.ideas),
        "row_id": result.data[0]["id"] if result.data else None,
    }



try:
    _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base") if tiktoken else None
except Exception as e:
    print(f"[TOKENS] failed to load tiktoken encoding, using fallback estimator: {e}")
    _TIKTOKEN_ENCODING = None


def _count_tokens(text_value: str) -> int:
    if not text_value:
        return 0
    if _TIKTOKEN_ENCODING is not None:
        return len(_TIKTOKEN_ENCODING.encode(text_value))
    return max(1, int(len(text_value.split()) * 1.3))


_request_token_log: contextvars.ContextVar = contextvars.ContextVar(
    "_request_token_log", default=None
)

_script_keywords_cache: contextvars.ContextVar = contextvars.ContextVar(
    "_script_keywords_cache", default=None
)


def _start_token_tracking() -> None:
    _request_token_log.set([])
    _script_keywords_cache.set({})


def _record_token_usage(label: str, completion) -> dict:
    input_tokens = output_tokens = total_tokens = None
    try:
        usage = completion.usage
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
    except Exception as e:
        print(f"[TOKENS] {label}: could not read usage off completion ({e})")

    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    print(
        f"[TOKENS] {label}: input_tokens={input_tokens} "
        f"output_tokens={output_tokens} total_tokens={total_tokens}"
    )

    entry = {
        "label": label,
        "input_tokens": input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "total_tokens": total_tokens or 0,
    }

    log = _request_token_log.get()
    if log is not None:
        log.append(entry)

    return entry


def _get_token_usage_summary() -> dict:
    log = _request_token_log.get()
    if not log:
        return {"calls": [], "total_input_tokens": 0, "total_output_tokens": 0, "total_tokens": 0}

    total_input = sum(c["input_tokens"] for c in log)
    total_output = sum(c["output_tokens"] for c in log)
    total = sum(c["total_tokens"] for c in log)

    print(
        f"[TOKENS] REQUEST TOTAL across {len(log)} call(s): "
        f"input={total_input} output={total_output} total={total}"
    )

    return {
        "calls": log,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total,
    }


HYDE_MAX_TOKENS = 70

def _cap_hyde_doc_tokens(text_value: str, max_tokens: int = HYDE_MAX_TOKENS) -> str:
    text_value = (text_value or "").strip()
    if not text_value:
        return text_value

    if _TIKTOKEN_ENCODING is None:
        words = text_value.split()
        max_words = max(1, int(max_tokens / 1.3))
        if len(words) <= max_words:
            return text_value
        print(f"[HYDE-CAP] (fallback estimator) trimming from {len(words)} words to ~{max_words} words")
        return " ".join(words[:max_words]).rstrip(",;:") + "."

    tokens = _TIKTOKEN_ENCODING.encode(text_value)
    if len(tokens) <= max_tokens:
        return text_value

    print(f"[HYDE-CAP] trimming from {len(tokens)} tokens to {max_tokens} tokens")
    truncated_tokens = tokens[:max_tokens]
    truncated_text = _TIKTOKEN_ENCODING.decode(truncated_tokens)
    return truncated_text.rstrip(",;: ") + "."


IDEAS_SYSTEM_PROMPT = """
You are a YouTube Content Ideation Engine.

## Inputs
1. User Topic
2. Retrieved knowledge chunks

## Objective
Synthesize all retrieved knowledge to generate high-quality YouTube video ideas.

Do NOT summarize individual chunks.
Instead, identify hidden stories, unanswered questions, conflicts, surprising insights, patterns, and opportunities that emerge only after combining information across multiple chunks.

Reason across dimensions such as:
- Historical evolution
- Current landscape
- Future implications
- Timeline of events
- People & organizations
- Winners & losers
- Political factors
- Economic consequences
- Scientific & technological significance
- Social & cultural impact
- Human stories
- Hidden incentives
- Power dynamics
- Ethical debates
- Myths vs Facts
- Unanswered questions
- Ripple effects
- Global & regional perspectives

Each idea must focus on ONE compelling narrative angle.

Diversify storytelling styles naturally across ideas, including:
- Documentary
- Historical Story
- Investigation
- Mystery
- Explainer
- Business Analysis
- Science
- Psychology
- Timeline
- Case Study
- Behind the Scenes
- Future Prediction
- Myth Busting
- Unexpected Facts
- What If

Prioritize ideas with:
- High curiosity
- Emotional engagement
- Strong storytelling potential
- Educational value
- Broad audience appeal
- Shareability

Avoid:
- Generic summaries
- Repeated angles
- Unsupported speculation
- Clickbait
- Duplicate ideas

Creativity:
- Temperature target: 0.7
- Be imaginative while remaining grounded in the provided evidence.

## Output

### Output 1 — Video Ideas
Generate exactly **10** ranked ideas (best first). Never generate fewer than 10.

For each idea provide a Title and a Description.

**Title**
- 8-15 words
- Natural, curiosity-driven YouTube title

**Description**
70-100 words explaining:
- Central story or question
- Main stakeholders
- Why it matters
- Historical and current context
- Future implications (if relevant)
- What viewers will discover

STRICT FORMATTING RULES — follow these exactly, with no deviation:
- Use plain text labels "Title:" and "Description:" — do not bold them, do not wrap the item number together with the label (e.g. never write "**1) Title:**").
- The title text must appear on the SAME line as "Title:", never on a separate line.
- The description text must appear on the SAME line as "Description:" (it may wrap naturally, but do not insert a blank line or line break between the label and the text).
- Number each idea with a plain "1." at the very start of the Title line, nothing bolded.
- Do not use any markdown bold (**), italics, or headers inside an idea's title or description text itself.
- Separate each idea from the next with exactly one blank line.

Output each idea in EXACTLY this format (this is a literal template — match it character for character, only replacing the placeholder text):

1. Title: <title text here>
Description: <description text here>

2. Title: <title text here>
Description: <description text here>

(continue through idea 10)

### Output 2 — Topic Summary

Write a concise **30-40 word** synthesis of the overall topic by combining insights from the user query and all retrieved chunks.

The summary should:
- Capture the core theme
- Highlight the biggest underlying narrative
- Avoid mentioning individual chunks
- Be suitable as a high-level overview for downstream content generation.

Output this section EXACTLY as:

Topic Summary: <summary text here>

Do not add any other headings, section titles, preambles, or closing remarks anywhere in the response. Output only the two sections above, in order, in the exact format specified.
"""

KEYWORD_GEN_PROMPT_TEMPLATE = """You are a Search Query Expansion Engine for automated web crawling.

Input:
A short user topic (2-10 words).

Goal:
Generate exactly 15 high-quality search engine keyword combinations that maximize information retrieval from Google, Bing, academic search engines, and news websites.

Requirements:
- Every phrase must incorporate the topic's core subject/entities naturally —
  do NOT output the raw topic string verbatim as one of the 15 lines by
  itself; each line must be a distinct EXPANDED search phrase, not a copy of
  the input.
- Generate search phrases, NOT sentences.
- Each phrase should target a unique research dimension.
- Cover:
  • latest news
  • history
  • timeline
  • root causes
  • stakeholders
  • government
  • companies
  • researchers
  • statistics
  • datasets
  • reports
  • research papers
  • expert opinions
  • controversies
  • future trends
- Include important entities when inferable.
- Avoid duplicate intent.
- Each keyword combination should contain 4-10 words.
- Return ONLY the 15 keyword combinations, nothing else — no preamble, no
  restating the topic on its own line.
- Number each result.

[TOPIC]: {topic}
"""


_sparse_vectorizer = None

def get_sparse_vectorizer() -> HashingVectorizer:
    global _sparse_vectorizer
    if _sparse_vectorizer is None:
        _sparse_vectorizer = HashingVectorizer(
            n_features=HASH_FEATURES,
            alternate_sign=False,
            norm="l2",
        )
    return _sparse_vectorizer


def _sparse_row_to_dict(sparse_row) -> dict:
    coo = sparse_row.tocoo()
    return {str(int(idx)): float(val) for idx, val in zip(coo.col, coo.data)}


def _sparse_cosine(query_sparse: dict, doc_sparse: dict) -> float:
    if not query_sparse or not doc_sparse:
        return 0.0
    shared_keys = query_sparse.keys() & doc_sparse.keys()
    return sum(query_sparse[k] * doc_sparse[k] for k in shared_keys)


TOPIC_SIMILARITY_THRESHOLD = 0.55
SUMMARY_SIMILARITY_THRESHOLD = 0.45
RPC_RAW_FETCH_THRESHOLD = 0.0

async def get_similar_saved_ideas(
    topic: str,
    hyde_doc: str,
    match_count: int = 10,
    topic_threshold: float = TOPIC_SIMILARITY_THRESHOLD,
    summary_threshold: float = SUMMARY_SIMILARITY_THRESHOLD,
) -> list[dict]:
    print(f"[MATCH] Searching saved_ideas for topic: '{topic}'")

    model = _get_st_model()
    topic_embedding, summary_query_embedding = await _run_encode(
        lambda: model.encode(
            [topic, hyde_doc],
            normalize_embeddings=True,
        )
    )

    try:
        result = await asyncio.to_thread(
            lambda: supabase.rpc(
                "match_saved_ideas",
                {
                    "query_topic_embedding": to_pgvector(topic_embedding),
                    "query_summary_embedding": to_pgvector(summary_query_embedding),
                    "match_count": match_count,
                    "similarity_threshold": RPC_RAW_FETCH_THRESHOLD,
                },
            ).execute()
        )
    except Exception as e:
        print(f"[MATCH] saved_ideas RPC call FAILED (not just 'no matches'): {e}")
        import traceback
        traceback.print_exc()
        return []

    candidates = result.data or []
    print(f"[MATCH] RPC returned {len(candidates)} raw candidates (unfiltered)")

    matches = []
    for row in candidates:
        t_sim = row.get("topic_similarity") or 0.0
        s_sim = row.get("summary_similarity") or 0.0
        if t_sim >= topic_threshold or s_sim >= summary_threshold:
            matches.append(row)

    matches.sort(
        key=lambda r: max(
            r.get("topic_similarity") or 0.0,
            r.get("summary_similarity") or 0.0,
        ),
        reverse=True,
    )

    print(f"[MATCH] {len(matches)}/{len(candidates)} candidates passed OR-threshold filter")
    return matches


async def select_table_for_topic(topic: str) -> str:
    table_selector_prompt = f"""
    You are a routing assistant. Given a topic, select the single most relevant
    table from the list below that would contain source documents for that topic.

    Available tables:
    - duplicate_RAG_Entrepreneurship: startups, business strategy, venture capital, founders
    - duplicate_RAG_Anthropology: human culture, society, archaeology, ethnography
    - duplicate_RAG_Biography: individual people's lives, histories, memoirs

    Topic: "{topic}"

    Respond with ONLY the exact table name from the list above, nothing else.
    """

    res = await _openai_create_with_timeout(
        lambda: openai_client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": table_selector_prompt}],
            stream=False,
        )
    )
    _record_token_usage("select_table_for_topic", res)
    table_name = res.choices[0].message.content.strip("`'\" \n")

    if table_name not in TABLES:
        print(f"[DB] table selector returned unexpected value '{table_name}', defaulting to {TABLES[0]}")
        table_name = TABLES[0]
    else:
        print(f"[DB] Selected table: {table_name}")

    return table_name


SCRIPT_TEMPLATE_MATCH_COUNT = 1


async def generate_topic_embedding(topic: str) -> np.ndarray:
    model = _get_st_model()
    embedding = await _run_encode(
        lambda: model.encode(topic, normalize_embeddings=True, convert_to_numpy=True)
    )
    return embedding


async def retrieve_best_script_template(topic: str) -> dict | None:
    print(f"[TEMPLATE] Embedding topic for template search: '{topic}'")
    topic_embedding = await generate_topic_embedding(topic)
    query_vector = to_pgvector(topic_embedding)

    try:
        result = await asyncio.to_thread(
            lambda: supabase.rpc(
                "match_script_structures",
                {
                    "query_embedding": query_vector,
                    "match_count": SCRIPT_TEMPLATE_MATCH_COUNT,
                },
            ).execute()
        )
    except Exception as e:
        print(f"[TEMPLATE] match_script_structures RPC FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

    rows = result.data or []
    if not rows:
        print("[TEMPLATE] no matching template found (empty result set)")
        return None

    best = rows[0]

    selected_template = {
        "key": best.get("key"),
        "title": best.get("title"),
        "cluster": best.get("cluster"),
        "about": best.get("about"),
        "best_fit_categories": best.get("best_fit_categories") or [],
        "human_texture_tier": best.get("human_texture_tier"),
        "segments": best.get("segments") or [],
        "template_text": best.get("template_text") or "",
        "similarity": best.get("similarity"),
    }

    print(
        f"[TEMPLATE] best match: key='{selected_template['key']}' "
        f"title='{selected_template['title']}' cluster='{selected_template['cluster']}' "
        f"similarity={selected_template['similarity']}"
    )

    return selected_template


def _word_count(text_value: str) -> int:
    return len(text_value.split())


async def _generate_length_constrained_hyde(
    client,
    model: str,
    prompt: str,
    label: str,
    hard_max_tokens: int = HYDE_MAX_TOKENS,
    first_max_tokens: int = 180,
    empty_retry_max_tokens: int = 1200,
) -> tuple[str, bool]:

    async def _call(messages: list[dict], max_tokens: int):
        completion = await _openai_create_with_timeout(
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                stream=False,
            )
        )
        _record_token_usage(f"{label} (max_tokens={max_tokens})", completion)
        choice = completion.choices[0]
        raw_content = (choice.message.content or "").strip()
        finish_reason = getattr(choice, "finish_reason", None)
        output_tokens = None
        try:
            output_tokens = completion.usage.completion_tokens
        except Exception:
            pass
        return raw_content, finish_reason, output_tokens

    messages = [{"role": "user", "content": prompt}]

    doc, finish_reason, output_tokens = await _call(messages, first_max_tokens)

    if not doc:
        print(
            f"[{label}] came back EMPTY (finish_reason={finish_reason}, "
            f"output_tokens={output_tokens}) — retrying with more headroom"
        )
        try:
            doc, finish_reason, output_tokens = await _call(messages, empty_retry_max_tokens)
        except Exception as retry_exc:
            print(f"[{label}] retry call raised: {retry_exc}")
            doc = ""

    if not doc:
        print(f"[{label}] still EMPTY after retry")
        return "", False

    tc = _count_tokens(doc)
    print(f"[{label}] draft: {tc} token(s) (local estimate), output_tokens={output_tokens}, finish_reason={finish_reason}")

    if tc <= hard_max_tokens:
        return doc, False

    print(f"[{label}] draft over the {hard_max_tokens}-token cap ({tc} tokens) — asking model to rewrite shorter")
    rewrite_request = (
        f"Your draft above is too long. Rewrite the SAME passage so it is "
        f"STRICTLY under {hard_max_tokens} tokens (aim for well under that, "
        f"e.g. 35-50 words). Keep the same information density and tone. "
        f"Output nothing but the rewritten passage — no preamble, no word "
        f"count, no notes."
    )
    messages2 = messages + [
        {"role": "assistant", "content": doc},
        {"role": "user", "content": rewrite_request},
    ]

    try:
        doc2, finish_reason2, output_tokens2 = await _call(messages2, max(first_max_tokens, 150))
    except Exception as exc:
        print(f"[{label}] rewrite call raised: {exc}")
        doc2 = ""

    if doc2:
        tc2 = _count_tokens(doc2)
        print(f"[{label}] rewrite: {tc2} token(s) (local estimate), output_tokens={output_tokens2}")
        if tc2 <= hard_max_tokens:
            return doc2, False
        print(f"[{label}] rewrite STILL over the cap ({tc2} tokens) — hard-trimming to {hard_max_tokens} tokens as last resort")
        return _cap_hyde_doc_tokens(doc2, max_tokens=hard_max_tokens), True

    print(f"[{label}] rewrite came back empty — hard-trimming the original draft to {hard_max_tokens} tokens as last resort")
    return _cap_hyde_doc_tokens(doc, max_tokens=hard_max_tokens), True


async def generate_hyde_document(topic: str, selected_template: dict) -> str:
    segments = selected_template.get("segments") or []
    segments_json = json.dumps(segments, indent=2, ensure_ascii=False)

    hyde_prompt = f"""
            You are generating a HyDE (Hypothetical Document Embedding) passage for
            a YouTube documentary research pipeline.

            Topic: "{topic}"

            You MUST strictly follow the script template below — do not invent a
            different structure. The "segments" JSON defines the exact structure
            the generated passage must mirror, section by section, in the same
            order as listed.

            Template title: "{selected_template.get('title')}"
            Template cluster: {selected_template.get('cluster')}
            Template purpose: {selected_template.get('about')}

            Template segments (JSON) — mirror this structure exactly:
            {segments_json}

            Template reference text:
            {selected_template.get('template_text')}

            Task:
            Write a short, factual, encyclopedia-style HyDE passage that provides
            direct, concrete, retrievable information relevant to the topic above,
            organized as one dense, information-rich block per segment, in the
            same order as the segments list. Include key terms a search/embedding
            system would match against. Do not write in a narrative or scripted
            tone — this is a retrieval seed document, not the script itself.

            Output only the passage, nothing else — no preamble, no headings or
            labels beyond what naturally separates each segment's content.

            STRICT LENGTH LIMIT: the passage must be under {HYDE_MAX_TOKENS} tokens
            (roughly 35-50 words). Do not exceed this under any circumstances.
""".strip()

    doc, was_hard_trimmed = await _generate_length_constrained_hyde(
        client=openai_client,
        model="gpt-5.4-mini",
        prompt=hyde_prompt,
        label="HYDE-TEMPLATE",
    )

    if not doc:
        print("[HYDE-TEMPLATE] falling back to topic")
        return topic

    if was_hard_trimmed:
        print("[HYDE-TEMPLATE] WARNING: had to hard-trim as a fallback (model didn't comply with length after rewrite ask)")

    doc = _cap_hyde_doc_tokens(doc)

    print(f"[HYDE-TEMPLATE] final: {_count_tokens(doc)} token(s) (local estimate)")
    print(f"[HYDE-TEMPLATE] {doc}")
    return doc


async def select_template_and_generate_hyde(topic: str) -> dict:
    selected_template = await retrieve_best_script_template(topic)

    if selected_template is None:
        print("[PIPELINE] no template matched — generating a template-less HyDE document")
        generated_hyde_document = topic
    else:
        generated_hyde_document = await generate_hyde_document(topic, selected_template)

    return {
        "selected_template": selected_template,
        "generated_hyde_document": generated_hyde_document,
    }


async def get_context_from_db(
    topic: str,
    hyde_doc: str = None,
    final_k: int = 7,
    table_name: str = None,
    similarity_threshold: float = DB_SIMILARITY_THRESHOLD,
    match_count: int = 20,
):
    print(f"[DB] Starting retrieval for topic: '{topic}'")

    if table_name is None:
        table_name = await select_table_for_topic(topic)
    else:
        print(f"[DB] Using pre-selected table: {table_name}")

    embedding_source = hyde_doc if hyde_doc else topic

    model = _get_st_model()
    dense_embedding = await _run_encode(
        lambda: model.encode(
            embedding_source,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()
    )
    print("[DB] Dense embedding computed")

    vectorizer = get_sparse_vectorizer()
    sparse_row = await asyncio.to_thread(lambda: vectorizer.transform([embedding_source]))
    query_sparse = _sparse_row_to_dict(sparse_row)
    print("[DB] Sparse embedding computed")

    try:
        result = await asyncio.to_thread(
            lambda: supabase.rpc(
                "match_documents",
                {
                    "query_dense_embedding": dense_embedding,
                    "match_table": table_name,
                    "match_count": match_count,
                    "similarity_threshold": similarity_threshold,
                },
            ).execute()
        )
    except Exception as e:
        print(f"[DB] RPC with similarity_threshold failed ({e}) — retrying with legacy 3-param signature")
        try:
            result = await asyncio.to_thread(
                lambda: supabase.rpc(
                    "match_documents",
                    {
                        "query_dense_embedding": dense_embedding,
                        "match_table": table_name,
                        "match_count": match_count,
                    },
                ).execute()
            )
            print("[DB] legacy RPC call succeeded — filtering by similarity client-side instead")
        except Exception as e2:
            print(f"[DB] vector search failed even with legacy signature: {e2}")
            import traceback
            traceback.print_exc()
            return []

    candidates = result.data or []
    print(
        f"[DB] RPC returned {len(candidates)} candidate(s) from {table_name} "
        f"(target similarity >= {similarity_threshold})"
    )

    reranked = []
    for row in candidates:
        doc_sparse = row.get("sparse_vector") or {}
        sparse_score = _sparse_cosine(query_sparse, doc_sparse)
        dense_score = row.get("dense_score", 0.0)
        combined = (0.7 * dense_score) + (0.3 * sparse_score)
        reranked.append({**row, "sparse_score": sparse_score, "combined_score": combined})

    reranked.sort(key=lambda r: r["combined_score"], reverse=True)

    above_threshold = [r for r in reranked if (r.get("dense_score") or 0.0) >= similarity_threshold]
    if len(above_threshold) != len(reranked):
        print(
            f"[DB] WARNING: {len(reranked) - len(above_threshold)} candidate(s) were below "
            f"{similarity_threshold} similarity despite coming from the RPC — check that the "
            f"match_documents SQL function is filtering on `embeddings` correctly."
        )

    matches = above_threshold[:final_k]

    print(f"[DB] Top {len(matches)} chunks after hybrid rerank + similarity filter:")
    for i, row in enumerate(matches, start=1):
        content = row.get("content")
        md5 = row.get("md5") or (
            hashlib.md5(content.encode("utf-8")).hexdigest() if content else None
        )
        print(
            f"  [DB-{i}] md5={md5} dense_score={row.get('dense_score')} "
            f"combined_score={row['combined_score']:.4f}"
        )
        print(f"    content: {content[:200]}{'...' if content and len(content) > 200 else ''}")

    for row in matches:
        row.pop("sparse_vector", None)

    return matches


def _ddgs_search_for_ideas(keyword: str, max_results: int) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.news(keyword, max_results=max_results):
                url = r.get("url")
                snippet = r.get("body", "") or r.get("title", "")
                if url:
                    results.append((url, snippet))
    except Exception as e:
        print(f"[DDGS] search failed for '{keyword}': {e}")
    return results


def _parse_keyword_lines(raw: str) -> list[str]:
    lines = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\u2022]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = line.strip("\"'` ")
        if line:
            lines.append(line)
    return lines


async def _generate_search_keywords(topic: str) -> list[str]:
    prompt = KEYWORD_GEN_PROMPT_TEMPLATE.format(topic=topic)

    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
        )
        _record_token_usage("_generate_search_keywords (ideas)", res)
        raw = res.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DDGS] keyword generation failed: {e}")
        return [f"{topic} latest news today", f"{topic} 2026 update"]

    keywords = _parse_keyword_lines(raw)

    topic_normalized = topic.strip().lower()
    keywords = [kw for kw in keywords if kw.strip().lower() != topic_normalized]
    keywords = keywords[:15]

    if not keywords:
        print("[DDGS] keyword generation returned nothing usable, using fallback")
        return [f"{topic} latest news today", f"{topic} 2026 update"]

    print(f"[DDGS] generated {len(keywords)} keywords")
    for i, kw in enumerate(keywords, start=1):
        print(f"  [KW-{i}] {kw}")

    return keywords


def _truncate_words(text_value: str, max_words: int = 400) -> str:
    words = text_value.split()
    if len(words) <= max_words:
        return text_value
    return " ".join(words[:max_words]) + "..."


def _split_into_chunks(text_value: str, max_words_per_chunk: int = 40) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text_value.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        if current and current_words + len(words) > max_words_per_chunk:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


_HASHTAG_PATTERN = re.compile(r"#(\w+)")


def _extract_hashtags(*texts: str) -> list[str]:
    found = []
    seen = set()
    for text_value in texts:
        if not text_value:
            continue
        for match in _HASHTAG_PATTERN.findall(text_value):
            tag = f"#{match}"
            if tag.lower() not in seen:
                seen.add(tag.lower())
                found.append(tag)
    return found


def _fetch_full_article_text(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text_value = trafilatura.extract(downloaded) or ""
        return text_value.strip()
    except Exception as e:
        print(f"[FETCH] failed to extract {url}: {e}")
        return ""


async def _fetch_full_article_text_with_timeout(url: str, timeout: float = 8.0) -> str:
    try:
        return await asyncio.wait_for(
            _run_scrape(_fetch_full_article_text, url),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        print(f"[FETCH] timed out fetching {url}")
        return ""


async def _generate_web_search_keywords(topic: str) -> list[str]:
    return await _generate_search_keywords(topic)


async def _generate_youtube_search_keywords(topic: str, description: str = "") -> list[str]:
    prompt = f"""
            You are a YouTube SEO strategist generating search queries to find the
            BEST-PERFORMING, most-optimized existing videos on a topic — the goal is
            to surface videos whose titles and descriptions are strong SEO examples,
            not just any video that happens to match.

            Idea Title: "{topic}"
            Idea Description: "{description or 'N/A'}"

            Use both the title and the description above to understand the true
            intent, entities, and angle of the idea before writing queries — the
            description often clarifies specific people, places, sub-topics, or
            framing that the title alone doesn't capture.

            YouTube search behaves differently from Google/web search:
            - People phrase queries like video titles, not keyword strings
              ("how X works", "X explained", "top 10 X", "X vs Y", "why does X
              happen", "X for beginners", "the truth about X")
            - High-performing videos usually rank for a clear, singular intent —
              write queries the same way, not stuffed with extra modifiers
            - Exact entities, names, places, or proper nouns from the title or
              description pull much more relevant, higher-quality results than
              generic phrasing — preserve and reuse them naturally instead of
              abstracting away

            Generate 10 distinct queries that together cover a SPREAD of these
            intents (use each intent at most once, don't repeat the same angle
            worded differently):
            - the single broad head-term query anyone searching this topic would type
            - a "how it works" / mechanism explainer query
            - a "X explained" / definition-style query
            - a beginner-friendly / "for beginners" query
            - an advanced / in-depth / expert-level query
            - a "top 10" or ranked-listicle query
            - a comparison ("X vs Y") query, if a natural comparison exists —
              otherwise substitute a myth-busting / "the truth about" query
            - a case-study, real-example, or "what happened when" query
            - a recent/current-year query (use the actual current year)
            - a question-phrased query (who/what/why/how) or "why does X happen"

            Rules:
            - Each query should be 3-7 words, phrased like a real YouTube search bar entry
            - No keyword stuffing, no boolean operators, no quotation marks
            - No duplicate intent — each query must target a genuinely different angle
            - Do not invent entities, names, or facts not implied by the title/description
            - Return ONLY the 10 queries, one per line, no numbering, no bullets, no commentary
""".strip()

    try:
        completion = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
        )
        _record_token_usage("_generate_youtube_search_keywords", completion)
        raw = completion.choices[0].message.content.strip()
        keywords = _parse_keyword_lines(raw)
        return keywords or [topic]
    except Exception as exc:
        print(f"--- YouTube keyword generation failed: {exc} ---")
        return [topic]


async def get_ddgs_news_context(
    topic: str,
    scraped_urls: set,
    hyde_doc: str,
    max_results: int = 10,
    similarity_threshold: float = WEB_CONTENT_SIMILARITY_THRESHOLD,
) -> list[dict]:

    print(f"[DDGS] Starting news search for topic: '{topic}'")

    keywords = await _generate_web_search_keywords(topic)

    model = _get_st_model()

    hyde_embedding = await _run_encode(
        lambda: model.encode(hyde_doc, normalize_embeddings=True, convert_to_numpy=True)
    )

    articles = []
    for keyword in keywords:
        if len(articles) >= MAX_WEB_SOURCES:
            print(f"[DDGS] Reached cap of {MAX_WEB_SOURCES} sources, stopping further keyword searches")
            break

        try:
            pairs = await _run_scrape(_ddgs_search_for_ideas, keyword, max_results)
            print(f"[DDGS] keyword '{keyword}' returned {len(pairs)} results")
        except Exception as e:
            print(f"[DDGS] thread failed for '{keyword}': {e}")
            pairs = []

        for url, snippet in pairs:
            if len(articles) >= MAX_WEB_SOURCES:
                break
            if url in scraped_urls:
                continue
            scraped_urls.add(url)

            full_text = await _fetch_full_article_text_with_timeout(url)
            used_source = "full" if full_text else "fallback"
            content = full_text if full_text else snippet

            if not content:
                print(f"[DDGS] SKIP (empty content, nothing to compare) {url}")
                continue

            content = _truncate_words(content, max_words=600)

            chunks = _split_into_chunks(content, max_words_per_chunk=40)
            if not chunks:
                print(f"[DDGS] SKIP (no chunks to compare) {url}")
                continue

            try:
                chunk_embeddings = await _run_encode(
                    lambda c=chunks: model.encode(c, normalize_embeddings=True, convert_to_numpy=True)
                )
            except Exception as e:
                print(f"[DDGS] SKIP (embedding failed: {e}) {url}")
                continue

            chunk_similarities = np.dot(chunk_embeddings, hyde_embedding)

            picked = [
                (chunk, float(sim))
                for chunk, sim in zip(chunks, chunk_similarities)
                if sim >= similarity_threshold
            ]

            if not picked:
                best_sim = float(np.max(chunk_similarities)) if len(chunk_similarities) else 0.0
                print(
                    f"[DDGS] SKIP (no passage cleared threshold, "
                    f"best_sim={best_sim:.4f} < {similarity_threshold}, "
                    f"{len(chunks)} passage(s) checked) {url}"
                )
                continue

            picked.sort(key=lambda p: p[1], reverse=True)

            picked_text = _truncate_words(" ".join(chunk for chunk, _ in picked), max_words=200)
            overall_similarity = picked[0][1]

            articles.append({
                "url": url,
                "snippet": picked_text,
                "source": used_source,
                "similarity": overall_similarity,
                "picked_passage_count": len(picked),
                "total_passage_count": len(chunks),
            })

    articles.sort(key=lambda a: a["similarity"], reverse=True)
    return articles


SCRIPT_KEYWORD_GEN_PROMPT_TEMPLATE = """You are a Search Query Expansion Engine for automated web crawling.

Input:
Everything known so far about a video that's about to be scripted: the idea's
title and description, the script template that was chosen for it (its title,
its purpose, and its ordered segment structure), and the target video
duration.

Idea Title: "{title}"
Idea Description: "{description}"
Target Video Duration: {time_minutes} minute(s)

Script Template Title: "{template_title}"
Script Template Purpose: {template_about}
Script Template Segments:
{segments_block}

Goal:
Generate exactly 15 high-quality search engine keyword combinations that
maximize information retrieval from Google, Bing, academic search engines,
and news websites, to gather source material for writing this script. Use
the segment structure and template purpose above to make sure the keywords
collectively cover what each part of the script will need, not just the
idea title in isolation.

Requirements:
- Every phrase must incorporate the topic's core subject/entities naturally —
  do NOT output the raw title string verbatim as one of the 15 lines by
  itself; each line must be a distinct EXPANDED search phrase, not a copy of
  the input.
- Generate search phrases, NOT sentences.
- Each phrase should target a unique research dimension.
- Cover:
  • latest news
  • history
  • timeline
  • root causes
  • stakeholders
  • government
  • companies
  • researchers
  • statistics
  • datasets
  • reports
  • research papers
  • expert opinions
  • controversies
  • future trends
- Include important entities when inferable from the title, description, or
  template segments.
- Avoid duplicate intent.
- Each keyword combination should contain 4-10 words.
- Return ONLY the 15 keyword combinations, nothing else — no preamble, no
  restating the title on its own line.
- Number each result.
"""


def _ddgs_search_for_script(keyword: str, max_results: int) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.news(keyword, max_results=max_results):
                url = r.get("url")
                snippet = r.get("body", "") or r.get("title", "")
                if url:
                    results.append((url, snippet))
    except Exception as e:
        print(f"[DDGS-SCRIPT] search failed for '{keyword}': {e}")
    return results


async def _generate_search_keywords_for_script(
    title: str,
    description: str = "",
    template: dict | None = None,
    time_minutes: int = 0,
) -> list[str]:
    template = template or {}

    cache = _script_keywords_cache.get()
    cache_key = "|".join([
        (title or "").strip().lower(),
        (description or "").strip().lower(),
        str(template.get("key") or "").strip().lower(),
        str(time_minutes),
    ])

    if cache is not None and cache_key in cache:
        cached_keywords = cache[cache_key]
        print(
            f"[DDGS-SCRIPT] keyword cache HIT for this request/topic — reusing "
            f"{len(cached_keywords)} keyword(s), skipping LLM call entirely"
        )
        return cached_keywords

    segments_block = _segments_brief(template.get("segments") or [])

    prompt = SCRIPT_KEYWORD_GEN_PROMPT_TEMPLATE.format(
        title=title,
        description=description or "N/A",
        time_minutes=time_minutes,
        template_title=template.get("title") or "N/A",
        template_about=template.get("about") or "N/A",
        segments_block=segments_block,
    )

    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
        )
        _record_token_usage("_generate_search_keywords_for_script", res)
        raw = res.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DDGS-SCRIPT] keyword generation failed: {e}")
        fallback = [f"{title} latest news today", f"{title} 2026 update"]
        if cache is not None:
            cache[cache_key] = fallback
        return fallback

    keywords = _parse_keyword_lines(raw)

    title_normalized = (title or "").strip().lower()
    keywords = [kw for kw in keywords if kw.strip().lower() != title_normalized]
    keywords = keywords[:15]

    if not keywords:
        print("[DDGS-SCRIPT] keyword generation returned nothing usable, using fallback")
        fallback = [f"{title} latest news today", f"{title} 2026 update"]
        if cache is not None:
            cache[cache_key] = fallback
        return fallback

    print(f"[DDGS-SCRIPT] generated {len(keywords)} keywords")
    for i, kw in enumerate(keywords, start=1):
        print(f"  [KW-SCRIPT-{i}] {kw}")

    if cache is not None:
        cache[cache_key] = keywords

    return keywords


async def get_ddgs_news_context_for_script(
    topic: str,
    scraped_urls: set,
    hyde_doc: str,
    max_results: int = 10,
    similarity_threshold: float = WEB_CONTENT_SIMILARITY_THRESHOLD,
    keywords: list[str] | None = None,
) -> list[dict]:

    print(f"[DDGS-SCRIPT] Starting news search for topic: '{topic}' (similarity_threshold={similarity_threshold})")

    if keywords is None:
        # Fallback path — no pre-generated keywords supplied, so we only have
        # the plain topic string to work with (no description/template/time).
        keywords = await _generate_search_keywords_for_script(topic, "", {}, 0)
    else:
        print(f"[DDGS-SCRIPT] reusing {len(keywords)} previously generated keyword(s) — skipping keyword regeneration")

    model = _get_st_model()

    hyde_embedding = await _run_encode(
        lambda: model.encode(hyde_doc, normalize_embeddings=True, convert_to_numpy=True)
    )

    articles = []
    for keyword in keywords:
        if len(articles) >= MAX_WEB_SOURCES:
            print(f"[DDGS-SCRIPT] Reached cap of {MAX_WEB_SOURCES} sources, stopping further keyword searches")
            break

        try:
            pairs = await _run_scrape(_ddgs_search_for_script, keyword, max_results)
            print(f"[DDGS-SCRIPT] keyword '{keyword}' returned {len(pairs)} results")
        except Exception as e:
            print(f"[DDGS-SCRIPT] thread failed for '{keyword}': {e}")
            pairs = []

        for url, snippet in pairs:
            if len(articles) >= MAX_WEB_SOURCES:
                break
            if url in scraped_urls:
                continue
            scraped_urls.add(url)

            full_text = await _fetch_full_article_text_with_timeout(url)
            used_source = "full" if full_text else "fallback"
            content = full_text if full_text else snippet

            if not content:
                print(f"[DDGS-SCRIPT] SKIP (empty content, nothing to compare) {url}")
                continue

            content = _truncate_words(content, max_words=600)

            chunks = _split_into_chunks(content, max_words_per_chunk=40)
            if not chunks:
                print(f"[DDGS-SCRIPT] SKIP (no chunks to compare) {url}")
                continue

            try:
                chunk_embeddings = await _run_encode(
                    lambda c=chunks: model.encode(c, normalize_embeddings=True, convert_to_numpy=True)
                )
            except Exception as e:
                print(f"[DDGS-SCRIPT] SKIP (embedding failed: {e}) {url}")
                continue

            chunk_similarities = np.dot(chunk_embeddings, hyde_embedding)

            picked = [
                (chunk, float(sim))
                for chunk, sim in zip(chunks, chunk_similarities)
                if sim >= similarity_threshold
            ]

            if not picked:
                best_sim = float(np.max(chunk_similarities)) if len(chunk_similarities) else 0.0
                print(
                    f"[DDGS-SCRIPT] SKIP (no passage cleared threshold, "
                    f"best_sim={best_sim:.4f} < {similarity_threshold}, "
                    f"{len(chunks)} passage(s) checked) {url}"
                )
                continue

            picked.sort(key=lambda p: p[1], reverse=True)

            picked_text = _truncate_words(" ".join(chunk for chunk, _ in picked), max_words=200)
            overall_similarity = picked[0][1]

            articles.append({
                "url": url,
                "snippet": picked_text,
                "source": used_source,
                "similarity": overall_similarity,
                "picked_passage_count": len(picked),
                "total_passage_count": len(chunks),
            })

    articles.sort(key=lambda a: a["similarity"], reverse=True)
    return articles



def _youtube_api_video_details(video_ids: list[str]) -> dict[str, dict]:
    if not YOUTUBE_API_KEY or not video_ids:
        return {}

    details: dict[str, dict] = {}

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        params = {
            "part": "snippet,statistics",
            "id": ",".join(batch),
            "key": YOUTUBE_API_KEY,
        }

        try:
            resp = _http_session.get(f"{YOUTUBE_API_BASE}/videos", params=params, timeout=15)
        except Exception as e:
            print(f"[YT-API] videos.list request failed: {e}")
            continue

        if resp.status_code != 200:
            print(f"[YT-API] videos.list HTTP {resp.status_code}: {resp.text[:300]}")
            continue

        try:
            data = resp.json()
        except Exception as e:
            print(f"[YT-API] failed to parse videos.list JSON: {e}")
            continue

        print(f"[YT-API] RAW videos.list response for batch of {len(batch)} id(s):")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        for item in data.get("items", []):
            vid = item.get("id")
            if not vid:
                continue
            snippet = item.get("snippet") or {}
            statistics = item.get("statistics") or {}

            view_count = None
            raw_views = statistics.get("viewCount")
            if raw_views is not None:
                try:
                    view_count = int(raw_views)
                except (TypeError, ValueError):
                    view_count = None

            details[vid] = {
                "title": snippet.get("title", "") or "",
                "description": snippet.get("description", "") or "",
                "channel": snippet.get("channelTitle", "") or "",
                "view_count": view_count,
                "tags": snippet.get("tags") or [],
            }

    return details



YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


def _youtube_api_search_ids(keyword: str, max_results: int = 1) -> list[str]:
    """Search YouTube via the Data API's search.list endpoint and return a
    list of video IDs. Costs 100 quota units per call regardless of
    max_results."""
    if not YOUTUBE_API_KEY:
        print("[YT-API] YOUTUBE_API_KEY not set, skipping search")
        return []

    params = {
        "part": "id",
        "q": keyword,
        "type": "video",
        "maxResults": max_results,
        "key": YOUTUBE_API_KEY,
        "safeSearch": "none",
        "order": "relevance",
    }

    try:
        resp = _http_session.get(f"{YOUTUBE_API_BASE}/search", params=params, timeout=15)
    except Exception as e:
        print(f"[YT-API] search request failed for '{keyword}': {e}")
        return []

    if resp.status_code != 200:
        print(f"[YT-API] search HTTP {resp.status_code} for '{keyword}': {resp.text[:300]}")
        return []

    try:
        data = resp.json()
    except Exception as e:
        print(f"[YT-API] failed to parse search JSON for '{keyword}': {e}")
        return []

    print(f"[YT-API] RAW search.list response for '{keyword}':")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    video_ids = []
    for item in data.get("items", []):
        vid = (item.get("id") or {}).get("videoId")
        if vid:
            video_ids.append(vid)

    return video_ids



def _youtube_search_via_api(keyword: str, max_results: int = 1) -> list[dict]:
    video_ids = _youtube_api_search_ids(keyword, max_results=max_results)
    if not video_ids:
        return []

    details_by_id = _youtube_api_video_details(video_ids)

    results = []
    for vid in video_ids:
        detail = details_by_id.get(vid)
        if not detail:
            continue
        results.append({
            "url": f"https://www.youtube.com/watch?v={vid}",
            "title": detail["title"],
            "description": detail["description"],
            "channel": detail["channel"],
            "view_count": detail["view_count"],
            "tags": detail["tags"],
        })

    return results


async def get_youtube_context(
    topic: str, description: str, scraped_urls: set, max_results: int = 10
) -> list[dict]:
    print(f"[YT] Starting YouTube search for topic: '{topic}'")

    if not YOUTUBE_API_KEY:
        print("[YT] YOUTUBE_API_KEY not set, skipping YouTube search")
        return []

    keywords = await _generate_youtube_search_keywords(topic, description)

    raw_candidates: list[dict] = []

    for keyword in keywords:
        try:
            results = await _run_scrape(_youtube_search_via_api, keyword, 1)
        except Exception as e:
            print(f"[YT] search failed for '{keyword}': {e}")
            results = []

        for r in results:
            url = r["url"]
            if url in scraped_urls:
                continue
            scraped_urls.add(url)

            title = r.get("title", "")
            desc = _truncate_words(r.get("description", ""), max_words=150)
            tags = r.get("tags") or []
            hashtags = _extract_hashtags(r.get("title", ""), r.get("description", ""))

            raw_candidates.append({
                "url": url,
                "title": title,
                "description": desc,
                "channel": r.get("channel", ""),
                "view_count": r.get("view_count"),
                "tags": tags,
                "hashtags": hashtags,
            })

    raw_candidates.sort(key=lambda v: v.get("view_count") or 0, reverse=True)
    videos = raw_candidates[:MAX_YOUTUBE_SOURCES]

    print(
        f"[YT] fetched {len(raw_candidates)} unique candidate video(s) via YouTube Data API "
        f"from {len(keywords)} keyword(s), returning top {len(videos)} "
        f"(capped at {MAX_YOUTUBE_SOURCES})"
    )

    return videos


def _build_ideas_context(db_results: list[dict], new_articles: list[dict]) -> str:
    parts = []

    if db_results:
        parts.append("=== KNOWLEDGE BASE EXCERPTS ===")
        for i, row in enumerate(db_results, start=1):
            content = row.get("content", "")
            parts.append(f"[KB-{i}] {content}")

    if new_articles:
        parts.append("\n=== RECENT NEWS ===")
        for i, article in enumerate(new_articles, start=1):
            snippet = article.get("snippet", "")
            url = article.get("url", "")
            parts.append(f"[NEWS-{i}] {snippet} (source: {url})")

    return "\n\n".join(parts) if parts else "No additional context available."


_SPLIT_ON_SUMMARY_HEADER = re.compile(
    r"\n\s*(?:#+\s*)?(?:\*\*)?"
    r"(?:Output\s*2\s*[-–—]?\s*)?"
    r"Topic\s*Summary"
    r"(?:\*\*)?:?\s*",
    re.IGNORECASE,
)

_TITLE_LABEL_CORE = r"\**\s*(?:#+\s*)?(?:\d+[\.\)]\s*)?\**\s*Title\**\s*:?\**"
_DESC_LABEL_CORE = r"\**\s*(?:\d+[\.\)]\s*)?\**\s*Description\**\s*:?\**"

_IDEA_PATTERN = re.compile(
    r"(?:^|\n)\s*" + _TITLE_LABEL_CORE + r"\s*"
    r"(?P<title>.+?)\s*\n+"
    r"\s*" + _DESC_LABEL_CORE + r"\s*"
    r"(?P<description>.+?)"
    r"(?=\n+\s*" + _TITLE_LABEL_CORE + r"|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _clean_idea_text(text_value: str) -> str:
    text_value = re.sub(r"\n?-{2,}\s*$", "", text_value)
    text_value = re.sub(r"^\s*(?:#+\s*)?(?:\*\*)?Output\s*1\b.*?\n", "", text_value, flags=re.IGNORECASE)
    text_value = text_value.strip("*_ \n")
    return text_value.strip()


def _split_ideas_and_summary(raw: str) -> tuple[str, str]:
    parts = _SPLIT_ON_SUMMARY_HEADER.split(raw, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    print("[IDEAS] no 'Topic Summary' header found, summary will be empty")
    return raw.strip(), ""


def _parse_ideas_markdown(raw: str) -> list[dict]:
    ideas = []
    for match in _IDEA_PATTERN.finditer(raw):
        title = _clean_idea_text(match.group("title"))
        description = _clean_idea_text(match.group("description"))
        if title and description:
            ideas.append({"title": title, "description": description})

    if ideas:
        return ideas

    print("[IDEAS] structured parse found nothing, attempting fallback split")
    blocks = re.split(r"\n\s*\n+", raw.strip())
    buffer_title = None
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if len(lines) == 1 and len(block) < 150 and buffer_title is None:
            buffer_title = _clean_idea_text(block)
            continue
        if buffer_title:
            ideas.append({"title": buffer_title, "description": _clean_idea_text(block)})
            buffer_title = None

    return ideas


def _clean_summary_text(text_value: str) -> str:
    text_value = text_value.strip()
    text_value = re.sub(r"^\s*(?:#+\s*)?(?:\*\*)?(?:Output\s*2\b.*?)?(?:\*\*)?:?\s*", "", text_value, flags=re.IGNORECASE)
    text_value = text_value.strip("*_ \n")
    return text_value.strip()


async def generate_ideas_from_context(
    topic: str, db_results: list[dict], new_articles: list[dict]
) -> dict:
    context_block = _build_ideas_context(db_results, new_articles)

    user_prompt = f"""Topic: "{topic}"

Content Chunks:
{context_block}
"""

    res = await _openai_create_with_timeout(
        lambda: openai_client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": IDEAS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )
    )
    _record_token_usage("generate_ideas_from_context", res)

    raw = res.choices[0].message.content.strip()

    ideas_block, summary_block = _split_ideas_and_summary(raw)

    ideas = _parse_ideas_markdown(ideas_block)
    topic_summary = _clean_summary_text(summary_block) if summary_block else ""

    return {"ideas": ideas, "topic_summary": topic_summary}


@app.post("/generate-ideas")
async def generate_ideas_endpoint(
    request: GenerateIdeasRequest,
):
    # GenerateIdeasRequest must carry a `userId` field for this check to work.
    # If your model doesn't have one yet, add: userId: str
    user_id = getattr(request, "userId", None)
    await require_valid_user(user_id)

    async with _pipeline_semaphore:
        return await _generate_ideas_endpoint_impl(request)


async def _generate_ideas_endpoint_impl(request: "GenerateIdeasRequest"):
    _start_token_tracking()

    topic = request.topic.strip()

    if not topic:
        raise HTTPException(status_code=400, detail="topic must be a non-empty string")

    try:
        hyde_prompt = f"""
        You are a Semantic Query Expansion Engine for a YouTube documentary research pipeline.

        Goal:
        Convert short user search queries (typically 3–7 keywords) into a natural-language semantic search paragraph optimized for vector search (RAG), NOT for humans.

        Input:
        User Query:
        "{topic}"
        A user query containing 3–7 keywords or a short topic.

        Task:
        1. Infer the user's true research intent.
        2. Expand the topic into a coherent natural-language paragraph.
        3. Preserve the original meaning while enriching context.
        4. Include synonyms, related concepts, alternate terminology, historical
           context, scientific concepts, geographical references, cultural
           context, causes/effects/mechanisms, notable events/discoveries/
           people/civilizations or theories when relevant.
        5. Include entities, alternate spellings and commonly searched phrases naturally.
        6. Do NOT invent unsupported facts. Expand only using generally accepted knowledge.
        7. Write as continuous natural language without bullets, lists or headings.
        8. Avoid conversational text, opinions, explanations or instructions.
        9. Maximize semantic richness and topical coverage for embedding similarity rather than keyword stuffing.
        10. Output only the expanded paragraph.
        11. STRICT LENGTH LIMIT: the output must be under {HYDE_MAX_TOKENS} tokens
        (roughly 35-50 words), a single short paragraph, with no additional text before or after it.
        """
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[{"role": "user", "content": hyde_prompt}],
                max_completion_tokens=400,
                stream=False,
            )
        )
        _record_token_usage("generate-ideas HYDE (initial)", res)

        raw_hyde_doc = (res.choices[0].message.content or "").strip()

        if not raw_hyde_doc:
            try:
                retry_res = await _openai_create_with_timeout(
                    lambda: openai_client.chat.completions.create(
                        model="gpt-5.4-mini",
                        messages=[{"role": "user", "content": hyde_prompt}],
                        max_completion_tokens=1500,
                        stream=False,
                    )
                )
                _record_token_usage("generate-ideas HYDE (retry)", retry_res)
                raw_hyde_doc = (retry_res.choices[0].message.content or "").strip()
            except Exception as retry_exc:
                print(f"[HYDE] retry call raised: {retry_exc}")
                raw_hyde_doc = ""

        hyde_doc = _cap_hyde_doc_tokens(raw_hyde_doc) if raw_hyde_doc else topic

        db_task = asyncio.create_task(get_context_from_db(topic, hyde_doc))
        similar_task = asyncio.create_task(get_similar_saved_ideas(topic, hyde_doc))

        done, pending = await asyncio.wait({db_task}, timeout=11)

        db_results = []
        new_articles = []
        scraped_urls = set()

        if db_task in done:
            try:
                db_results = db_task.result()
            except Exception as e:
                print(f"[MAIN] DB task raised an error: {e}")
                db_results = []

        try:
            new_articles = await get_ddgs_news_context(topic, scraped_urls, hyde_doc)
        except Exception as exc:
            print(f"[MAIN] web search (DDGS) failed: {exc}")
            new_articles = []

        if not db_results and db_task not in done:
            try:
                db_results = await asyncio.wait_for(asyncio.shield(db_task), timeout=5)
            except asyncio.TimeoutError:
                db_results = []
            except Exception as e:
                print(f"[MAIN] DB task raised an error on late check: {e}")
                db_results = []

        try:
            similar_saved_ideas = await asyncio.wait_for(similar_task, timeout=5)
        except asyncio.TimeoutError:
            similar_saved_ideas = []
        except Exception as e:
            print(f"[MAIN] similar_task raised an error: {e}")
            similar_saved_ideas = []

        try:
            result = await generate_ideas_from_context(topic, db_results, new_articles)
            ideas = result["ideas"]
            topic_summary = result["topic_summary"]
        except Exception as exc:
            print(f"[MAIN] idea generation failed: {exc}")
            ideas = []
            topic_summary = ""

        token_usage = _get_token_usage_summary()

        return {
            "topic": topic,
            "topic_summary": topic_summary,
            "ideas": ideas,
            "similar_past_ideas": similar_saved_ideas,
            "token_usage": token_usage,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] /generate-ideas failed: {e}")
        traceback.print_exc()
        return {
            "error": "An error occurred in the idea generation pipeline.",
            "detail": str(e),
            "token_usage": _get_token_usage_summary(),
        }


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

        response = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": "Return only the category name."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )
        )
        _record_token_usage("get_structure", response)

        category = response.choices[0].message.content.strip()
        return {"category": category}

    except Exception as e:
        return {"category": "UNKNOWN", "error": str(e)}


async def get_channel_profile(userId: str):
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
        sub_res = supabase.table('subscriptions') \
            .select('id, credits, purchased_date') \
            .eq('userId', request.userId) \
            .order('purchased_date', desc=True) \
            .limit(1) \
            .execute()

        if sub_res.data and len(sub_res.data) > 0:
            latest_subscription = sub_res.data[0]
            subscription_id = latest_subscription["id"]
            subscription_credits = latest_subscription["credits"]

            if subscription_credits <= 0 or subscription_credits < request.duration:
                return {"message": "credits not sufficient"}

            new_subscription_credits = subscription_credits - request.duration

            supabase.table('subscriptions') \
                .update({'credits': new_subscription_credits}) \
                .eq('id', subscription_id) \
                .execute()

            return {
                "message": "success",
                "source": "subscription",
                "remaining_credits": new_subscription_credits,
            }

        profile_res = supabase.table('user_profiles') \
            .select('credits_remaining') \
            .eq('id', request.userId) \
            .single() \
            .execute()

        old_credits = profile_res.data["credits_remaining"]

        if old_credits <= 0 or old_credits < request.duration:
            return {"message": "credits not sufficient"}

        new_credits = old_credits - request.duration

        supabase.table('user_profiles') \
            .update({'credits_remaining': new_credits}) \
            .eq('id', request.userId) \
            .execute()

        return {
            "message": "success",
            "source": "profile",
            "remaining_credits": new_credits,
        }

    except Exception as e:
        print("error:", e)
        raise HTTPException(status_code=500, detail=str(e))


def target_word_count_for_time(minutes: int) -> int:
    return max(50, int(minutes * WORDS_PER_MINUTE))


class ScriptRequest(BaseModel):
    userId: str
    title: str
    description: str
    time: int

def build_topic_text(request: "ScriptRequest") -> str:
    return f"{request.title}\n\n{request.description}".strip()


def bucket_segments_by_time(segments: list[dict], num_docs: int) -> list[list[dict]]:
    if not segments:
        return [[]]

    num_docs = max(1, min(num_docs, len(segments)) if num_docs <= len(segments) else num_docs)

    if num_docs >= len(segments):
        return [[s] for s in segments]

    total_pct = sum(s.get("percentage", 0) for s in segments) or 100
    target_per_bucket = total_pct / num_docs

    buckets: list[list[dict]] = []
    current_bucket: list[dict] = []
    running_pct = 0.0

    for seg in segments:
        current_bucket.append(seg)
        running_pct += seg.get("percentage", 0)
        if running_pct >= target_per_bucket and len(buckets) < num_docs - 1:
            buckets.append(current_bucket)
            current_bucket = []
            running_pct = 0.0

    if current_bucket:
        buckets.append(current_bucket)

    while len(buckets) < num_docs and len(buckets) > 0:
        buckets.append(buckets[-1])
    while len(buckets) > num_docs:
        buckets[-2].extend(buckets[-1])
        buckets.pop()

    return buckets


def num_hyde_docs_for_time(minutes: int) -> int:
    return max(1, math.ceil(minutes / 2))


async def generate_hyde_doc_for_segments(
    title: str,
    description: str,
    template: dict,
    segment_group: list[dict],
    time_minutes: int,
) -> str:
    segment_briefs = "\n".join(
        f"- {seg.get('name', 'segment')} ({seg.get('percentage', 0)}%): {seg.get('brief', '')}"
        for seg in segment_group
    )

    fallback_text = f"{title}\n\n{description}".strip()

    hyde_prompt = f"""
            You are generating a HyDE (Hypothetical Document Embedding) passage to
            drive retrieval for one part of a video script.

            Idea Title: "{title}"
            Idea Description: "{description}"
            Target Video Duration: {time_minutes} minute(s)

            This passage must strictly follow the structure of the retrieved script
            template below — do not invent a different structure.

            Template: "{template.get('title')}" (cluster: {template.get('cluster')})
            Template purpose: {template.get('about')}

            This HyDE document should specifically support retrieval for the
            following segment(s) of that template:
            {segment_briefs}

            Write a short, factual, encyclopedia-style paragraph that provides direct,
            concrete, retrievable information relevant to the idea and the segment(s)
            above. Be concise, information-dense, and include key terms a search/embedding
            system would match against. Do not write in a narrative or scripted tone — this is
            a retrieval seed document, not the script itself.

            STRICT LENGTH LIMIT: output must be under {HYDE_MAX_TOKENS} tokens
            (roughly 35-50 words, a single short paragraph). Do not exceed this.
            Output only the paragraph, nothing else.
""".strip()

    segment_label = ', '.join(s.get('name', 'segment') for s in segment_group)

    async def _call(max_tokens: int):
        completion = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[{"role": "user", "content": hyde_prompt}],
                max_completion_tokens=max_tokens,
                stream=False,
            )
        )
        _record_token_usage(f"generate_hyde_doc_for_segments[{segment_label}] (max_tokens={max_tokens})", completion)
        choice = completion.choices[0]
        raw_content = (choice.message.content or "").strip()
        finish_reason = getattr(choice, "finish_reason", None)
        output_tokens = None
        try:
            output_tokens = completion.usage.completion_tokens
        except Exception:
            pass
        return raw_content, finish_reason, output_tokens

    try:
        raw_doc, finish_reason, output_tokens = await _call(max_tokens=900)
        doc = _cap_hyde_doc_tokens(raw_doc) if raw_doc else ""

        if not doc:
            try:
                raw_doc, finish_reason, output_tokens = await _call(max_tokens=2000)
                doc = _cap_hyde_doc_tokens(raw_doc) if raw_doc else ""
            except Exception as retry_exc:
                print(f"--- HyDE DOC [{segment_label}] retry call raised: {retry_exc} ---")
                doc = ""

        if not doc:
            print(f"--- HyDE DOC [{segment_label}] still EMPTY after retry, falling back to title/description ---")
            return _cap_hyde_doc_tokens(fallback_text)

        return doc
    except Exception as exc:
        print(f"--- HyDE generation failed for segment group [{segment_label}]: {type(exc).__name__}: {exc} ---")
        return _cap_hyde_doc_tokens(fallback_text)


async def get_context_with_timeout(
    topic_text: str, hyde_document: str, table_name: str = None, timeout: float = 20.0
) -> list:
    task = asyncio.create_task(get_context_from_db(topic_text, hyde_document, table_name=table_name))
    done, pending = await asyncio.wait({task}, timeout=timeout)

    if task in done:
        try:
            result = task.result()
            print(f"[DB] task finished within timeout. Found {len(result)} documents.")
            return result
        except Exception as e:
            print(f"[DB] task raised an error: {e}")
            return []
    else:
        print("[DB] task still running after timeout, proceeding without it for now.")
        return []


SCRIPT_SYSTEM_PROMPT = """
You are a YouTube Script Writer for long-form documentary-style videos.

## Inputs
1. Video Title & Description
2. A script template (title, cluster, purpose, and an ordered list of segments,
   each with a name, target percentage of runtime, and a brief describing what
   that segment should accomplish)
3. Retrieved knowledge chunks and recent news snippets — ONLY high-confidence,
   semantically relevant material. Every chunk you are given already cleared a
   similarity bar against the topic (see DB_SIMILARITY_THRESHOLD /
   WEB_CONTENT_SIMILARITY_THRESHOLD), so treat all of it as
   trustworthy, on-topic source material.
4. A target total word count for the finished script (derived from the
   requested video duration)

## Objective
Write a complete, narration-ready YouTube script that:
- Strictly follows the template's segments, IN ORDER, using each segment's
  brief as its creative direction
- Allocates word count across segments roughly proportional to each
  segment's target percentage of runtime
- Weaves in concrete facts, figures, names, and details from the retrieved
  knowledge chunks and news snippets — grounded, not vague or generic
- Reads naturally aloud: conversational spoken-word rhythm, not essay prose
- Opens with a strong hook in the first segment and maintains narrative
  momentum throughout
- Lands within about 10% of the target word count
- Uses ONLY the provided source material for facts — do not invent
  statistics, quotes, or events not supported by the retrieved context

## Output
Output ONLY the finished script text, written as continuous narration broken
into paragraphs per segment. Prefix each segment with its name in brackets on
its own line (e.g. "[Hook]"), followed by the narration for that segment.
No preamble, no meta-commentary, no word-count notes, no markdown headers
beyond the segment name markers.
"""


def _build_script_context(db_results: list[dict], new_articles: list[dict]) -> str:
    parts = []

    if db_results:
        parts.append(f"=== KNOWLEDGE BASE EXCERPTS (dense similarity >= {DB_SIMILARITY_THRESHOLD}) ===")
        for i, row in enumerate(db_results, start=1):
            content = row.get("content", "")
            dense_score = row.get("dense_score")
            parts.append(f"[KB-{i}] (similarity={dense_score}) {content}")

    if new_articles:
        parts.append(f"\n=== RECENT NEWS / WEB (similarity >= {WEB_CONTENT_SIMILARITY_THRESHOLD}) ===")
        for i, article in enumerate(new_articles, start=1):
            snippet = article.get("snippet", "")
            url = article.get("url", "")
            similarity = article.get("similarity")
            parts.append(f"[NEWS-{i}] (similarity={similarity}) {snippet} (source: {url})")

    return "\n\n".join(parts) if parts else "No high-confidence source material available."


def _segments_brief(segments: list[dict]) -> str:
    if not segments:
        return "No template segments available — write a natural documentary-style structure."
    return "\n".join(
        f"- {seg.get('name', 'segment')} ({seg.get('percentage', 0)}% of runtime): {seg.get('brief', '')}"
        for seg in segments
    )


async def generate_script_from_context(
    request: "ScriptRequest",
    selected_template: dict,
    db_results: list[dict],
    new_articles: list[dict],
    target_word_count: int,
) -> str:
    context_block = _build_script_context(db_results, new_articles)
    segments_block = _segments_brief(selected_template.get("segments") or [])

    user_prompt = f"""
Video Title: "{request.title}"
Video Description: "{request.description}"
Target Duration: {request.time} minute(s)
Target Word Count: approximately {target_word_count} words

Template: "{selected_template.get('title')}" (cluster: {selected_template.get('cluster')})
Template Purpose: {selected_template.get('about')}

Segments (write the script in this exact order):
{segments_block}

Source Material:
{context_block}
"""

    try:
        completion = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": SCRIPT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
            ),
            timeout=max(OPENAI_CALL_TIMEOUT, 90.0),
        )
        _record_token_usage("generate_script_from_context", completion)
        script_text = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[SCRIPT] generation failed: {e}")
        script_text = ""

    return script_text


YOUTUBE_SEO_SYSTEM_PROMPT = """
You are a YouTube SEO metadata specialist.

## Inputs
1. The finished video script (or its title/description if no script is
   available)
2. Reference metadata scraped from real, currently-ranking YouTube videos on
   the same topic (titles, tags, hashtags) — use these to understand what
   keywords and phrasing already perform well. Do not copy them verbatim.

## Objective
Generate SEO-optimized YouTube metadata options for this video. Produce
EXACTLY:
- 3 alternative TITLES (each 40-70 characters, curiosity-driven, includes a
  primary keyword naturally, no clickbait or false claims)
- 3 alternative DESCRIPTIONS (each 60-100 words, opens with a hook sentence
  containing the primary keyword, naturally works in supporting keywords,
  ends with a soft call-to-action)
- 3 HASHTAG SETS (each set is 8-15 distinct hashtags suited for a YouTube
  video's hashtag field — every entry MUST start with "#", use camelCase for
  multi-word phrases (e.g. "#artificialIntelligence"), no spaces, no
  punctuation besides the leading "#", mix broad and long-tail/specific
  terms, no duplicate hashtags within a set)
- 3 THUMBNAIL TEXTS (each 4-8 words, punchy and readable at a glance, no
  full sentences)

## Output Format
Respond with ONLY valid JSON, no markdown code fences, no preamble, no
trailing commentary, in exactly this shape:

{
  "titles": ["...", "...", "..."],
  "descriptions": ["...", "...", "..."],
  "hashtags": [["#...", "#..."], ["#...", "#..."], ["#...", "#..."]],
  "thumbnail_text": ["...", "...", "..."]
}
"""


def _build_youtube_reference_block(new_videos: list[dict]) -> str:
    if not new_videos:
        return "No reference video metadata available."
    parts = []
    for i, v in enumerate(new_videos, start=1):
        tags = ", ".join(v.get("tags") or [])
        hashtags = ", ".join(v.get("hashtags") or [])
        parts.append(f"[REF-{i}] title: {v.get('title')} | tags: {tags} | hashtags: {hashtags}")
    return "\n".join(parts)


def _parse_json_block(raw: str) -> dict:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _keyword_to_hashtag(keyword: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", keyword)
    if not words:
        return ""
    first, rest = words[0].lower(), words[1:]
    camel = first + "".join(w.capitalize() for w in rest)
    return f"#{camel}" if camel else ""


def _build_hashtags_from_keywords(keywords: list[str]) -> list[str]:
    hashtags = []
    seen = set()
    for kw in keywords:
        tag = _keyword_to_hashtag(kw)
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            hashtags.append(tag)
    return hashtags


def _keyword_sets_to_hashtag_sets(keyword_sets: list[list[str]]) -> list[list[str]]:
    return [_build_hashtags_from_keywords(s) for s in (keyword_sets or [])]


def _build_fallback_youtube_metadata(request: "ScriptRequest") -> dict:
    base = (request.title or "").strip() or "This Topic"
    desc_hint = (request.description or "").strip()
    short_desc = (desc_hint[:140] + "...") if len(desc_hint) > 140 else desc_hint
    base_lower = base.lower()

    return {
        "titles": [
            f"{base}: The Full Story Explained",
            f"Why {base} Matters More Than You Think",
            f"The Truth Behind {base} (In-Depth)",
        ],
        "descriptions": [
            f"{base} — {short_desc or 'a deep dive into everything you need to know about this topic.'} Watch till the end for the full picture.",
            f"Everything you need to know about {base_lower}, explained clearly with real context and evidence. Subscribe for more deep dives like this.",
            f"A closer look at {base_lower}: what happened, why it matters, and what comes next. Let us know your thoughts in the comments.",
        ],
        "hashtags": _keyword_sets_to_hashtag_sets([
            [base_lower, "documentary", "explained", "deep dive", "full story"],
            [base_lower, "analysis", "breakdown", "case study", "explainer"],
            [base_lower, "facts", "history", "insight", "overview"],
        ]),
        "thumbnail_text": [
            (base[:28] or "Watch Now"),
            "The Full Story",
            "What Really Happened",
        ],
    }


async def generate_youtube_seo_metadata(
    request: "ScriptRequest",
    script_text: str,
    new_videos: list[dict],
) -> dict:
    reference_block = _build_youtube_reference_block(new_videos)
    script_excerpt = _truncate_words(script_text, max_words=300) if script_text else ""
    fallback = _build_fallback_youtube_metadata(request)

    user_prompt = f"""
Video Title: "{request.title}"
Video Description: "{request.description}"

Script excerpt (for context on the actual content/angle):
{script_excerpt or "No script available — base metadata on the title/description alone."}

Reference metadata from currently-ranking videos on this topic:
{reference_block}
"""

    metadata = None

    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": YOUTUBE_SEO_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
            )
        )
        _record_token_usage("generate_youtube_seo_metadata", res)
        raw = (res.choices[0].message.content or "").strip()
        metadata = _parse_json_block(raw)
    except Exception as e:
        print(f"[SEO] generation/parse failed: {e} — retrying once")
        try:
            res = await _openai_create_with_timeout(
                lambda: openai_client.chat.completions.create(
                    model="gpt-5.4-mini",
                    messages=[
                        {"role": "system", "content": YOUTUBE_SEO_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=1500,
                    stream=False,
                )
            )
            _record_token_usage("generate_youtube_seo_metadata (retry)", res)
            raw = (res.choices[0].message.content or "").strip()
            metadata = _parse_json_block(raw)
        except Exception as e2:
            print(f"[SEO] retry also failed: {e2} — using deterministic fallback")
            metadata = {}

    if not isinstance(metadata, dict):
        metadata = {}

    raw_hashtag_sets = metadata.get("hashtags")
    if not isinstance(raw_hashtag_sets, list) or not raw_hashtag_sets:
        raw_hashtag_sets = metadata.get("keywords")
    metadata["hashtags"] = _keyword_sets_to_hashtag_sets(raw_hashtag_sets or [])
    metadata.pop("keywords", None)

    for key, fallback_values in fallback.items():
        values = metadata.get(key)
        if not isinstance(values, list) or len(values) == 0:
            print(f"[SEO] '{key}' missing/empty in generated metadata, using fallback")
            metadata[key] = fallback_values
        else:
            i = 0
            while len(metadata[key]) < 3 and i < len(fallback_values):
                metadata[key].append(fallback_values[i])
                i += 1

    return metadata


def _extract_source_website_names(articles: list[dict]) -> list[str]:
    names = []
    seen = set()
    for article in articles:
        url = article.get("url", "")
        if not url:
            continue
        try:
            netloc = urlparse(url).netloc.lower()
        except Exception:
            continue
        if netloc.startswith("www."):
            netloc = netloc[4:]
        if netloc and netloc not in seen:
            seen.add(netloc)
            names.append(netloc)
    return names


def _unique_url_count(articles: list[dict]) -> int:
    return len({a.get("url") for a in articles if a.get("url")})


async def _backfill_sources_to_target(
    new_articles: list[dict],
    scraped_urls: set,
    title: str,
    hyde_doc: str,
    keywords: list[str] | None = None,
    target_count: int = MAX_WEB_SOURCES,
    max_rounds: int = 8,
) -> list[dict]:

    def _existing_urls() -> set:
        return {a.get("url") for a in new_articles if a.get("url")}

    round_num = 0
    while _unique_url_count(new_articles) < target_count and round_num < max_rounds:
        round_num += 1
        added_this_round = 0

        if round_num == 1:
            try:
                relaxed = await get_ddgs_news_context_for_script(
                    title, scraped_urls, hyde_doc, similarity_threshold=0.0, keywords=keywords,
                )
            except Exception as e:
                print(f"[MAIN] sources backfill round {round_num} failed: {e}")
                relaxed = []

            known_domains = set(_extract_source_website_names(new_articles))
            existing_urls = _existing_urls()
            for article in relaxed:
                if _unique_url_count(new_articles) >= target_count:
                    break
                url = article.get("url", "")
                if not url or url in existing_urls:
                    continue
                netloc = urlparse(url).netloc.lower()
                if netloc.startswith("www."):
                    netloc = netloc[4:]
                new_articles.append(article)
                existing_urls.add(url)
                if netloc:
                    known_domains.add(netloc)
                added_this_round += 1

        else:
            generic_queries = [
                title,
                f"{title} news",
                f"{title} latest update",
                f"{title} analysis",
                f"{title} explained",
                f"{title} report",
                f"{title} overview",
                f"{title} background",
                f"{title} facts",
                f"{title} details",
            ]
            existing_urls = _existing_urls()
            for query in generic_queries:
                if _unique_url_count(new_articles) >= target_count:
                    break
                try:
                    pairs = await _run_scrape(_ddgs_search_for_script, query, 20)
                except Exception as e:
                    print(f"[MAIN] sources backfill generic query '{query}' failed: {e}")
                    continue
                for url, snippet in pairs:
                    if _unique_url_count(new_articles) >= target_count:
                        break
                    if not url or url in scraped_urls or url in existing_urls:
                        continue
                    scraped_urls.add(url)
                    existing_urls.add(url)
                    new_articles.append({
                        "url": url,
                        "snippet": _truncate_words(snippet or title, max_words=200),
                        "source": "fallback-backfill",
                        "similarity": 0.0,
                        "picked_passage_count": 0,
                        "total_passage_count": 0,
                    })
                    added_this_round += 1

        if added_this_round == 0:
            print(f"[MAIN] sources backfill round {round_num} added 0 new source(s), stopping this round type")

    final_count = _unique_url_count(new_articles)
    if final_count < target_count:
        print(
            f"[MAIN] WARNING: could only find {final_count}/{target_count} unique source URL(s) "
            f"for this topic after {round_num} backfill round(s) — the web genuinely doesn't have "
            f"more distinct results to offer for this query."
        )
    else:
        print(f"[MAIN] sources backfill reached the compulsory target: {final_count}/{target_count} unique source URL(s)")

    return new_articles


SCRIPT_METRICS_SYSTEM_PROMPT = """
You are a content analyst reviewing a finished YouTube documentary script.

## Task
Read the script and count/score the following content elements exactly as
they appear in the script — do not estimate generically, actually look at
what's present in the text.

- emotionalDepth: a 1-10 score for how emotionally engaging/resonant the
  script is (human stakes, tension, vivid imagery), not just informational
- generalExamples: count of general illustrative examples used (concrete
  scenarios, comparisons, "for example" style illustrations) that are NOT
  historical events
- proverbs_count: count of proverbs, sayings, quotes, or aphorisms used
- historicalExamples: count of specific historical events, figures, or
  eras referenced as examples
- researchFacts: count of distinct research-backed facts, statistics, or
  studies cited
- keywords: 8-15 distinct topical keywords/phrases that best represent the
  script's subject matter — real subject-matter terms only, never a segment
  label like "[Hook]" or "[Intro]" and never a generic filler word

## Scoring rules
Every numeric field below is on a MINIMUM scale of 1. A value of 0 is NEVER
valid output for any numeric field.

## Output Format
Respond with ONLY valid JSON, no markdown fences, no preamble, in exactly
this shape:

{
  "emotionalDepth": <number, 1-10>,
  "generalExamples": <number, >= 1>,
  "proverbs_count": <number, >= 1>,
  "historicalExamples": <number, >= 1>,
  "researchFacts": <number, >= 1>,
  "keywords": ["...", "..."]
}
"""

_METRIC_MIN_VALUES = {
    "emotionalDepth": 1,
    "generalExamples": 1,
    "proverbs_count": 1,
    "historicalExamples": 1,
    "researchFacts": 1,
}

_DEFAULT_SCRIPT_METRICS = {
    "emotionalDepth": 1,
    "generalExamples": 1,
    "proverbs_count": 1,
    "historicalExamples": 1,
    "researchFacts": 1,
}

_KEYWORD_STOPWORDS = {
    "hook", "intro", "introduction", "climax", "outro", "conclusion",
    "segment", "around", "simple", "because", "before", "after", "there",
    "their", "which", "would", "could", "should", "these", "those",
    "where", "while", "about", "through", "during", "again", "still",
}


def _clean_keyword_token(token: str) -> str:
    token = token.strip()
    token = re.sub(r"^\[|\]$", "", token)
    token = token.strip(".,!?\"'*:; \n").lower()
    return token


def _clamp_metric_value(key: str, value) -> int:
    floor = _METRIC_MIN_VALUES.get(key, 1)
    try:
        num = int(round(float(value)))
    except (TypeError, ValueError):
        num = floor
    if num < floor:
        num = floor
    if key in ("emotionalDepth",) and num > 10:
        num = 10
    return num


def _fallback_keywords_from_text(script_text: str, topic_text: str, min_count: int = 8) -> list[str]:
    words = [_clean_keyword_token(w) for w in script_text.split()]
    seen = set()
    fallback_kw = []
    for w in words:
        if not w or len(w) <= 5 or w in _KEYWORD_STOPWORDS or not w.isalpha():
            continue
        if w not in seen:
            seen.add(w)
            fallback_kw.append(w)
        if len(fallback_kw) >= min_count:
            break
    if not fallback_kw:
        fallback_kw = [w.strip().lower() for w in re.split(r"[,\n]", topic_text) if w.strip()][:min_count]
    return fallback_kw or [topic_text.strip().lower() or "topic"]


async def generate_script_metrics(script_text: str, topic_text: str = "") -> dict:
    if not script_text:
        metrics = dict(_DEFAULT_SCRIPT_METRICS)
        metrics["keywords"] = _fallback_keywords_from_text("", topic_text)
        return metrics

    user_prompt = f"Script:\n{script_text}"

    metrics = None
    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": SCRIPT_METRICS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
            )
        )
        _record_token_usage("generate_script_metrics", res)
        raw = (res.choices[0].message.content or "").strip()
        metrics = _parse_json_block(raw)
    except Exception as e:
        print(f"[METRICS] generation/parse failed: {e}")
        metrics = {}

    if not isinstance(metrics, dict):
        metrics = {}

    for key in _METRIC_MIN_VALUES:
        raw_value = metrics.get(key)
        clamped = _clamp_metric_value(key, raw_value)
        metrics[key] = clamped

    raw_keywords = metrics.get("keywords") or []
    cleaned_keywords = []
    seen_kw = set()
    for kw in raw_keywords:
        if not isinstance(kw, str):
            continue
        cleaned = _clean_keyword_token(kw)
        if not cleaned or cleaned in _KEYWORD_STOPWORDS or not re.match(r"^[a-z][a-z\s\-']*$", cleaned):
            continue
        if cleaned not in seen_kw:
            seen_kw.add(cleaned)
            cleaned_keywords.append(cleaned)

    if len(cleaned_keywords) < 8:
        for kw in _fallback_keywords_from_text(script_text, topic_text, min_count=8):
            if kw not in seen_kw:
                seen_kw.add(kw)
                cleaned_keywords.append(kw)
            if len(cleaned_keywords) >= 8:
                break

    metrics["keywords"] = cleaned_keywords

    return metrics


MYSQL_URL = os.getenv("MYSQL_URL")

_mysql_engine = None


def get_mysql_engine():
    global _mysql_engine
    if _mysql_engine is None:
        print("[MYSQL] Creating engine")
        _mysql_engine = create_engine(MYSQL_URL, pool_pre_ping=True, pool_recycle=280)
    return _mysql_engine


def _fetch_books_by_md5_sync(md5_list: list[str]) -> list[dict]:
    if not md5_list:
        return []

    engine = get_mysql_engine()
    query = text(
        f"SELECT Title, Author, Year, md5 FROM {BOOKS_TABLE_NAME} WHERE md5 IN :md5_list"
    ).bindparams(bindparam("md5_list", expanding=True))

    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"md5_list": md5_list})
            rows = [dict(r._mapping) for r in result]
        return rows
    except Exception as e:
        print(f"[MYSQL] book lookup failed: {e}")
        return []


def _normalize_book_year(raw_year) -> str | None:
    """Normalize whatever comes back from MySQL for the Year column into a
    clean display string (e.g. '2014'), or None if there's no usable year."""
    if raw_year is None:
        return None
    year_str = str(raw_year).strip()
    if not year_str or year_str.lower() in ("none", "null", "0", "0000"):
        return None
    match = re.match(r"(\d{3,4})", year_str)
    return match.group(1) if match else year_str


async def get_books_for_chunks(
    all_db_chunks: list[dict],
    topic_text: str = "",
    script_text: str = "",
    max_books: int = MAX_BOOKS,
) -> list[dict]:
    """
    DB-ONLY. Pulls book Title/Author/Year entries strictly from the
    `english_books` MySQL table by matching chunk md5 values.
    """
    md5_list = []
    seen_md5 = set()
    for row in all_db_chunks:
        md5 = row.get("md5")
        if md5 and md5 not in seen_md5:
            seen_md5.add(md5)
            md5_list.append(md5)

    books: list[dict] = []
    if md5_list:
        print(f"[MYSQL] looking up {len(md5_list)} unique md5(s) for book Title/Author/Year in '{BOOKS_TABLE_NAME}'")
        rows = await asyncio.to_thread(_fetch_books_by_md5_sync, md5_list)

        seen_books = set()
        for row in rows:
            title = row.get("Title")
            author = row.get("Author")
            if not title or not author:
                continue
            title = str(title).strip()
            author = str(author).strip()
            if not title or not author:
                continue
            year = _normalize_book_year(row.get("Year"))
            key = (title, author)
            if key not in seen_books:
                seen_books.add(key)
                books.append({"title": title, "author": author, "year": year})

        print(f"[MYSQL] resolved {len(books)} unique book(s) with title+author (+year where available) from {len(md5_list)} md5(s)")
    else:
        print("[MYSQL] no md5s found on retrieved DB chunks, skipping direct lookup")

    books = books[:max_books]
    print(f"[MYSQL] final books list: {len(books)} entries (no placeholder padding)")
    return books


async def _backfill_books_to_target(
    current_books: list[dict],
    known_md5s: set,
    topic_text: str,
    hyde_doc: str,
    table_name: str,
    target_count: int = MAX_BOOKS,
    max_rounds: int = 4,
) -> list[dict]:
    books = list(current_books)
    seen_book_keys = {(b["title"], b["author"]) for b in books}
    seen_md5 = set(known_md5s)

    query_variants = [q for q in (hyde_doc, topic_text) if q]

    for round_num, query in enumerate(query_variants, start=1):
        if len(books) >= target_count or round_num > max_rounds:
            break

        print(f"[MYSQL-BACKFILL] round {round_num}: widening DB search (threshold=0.0, match_count=100)")
        try:
            candidates = await get_context_from_db(
                topic_text,
                query,
                final_k=100,
                table_name=table_name,
                similarity_threshold=0.0,
                match_count=100,
            )
        except Exception as e:
            print(f"[MYSQL-BACKFILL] round {round_num} retrieval failed: {e}")
            continue

        new_md5s = []
        for item in candidates:
            md5 = item.get("md5")
            if md5 and md5 not in seen_md5:
                seen_md5.add(md5)
                new_md5s.append(md5)

        if not new_md5s:
            print(f"[MYSQL-BACKFILL] round {round_num}: no new candidate md5(s), skipping")
            continue

        print(f"[MYSQL-BACKFILL] round {round_num}: checking {len(new_md5s)} new candidate md5(s) against MySQL")
        rows = await asyncio.to_thread(_fetch_books_by_md5_sync, new_md5s)

        for row in rows:
            if len(books) >= target_count:
                break
            title = (row.get("Title") or "").strip()
            author = (row.get("Author") or "").strip()
            if not title or not author:
                continue
            key = (title, author)
            if key in seen_book_keys:
                continue
            year = _normalize_book_year(row.get("Year"))
            seen_book_keys.add(key)
            books.append({"title": title, "author": author, "year": year})

        print(f"[MYSQL-BACKFILL] round {round_num} done — now {len(books)}/{target_count} book(s)")

    if len(books) < target_count:
        print(
            f"[MYSQL-BACKFILL] stopped after exhausting query variants — only {len(books)}/"
            f"{target_count} distinct book(s) with both title+author exist in the DB for this topic."
        )

    return books[:target_count]


FACE_THUMBNAILS_TABLE = "user_profiles"
FACE_PHOTO_DEFAULT_KEY = "photo1"


async def get_user_face_photo_url(user_id: str, photo_key: str = FACE_PHOTO_DEFAULT_KEY) -> str | None:
    try:
        res = await asyncio.to_thread(
            lambda: supabase.table(FACE_THUMBNAILS_TABLE)
            .select("thumbnail_images")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        print(f"[FACE] Supabase query for user_profiles.thumbnail_images failed for user {user_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

    rows = res.data or []
    print(f"[FACE] user_profiles lookup for id={user_id} returned {len(rows)} row(s)")

    if not rows:
        print(
            f"[FACE] no row in '{FACE_THUMBNAILS_TABLE}' with id={user_id} — "
            f"double check this matches the primary key column actually used "
            f"in that table (currently querying column 'id')"
        )
        return None

    row = rows[0]
    thumbnail_images = row.get("thumbnail_images")
    print(f"[FACE] raw thumbnail_images for user {user_id}: {thumbnail_images!r}")

    if isinstance(thumbnail_images, str):
        try:
            thumbnail_images = json.loads(thumbnail_images)
        except Exception as e:
            print(f"[FACE] thumbnail_images for user {user_id} is a string but not valid JSON: {e}")
            return None

    if not isinstance(thumbnail_images, dict):
        print(f"[FACE] thumbnail_images for user {user_id} is not a dict/object (got {type(thumbnail_images)})")
        return None

    photo_url = thumbnail_images.get(photo_key)
    if not photo_url:
        print(
            f"[FACE] thumbnail_images for user {user_id} has no '{photo_key}' entry — "
            f"available keys: {list(thumbnail_images.keys())}"
        )
        return None

    print(f"[FACE] resolved '{photo_key}' URL for user {user_id}: {photo_url}")
    return photo_url


def _download_image_bytes_sync(url: str, timeout: float = 15.0) -> bytes | None:
    try:
        response = _http_session.get(url, timeout=timeout)
    except Exception as e:
        print(f"[FACE] failed to download photo from {url}: {e}")
        return None

    if response.status_code != 200:
        print(f"[FACE] photo download returned HTTP {response.status_code} for {url} — body: {response.text[:300]}")
        return None

    content = response.content
    content_type = response.headers.get("Content-Type", "unknown")
    print(f"[FACE] downloaded photo: {len(content)} bytes, content-type={content_type}, url={url}")

    if not content or len(content) < 100:
        print(f"[FACE] downloaded photo looks suspiciously small/empty ({len(content)} bytes) — treating as failed download")
        return None

    return content


async def get_user_face_photo_bytes(user_id: str, photo_key: str = FACE_PHOTO_DEFAULT_KEY) -> bytes | None:
    photo_url = await get_user_face_photo_url(user_id, photo_key=photo_key)
    if not photo_url:
        return None

    photo_bytes = await asyncio.to_thread(_download_image_bytes_sync, photo_url)
    if not photo_bytes:
        print(f"[FACE] could not download usable photo bytes for user {user_id} from {photo_url}")
        return None

    return photo_bytes



THUMBNAIL_PROMPT_SYSTEM_PROMPT = """
You are a YouTube thumbnail art director.

## Inputs
1. Video title & description
2. The finished narration script (for grounding the scene/mood/story beats
   — use it to pick a moment or visual that actually represents the video,
   not just the title)
3. ONE specific short thumbnail text phrase that has already been chosen
   and MUST be rendered as real, legible text inside the generated image

## Objective
Write ONE image-generation prompt for a high-CTR YouTube thumbnail that
visually represents the video's central story or hook, AND explicitly
instructs the image model to render the given thumbnail text phrase as
bold, large, high-contrast text baked directly into the image.

Guidelines:
- Describe a single clear focal subject with strong emotional read, grounded
  in a specific beat, fact, or image pulled from the script — not just the
  title restated
- Specify composition, lighting and color mood, and a photographic or
  digital-art style appropriate to the content
- Explicitly state the exact thumbnail text phrase (quote it verbatim) and
  instruct it to appear as bold, punchy, high-contrast typography — e.g.
  thick sans-serif or condensed impact-style lettering, drop shadow or
  outline for readability, positioned in a clear area of the composition
  (top, bottom, or one side) that doesn't overlap the focal subject
- Do NOT add any other text, letters, numbers, logos, or watermarks beyond
  that exact phrase
- Do NOT depict real, named, identifiable public figures
- Ground the scene in concrete, specific details from the input rather than
  generic stock-photo phrasing

## Output
Output ONLY the finished image-generation prompt as a single dense
paragraph, 50-90 words. No preamble, no labels, no markdown, no explanation
of your choices. You may use quotation marks only around the thumbnail text
phrase itself.
"""

THUMBNAIL_PROMPT_SYSTEM_PROMPT_WITH_FACE = """
You are a YouTube thumbnail art director.

## Inputs
1. Video title & description
2. The finished narration script (for grounding the scene/mood/story beats
   — use it to pick a moment or visual that actually represents the video,
   not just the title)
3. ONE specific short thumbnail text phrase that has already been chosen
   and MUST be rendered as real, legible text inside the generated image

## Context
This prompt will be used for an IMAGE-TO-IMAGE edit starting from a real
photo of the video's creator. The creator's face and likeness from that
source photo will be preserved and placed into the scene you describe —
your job is to describe the SCENE, POSE, EXPRESSION, STYLING, and the
overlaid TEXT around them, not to invent a different person.

## Objective
Write ONE image-generation prompt for a high-CTR YouTube thumbnail where
the creator is the central subject, reacting to or standing in front of a
scene grounded in the script, AND the given thumbnail text phrase is
rendered as bold, large, high-contrast text baked directly into the image.

Guidelines:
- Keep the creator as a single, clear, front-and-center focal subject with
  a strong, readable facial expression (e.g. shocked, intrigued, excited,
  concerned — pick what fits the topic and the script's tone)
- Describe the background/scene behind or around them, grounded in a
  specific beat or detail from the script, plus composition, lighting, and
  color mood
- Explicitly state the exact thumbnail text phrase (quote it verbatim) and
  instruct it to appear as bold, punchy, high-contrast typography — thick
  sans-serif or condensed impact-style lettering, drop shadow or outline
  for readability, positioned in a clear area (top, bottom, or one side)
  that doesn't overlap the creator's face
- Do NOT add any other text, letters, numbers, logos, or watermarks beyond
  that exact phrase
- Do not describe specific facial features, ethnicity, age, or identity
  details — the real photo already provides those; focus only on
  expression, pose, styling, surroundings, and the text overlay
- Ground the scene in concrete, specific details from the input rather than
  generic stock-photo phrasing

## Output
Output ONLY the finished image-generation prompt as a single dense
paragraph, 50-90 words. No preamble, no labels, no markdown, no explanation
of your choices. You may use quotation marks only around the thumbnail text
phrase itself.
"""


def _pick_thumbnail_text(youtube_metadata: dict, request) -> str:
    candidates = youtube_metadata.get("thumbnail_text") or []
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    base = (request.title or "Watch Now").strip()
    return base[:28] if base else "Watch Now"


def _build_thumbnail_context(
    request,
    script_text: str,
    chosen_thumbnail_text: str,
) -> str:
    script_excerpt = _truncate_words(script_text, max_words=350) if script_text else "No script available."

    return f"""
Video Title: "{request.title}"
Video Description: "{request.description}"

Finished narration script (excerpt, for grounding the visual):
{script_excerpt}

Thumbnail text phrase that MUST be rendered inside the image (verbatim):
"{chosen_thumbnail_text}"
"""


def _fallback_thumbnail_prompt(request, chosen_thumbnail_text: str = None) -> str:
    base = (request.title or "this topic").strip()
    text_phrase = chosen_thumbnail_text or (base[:28] if base else "Watch Now")
    return (
        f"Cinematic, high-contrast documentary-style photo representing the story of {base}. "
        f"Dramatic lighting, bold saturated colors, strong single focal subject shot from a "
        f"dynamic angle, deep shadows, shallow depth of field. Render the text \"{text_phrase}\" "
        f"as bold, large, high-contrast impact-style typography with a drop shadow, positioned "
        f"in a clear area of the composition that doesn't overlap the main subject. No other "
        f"text, letters, numbers, logos, or watermarks anywhere in the image."
    )


async def generate_thumbnail_prompt(
    request,
    script_text: str,
    chosen_thumbnail_text: str,
    with_face: bool = False,
) -> str:
    context_block = _build_thumbnail_context(request, script_text, chosen_thumbnail_text)
    system_prompt = THUMBNAIL_PROMPT_SYSTEM_PROMPT_WITH_FACE if with_face else THUMBNAIL_PROMPT_SYSTEM_PROMPT

    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_block},
                ],
                stream=False,
            )
        )
        _record_token_usage("generate_thumbnail_prompt", res)
        image_prompt = (res.choices[0].message.content or "").strip()
        print(image_prompt)
    except Exception as e:
        print(f"[THUMBNAIL] prompt generation failed: {e}")
        image_prompt = ""

    if not image_prompt:
        image_prompt = _fallback_thumbnail_prompt(request, chosen_thumbnail_text)

    if chosen_thumbnail_text.lower() not in image_prompt.lower():
        print("[THUMBNAIL] generated prompt didn't mention the chosen thumbnail text — appending it explicitly")
        image_prompt = (
            f'{image_prompt} Render the text "{chosen_thumbnail_text}" as bold, large, '
            f"high-contrast typography baked into the image, in a clear area that doesn't "
            f"overlap the main subject."
        )

    return image_prompt


def _generate_thumbnail_image_gpt_image_sync(
    prompt: str,
    face_image_bytes: bytes | None = None,
    size: str = GPT_IMAGE_SIZE,
    quality: str = GPT_IMAGE_QUALITY,
) -> dict:
    try:
        if face_image_bytes:
            print(f"[THUMBNAIL-GPT] editing WITH user face photo (image-to-image, model='{GPT_IMAGE_MODEL}')")
            face_file = io.BytesIO(face_image_bytes)
            face_file.name = "face.jpg"

            response = openai_client.images.edit(
                model=GPT_IMAGE_MODEL,
                image=face_file,
                prompt=prompt,
                size=size,
                quality=quality,
            )
        else:
            print(f"[THUMBNAIL-GPT] generating text-to-image (model='{GPT_IMAGE_MODEL}')")

            response = openai_client.images.generate(
                model=GPT_IMAGE_MODEL,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1,
            )
    except Exception as e:
        print(f"[THUMBNAIL-GPT] request to GPT Image 2 failed: {e}")
        return {"image_base64": None, "error": f"request failed: {e}"}

    try:
        image_base64 = response.data[0].b64_json
    except Exception as e:
        return {"image_base64": None, "error": f"failed to parse GPT Image 2 response: {e}"}

    if not image_base64:
        print("[THUMBNAIL-GPT] GPT Image 2 returned no image data (b64_json empty)")
        return {"image_base64": None, "error": "empty image data in response"}

    print(f"[THUMBNAIL-GPT] received image ({len(image_base64)} base64 chars)")
    return {"image_base64": image_base64, "error": None}


async def generate_thumbnail_image(prompt: str, face_image_bytes: bytes | None = None) -> dict:
    try:
        result = await asyncio.to_thread(
            _generate_thumbnail_image_gpt_image_sync,
            prompt,
            face_image_bytes,
        )
    except Exception as e:
        print(f"[THUMBNAIL] GPT Image 2 image generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"image_base64": None, "prompt": prompt, "error": str(e)}

    if not result.get("image_base64"):
        print(f"[THUMBNAIL] GPT Image 2 returned no image: {result.get('error')}")
        return {"image_base64": None, "prompt": prompt, "error": result.get("error") or "empty image data"}

    return {"image_base64": result["image_base64"], "prompt": prompt, "error": None}


async def generate_thumbnail_for_script(
    request,
    script_text: str,
    youtube_metadata: dict,
) -> dict:
    chosen_thumbnail_text = _pick_thumbnail_text(youtube_metadata, request)
    print(f"[THUMBNAIL] chosen thumbnail text to render into image: '{chosen_thumbnail_text}'")

    face_image_bytes = None

    if getattr(request, "isFace", False):
        try:
            face_image_bytes = await get_user_face_photo_bytes(request.userId, photo_key=FACE_PHOTO_DEFAULT_KEY)
        except Exception as exc:
            print(f"[THUMBNAIL] face photo lookup/download failed, falling back to face-less thumbnail: {exc}")
            face_image_bytes = None

        if face_image_bytes:
            print(f"[THUMBNAIL] isFace=True — using user {request.userId}'s '{FACE_PHOTO_DEFAULT_KEY}' for thumbnail")
        else:
            print(f"[THUMBNAIL] isFace=True but no usable photo found for user {request.userId} — using text-to-image instead")

    image_prompt = await generate_thumbnail_prompt(
        request,
        script_text,
        chosen_thumbnail_text,
        with_face=bool(face_image_bytes),
    )
    result = await generate_thumbnail_image(image_prompt, face_image_bytes=face_image_bytes)
    return result




def _build_structure_response(selected_template: dict) -> list[dict]:
    segments = selected_template.get("segments") or []
    return [
        {"name": seg.get("name"), "percentage": seg.get("percentage")}
        for seg in segments
    ]


async def save_thumbnail_to_supabase(user_id: str, image_base64: str) -> str | None:
    if not image_base64:
        return None

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception as e:
        print(f"[THUMBNAIL-SAVE] failed to decode base64: {e}")
        return None

    file_path = f"{user_id}/{uuid.uuid4().hex}.png"

    try:
        await asyncio.to_thread(
            lambda: supabase.storage.from_(THUMBNAILS_BUCKET).upload(
                file_path,
                image_bytes,
                {"content-type": "image/png"},
            )
        )
    except Exception as e:
        print(f"[THUMBNAIL-SAVE] storage upload failed: {e}")
        return None

    try:
        url_res = supabase.storage.from_(THUMBNAILS_BUCKET).get_public_url(file_path)
        if isinstance(url_res, str):
            public_url = url_res
        elif isinstance(url_res, dict):
            public_url = url_res.get("publicUrl") or (url_res.get("data") or {}).get("publicUrl")
        else:
            public_url = getattr(url_res, "public_url", None)
    except Exception as e:
        print(f"[THUMBNAIL-SAVE] failed to get public url: {e}")
        return None

    if not public_url:
        return None

    try:
        existing = await asyncio.to_thread(
            lambda: supabase.table("thumbnails").select("images").eq("userId", user_id).execute()
        )
        if existing.data:
            current_images = existing.data[0].get("images") or []
            updated_images = current_images + [public_url]
            await asyncio.to_thread(
                lambda: supabase.table("thumbnails")
                .update({"images": updated_images})
                .eq("userId", user_id)
                .execute()
            )
        else:
            await asyncio.to_thread(
                lambda: supabase.table("thumbnails")
                .insert({"userId": user_id, "images": [public_url]})
                .execute()
            )
    except Exception as e:
        print(f"[THUMBNAIL-SAVE] failed to update thumbnails table: {e}")
        return public_url

    return public_url



class ThumbnailRequest(BaseModel):
    userId: str
    title: str
    description: str
    isFace: bool
    script: str = ""
    thumbnail_text: list[str] | None = None


@app.post("/generate-thumbnail")
async def generate_thumbnail_endpoint(request: ThumbnailRequest):
    await require_valid_user(request.userId)

    async with _pipeline_semaphore:
        return await _generate_thumbnail_endpoint_impl(request)


FREE_TIER_LABELS = {"free", "free_tier", "free-tier", "trial", "none", ""}


async def _get_user_tier(user_id: str) -> str:
    try:
        result = await asyncio.to_thread(
            lambda: supabase.table("user_profiles")
            .select("user_tier")
            .eq("id", user_id)
            .single()
            .execute()
        )
        raw_tier = (result.data or {}).get("user_tier")
        tier = (raw_tier or "").strip().lower()
        print(f"[CREDITS] user {user_id} user_tier='{tier or 'free (default)'}'")
        return tier
    except Exception as exc:
        print(f"[CREDITS] failed to fetch user_tier for user {user_id}, defaulting to 'free': {exc}")
        return "free"


async def _deduct_profile_credits(user_id: str, amount: int = 20):
    try:
        result = (
            supabase.table("user_profiles")
            .select("credits_remaining")
            .eq("id", user_id)
            .single()
            .execute()
        )

        current_credits = (result.data or {}).get("credits_remaining")
        if current_credits is None:
            print(f"[CREDITS] No credits_remaining found for user {user_id}, skipping deduction.")
            return

        new_credits = max(current_credits - amount, 0)

        supabase.table("user_profiles").update(
            {"credits_remaining": new_credits}
        ).eq("id", user_id).execute()

        print(f"[CREDITS] (free tier / user_profiles) Deducted {amount} credits from user {user_id}: {current_credits} -> {new_credits}")
    except Exception as exc:
        print(f"[CREDITS] Failed to deduct user_profiles credits for user {user_id}: {exc}")
        import traceback
        traceback.print_exc()


async def _deduct_subscription_credits(user_id: str, amount: int = 20):
    try:
        sub_res = (
            supabase.table("subscriptions")
            .select("id, credits, created_at")
            .eq("userId", user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        rows = sub_res.data or []
        if not rows:
            print(f"[CREDITS] No subscription rows found for user {user_id} (non-free tier) — skipping deduction.")
            return

        latest_subscription = rows[0]
        subscription_id = latest_subscription["id"]
        current_credits = latest_subscription.get("credits")
        if current_credits is None:
            print(f"[CREDITS] Latest subscription {subscription_id} for user {user_id} has no 'credits' value, skipping deduction.")
            return

        new_credits = max(current_credits - amount, 0)

        supabase.table("subscriptions").update(
            {"credits": new_credits}
        ).eq("id", subscription_id).execute()

        print(
            f"[CREDITS] (non-free tier / subscriptions, most recent by created_at) "
            f"Deducted {amount} credits from subscription {subscription_id} "
            f"(user {user_id}): {current_credits} -> {new_credits}"
        )
    except Exception as exc:
        print(f"[CREDITS] Failed to deduct subscription credits for user {user_id}: {exc}")
        import traceback
        traceback.print_exc()


async def _deduct_thumbnail_credits(user_id: str, amount: int = 20):
    tier = await _get_user_tier(user_id)

    if tier in FREE_TIER_LABELS:
        await _deduct_profile_credits(user_id, amount)
    else:
        await _deduct_subscription_credits(user_id, amount)


async def _generate_thumbnail_endpoint_impl(request: "ThumbnailRequest"):
    _start_token_tracking()

    total_start_time = time.time()
    script_text = request.script or ""

    youtube_metadata_stub = {
        "thumbnail_text": request.thumbnail_text or [],
    }

    thumbnail_result = {"image_base64": None, "prompt": None, "error": "not attempted"}

    try:
        print("[MAIN] Generating thumbnail prompt + image (standalone endpoint).")
        thumbnail_result = await generate_thumbnail_for_script(
            request, script_text, youtube_metadata_stub
        )
        thumbnail_url = None
        if thumbnail_result.get("image_base64"):
            thumbnail_url = await save_thumbnail_to_supabase(
                request.userId, thumbnail_result["image_base64"]
            )
            await _deduct_thumbnail_credits(request.userId, 20)
        thumbnail_result["public_url"] = thumbnail_url
    except Exception as exc:
        print(f"--- thumbnail generation failed: {exc} ---")
        import traceback
        traceback.print_exc()

        chosen_text = next(
            (t.strip() for t in (request.thumbnail_text or []) if isinstance(t, str) and t.strip()),
            None,
        ) or _pick_thumbnail_text(youtube_metadata_stub, request)

        thumbnail_result = {
            "image_base64": None,
            "prompt": _fallback_thumbnail_prompt(request, chosen_text),
            "error": str(exc),
            "public_url": None,
        }

    token_usage = _get_token_usage_summary()

    print(f"[/generate-thumbnail] total time: {time.time() - total_start_time:.2f}s")

    return {
        "thumbnail": {
            "prompt": thumbnail_result.get("prompt"),
            "public_url": thumbnail_result.get("public_url"),
            "error": thumbnail_result.get("error"),
        },
        "token_usage": token_usage,
    }






@app.post("/generate-script")
async def generate_script(request: ScriptRequest):
    await require_valid_user(request.userId)

    async with _pipeline_semaphore:
        return await _generate_script_impl(request)


_DEFAULT_CLASSIFICATION = {"category": "UNKNOWN", "subcategories": []}

async def generate_category_and_subcategory(
    title: str,
    description: str | None,
    script_text: str,
) -> dict:
    """
    Uses an LLM call to classify the content into exactly 1 category
    and up to 5 subcategories, based on the title, description, and script.
    """
    script_excerpt = (script_text or "")[:6000]

    classification_prompt = f"""You are a strict content classifier for YouTube-style video scripts.

Given the title, description, and script below, return exactly ONE top-level category and UP TO FIVE relevant subcategories.

Respond ONLY with valid JSON in this exact shape, no preamble, no markdown fences, no extra text:
{{"category": "string", "subcategories": ["string", "string"]}}

Rules:
- "category" must be a single, concise label (1-3 words).
- "subcategories" must be an array of 0 to 5 short, concise labels (1-3 words each).
- Do not include duplicates.
- Do not include any text outside the JSON object.

Title: {title}

Description: {description or "N/A"}

Script:
{script_excerpt}

Classify this content now. Return only the JSON object."""

    raw_text = ""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": classification_prompt}],
            stream=False,
        )

        raw_text = response.choices[0].message.content.strip()
        print(f"[CLASSIFY] raw LLM output: {raw_text!r}")

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        parsed = json.loads(cleaned)

        category = parsed.get("category")
        if not isinstance(category, str) or not category.strip():
            category = "UNKNOWN"
        else:
            category = category.strip()

        subcategories_raw = parsed.get("subcategories", [])
        if not isinstance(subcategories_raw, list):
            subcategories_raw = []

        subcategories = []
        seen_lower = set()
        for item in subcategories_raw:
            if not isinstance(item, str):
                continue
            clean_item = item.strip()
            if not clean_item:
                continue
            key = clean_item.lower()
            if key in seen_lower:
                continue
            seen_lower.add(key)
            subcategories.append(clean_item)
            if len(subcategories) >= 5:
                break

        return {"category": category, "subcategories": subcategories}

    except Exception as exc:
        print(f"--- category/subcategory classification failed: {exc!r} ---")
        print(f"--- raw_text at time of failure: {raw_text!r} ---")
        import traceback
        traceback.print_exc()
        return dict(_DEFAULT_CLASSIFICATION)


def _extract_source_links(articles: list[dict]) -> list[str]:
    links = []
    seen = set()
    for article in articles:
        url = article.get("url", "")
        if url and url not in seen:
            seen.add(url)
            links.append(url)
        if len(links) >= MAX_WEB_SOURCES:
            break
    return links



async def _generate_script_impl(request: "ScriptRequest"):
    _start_token_tracking()

    total_start_time = time.time()
    topic_text = build_topic_text(request)
    print(f"SCRIPT GENERATION: Received request for title: '{request.title}'")

    selected_template = await retrieve_best_script_template(topic_text)

    if selected_template is None:
        print("[SCRIPT] no template matched via embedding search — proceeding with an empty structure")
        selected_template = {
            "key": None, "title": None, "cluster": None, "about": None,
            "best_fit_categories": [], "human_texture_tier": None,
            "segments": [], "template_text": "", "similarity": None,
        }

    category = selected_template.get("cluster") or (
        (selected_template.get("best_fit_categories") or ["UNKNOWN"])[0]
    )
    print(f"Category (from selected template): {category}")

    segments = selected_template.get("segments", [])

    try:
        channel_profile = await get_channel_profile(request.userId)
        summary = channel_profile[0]["Summary"] if channel_profile else None
    except Exception as exc:
        print(f"--- error fetching channel profile: {exc} ---")
        summary = None

    num_docs = num_hyde_docs_for_time(request.time)
    segment_buckets = bucket_segments_by_time(segments, num_docs)
    print(f"Generating {len(segment_buckets)} HyDE doc(s) for a {request.time}-minute script")

    hyde_documents: list[str] = []
    db_results: list = []
    all_db_chunks_seen: list = []
    all_db_md5s_seen: set = set()
    new_articles: list = []
    new_videos = []
    scraped_urls = set()
    script_text = ""
    youtube_metadata = {"titles": [], "descriptions": [], "hashtags": [], "thumbnail_text": []}
    script_metrics = dict(_DEFAULT_SCRIPT_METRICS)
    sources: list[str] = []
    books: list[dict] = []
    table_name = None
    classification = dict(_DEFAULT_CLASSIFICATION)

    try:
        table_name = await select_table_for_topic(topic_text)

        hyde_documents = await asyncio.gather(
            *[
                generate_hyde_doc_for_segments(
                    request.title,
                    request.description,
                    selected_template,
                    bucket,
                    request.time,
                )
                for bucket in segment_buckets
            ]
        )
    except Exception as exc:
        print(f"--- table selection / HyDE generation failed: {exc} ---")
        import traceback
        traceback.print_exc()

    try:
        db_results_per_doc = await asyncio.gather(
            *[get_context_with_timeout(topic_text, doc, table_name=table_name) for doc in hyde_documents]
        )

        seen_md5_all = set()
        for doc_results in db_results_per_doc:
            for item in doc_results:
                key = item.get("md5") or item.get("content")
                if key and key not in seen_md5_all:
                    seen_md5_all.add(key)
                    all_db_chunks_seen.append(item)
                    md5_val = item.get("md5")
                    if md5_val:
                        all_db_md5s_seen.add(md5_val)

        db_results = []
        seen_md5_context = set()
        max_len = max((len(d) for d in db_results_per_doc), default=0)
        round_idx = 0
        while len(db_results) < MAX_SCRIPT_CONTEXT_CHUNKS and round_idx < max_len:
            for doc_results in db_results_per_doc:
                if len(db_results) >= MAX_SCRIPT_CONTEXT_CHUNKS:
                    break
                if round_idx >= len(doc_results):
                    continue
                item = doc_results[round_idx]
                key = item.get("md5") or item.get("content")
                if key and key not in seen_md5_context:
                    seen_md5_context.add(key)
                    db_results.append(item)
            round_idx += 1

        print(
            f"[MAIN] Combined DB context: {len(db_results)} unique chunk(s) "
            f"(round-robin across {len(hyde_documents)} HyDE doc(s), capped at "
            f"{MAX_SCRIPT_CONTEXT_CHUNKS}, target >= {DB_SIMILARITY_THRESHOLD} similarity). "
            f"{len(all_db_chunks_seen)} unique chunk(s) seen in total across all docs "
            f"(used for book lookups)."
        )
    except Exception as exc:
        print(f"--- DB retrieval failed: {exc} ---")
        import traceback
        traceback.print_exc()

    combined_hyde_doc = "\n\n".join(doc for doc in hyde_documents if doc) or topic_text

    try:
        print("[MAIN] Generating search keywords ONCE for script web search.")
        script_search_keywords = await _generate_search_keywords_for_script(
            request.title, request.description, selected_template, request.time
        )
    except Exception as exc:
        print(f"--- script search keyword generation failed: {exc} ---")
        script_search_keywords = [f"{request.title} latest news today", f"{request.title} 2026 update"]

    try:
        print("[MAIN] Performing web search (script-specific DDGS pipeline, reusing the keywords above).")
        new_articles = await get_ddgs_news_context_for_script(
            request.title, scraped_urls, combined_hyde_doc, keywords=script_search_keywords,
        )

        unique_source_count = _unique_url_count(new_articles)
        if unique_source_count < MAX_WEB_SOURCES:
            print(
                f"[MAIN] Only {unique_source_count} unique source URL(s) found, "
                f"running multi-round backfill to reach the compulsory {MAX_WEB_SOURCES}."
            )
            try:
                new_articles = await _backfill_sources_to_target(
                    new_articles, scraped_urls, request.title, combined_hyde_doc,
                    keywords=script_search_keywords,
                    target_count=MAX_WEB_SOURCES,
                )
            except Exception as backfill_exc:
                print(f"[MAIN] sources backfill failed: {backfill_exc}")

        print(f"[MAIN] Final unique source count: {_unique_url_count(new_articles)}/{MAX_WEB_SOURCES}")
    except Exception as exc:
        print(f"--- web search (DDGS) failed: {exc} ---")
        import traceback
        traceback.print_exc()

    try:
        # YouTube keyword generation now also takes the idea description.
        print("[MAIN] Performing YouTube search.")
        new_videos = await get_youtube_context(request.title, request.description, scraped_urls)
    except Exception as exc:
        print(f"--- YouTube search failed: {exc} ---")
        import traceback
        traceback.print_exc()

    try:
        target_word_count = target_word_count_for_time(request.time)
        script_text = await generate_script_from_context(
            request, selected_template, db_results, new_articles, target_word_count
        )
    except Exception as exc:
        print(f"--- script generation failed: {exc} ---")
        import traceback
        traceback.print_exc()

    try:
        books = await get_books_for_chunks(
            all_db_chunks_seen, topic_text=topic_text, script_text=script_text
        )

        if len(books) < MAX_BOOKS:
            print(
                f"[MAIN] Only {len(books)}/{MAX_BOOKS} real book(s) found from the initial "
                f"chunk pool — widening DB search to try to reach {MAX_BOOKS}."
            )
            books = await _backfill_books_to_target(
                books,
                all_db_md5s_seen,
                topic_text,
                combined_hyde_doc,
                table_name,
                target_count=MAX_BOOKS,
            )
    except Exception as exc:
        print(f"--- MySQL book lookup failed: {exc} ---")
        import traceback
        traceback.print_exc()
        books = []

    try:
        print("[MAIN] Generating YouTube SEO metadata.")
        youtube_metadata = await generate_youtube_seo_metadata(request, script_text, new_videos)
    except Exception as exc:
        print(f"--- YouTube SEO metadata generation failed: {exc} ---")
        import traceback
        traceback.print_exc()
        youtube_metadata = _build_fallback_youtube_metadata(request)

    try:
        print("[MAIN] Generating script content metrics.")
        script_metrics = await generate_script_metrics(script_text, topic_text=topic_text)
    except Exception as exc:
        print(f"--- script metrics generation failed: {exc} ---")
        import traceback
        traceback.print_exc()

    try:
        print("[MAIN] Generating category and subcategory classification.")
        classification = await generate_category_and_subcategory(
            request.title, request.description, script_text
        )
        print(
            f"[MAIN] Classification -> category: {classification.get('category')}, "
            f"subcategories: {classification.get('subcategories')}"
        )
    except Exception as exc:
        print(f"--- category/subcategory classification failed: {exc} ---")
        import traceback
        traceback.print_exc()
        classification = dict(_DEFAULT_CLASSIFICATION)

    sources = _extract_source_links(new_articles)

    total_words = _word_count(script_text) if script_text else 0
    video_length = round(total_words / WORDS_PER_MINUTE, 2) if total_words else 0

    structure = _build_structure_response(selected_template)

    token_usage = _get_token_usage_summary()

    print(f"Total time so far: {time.time() - total_start_time:.2f}s")
    print(
        f"[TOKENS] /generate-script total — input: {token_usage['total_input_tokens']}, "
        f"output: {token_usage['total_output_tokens']}, total: {token_usage['total_tokens']} "
        f"across {len(token_usage['calls'])} LLM call(s)"
    )

    return {
        "script": script_text,
        "youtube_metadata": {
            "titles": youtube_metadata.get("titles", []),
            "descriptions": youtube_metadata.get("descriptions", []),
            "hashtags": youtube_metadata.get("hashtags", []),
            "thumbnail_text": youtube_metadata.get("thumbnail_text", []),
        },
        "metrics": {
            "totalWords": total_words,
            "videoLength": video_length,
            "emotionalDepth": script_metrics.get("emotionalDepth", 0),
            "generalExamples": script_metrics.get("generalExamples", 0),
            "proverbs_count": script_metrics.get("proverbs_count", 0),
            "historical_facts": script_metrics.get("historicalExamples", 0),
            "researchFacts": script_metrics.get("researchFacts", 0),
        },
        "sources": sources,
        "books": books,
        "structure": structure,
        "category": classification.get("category", "UNKNOWN"),
        "subcategories": classification.get("subcategories", []),
        "token_usage": token_usage,
    }






from typing import Optional

from pydantic import Field

_http_session = requests.Session()
_http_adapter = requests.adapters.HTTPAdapter(
    pool_connections=20, pool_maxsize=20, max_retries=1
)
_http_session.mount("https://", _http_adapter)
_http_session.mount("http://", _http_adapter)


PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"


class PexelsVideoSearchRequest(BaseModel):
    userId : str
    query: str = Field(..., description="Search term, e.g. 'ocean waves'")
    per_page: int = Field(50, ge=1, le=80, description="Results per page (max 80)")
    page: int = Field(1, ge=1, description="Page number")
    orientation: Optional[str] = Field(
        None, description="landscape | portrait | square (optional)"
    )
    size: Optional[str] = Field(
        None, description="large | medium | small (optional, min video resolution)"
    )


def _pexels_search_videos_sync(
    query: str,
    per_page: int,
    page: int,
    orientation: Optional[str],
    size: Optional[str],
) -> dict:
    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY not set")

    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": query,
        "per_page": per_page,
        "page": page,
    }
    if orientation:
        params["orientation"] = orientation
    if size:
        params["size"] = size

    resp = _http_session.get(
        PEXELS_VIDEO_SEARCH_URL, headers=headers, params=params, timeout=15
    )
    resp.raise_for_status()
    return resp.json()


def _extract_video_files(video: dict) -> list[dict]:
    """Pick out the useful video_files entries (quality/resolution/link)."""
    files = []
    for vf in video.get("video_files", []):
        files.append({
            "quality": vf.get("quality"),
            "width": vf.get("width"),
            "height": vf.get("height"),
            "file_type": vf.get("file_type"),
            "link": vf.get("link"),
        })
    return files


def _format_video_result(video: dict) -> dict:
    return {
        "id": video.get("id"),
        "url": video.get("url"),
        "width": video.get("width"),
        "height": video.get("height"),
        "duration": video.get("duration"),
        "thumbnail": video.get("image"),
        "user": {
            "name": (video.get("user") or {}).get("name"),
            "url": (video.get("user") or {}).get("url"),
        },
        "video_files": _extract_video_files(video),
    }


@app.post("/search-pexels-videos")
async def search_pexels_videos(request: PexelsVideoSearchRequest):
    await require_valid_user(request.userId)

    if not PEXELS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="PEXELS_API_KEY is not configured on the server.",
        )

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must be a non-empty string")

    try:
        data = await asyncio.to_thread(
            _pexels_search_videos_sync,
            query,
            request.per_page,
            request.page,
            request.orientation,
            request.size,
        )
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 502
        detail = e.response.text[:300] if e.response is not None else str(e)
        raise HTTPException(status_code=status, detail=f"Pexels API error: {detail}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pexels video search failed: {e}")

    videos = [_format_video_result(v) for v in (data.get("videos") or [])]

    return {
        "query": query,
        "page": data.get("page", request.page),
        "per_page": data.get("per_page", request.per_page),
        "total_results": data.get("total_results", 0),
        "videos": videos,
    }







































































































































































import datetime
import string
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)


def generate_invoice_number():
    random_part = ''.join(random.choices(string.digits, k=6))
    year = datetime.datetime.now().year
    return f"INV-{year}-{random_part}"


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

    elements.append(Paragraph("BILL TO", section_header_style))
    elements.append(Spacer(1, 1*mm))
    elements.append(Paragraph(f"<b>{customer_name}</b>", body_bold_style))
    elements.append(Paragraph(customer_address, body_style))
    elements.append(Paragraph(f"Phone: {customer_phone}", body_style))
    elements.append(Spacer(1, 7*mm))

    grand_total = round(amount, 2)
    base_price  = round(amount / 1.18, 2)
    gst_amount  = round(grand_total - base_price, 2)

    CW = [W*0.34, W*0.12, W*0.18, W*0.12, W*0.24]

    WHITE      = colors.white
    DARK       = colors.HexColor('#1a1a2e')
    LIGHT_GRAY = colors.HexColor('#f0f0f0')
    MID_GRAY   = colors.HexColor('#e8e8e8')
    TEXT_DARK  = colors.HexColor('#1a1a2e')

    def lp(bold=False):
        return ParagraphStyle('_l', parent=styles['Normal'], fontSize=10,
                              fontName='Helvetica-Bold' if bold else 'Helvetica',
                              textColor=TEXT_DARK, alignment=TA_LEFT)

    def cp(bold=False, tc=TEXT_DARK):
        return ParagraphStyle('_c', parent=styles['Normal'], fontSize=10,
                              fontName='Helvetica-Bold' if bold else 'Helvetica',
                              textColor=tc, alignment=TA_CENTER)

    def rp(bold=False, tc=TEXT_DARK):
        return ParagraphStyle('_r', parent=styles['Normal'], fontSize=10,
                              fontName='Helvetica-Bold' if bold else 'Helvetica',
                              textColor=tc, alignment=TA_RIGHT)

    table_data = [
        ['ITEM', 'PLAN', 'RATE', 'QTY', 'TOTAL'],
        [item_name, plan.title(), f"Rs. {base_price:.2f}", "1", f"Rs. {base_price:.2f}"],
        [
            Paragraph("", lp()),
            "",
            "",
            Paragraph("GST (18%)", rp(False, colors.HexColor('#555555'))),
            Paragraph(f"Rs. {gst_amount:.2f}", rp(False, TEXT_DARK)),
        ],
        [
            Paragraph("GRAND TOTAL", lp(True)),
            "",
            "",
            "",
            Paragraph(f"Rs. {grand_total:.2f}", rp(True, TEXT_DARK)),
        ],
    ]

    ts = TableStyle([
        ('BACKGROUND',    (0,0),(-1,0), LIGHT_GRAY),
        ('TEXTCOLOR',     (0,0),(-1,0), TEXT_DARK),
        ('FONTNAME',      (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,0), 9),
        ('TOPPADDING',    (0,0),(-1,0), 9),
        ('BOTTOMPADDING', (0,0),(-1,0), 9),
        ('ALIGN',         (0,0),(0,0),  'LEFT'),
        ('ALIGN',         (1,0),(1,0),  'CENTER'),
        ('ALIGN',         (2,0),(2,0),  'RIGHT'),
        ('ALIGN',         (3,0),(3,0),  'CENTER'),
        ('ALIGN',         (4,0),(4,0),  'RIGHT'),
        ('BACKGROUND',    (0,1),(-1,1), colors.HexColor('#f8f8fb')),
        ('FONTNAME',      (0,1),(-1,1), 'Helvetica'),
        ('FONTSIZE',      (0,1),(-1,1), 10),
        ('TOPPADDING',    (0,1),(-1,1), 10),
        ('BOTTOMPADDING', (0,1),(-1,1), 10),
        ('ALIGN',         (0,1),(0,1),  'LEFT'),
        ('ALIGN',         (1,1),(1,1),  'CENTER'),
        ('ALIGN',         (2,1),(-1,1), 'RIGHT'),
        ('ALIGN',         (3,1),(3,1),  'CENTER'),
        ('GRID',          (0,0),(-1,1), 0.5, colors.HexColor('#dddddd')),
        ('SPAN',          (0,2),(2,2)),
        ('BACKGROUND',    (0,2),(-1,2), LIGHT_GRAY),
        ('TOPPADDING',    (0,2),(-1,2), 8),
        ('BOTTOMPADDING', (0,2),(-1,2), 8),
        ('LINEBELOW',     (0,2),(-1,2), 0.5, colors.HexColor('#dddddd')),
        ('LINEABOVE',     (0,2),(-1,2), 0.5, colors.HexColor('#dddddd')),
        ('VALIGN',        (0,2),(-1,2), 'MIDDLE'),
        ('SPAN',          (0,3),(3,3)),
        ('BACKGROUND',    (0,3),(-1,3), MID_GRAY),
        ('TOPPADDING',    (0,3),(-1,3), 10),
        ('BOTTOMPADDING', (0,3),(-1,3), 10),
        ('LINEBELOW',     (0,3),(-1,3), 1.0, colors.HexColor('#cccccc')),
        ('VALIGN',        (0,3),(-1,3), 'MIDDLE'),
        ('LEFTPADDING',   (0,0),(-1,-1), 8),
        ('RIGHTPADDING',  (0,0),(-1,-1), 8),
    ])

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
    if request_data.target_tier not in ['plus', 'pro']:
        raise HTTPException(status_code=400, detail="Invalid target tier.")

    order_data = {
        "amount": int(float(amount) * 100),
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
                'plus': {'credits': 100, 'validity_days': 30},
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
                    .insert(subscription_row)
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
                            due_date=validity_date,
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

        elif event_type == 'payment.captured':
            print("Ignoring 'payment.captured' (handled by 'order.paid').")

        elif event_type == 'payment.failed':
            payment_entity    = event_data['payload']['payment']['entity']
            failed_order_id   = payment_entity.get('order_id', 'unknown')
            failed_payment_id = payment_entity.get('id', 'unknown')
            error_desc        = payment_entity.get('error_description', 'No description')

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

        else:
            print(f"Ignoring unhandled event: {event_type}")

        return {"invoice_url": invoice_url}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    except Exception as e:
        print(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


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

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: supabase.table('user_channel_memory').upsert(
            chunks,
            on_conflict="chunk_id"
        ).execute()
    )

    asyncio.create_task(run_intelligence_for_user(userId))

    return {"message": "Uploaded and processed"}