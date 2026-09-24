from fastapi import Depends, HTTPException, Request, Header,UploadFile, File,Form
from fastapi import FastAPI
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
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
from fishaudio import FishAudio
import whisperx
import tempfile
import uuid


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

whisper_model = whisperx.load_model(
    "small",
    device="cpu",
    compute_type="int8"
)

align_model, align_metadata = whisperx.load_align_model(
    language_code="en",
    device="cpu"
)


def _get_st_model():
    global _bge_model
    if _bge_model is None:
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(2)  
        print("[MODEL] Loading BAAI/bge-m3")
        _bge_model = SentenceTransformer("BAAI/bge-m3")
        print("[MODEL] BAAI/bge-m3 loaded")
    return _bge_model


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
    worker_task = asyncio.create_task(_render_queue_worker())
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    print("[Lifespan] Shutting down.")


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "https://www.storio.tech",
    "https://api.storio.tech",
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



from typing import Literal

class CreateOrderRequest(BaseModel):
    amount: float = Field(gt=0)
    currency: Literal["INR", "USD"] = "USD"
    billing_cycle: Literal["monthly", "annual"] = "monthly"
    receipt: str | None = None
    target_tier: Literal["plus", "pro"]


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class GenerateIdeasRequest(BaseModel):
    userId: str
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
import json
import time
import math
import base64
import hashlib
import contextvars
import concurrent.futures
from urllib.parse import urlparse


import requests
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from fastapi import HTTPException
from openai import OpenAI
import trafilatura



SCRIPTS_UNIVERSAL_TABLE = "scripts_universal"
IDEAS_HYDE_DOC_COUNT = 5


SUPPORTED_LANGUAGES = {
    "afrikaans": "af",
    "albanian": "sq",
    "amharic": "am",
    "arabic": "ar",
    "armenian": "hy",
    "assamese": "as",
    "azerbaijani": "az",
    "basque": "eu",
    "belarusian": "be",
    "bengali": "bn",
    "bosnian": "bs",
    "breton": "br",
    "bulgarian": "bg",
    "burmese": "my",
    "catalan": "ca",
    "chinese": "zh",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "faroese": "fo",
    "finnish": "fi",
    "french": "fr",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "haitian creole": "ht",
    "hebrew": "iw",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "kannada": "kn",
    "kazakh": "kk",
    "khmer": "km",
    "korean": "ko",
    "latin": "la",
    "latvian": "lv",
    "lithuanian": "lt",
    "macedonian": "mk",
    "malay": "ms",
    "malayalam": "ml",
    "maori": "mi",
    "marathi": "mr",
    "mongolian": "mn",
    "nepali": "ne",
    "norwegian": "no",
    "norwegian nynorsk": "nn",
    "pashto": "ps",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "sanskrit": "sa",
    "serbian": "sr",
    "shona": "sn",
    "sindhi": "sd",
    "sinhala": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "spanish": "es",
    "swahili": "sw",
    "swedish": "sv",
    "filipino": "tl",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
    "tibetan": "bo",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "vietnamese": "vi",
    "welsh": "cy",
    "yiddish": "yi",
    "yoruba": "yo",
}



from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

QDRANT_URL = "http://37.27.101.243:6333"

QDRANT_DENSE_VECTOR_NAME = "dense"
QDRANT_SPARSE_VECTOR_NAME = "sparse"

_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


RAG_CATEGORIES = [
    "Anthropology", "Biography", "Business", "Economics", "Entrepreneurship",
    "Finance", "Geography", "Health", "History", "Knowledge", "Law",
    "Personal_Development", "Philosophy", "Politics", "Psychology",
    "Religion", "Science", "SocialScience", "Sports", "Technology",
]


def _qdrant_collection_name(category: str) -> str:
    return f"RAG_{category}_supabase"


def _supabase_content_table_name(category: str) -> str:
    return f"RAG_{category}"


 
DEFAULT_LANGUAGE = "English"
 
TRANSLATE_CHUNK_MAX_CHARS = 4000


_HASHTAG_WORD_RE = re.compile(r"[^\s#]+", re.UNICODE)
 

def _keyword_to_hashtag(keyword: str) -> str:
    words = _HASHTAG_WORD_RE.findall(keyword or "")
    if not words:
        return ""
    if all(w.isascii() for w in words):
        first, rest = words[0].lower(), words[1:]
        camel = first + "".join(w.capitalize() for w in rest)
        return f"#{camel}" if camel else ""
    joined = "".join(words)
    return f"#{joined}" if joined else ""    



def _normalize_language(language: str | None) -> str:
    if not language or not language.strip():
        return DEFAULT_LANGUAGE
    key = language.strip().lower()
    if key not in SUPPORTED_LANGUAGES:
        print(f"[LANG] unrecognized language '{language}', defaulting to English")
        return DEFAULT_LANGUAGE
    return "Odia" if key == "odia" else language.strip().title()
 



_HYDE_DOC_SPLIT_RE = re.compile(r"Document\s*\d+\s*:\s*", re.IGNORECASE)


def _parse_hyde_documents(raw: str, expected_count: int) -> list[str]:
    """Splits a single completion containing 'Document 1: ...  Document 2: ...'
    into individual document strings, in order."""
    if not raw:
        return []
    parts = [p.strip() for p in _HYDE_DOC_SPLIT_RE.split(raw) if p.strip()]
    return parts[:expected_count]


async def generate_ideas_hyde_documents(topic: str, num_docs: int = IDEAS_HYDE_DOC_COUNT) -> list[str]:
    hyde_prompt = f"""
# HYPOTHETICAL DOCUMENT GENERATOR (HDG v2)

## ROLE

You are a **Hypothetical Document Generator** for a Retrieval-Augmented Generation (RAG) system.

Given a user query, generate **exactly {num_docs} hypothetical documents** that maximize semantic similarity with authoritative source material likely to exist in a large corpus of books.

These documents are **retrieval anchors**, not answers. They will be embedded and used to retrieve the most relevant passages from a vector database.

---

## INPUT

Topic: "{topic}"

---

## TASK

First infer the query's primary knowledge domain(s). Then generate **{num_docs} complementary hypothetical documents**, each representing a distinct perspective naturally suited to the query (e.g. historical context, conceptual foundations, mechanisms, stakeholders, controversies, future implications — choose perspectives dynamically based on the topic).

Each document must:

* focus on a unique semantic perspective, with no repetition across documents
* naturally include relevant domain terminology, synonyms, and related concepts
* naturally reference important entities strongly implied by the query
* emphasize conceptual relationships instead of isolated keywords
* avoid conversational language and avoid directly answering the user's question

## FACTUAL DISCIPLINE

Do not invent specific dates, statistics, quotations, study results, citations, named publications, financial figures, researcher names, or organizations unless explicitly present in the query or universally inseparable from the topic. When uncertain, describe concepts generically.

## STYLE

Objective, information-dense academic style, like genuine reference material. Avoid opinions, recommendations, storytelling, introductions, conclusions, speculative language.

## LENGTH

Each document must be **35-50 words** — short and strict. Do not exceed this.

## OUTPUT

Generate exactly {num_docs} documents, in this exact format and nothing else:

Document 1: <text>

Document 2: <text>

Document 3: <text>

Document 4: <text>

Document 5: <text>

    """.strip()

    async def _call(max_tokens: int):
        completion = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[{"role": "user", "content": hyde_prompt}],
                max_completion_tokens=max_tokens,
                stream=False,
                temperature=0.25,
                top_p=0.9,
            )
        )
        _record_token_usage(f"generate-ideas HYDE (batch, max_tokens={max_tokens})", completion)
        print("=============================================")
        print(completion.choices[0].message.content)
        print("=============================================")
        return (completion.choices[0].message.content or "").strip()

    first_pass_tokens = max(400, num_docs * 100)
    raw = await _call(first_pass_tokens)

    docs = _parse_hyde_documents(raw, num_docs)

    if len(docs) < num_docs:
        print(
            f"[IDEAS-HYDE] batch call returned {len(docs)}/{num_docs} parsed document(s) "
            f"— retrying with more headroom"
        )
        try:
            raw_retry = await _call(max(first_pass_tokens * 2, 1200))
            retry_docs = _parse_hyde_documents(raw_retry, num_docs)
            if len(retry_docs) > len(docs):
                docs = retry_docs
        except Exception as exc:
            print(f"[IDEAS-HYDE] batch retry failed: {exc}")

    capped_docs = [_cap_hyde_doc_tokens(d) for d in docs]
    while len(capped_docs) < num_docs:
        print(f"[IDEAS-HYDE] missing document #{len(capped_docs) + 1} after parse/retry, falling back to topic")
        capped_docs.append(topic)

    return capped_docs

 


_BLOCKED_SOURCE_DOMAINS = {
   "aol.com", "flipboard.com",
    "pinterest.com",
    "thefreedictionary.com", "dictionary.com",
    "thesaurus.com", "vocabulary.com", "urbandictionary.com","satta-king-fast.com"
}

_BLOCKED_DOMAIN_SUBSTRINGS = (
    "dictionary", "thesaurus", "wiktionary", "definition",
)


def _is_blocked_source_url(url: str) -> bool:
    if not url:
        return True
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return True
    if netloc.startswith("www."):
        netloc = netloc[4:]

    for blocked in _BLOCKED_SOURCE_DOMAINS:
        if netloc == blocked or netloc.endswith("." + blocked):
            return True

    if any(sub in netloc for sub in _BLOCKED_DOMAIN_SUBSTRINGS):
        return True

    if "search.yahoo" in netloc or netloc.startswith("r.") or "/RU=" in url:
        return True

    return False



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

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0, max_retries=1)

GPT_IMAGE_MODEL = os.getenv("GPT_IMAGE_MODEL", "gpt-image-2")
GPT_IMAGE_SIZE = os.getenv("GPT_IMAGE_SIZE", "1536x1024")
GPT_IMAGE_QUALITY = os.getenv("GPT_IMAGE_QUALITY", "high")


_ENCODE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("ENCODE_EXECUTOR_WORKERS", "1")),
    thread_name_prefix="encode",
)

_IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("IO_EXECUTOR_WORKERS", "8")),
    thread_name_prefix="io",
)

async def _run_io(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_IO_EXECUTOR, lambda: fn(*args, **kwargs))

_http_session = requests.Session()
_http_adapter = requests.adapters.HTTPAdapter(
    pool_connections=20, pool_maxsize=20, max_retries=1
)
_http_session.mount("https://", _http_adapter)
_http_session.mount("http://", _http_adapter)

_MAX_CONCURRENT_PIPELINES = int(os.getenv("MAX_CONCURRENT_PIPELINES", "2"))
_pipeline_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_PIPELINES)

_MAX_CONCURRENT_ENCODES = int(os.getenv("MAX_CONCURRENT_ENCODES", "4"))
_encode_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_ENCODES)
_MAX_CONCURRENT_SCRAPES = int(os.getenv("MAX_CONCURRENT_SCRAPES", "6")) 
_scrape_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SCRAPES)

OPENAI_CALL_TIMEOUT = float(os.getenv("OPENAI_CALL_TIMEOUT", "45"))


async def _run_encode(fn):
    async with _encode_semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_ENCODE_EXECUTOR, fn)


async def _openai_create_with_timeout(call_fn, timeout: float = OPENAI_CALL_TIMEOUT):
    return await asyncio.wait_for(_run_io(call_fn), timeout=timeout)

USER_PROFILES_TABLE = "user_profiles"
USER_PROFILES_ID_COLUMN = "id"


async def _user_exists_in_profiles(user_id: str | None) -> bool:
    if not user_id or not str(user_id).strip():
        return False

    user_id = str(user_id).strip()

    try:
        result = await _run_io(
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

IDEAS_DB_CHUNKS_TO_LLM = IDEAS_HYDE_DOC_COUNT * 2      
IDEAS_WEB_SOURCES_TO_LLM = IDEAS_HYDE_DOC_COUNT * 2   
IDEAS_SEARCH_KEYWORD_COUNT = 10
SCRIPT_SEARCH_KEYWORD_COUNT = 10

IDEAS_RAG_POOL_PER_DOC = 10
IDEAS_WEB_POOL_PER_DOC = 10

IDEAS_TOP_K_PER_DOC = 2


MAX_YOUTUBE_SOURCES = 7
MAX_DB_CHUNKS = 7
MAX_SCRIPT_CONTEXT_CHUNKS = 20

MAX_BOOKS = 7

WEB_CONTENT_SIMILARITY_THRESHOLD = 0.4
DB_SIMILARITY_THRESHOLD = 0.5

WORDS_PER_MINUTE = 140


BOOKS_TABLE_NAME = "english_books"
THUMBNAILS_BUCKET = "generated-thumbnails"
FETCH_TIMEOUT_SECONDS = float(os.getenv("FETCH_TIMEOUT_SECONDS", "6"))  

def to_pgvector(embedding) -> str:
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"



_bge_model = None

from typing import List, Dict, Any


# =============================================================================
# _LogBuffer — collects log lines for one concurrently-running task and
# flushes them as a single contiguous block when the task finishes. This is
# purely cosmetic: it stops parallel tasks (Stage 2 / Stage 3 / Stage 4)
# from interleaving their print() output on stdout. No timing, concurrency,
# or retrieval logic is affected by this — lines are just buffered and
# printed together instead of streamed live.
# =============================================================================
class _LogBuffer:
    def __init__(self, label: str):
        self.label = label
        self.lines: list[str] = []

    def log(self, msg: str):
        self.lines.append(msg)

    def flush(self):
        print(f"\n{'=' * 90}")
        print(f"[{self.label}] BEGIN")
        print("=" * 90)
        print("\n".join(self.lines))
        print(f"[{self.label}] END")
        print("=" * 90)


class Idea(BaseModel):
    title: str
    description: str


class SaveIdeasRequest(BaseModel):
    userId: str
    topic: str
    topic_summary: str
    sources: List[Dict[str, Any]] = []
    books: List[Dict[str, Any]] = []
    ideas: List[Idea]

@app.post("/save-ideas")
async def save_ideas(data: SaveIdeasRequest):
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
        "topic_summary": data.topic_summary,     
        "ideas": ideas_payload,
        "sources": data.sources,                
        "books": data.books,                    
        "topic_embeddings": to_pgvector(topic_embedding),
        "summary_embeddings": to_pgvector(summary_embedding),
    }

    try:
        result = supabase.table("saved_ideas").insert(row).execute()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Supabase insert failed: {e}"
        )

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

KEYWORD_GEN_PROMPT_TEMPLATE = """
You are a **Search Query Expansion Engine** for automated web crawling and knowledge retrieval.

Your task is to convert a short user topic into **15 high-quality search engine keyword combinations** that maximize information retrieval from Google, Bing, academic search engines, news websites, government portals, company websites, research repositories, statistical databases, think tanks, digital libraries, and other authoritative sources.

The output will be used by an automated crawler to collect information for creating high-quality YouTube documentaries, educational videos, business stories, biographies, explainers, analytical reports, and research-driven content.

---

## INPUT

[TOPIC]: {topic}

---

## OBJECTIVE

Generate **exactly 15 expanded search phrases** that comprehensively explore the topic while adapting intelligently to its subject matter.

Before generating queries, infer the topic's domain(s) and prioritize the search dimensions that naturally matter most.

A topic may belong to multiple domains.

Possible domains include (not limited to):

* History
* Politics
* Business
* Economics
* Finance
* Technology
* Science
* Psychology
* Neuroscience
* Philosophy
* Religion
* Health
* Biography
* Law
* Sociology
* Anthropology
* Cultural Studies
* Geography
* Travel
* Astronomy
* Sports
* Communication
* Film & Theatre
* Entrepreneurship
* Personal Development
* Self Help

---

## DOMAIN-AWARE RETRIEVAL

Adapt the search strategy to the inferred domain instead of using identical patterns for every topic.

Examples of domain emphasis:

* **Business / Economics / Finance:** revenue, profit, valuation, market share, business model, competition, industry trends, financial statements, annual reports, earnings, investors, strategy, growth, acquisitions, statistics.
* **History:** origins, chronology, timeline, causes, consequences, historical documents, historians, civilizations, primary sources, archaeological evidence.
* **Politics / Law:** legislation, governance, elections, policies, constitutional aspects, court judgments, government reports, international relations, public opinion.
* **Psychology / Neuroscience:** theories, experiments, researchers, journals, clinical evidence, behavioural studies, cognitive science, neuroscience.
* **Science / Health:** peer-reviewed research, journals, experiments, datasets, systematic reviews, clinical trials, consensus, discoveries.
* **Technology:** architecture, standards, patents, technical documentation, benchmarks, research papers, companies, adoption, innovations, security.
* **Religion / Philosophy:** scriptures, philosophical schools, interpretations, scholars, ethics, debates, historical context, influence, criticisms.
* **Biography:** early life, career timeline, achievements, failures, interviews, speeches, writings, legacy, turning points.
* **Social Sciences:** demographics, field studies, ethnography, social movements, cultural evolution, societal impact.
* **Sports:** performance statistics, tournaments, rankings, governing bodies, analytics, controversies.
* **Film & Theatre:** production history, creators, critics, awards, box office, audience reception, cultural impact.
* **Personal Development / Entrepreneurship / Communication:** frameworks, evidence-based methods, experts, books, practical applications, measurable outcomes.

Use similar reasoning for any other domain not explicitly listed.

---

## RETRIEVAL DIMENSIONS

Choose the dimensions that best fit the topic rather than forcing all of them.

Possible dimensions include:

* latest developments
* history
* timeline
* origins
* evolution
* root causes
* major events
* stakeholders
* governments
* organizations
* companies
* institutions
* researchers
* influential people
* statistics
* datasets
* surveys
* reports
* white papers
* academic research
* journals
* books
* primary sources
* interviews
* speeches
* expert opinions
* controversies
* criticisms
* myths vs facts
* challenges
* opportunities
* future trends
* predictions
* comparative analysis
* regional perspectives
* global perspectives

Prioritize the dimensions that maximize useful information for the inferred domain.

---

## ENTITY EXPANSION

Whenever possible, naturally expand the search phrases using relevant entities such as:

* people
* companies
* governments
* organizations
* institutions
* products
* technologies
* legislation
* historical events
* locations
* books
* research laboratories
* scientific concepts

Infer these entities whenever they are reasonably obvious from the topic.

---

## SEARCH QUALITY RULES

Each search phrase should:

* naturally include the topic's core subject or entities
* retrieve a different aspect of the topic
* maximize diversity of retrieved information
* balance authoritative, academic, governmental, industrial, historical, and recent sources where applicable
* retrieve quantitative information when relevant
* retrieve qualitative analysis when relevant
* favor primary sources whenever available
* avoid duplicate retrieval intent

---

## OUTPUT REQUIREMENTS

* Generate **exactly 15** search keyword combinations.
* Do **NOT** output the original topic by itself.
* Generate **search phrases**, not sentences.
* Every phrase must represent a unique retrieval intent.
* Each phrase must contain **4–10 words**.
* Number each result from **1–15**.
* Return **ONLY** the numbered search keyword combinations.
* Do **NOT** include explanations, headings, reasoning, or additional text.


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


QUERY_SPARSE_TOP_K = 100
def _sparse_row_to_dict(sparse_row, top_k: int = QUERY_SPARSE_TOP_K) -> dict:
    coo = sparse_row.tocoo()
    indices = coo.col
    values = coo.data

    if len(values) > top_k:
        keep = np.argpartition(-np.abs(values), top_k - 1)[:top_k]
    else:
        keep = np.arange(len(values))

    return {str(int(indices[i])): float(values[i]) for i in keep}

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
    categories_block = "\n".join(f"- {c}" for c in RAG_CATEGORIES)
    table_selector_prompt = f"""
    You are a routing assistant. Given a topic, select the single most relevant
    category from the list below that would contain source documents for that topic.

    Available categories:
    {categories_block}

    Topic: "{topic}"

    Respond with ONLY the exact category name from the list above, nothing else.
    """

    res = await _openai_create_with_timeout(
        lambda: openai_client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": table_selector_prompt}],
            stream=False,
        )
    )
    _record_token_usage("select_table_for_topic", res)
    category = res.choices[0].message.content.strip("`'\" \n")

    if category not in RAG_CATEGORIES:
        print(f"[QDRANT] table selector returned unexpected value '{category}', defaulting to {RAG_CATEGORIES[0]}")
        category = RAG_CATEGORIES[0]
    else:
        print(f"[QDRANT] Selected category: {category}")

    return category

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
    match_count: int = 30,
):
    if table_name is None:
        table_name = await select_table_for_topic(topic)

    category = table_name
    collection_name = _qdrant_collection_name(category)
    supabase_table = _supabase_content_table_name(category)

    embedding_source = hyde_doc if hyde_doc else topic

    model = _get_st_model()
    dense_embedding = await _run_encode(
        lambda: model.encode(
            embedding_source,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()
    )

    vectorizer = get_sparse_vectorizer()
    sparse_row = await asyncio.to_thread(lambda: vectorizer.transform([embedding_source]))
    query_sparse_dict = _sparse_row_to_dict(sparse_row)
    sparse_indices = [int(k) for k in query_sparse_dict.keys()]
    sparse_values = [float(v) for v in query_sparse_dict.values()]

    client = get_qdrant_client()

    try:
        qdrant_result = await _run_io(
            lambda: client.query_points(
                collection_name=collection_name,
                prefetch=[
                    qdrant_models.Prefetch(
                        query=dense_embedding,
                        using=QDRANT_DENSE_VECTOR_NAME,
                        limit=match_count,
                    ),
                    qdrant_models.Prefetch(
                        query=qdrant_models.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                        using=QDRANT_SPARSE_VECTOR_NAME,
                        limit=match_count,
                    ),
                ],
                query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
                limit=match_count,
                with_payload=True,
            )
        )
        points = qdrant_result.points
    except Exception as e:
        print(f"[QDRANT] hybrid query FAILED against '{collection_name}': {e}")
        return []

    if not points:
        return []

    chunk_ids = []
    score_by_chunk_id: dict = {}
    for point in points:
        payload = point.payload or {}
        chunk_id = payload.get("chunk_id")
        if chunk_id is None:
            continue
        chunk_ids.append(chunk_id)
        score_by_chunk_id[chunk_id] = point.score

    if not chunk_ids:
        return []

    try:
        supabase_rows = await _run_io(
            lambda: supabase.table(supabase_table)
            .select("chunk_id, md5, content")
            .in_("chunk_id", chunk_ids)
            .execute()
        )
        rows = supabase_rows.data or []
    except Exception as e:
        print(f"[QDRANT] Supabase content lookup against '{supabase_table}' failed: {e}")
        return []

    matches = []
    for row in rows:
        chunk_id = row.get("chunk_id")
        content = row.get("content")
        md5 = row.get("md5")
        if not content:
            continue
        score = score_by_chunk_id.get(chunk_id, 0.0)
        matches.append({
            "content": content,
            "md5": md5,
            "chunk_id": chunk_id,
            "dense_score": score,
            "combined_score": score,
        })

    matches.sort(key=lambda r: r["combined_score"], reverse=True)
    return matches[:final_k]


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
                temperature=0.45,  
                top_p=0.9,

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
    keywords = keywords[:IDEAS_SEARCH_KEYWORD_COUNT]

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


from trafilatura.settings import use_config

_TRAFILATURA_CONFIG = use_config()
_TRAFILATURA_CONFIG.set("DEFAULT", "DOWNLOAD_TIMEOUT", "4")   # was 8
_TRAFILATURA_CONFIG.set("DEFAULT", "MAX_REDIRECTS", "2")      # was 3, fewer hops = faster failure on redirect chains

def _fetch_full_article_text(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url, config=_TRAFILATURA_CONFIG)
        if not downloaded:
            return ""
        text_value = trafilatura.extract(downloaded) or ""
        return text_value.strip()
    except Exception as e:
        print(f"[FETCH] failed to extract {url}: {e}")
        return ""


PER_KEYWORD_SCRAPE_COUNT = 5

_SCRAPE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("SCRAPE_EXECUTOR_WORKERS", "8")),
    thread_name_prefix="scrape",
)


async def _run_scrape(fn, *args, **kwargs):
    """Run a blocking network call (DDGS search, trafilatura fetch,
    scrapetube search) gated by a semaphore to cap total concurrent
    outbound connections."""
    async with _scrape_semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_SCRAPE_EXECUTOR, lambda: fn(*args, **kwargs))


async def _fetch_full_article_text_with_timeout(url: str, timeout: float = FETCH_TIMEOUT_SECONDS) -> str:
    try:
        return await asyncio.wait_for(
            _run_scrape(_fetch_full_article_text, url),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
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
                temperature=0.35,  
                top_p=0.9,
                
            )
        )
        _record_token_usage("_generate_youtube_search_keywords", completion)
        raw = completion.choices[0].message.content.strip()
        keywords = _parse_keyword_lines(raw)
        return keywords or [topic]
    except Exception as exc:
        print(f"--- YouTube keyword generation failed: {exc} ---")
        return [topic]


async def build_shared_web_pool(
    keywords: list[str],
    scraped_urls: set,
    per_keyword_results: int = PER_KEYWORD_SCRAPE_COUNT,
    overall_timeout: float = 120.0,   # NEW — hard cap for the whole pool-build stage
) -> list[dict]:
    model = _get_st_model()

    per_kw_results = await asyncio.gather(
        *[_run_scrape(_ddgs_search_for_script, kw, per_keyword_results) for kw in keywords],
        return_exceptions=True,
    )

    candidates: list[tuple[str, str]] = []
    seen_local = set()
    kw_failures = 0
    for kw, res in zip(keywords, per_kw_results):
        if isinstance(res, Exception):
            kw_failures += 1
            continue
        for url, snippet in res:
            if not url or url in scraped_urls or url in seen_local or _is_blocked_source_url(url):
                continue
            seen_local.add(url)
            candidates.append((url, snippet))

    for url, _ in candidates:
        scraped_urls.add(url)

    print(f"[POOL] {len(keywords)} keyword(s) ({kw_failures} failed) -> {len(candidates)} unique URL(s) to prepare")

    async def _prepare(url: str, fallback_snippet: str) -> dict | None:
        t0 = time.time()
        full_text = await _fetch_full_article_text_with_timeout(url)
        fetch_time = time.time() - t0
        used_source = "full" if full_text else "fallback"
        content = full_text if full_text else fallback_snippet
        if not content:
            print(f"[POOL] SKIP (no content, {fetch_time:.1f}s) {url}")
            return None
        content = _truncate_words(content, max_words=600)
        chunks = _split_into_chunks(content, max_words_per_chunk=40)
        if not chunks:
            print(f"[POOL] SKIP (no chunks, {fetch_time:.1f}s) {url}")
            return None
        try:
            chunk_embeddings = await _run_encode(
                lambda c=chunks: model.encode(c, normalize_embeddings=True, convert_to_numpy=True)
            )
        except Exception as e:
            print(f"[POOL] embedding failed for {url}: {e}")
            return None
        print(f"[POOL] OK ({fetch_time:.1f}s, {len(chunks)} chunks, source={used_source}) {url}")
        return {"url": url, "chunks": chunks, "chunk_embeddings": chunk_embeddings, "source": used_source}

    tasks = [asyncio.create_task(_prepare(u, s)) for u, s in candidates]

    stage_start = time.time()
    done, pending = await asyncio.wait(tasks, timeout=overall_timeout)

    for t in pending:
        t.cancel()
    if pending:
        print(f"[POOL] deadline hit at {overall_timeout}s — cancelled {len(pending)}/{len(tasks)} still-in-flight fetch(es)")

    pool = []
    for t in done:
        try:
            r = t.result()
            if r is not None:
                pool.append(r)
        except Exception:
            pass

    print(f"[POOL] prepared {len(pool)}/{len(candidates)} usable article(s) in {time.time() - stage_start:.1f}s (deadline={overall_timeout}s)")
    return pool

def rank_pool_for_hyde_doc(
    pool: list[dict],
    hyde_embedding: np.ndarray,
    similarity_threshold: float,
    top_k: int,
    exclude_urls: set | None = None,
) -> list[dict]:
    """Score an already-prepared shared pool against ONE HyDE doc's own
    embedding (no re-fetching, no re-embedding) and return its own top_k,
    skipping any URL already claimed by a previous HyDE doc in this batch."""
    exclude_urls = exclude_urls or set()
    scored = []
    for entry in pool:
        if entry["url"] in exclude_urls:
            continue
        sims = np.dot(entry["chunk_embeddings"], hyde_embedding)
        picked = [
            (chunk, float(sim))
            for chunk, sim in zip(entry["chunks"], sims)
            if sim >= similarity_threshold
        ]
        if not picked:
            continue
        picked.sort(key=lambda p: p[1], reverse=True)
        scored.append({
            "url": entry["url"],
            "snippet": _truncate_words(" ".join(c for c, _ in picked), max_words=200),
            "source": entry["source"],
            "similarity": picked[0][1],
            "picked_passage_count": len(picked),
            "total_passage_count": len(entry["chunks"]),
        })
    scored.sort(key=lambda a: a["similarity"], reverse=True)
    return scored[:top_k]



SCRIPT_KEYWORD_GEN_PROMPT_TEMPLATE = """
## ROLE

You are a **Search Query Expansion Engine** for automated web crawling.

Your purpose is to generate high-quality search engine keyword combinations that maximize retrieval from Google, Bing, academic search engines, news websites, government portals, company websites, research repositories, digital libraries, statistical databases, and other authoritative sources.

These search queries will be used to gather evidence for writing a complete video script.

---

## INPUT

**Idea Title:** `{title}`

**Idea Description:** `{description}`

**Target Video Duration:** `{time_minutes}` minute(s)

**Script Template Title:** `{template_title}`

**Script Template Purpose:** `{template_about}`

**Template Segments:**

`{segments_block}`

Template segments provide structural context for understanding the knowledge progression of the topic. They are **not** used to determine the number of search queries.

---

## OBJECTIVE

Generate **exactly 10 unique search keyword combinations** that collectively retrieve all knowledge required to write the complete script.

First infer the topic's primary domain(s) (such as history, business, technology, science, psychology, philosophy, politics, finance, biography, health, law, engineering, economics, medicine, sociology, or other academic disciplines).

Then analyze the idea title, idea description, template purpose, and template segments to understand the complete scope of the topic.

Generate search queries that collectively maximize factual coverage across the entire subject rather than focusing on individual template segments.

---

## TEMPLATE AWARENESS

Treat the template as a **high-level knowledge map**, not as individual retrieval tasks.

Do **not** generate search queries for each segment.

Instead, use the combined information from the title, description, template purpose, and template progression to infer all major knowledge areas required for the complete script.

Ignore storytelling instructions such as:

* hooks
* suspense
* pacing
* callbacks
* emotional engagement
* curiosity gaps

Instead, infer the underlying information needs across the complete topic.

These may naturally include, where relevant:

* definitions
* origins
* chronology
* mechanisms
* historical context
* evidence
* research
* case studies
* business strategy
* financial performance
* legislation
* scientific explanations
* expert opinions
* competing theories
* controversies
* limitations
* practical applications
* future developments

Only include knowledge areas that are genuinely relevant to the topic.

---

## SEARCH QUALITY

Collectively, the 10 search keyword combinations should maximize retrieval diversity while minimizing overlap.

Each keyword combination should:

* naturally include the core topic or closely related entities
* target one distinct retrieval intent
* maximize semantic diversity
* retrieve authoritative information
* retrieve primary sources whenever applicable
* retrieve academic, governmental, industrial, historical, technical, legal, or scientific sources where relevant
* naturally include important people, organizations, companies, technologies, theories, locations, historical events, legislation, institutions, standards, or frameworks when strongly implied by the topic

Ensure the complete set of searches covers the topic from foundational knowledge through advanced, contextual, analytical, and contemporary aspects.

Do **not** generate duplicate retrieval intents.

Do **not** keyword stuff.

---

## OUTPUT REQUIREMENTS

* Generate **exactly 10 unique search keyword combinations**.
* Cover the complete topic as comprehensively as possible.
* Number the results sequentially from **1** to **10**.
* Every keyword combination must contain **4–10 words**.
* Generate **search phrases**, not sentences.
* Do **not** output the raw idea title by itself.
* Do **not** include headings, explanations, grouping labels, reasoning, or additional text.

Output format:

```text
1. keyword combination

2. keyword combination

3. keyword combination

...

10. keyword combination
```

Return **only** the numbered keyword combinations.

"""


def _ddgs_search_for_script(keyword: str, max_results: int) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    try:
        with DDGS(timeout=10) as ddgs:  
            for r in ddgs.text(keyword, max_results=max_results * 2, backend="html"):
                url = r.get("href") or r.get("url")
                snippet = r.get("body", "") or r.get("title", "")
                if not url or _is_blocked_source_url(url):
                    continue
                results.append((url, snippet))
                if len(results) >= max_results:
                    break
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

    segments_block = _segments_brief(template.get("segments") or [], brief_field="hyde_brief")

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
                temperature=0.3,  
                top_p=0.9        
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
    keywords = keywords[:SCRIPT_SEARCH_KEYWORD_COUNT]

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

    print(f"[YT-API] search.list for '{keyword}' returned {len(data.get('items', []))} item(s)")

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


# =============================================================================
# get_youtube_context — scraping-optimized: the 10 keyword searches are now
# fired CONCURRENTLY via asyncio.gather instead of sequentially in a for-loop
# (each search.list + videos.list round trip no longer blocks the next).
# All log lines are buffered and flushed as one block so they don't
# interleave with Stage 2/3 output that runs at the same time. Ranking,
# dedup, and truncation logic is unchanged.
# =============================================================================
async def get_youtube_context(
    topic: str, description: str, scraped_urls: set, max_results: int = 10
) -> list[dict]:
    buf = _LogBuffer("STAGE 4 - YOUTUBE")
    buf.log(f"Starting YouTube search for topic: '{topic}'")

    if not YOUTUBE_API_KEY:
        buf.log("YOUTUBE_API_KEY not set, skipping YouTube search")
        buf.flush()
        return []

    keywords = await _generate_youtube_search_keywords(topic, description)

    async def _search_one(keyword: str) -> list[dict]:
        try:
            return await _run_scrape(_youtube_search_via_api, keyword, 1)
        except Exception as e:
            buf.log(f"search failed for '{keyword}': {e}")
            return []

    all_results = await asyncio.gather(*[_search_one(kw) for kw in keywords])

    raw_candidates: list[dict] = []
    for keyword, results in zip(keywords, all_results):
        buf.log(f"search.list for '{keyword}' returned {len(results)} item(s)")
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

    buf.log(
        f"fetched {len(raw_candidates)} unique candidate video(s) via YouTube Data API "
        f"from {len(keywords)} keyword(s), returning top {len(videos)} "
        f"(capped at {MAX_YOUTUBE_SOURCES})"
    )
    buf.flush()

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

    print("\n" + "=" * 100)
    print("[IDEAS-PROMPT] FINAL LLM PROMPT — generate_ideas_from_context")
    print("=" * 100)
    print("----- SYSTEM PROMPT -----")
    print(IDEAS_SYSTEM_PROMPT)
    print("----- USER PROMPT (topic + RAG/web chunks) -----")
    print(user_prompt)
    print("=" * 100 + "\n")


    res = await _openai_create_with_timeout(
        lambda: openai_client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": IDEAS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            temperature=0.55,   
            top_p=0.95     
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
        hyde_documents = await generate_ideas_hyde_documents(topic, num_docs=IDEAS_HYDE_DOC_COUNT)
        combined_hyde_doc = "\n\n".join(d for d in hyde_documents if d) or topic

        try:
            table_name = await select_table_for_topic(topic)
        except Exception as exc:
            print(f"[MAIN] table selection failed, defaulting to {exc}")

        similar_task = asyncio.create_task(get_similar_saved_ideas(topic, combined_hyde_doc))

        db_results_per_doc = []
        try:
            db_results_per_doc = await asyncio.gather(
                *[
                    get_context_from_db(
                        topic, doc, table_name=table_name,
                        final_k=IDEAS_RAG_POOL_PER_DOC,
                    )
                    for doc in hyde_documents
                ]
            )
        except Exception as exc:
            print(f"[MAIN] DB retrieval failed: {exc}")
            db_results_per_doc = []

        all_db_chunks_seen: list = []
        seen_md5_all = set()
        for doc_results in db_results_per_doc:
            for item in doc_results:
                key = item.get("md5") or item.get("content")
                if key and key not in seen_md5_all:
                    seen_md5_all.add(key)
                    all_db_chunks_seen.append(item)

        # STRICT pass: exactly top-K per doc, deduped, later docs skip URLs/chunks
        # already claimed by earlier docs so they're forced to their next-best pick.
        db_results = []
        seen_md5_context = set()
        for doc_idx, doc_results in enumerate(db_results_per_doc, start=1):
            top_for_doc = []
            for item in doc_results:
                if len(top_for_doc) >= IDEAS_TOP_K_PER_DOC:
                    break
                key = item.get("md5") or item.get("content")
                if key and key in seen_md5_context:
                    continue
                top_for_doc.append(item)

            for item in top_for_doc:
                key = item.get("md5") or item.get("content")
                if key and key not in seen_md5_context:
                    seen_md5_context.add(key)
                    db_results.append(item)

            print(
                f"[MAIN] HyDE doc #{doc_idx}: picked top {len(top_for_doc)} "
                f"RAG chunk(s) from a pool of {len(doc_results)} "
                f"(target pool size {IDEAS_RAG_POOL_PER_DOC}, excluding already-claimed chunks)"
            )

        print(
            f"[MAIN] Combined DB context (ideas): {len(db_results)} unique chunk(s) "
            f"— top {IDEAS_TOP_K_PER_DOC} picked independently from each of the "
            f"{len(hyde_documents)} HyDE doc(s), each drawn from a pool of "
            f"{IDEAS_RAG_POOL_PER_DOC}, with cross-doc dedup."
        )

        if len(db_results) < IDEAS_DB_CHUNKS_TO_LLM:
            max_pool_len = max((len(d) for d in db_results_per_doc), default=0)
            rank = IDEAS_TOP_K_PER_DOC
            while len(db_results) < IDEAS_DB_CHUNKS_TO_LLM and rank < max_pool_len:
                for doc_results in db_results_per_doc:
                    if len(db_results) >= IDEAS_DB_CHUNKS_TO_LLM:
                        break
                    if rank >= len(doc_results):
                        continue
                    item = doc_results[rank]
                    key = item.get("md5") or item.get("content")
                    if key and key not in seen_md5_context:
                        seen_md5_context.add(key)
                        db_results.append(item)
                rank += 1
            print(
                f"[MAIN] RAG backfill (ideas): now {len(db_results)}/{IDEAS_DB_CHUNKS_TO_LLM} "
                f"unique chunk(s) after pulling deeper into each doc's own pool"
            )

        db_results = db_results[:IDEAS_DB_CHUNKS_TO_LLM]
        if len(db_results) < IDEAS_DB_CHUNKS_TO_LLM:
            print(
                f"[MAIN] WARNING: only {len(db_results)}/{IDEAS_DB_CHUNKS_TO_LLM} unique RAG "
                f"chunk(s) available for this topic even after backfill — DB genuinely doesn't "
                f"have more distinct on-topic chunks across the HyDE docs' pools."
            )

        scraped_urls = set()
        try:
            ideas_search_keywords = await _generate_web_search_keywords(topic)
        except Exception as exc:
            print(f"[MAIN] ideas keyword generation failed: {exc}")
            ideas_search_keywords = [f"{topic} latest news today", f"{topic} 2026 update"]

        shared_pool: list[dict] = []
        try:
            shared_pool = await build_shared_web_pool(ideas_search_keywords, scraped_urls)
        except Exception as exc:
            print(f"[MAIN] shared web pool build failed: {exc}")
            shared_pool = []

        model = _get_st_model()
        new_articles: list[dict] = []
        seen_urls_final: set = set()

        for doc_idx, doc in enumerate(hyde_documents, start=1):
            try:
                hyde_embedding = await _run_encode(
                    lambda d=doc: model.encode(d, normalize_embeddings=True, convert_to_numpy=True)
                )
                top_for_doc = rank_pool_for_hyde_doc(
                    shared_pool,
                    hyde_embedding,
                    WEB_CONTENT_SIMILARITY_THRESHOLD,
                    IDEAS_TOP_K_PER_DOC,
                    exclude_urls=seen_urls_final,  # forces this doc to pick URLs not already claimed
                )
            except Exception as exc:
                print(f"[MAIN] ranking shared pool for HyDE doc #{doc_idx} failed: {exc}")
                top_for_doc = []

            newly_added = 0
            for article in top_for_doc:
                url = article.get("url")
                if url and url not in seen_urls_final:
                    seen_urls_final.add(url)
                    new_articles.append(article)
                    newly_added += 1

            print(
                f"[MAIN] HyDE doc #{doc_idx}: picked top {newly_added} NEW unique "
                f"web source(s) from the shared pool of {len(shared_pool)} "
                f"(target top-{IDEAS_TOP_K_PER_DOC}, {len(seen_urls_final)} unique claimed so far)"
            )

        print(
            f"[MAIN] Combined web context (ideas): {len(new_articles)} unique source(s) "
            f"— top {IDEAS_TOP_K_PER_DOC} picked independently (with cross-doc dedup) from "
            f"each of the {len(hyde_documents)} HyDE doc(s) against the SAME shared pool of "
            f"{len(shared_pool)} pre-fetched article(s)."
        )

        if _unique_url_count(new_articles) < IDEAS_WEB_SOURCES_TO_LLM:
            print(
                f"[MAIN] Only {_unique_url_count(new_articles)} unique source URL(s) found "
                f"(ideas), backfilling with generic queries against a relaxed threshold."
            )
            try:
                generic_queries = [
                    topic, f"{topic} history", f"{topic} overview", f"{topic} explained",
                    f"{topic} background", f"{topic} facts", f"{topic} details",
                    f"{topic} analysis", f"{topic} biography", f"{topic} encyclopedia",
                ]
                extra_pool = await build_shared_web_pool(generic_queries, scraped_urls)
                combined_pool = shared_pool + extra_pool

                for doc in hyde_documents:
                    if _unique_url_count(new_articles) >= IDEAS_WEB_SOURCES_TO_LLM:
                        break
                    hyde_embedding = await _run_encode(
                        lambda d=doc: model.encode(d, normalize_embeddings=True, convert_to_numpy=True)
                    )
                    for article in rank_pool_for_hyde_doc(
                        combined_pool,
                        hyde_embedding,
                        _MIN_ACCEPTABLE_SIMILARITY,
                        IDEAS_TOP_K_PER_DOC,
                        exclude_urls=seen_urls_final,  # same dedup guard during backfill
                    ):
                        if _unique_url_count(new_articles) >= IDEAS_WEB_SOURCES_TO_LLM:
                            break
                        url = article.get("url")
                        if url and url not in seen_urls_final:
                            seen_urls_final.add(url)
                            new_articles.append(article)
                print(
                    f"[MAIN] backfill done — now {_unique_url_count(new_articles)}/"
                    f"{IDEAS_WEB_SOURCES_TO_LLM} unique source(s)"
                )
            except Exception as exc:
                print(f"[MAIN] ideas web backfill failed: {exc}")

        new_articles.sort(key=lambda a: a.get("similarity", 0.0), reverse=True)
        new_articles = new_articles[:IDEAS_WEB_SOURCES_TO_LLM]
        if _unique_url_count(new_articles) < IDEAS_WEB_SOURCES_TO_LLM:
            print(
                f"[MAIN] WARNING: only {_unique_url_count(new_articles)}/{IDEAS_WEB_SOURCES_TO_LLM} "
                f"unique web source(s) available even after backfill — the web genuinely doesn't "
                f"have more relevant, distinct sources for this topic."
            )

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

        sources = _extract_source_links(new_articles)

        books: list[dict] = []
        try:
            books = await get_books_for_chunks(
                all_db_chunks_seen, topic_text=topic, script_text=""
            )

            if len(books) < MAX_BOOKS:
                print(
                    f"[MAIN] Only {len(books)}/{MAX_BOOKS} real book(s) found from the "
                    f"initial chunk pool (ideas) — widening DB search to try to reach {MAX_BOOKS}."
                )
                known_md5s = {r.get("md5") for r in all_db_chunks_seen if r.get("md5")}
                books = await _backfill_books_to_target(
                    books,
                    known_md5s,
                    topic,
                    combined_hyde_doc,
                    table_name,
                    target_count=MAX_BOOKS,
                )
        except Exception as exc:
            print(f"--- MySQL book lookup failed (ideas): {exc} ---")
            import traceback
            traceback.print_exc()
            books = []

        token_usage = _get_token_usage_summary()

        return {
            "topic": topic,
            "topic_summary": topic_summary,
            "ideas": ideas,
            "similar_past_ideas": similar_saved_ideas,
            "sources": sources,
            "books": books,
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



async def get_channel_profile(userId: str):
    try:
        channel_profile = await _run_io(
            lambda: supabase
            .table("user_channel_memory_input")
            .select("Summary")
            .eq("userId", userId)
            .execute()
        )
        return channel_profile.data
    except Exception as e:
        print(e)
        return None
    
from fastapi import HTTPException


class UnlockRequest(BaseModel):
    userId: str
    duration: float 

CREDITS_PER_MINUTE = 1


class CheckCreditsRequest(BaseModel):
    userId: str


@app.post("/check-credits")
async def check_credits(request: CheckCreditsRequest):
    try:
        profile_res = supabase.table('user_profiles') \
            .select('id, credit_batches') \
            .eq('id', request.userId) \
            .maybe_single() \
            .execute()

        if not profile_res.data:
            raise HTTPException(status_code=404, detail="user profile not found")

        batches = profile_res.data.get('credit_batches') or []
        now = datetime.datetime.now(datetime.timezone.utc)
        active_batches = _expire_stale_batches(batches, now)

        new_total = _sum_batches(active_batches)

        if active_batches != batches:
            supabase.table('user_profiles').update({
                'credit_batches': active_batches,
                'credits_remaining': new_total,
            }).eq('id', request.userId).execute()

        return {
            "message": "success",
            "remaining_credits": new_total,
            "expired_removed": len(batches) - len(active_batches),
        }

    except HTTPException:
        raise
    except Exception as e:
        print("error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unlock")
async def cut_credits(request: UnlockRequest):
    if request.duration <= 0:
        raise HTTPException(status_code=400, detail="duration must be positive")

    cost = round(request.duration * CREDITS_PER_MINUTE)

    try:
        profile_res = supabase.table('user_profiles') \
            .select('id, credit_batches') \
            .eq('id', request.userId) \
            .maybe_single() \
            .execute()

        if not profile_res.data:
            raise HTTPException(status_code=404, detail="user profile not found")

        batches = profile_res.data.get('credit_batches') or []
        now = datetime.datetime.now(datetime.timezone.utc)
        active_batches = _expire_stale_batches(batches, now)

        updated_batches, deducted = _deduct_from_batches(active_batches, cost)
        if deducted == 0:
            return {"message": "credits not sufficient"}

        new_total = _sum_batches(updated_batches)

        supabase.table('user_profiles').update({
            'credit_batches': updated_batches,
            'credits_remaining': new_total,
        }).eq('id', request.userId).execute()

        return {
            "message": "success",
            "remaining_credits": new_total,
        }

    except HTTPException:
        raise
    except Exception as e:
        print("error:", e)
        raise HTTPException(status_code=500, detail=str(e))



def target_word_count_for_time(minutes: float) -> int:
    return max(50, int(minutes * WORDS_PER_MINUTE))


class ScriptRequest(BaseModel):
    userId: str
    title: str
    description: str
    time: int
    topic: str
    language: str


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


def num_hyde_docs_for_time(minutes: float) -> int:
    return max(1, math.ceil(minutes / 2))


def _strip_json_fences(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


RRF_K = 60  
SCRIPT_RAG_POOL_PER_DOC = 40
SCRIPT_TOP_K_PER_DOC = 2       

DENSE_SCORE_THRESHOLD = 0.3
SPARSE_SCORE_THRESHOLD = 0.2

async def get_context_from_db_segment(
    hyde_document: str,
    keywords: list[str],
    table_name: str,
    dense_k: int = SCRIPT_RAG_POOL_PER_DOC,
    sparse_k: int = SCRIPT_RAG_POOL_PER_DOC,
    rrf_k: int = RRF_K,
    dense_score_threshold: float = DENSE_SCORE_THRESHOLD,
    sparse_score_threshold: float = SPARSE_SCORE_THRESHOLD,
) -> list[dict]:
    category = table_name
    collection_name = _qdrant_collection_name(category)
    supabase_table = _supabase_content_table_name(category)

    model = _get_st_model()
    client = get_qdrant_client()

    dense_embedding = await _run_encode(
        lambda: model.encode(
            hyde_document, convert_to_numpy=True, normalize_embeddings=True
        ).tolist()
    )

    try:
        dense_result = await _run_io(
            lambda: client.query_points(
                collection_name=collection_name,
                query=dense_embedding,
                using=QDRANT_DENSE_VECTOR_NAME,
                limit=dense_k,
                score_threshold=dense_score_threshold,
                with_payload=True,
            )
        )
        dense_points = dense_result.points
    except Exception as e:
        print(f"[QDRANT-SEG] dense query FAILED against '{collection_name}': {e}")
        dense_points = []

    keyword_text = " ".join(k for k in (keywords or []) if k) or hyde_document
    vectorizer = get_sparse_vectorizer()
    sparse_row = await asyncio.to_thread(lambda: vectorizer.transform([keyword_text]))
    query_sparse_dict = _sparse_row_to_dict(sparse_row)
    sparse_indices = [int(k) for k in query_sparse_dict.keys()]
    sparse_values = [float(v) for v in query_sparse_dict.values()]

    try:
        sparse_result = await _run_io(
            lambda: client.query_points(
                collection_name=collection_name,
                query=qdrant_models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using=QDRANT_SPARSE_VECTOR_NAME,
                limit=sparse_k,
                score_threshold=sparse_score_threshold,
                with_payload=True,
            )
        )
        sparse_points = sparse_result.points
    except Exception as e:
        print(f"[QDRANT-SEG] sparse query FAILED against '{collection_name}': {e}")
        sparse_points = []

    dense_raw_scores: dict = {}
    sparse_raw_scores: dict = {}
    dense_ranks: dict = {}
    sparse_ranks: dict = {}
    chunk_ids: set = set()

    for rank, point in enumerate(dense_points, start=1):
        payload = point.payload or {}
        chunk_id = payload.get("chunk_id")
        if chunk_id is None:
            continue
        dense_ranks[chunk_id] = rank
        dense_raw_scores[chunk_id] = point.score
        chunk_ids.add(chunk_id)

    for rank, point in enumerate(sparse_points, start=1):
        payload = point.payload or {}
        chunk_id = payload.get("chunk_id")
        if chunk_id is None:
            continue
        sparse_ranks[chunk_id] = rank
        sparse_raw_scores[chunk_id] = point.score
        chunk_ids.add(chunk_id)

    if not chunk_ids:
        print(f"[QDRANT-SEG] dense={len(dense_points)} sparse={len(sparse_points)} -> 0 candidate(s)")
        return []

    try:
        supabase_rows = await _run_io(
            lambda: supabase.table(supabase_table)
            .select("chunk_id, md5, content")
            .in_("chunk_id", list(chunk_ids))
            .execute()
        )
        rows = supabase_rows.data or []
    except Exception as e:
        print(f"[QDRANT-SEG] Supabase content lookup against '{supabase_table}' failed: {e}")
        return []

    matches = []
    for row in rows:
        chunk_id = row.get("chunk_id")
        content = row.get("content")
        md5 = row.get("md5")
        if not content:
            continue

        d_rank = dense_ranks.get(chunk_id)
        s_rank = sparse_ranks.get(chunk_id)

        rrf_score = 0.0
        if d_rank is not None:
            rrf_score += 1.0 / (rrf_k + d_rank)
        if s_rank is not None:
            rrf_score += 1.0 / (rrf_k + s_rank)

        matches.append({
            "content": content,
            "md5": md5,
            "chunk_id": chunk_id,
            "dense_score": dense_raw_scores.get(chunk_id),
            "sparse_score": sparse_raw_scores.get(chunk_id),
            "dense_rank": d_rank,
            "sparse_rank": s_rank,
            "combined_score": rrf_score,
            "matched_via": "both" if (d_rank is not None and s_rank is not None) else ("dense" if d_rank is not None else "sparse"),
        })

    matches.sort(key=lambda r: r["combined_score"], reverse=True)

    print(
        f"[QDRANT-SEG] dense={len(dense_points)}(thr={dense_score_threshold}) "
        f"sparse={len(sparse_points)}(thr={sparse_score_threshold}) "
        f"-> {len(chunk_ids)} unique candidate(s) in combined pool (RRF, k={rrf_k})"
    )
    for i, m in enumerate(matches[:5], start=1):
        print(
            f"    [{i}] chunk_id={m['chunk_id']} via={m['matched_via']} "
            f"dense_rank={m['dense_rank']} sparse_rank={m['sparse_rank']} "
            f"rrf={m['combined_score']:.5f}"
        )

    return matches

async def get_context_from_db_segment_with_timeout(
    hyde_document: str,
    keywords: list[str],
    table_name: str,
    timeout: float = 60.0,
    dense_k: int = SCRIPT_RAG_POOL_PER_DOC,   # was 10
    sparse_k: int = SCRIPT_RAG_POOL_PER_DOC,  # was 10
) -> list[dict]:
    task = asyncio.create_task(
        get_context_from_db_segment(hyde_document, keywords, table_name, dense_k, sparse_k)
    )
    done, pending = await asyncio.wait({task}, timeout=timeout)

    if task in done:
        try:
            result = task.result()
            print(f"[DB-SEG] task finished within timeout. Pool has {len(result)} candidate(s).")
            return result
        except Exception as e:
            print(f"[DB-SEG] task raised an error: {e}")
            return []
    else:
        print("[DB-SEG] task still running after timeout, proceeding without it for now.")
        return []


async def generate_hyde_docs_for_script(
    title: str,
    description: str,
    template: dict,
    segments: list[dict],
) -> list[dict]:
    """
    Returns a list of {"hyde_document": str, "keywords": list[str],
    "segment_index": int, "segment_name": str, "hyde_brief": str,
    "llm_brief": str} — one entry per template segment, in order.

    Only `segment_name` and `hyde_brief` are pulled from each segment to
    build the HyDE generation prompt (llm_brief / engagement_craft are never
    shown to the HyDE generator — they are for the final script-writing LLM
    only). segment_name / hyde_brief / llm_brief are carried through onto
    the returned entry (not invented by the model) so downstream retrieval
    can tag each chunk with the segment it was retrieved for.
    """
    segment_briefs = "\n".join(
        f"- {seg.get('segment_name', 'segment')} ({seg.get('percentage', 0)}%): {seg.get('hyde_brief', '')}"
        for seg in segments
    )

    fallback_text = f"{title}\n\n{description}".strip()
    fallback_docs = lambda: [
        {
            "hyde_document": fallback_text,
            "keywords": [],
            "segment_index": idx,
            "segment_name": seg.get("segment_name", f"segment_{idx}"),
            "hyde_brief": seg.get("hyde_brief", ""),
            "llm_brief": seg.get("llm_brief", ""),
        }
        for idx, seg in enumerate(segments, start=1)
    ] or [{
        "hyde_document": fallback_text,
        "keywords": [],
        "segment_index": 1,
        "segment_name": "segment_1",
        "hyde_brief": "",
        "llm_brief": "",
    }]

    if not segments:
        return [{
            "hyde_document": fallback_text,
            "keywords": [],
            "segment_index": 1,
            "segment_name": "segment_1",
            "hyde_brief": "",
            "llm_brief": "",
        }]

    template_title = template.get('title')
    template_about = template.get('about')

    hyde_prompt = f"""


# SS-HDG v7 — Script Segment HyDE + Keyword Generator

ROLE: For each script segment, generate TWO retrieval artifacts:
(1) a HyDE paragraph for dense embedding, and
(2) a keyword set for sparse/lexical retrieval.
These serve different retrieval mechanisms and must be optimized differently — do not just extract keywords from the paragraph afterward; reason about each independently.

INPUT
Idea Title: {title}
Idea Description: {description}
Template Title: {template_title}
Template Purpose: {template_about}
Segments: {segment_briefs}

Each segment in {segment_briefs} includes a precomputed `keyword_count` — the exact number of keywords required for that segment (derived from its % of script and retrieval objective type). Use it exactly as given; do not decide the count yourself.

RULES

1. RETRIEVAL OBJECTIVE PER SEGMENT — infer from segment purpose, e.g.:
- Hook → memorable incidents, founder stories, verifiable quotes
- Definition → authoritative definitions, terminology, taxonomy
- Mechanism/Mapping → causal processes, technical detail
- Evidence → studies, benchmarks, institutional reports
- Applications → real-world deployments, organizations, products
- Comparison → alternative approaches, contrasts
- Limitations → criticisms, failure modes, open questions
- Synthesis → integrative framework (write LAST, must add an angle not covered above)
Infer objective for any other segment from its brief. This objective guides both outputs.

2. HYDE_DOCUMENT:
- No meta-language: never mention "retrieval," "should target," "sources on," or reference itself as a document. Write as a standalone excerpt from an authoritative source.
- No shared sentence-opening pattern across segments.
- Only the Definition segment may formally introduce/define the topic.
- Word budget scales with segment %: ≤10% → 50-60 words | 11-20% → 90-110 words | 21-30%+ → 130-150 words.
- Factual discipline: never fabricate statistics, dates, quotes, studies, citations.

3. KEYWORDS — count is fixed per segment via `keyword_count`, not open-ended:
- Each entry is a short phrase (1-4 words), NOT a sentence.
- Include: named frameworks/methods relevant to this segment, technical terminology, named people/organizations if factually associated with this segment's objective, synonyms and alternate phrasings a source might use for the same concept, closely related sub-concepts.
- Do NOT include: generic filler words, stopword-heavy phrases, terms already used in this exact form by an earlier segment's keyword list.
- Prioritize coverage breadth over redundancy: if genuinely distinct terms run out before reaching the count, prefer a slightly shorter list over padding with near-duplicate phrasings.
- Keywords may share a few terms with the hyde_document's vocabulary, but should extend beyond it — synonyms and adjacent terms the paragraph didn't use, since the goal is catching lexical matches the dense channel might miss.

4. SEMANTIC ISOLATION — applies to both outputs. Before finalizing a segment, check:
- Does another segment's hyde_document cover largely the same ground? Rewrite if so.
- Does another segment's keyword list overlap heavily (same terms in same form)? Trim/replace overlapping terms with segment-specific alternatives.

5. STYLE (hyde_document only) — objective, reference-book tone. No storytelling, opinions, second-person language, intros, conclusions, or self-referential commentary. Keywords are exempt from prose style rules per Rule 3.

OUTPUT — valid JSON only, no markdown fences, no preamble, no trailing text:

{{
  "documents": [
    {{
      "segment": "<Segment Name>",
      "focus": "<one short phrase, internal logging only>",
      "hyde_document": "<standalone paragraph, zero meta-language>",
      "keywords": ["<term 1>", "<term 2>", "..."]
    }}
  ]
}}


""".strip()

    try:
        completion = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[{"role": "user", "content": hyde_prompt}],
                stream=False,
                temperature=0.3,
                top_p=0.9,
            )
        )
        _record_token_usage("generate_hyde_docs_for_script", completion)

        choice = completion.choices[0]
        raw = (choice.message.content or "").strip()

        print(f"\n{'#' * 80}")
        print(f"HYDE RAW OUTPUT — {len(segments)} segment(s) requested")
        print(f"{'#' * 80}")
        print(raw if raw else "(empty response from model)")
        print(f"{'#' * 80}\n")

        if not raw:
            print("--- HyDE generation EMPTY, falling back to title/description ---")
            return fallback_docs()

        try:
            parsed = json.loads(_strip_json_fences(raw))
        except Exception as parse_exc:
            print(f"--- HyDE JSON parse failed: {parse_exc} — falling back ---")
            return fallback_docs()

        raw_docs = parsed.get("documents") if isinstance(parsed, dict) else None
        if not isinstance(raw_docs, list) or not raw_docs:
            print("--- HyDE JSON had no usable 'documents' list, falling back ---")
            return fallback_docs()

        docs = []
        for i, entry in enumerate(raw_docs):
            if not isinstance(entry, dict):
                continue
            text_value = (entry.get("hyde_document") or "").strip()
            raw_keywords = entry.get("keywords") or []
            if not isinstance(raw_keywords, list):
                raw_keywords = []
            kw_clean = [str(k).strip() for k in raw_keywords if str(k).strip()]
            if not text_value:
                continue

            # Positional match to the original template segment — segment
            # metadata (segment_name / hyde_brief / llm_brief) always comes
            # from the input `segments` list, never from what the model
            # returned, so it can be trusted for chunk provenance tagging.
            seg_meta = segments[i] if i < len(segments) else {}
            docs.append({
                "hyde_document": text_value,
                "keywords": kw_clean,
                "segment_index": i + 1,
                "segment_name": seg_meta.get("segment_name") or entry.get("segment") or f"segment_{i + 1}",
                "hyde_brief": seg_meta.get("hyde_brief", ""),
                "llm_brief": seg_meta.get("llm_brief", ""),
            })

        if len(docs) != len(segments):
            print(
                f"--- HyDE parse WARNING: got {len(docs)} document(s) but "
                f"expected {len(segments)} — using what was parsed anyway ---"
            )

        if not docs:
            print("--- HyDE parse produced 0 documents, falling back ---")
            return fallback_docs()

        return docs

    except Exception as exc:
        print(f"--- HyDE generation failed: {type(exc).__name__}: {exc} ---")
        return fallback_docs()
                        
async def get_context_with_timeout(
    topic_text: str,
    hyde_document: str,
    table_name: str = None,
    timeout: float = 20.0,
    final_k: int = 10,
) -> list:
    task = asyncio.create_task(
        get_context_from_db(topic_text, hyde_document, table_name=table_name, final_k=final_k)
    )
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
# YOUTUBE DOCUMENTARY SCRIPT GENERATION AGENT

## ROLE

You are a production-grade YouTube Documentary Script Generation Agent.

Turn the supplied idea, template, retrieved knowledge, and recent web/news material into a highly engaging documentary narration for a broad audience. The result must feel researched but human, cinematic but conversational, educational without sounding like an article, and structured for strong audience retention.

## INPUT

You will receive:

* *Idea Title*
* *Idea Description*
* *Target Duration (minutes)*
* *Output Language:* language required for the final narration
* *Script Template:* title, cluster, purpose, and ordered segments
* *Retrieved Knowledge Chunks with sources:* book name, author, published year
* *Recent Web/News Chunks with source details:* including the web article link

The *Output Language* is mandatory. Generate the complete `script` in that language while preserving the supplied meaning, narrative flow, factual accuracy, analysis, emotional continuity, and engagement intent.

Treat the supplied template and retrieved material as the primary source of truth. Never invent facts, sources, quotations, dates, numbers, events, or claims.

## INSTRUCTION HIERARCHY

Apply all layers together:

1. *Template Structure* = WHAT happens and WHEN.
2. *LLM Brief / Retrieval Directive* = WHICH source information belongs in each segment.
3. *Engagement Craft* = HOW that information works inside the segment.
4. *Global Storytelling Rules* = HOW the complete script sounds and flows.
5. *Output Language + TTS Rules* = HOW the narration is expressed and spoken.

Never replace the template with generic storytelling rules or ignore a segment's LLM Brief or retrieval purpose.

## SOURCE → STORY

Use retrieved material as *raw material, not text to summarize*. Select only information serving the current segment. Preserve names, dates, numbers, places, evidence, and causal relationships accurately.

Do not force every chunk into the script. Prefer concrete, specific, story-worthy details over broad background.

Use recent web/news material when relevant. For time-sensitive information, retain its appropriate date and context rather than presenting it as timeless fact.

## OUTPUT LANGUAGE + TTS-READY NARRATION

Generate the entire `script` in the supplied *Output Language*.

Preserve the original story logic, analysis, segment progression, emotional flow, and factual relationships while expressing them naturally in the requested language.

Translate for *meaning and narration*, not word-for-word literal translation. The result must sound like it was naturally written by a skilled native-language documentary narrator.

Use:

* natural grammar, word order, vocabulary, and transitions
* language-appropriate emotional expression
* consistent terminology
* natural spoken rhythm and pronunciation
* culturally natural expressions where appropriate

Do not produce robotic, mechanical, academic, or literal translation.

The script must be directly usable as Text-to-Speech input without additional preprocessing.

### TTS NUMBER RULES

Write every numerical expression in the `script` as its *spoken-word form in the Output Language*. Never use numerical digits in the narration.

This applies to:

* numbers and quantities
* years and dates
* ages
* percentages
* statistics
* currencies and monetary values
* measurements and units
* decimals
* ranges
* rankings and numbered references

Examples:

* `1942` → spoken year in the Output Language
* `10 minutes` → spoken number and unit
* `₹10,000` → spoken currency amount
* `25%` → spoken percentage
* `3.5` → spoken decimal
* `2–3` → spoken range

Preserve the exact factual value while converting its written representation into natural speech.

Avoid symbols, abbreviations, or notation that could produce unnatural TTS pronunciation. Express them as words when appropriate.

Keep proper nouns, brands, places, technical terms, and established names factually correct while using the form that produces natural pronunciation in the requested language.

Do not add phonetic annotations, pronunciation instructions, SSML, brackets, or meta-commentary unless explicitly requested.

## OPENING + OVERALL VIDEO PROMISE

After the initial engaging hook, naturally give the viewer a *brief overview of what the video will explore*.

It should feel like a conversational orientation, not an agenda.

* Establish the overall subject, journey, question, or discovery.
* Tell the viewer what they will learn, discover, or experience by staying.
* Keep it warm, natural, and curiosity-driven.
* Connect it directly to the Hook.
* Do not list sections or reveal the final answer, major reveal, or full conclusion.
* Avoid generic phrases such as “In this video, we will discuss…” unless naturally rephrased.

The opening should quickly accomplish:

*Hook the viewer → orient them to the journey.*

## HOOK

Open immediately with the strongest supported incident, fact, contradiction, image, result, or human moment available for the template.

The Hook must be a *YouTube hook, not a topic introduction*.

* No “today we're going to…”
* No generic greeting.
* No broad textbook definition.
* No empty scene-setting.
* Put a concrete detail in the opening lines.
* Create curiosity, tension, contradiction, uncertainty, or a held-back promise.
* Make the viewer want the next sentence.
* Follow the template's Hook LLM Brief and Engagement Craft precisely.

A natural audience-facing line may appear later if it does not weaken the opening.

## CONTINUOUS ENGAGEMENT

Every segment must earn its place.

Create forward motion through causation, contrast, escalation, unanswered questions, expectations, consequences, and revelations.

Prefer *“but,” “therefore,” and “because”* logic over disconnected “and then” accumulation.

Plant details that can pay off later. Use Breadcrumbs, Backpacks, open questions, or other Engagement Craft techniques when required by the template.

Do not manufacture suspense unsupported by evidence.

## STORY + EXPLANATION

Teach through story whenever possible.

Prefer:

*concrete example → context → meaning*

over:

*background → explanation → facts*

Explain difficult ideas in plain language. Use intuitive analogies, comparisons, or hypothetical examples when useful without distorting evidence.

Avoid list-like narration unless the template explicitly requires a list, ranking, checklist, timeline, or similar structure.

## HUMAN VOICE

Write like an intelligent human narrator speaking naturally to a real audience:

* conversational, confident, curious, vivid, and natural
* varied sentence length and rhythm
* occasional short sentences for emphasis
* concrete verbs and supported sensory detail
* natural transitions
* no corporate, academic, robotic, or Wikipedia-like phrasing
* no repetitive “this is important because…” constructions
* no excessive rhetorical questions
* no filler or ornamental padding

Do not fabricate dialogue or inner thoughts.

Any quotation must be supported by the retrieved material. Do not present wording as an exact quotation unless the source supports it.

## PACING + TEMPLATE

Use the template's ordered segments and proportions as the narrative architecture.

Segment proportions guide *pacing only* and must not appear as labels in the final script.

Maintain natural transitions so the story feels continuous rather than stitched together.

Do not over-explain merely to satisfy a segment percentage.

### HARD WORD-COUNT REQUIREMENT

The final script must contain exactly:

*Target Duration × 130 words*

This is a hard production requirement.

Internally revise, compress, or expand the narration until the required word count is reached while preserving factual accuracy, narrative quality, pacing, template structure, and natural language.

## ENDING + CTA

Complete the narrative before asking for engagement.

The ending must pay off the Hook's question, promise, image, contradiction, or tension through the template's final synthesis/close.

When the template contains a CTA:

* make it earned and connected to the story
* explicitly invite thoughts, interpretation, opinion, experience, or feedback in the comments
* use a specific question when appropriate
* callback to a concrete earlier detail when required

Do not use a generic “like and subscribe” unless the supplied template specifically requires it.

## FACTUAL INTEGRITY

Never invent unsupported information to improve drama or word count.

Do not merge separate facts into a false causal relationship.

Clearly distinguish documented fact from interpretation, hypothesis, prediction, or uncertainty.

When sources disagree, preserve the relevant uncertainty rather than silently selecting a convenient version.

## FINAL INTERNAL QA

Before returning the answer, silently verify:

* template order and segment purposes are respected
* each segment follows its LLM Brief and retrieval purpose
* Engagement Craft is reflected in execution
* Hook is genuinely a hook
* opening naturally provides the required overview
* overview does not spoil the major reveal or conclusion
* narrative is coherent and progressively engaging
* entire `script` is in the requested Output Language
* language is natural rather than mechanically translated
* narration flow, analysis, and factual meaning are preserved
* script is TTS-ready
* all numerical expressions in the script are written as spoken words
* no unsupported claims or fabricated quotations
* script contains exactly *Target Duration × 130 words*
* CTA explicitly invites comments/feedback when applicable
* metrics are calculated from the actual generated script
* JSON is valid
* nothing appears outside the JSON object

## SCRIPT ANALYTICS

Return:

* *Every numeric value must be at least 1.*
* videoLengthMinutes: copy *Target Duration exactly*.
* wordCount: total words in the generated script.
* emotionalDepth (1–10): emotional engagement, storytelling quality, tension, vivid imagery, and human resonance.
* generalExamples: illustrative examples, analogies, comparisons, or hypothetical scenarios that are *NOT historical examples*.
* proverbs_count: proverbs, sayings, quotations, aphorisms, and memorable quotes.
* historicalExamples: historical events, figures, discoveries, civilizations, companies, or eras.
* researchFacts: distinct research-backed facts, reports, studies, surveys, or statistics.

## CONTENT CLASSIFICATION

Return:

* category: one primary category, *1–3 words*.
* subcategories: up to five concise subcategories, *1–3 words each*.
* Do not repeat the category inside subcategories.

## OUTPUT

Return *only valid JSON*.

The JSON example below is a *STRUCTURE/SCHEMA EXAMPLE ONLY*. It is not a fixed set of values, wording, counts, or classifications. Do not copy its sample content. Generate every value dynamically from the actual input and generated script.

json id="rpj990"
{
"script": "Complete documentary narration in the requested Output Language, written as natural TTS-ready continuous paragraphs.",
"metrics": {
"videoLengthMinutes": 10,
"wordCount": 1300,
"emotionalDepth": 1,
"generalExamples": 1,
"proverbs_count": 1,
"historicalExamples": 1,
"researchFacts": 1
},
"classification": {
"category": "string",
"subcategories": [
"string"
]
}
}
"""




def _segments_brief(segments: list[dict], brief_field: str = "hyde_brief") -> str:
    if not segments:
        return "No template segments available — write a natural documentary-style structure."
    return "\n".join(
        f"- {seg.get('segment_name', 'segment')} ({seg.get('percentage', 0)}% of runtime): {seg.get(brief_field, '')}"
        for seg in segments
    )




def _fetch_books_by_md5_map_sync(md5_list: list[str]) -> dict[str, dict]:
    if not md5_list:
        return {}

    try:
        result = (
            supabase.table(BOOKS_SUPABASE_TABLE)
            .select(f"Title, Author, Year, {BOOKS_SUPABASE_MD5_COLUMN}")
            .in_(BOOKS_SUPABASE_MD5_COLUMN, md5_list)
            .execute()
        )
        rows = result.data or []
    except Exception as e:
        print(f"[SUPABASE] md5->book map lookup against '{BOOKS_SUPABASE_TABLE}' failed: {e}")
        return {}

    book_map: dict[str, dict] = {}
    for row in rows:
        md5 = row.get(BOOKS_SUPABASE_MD5_COLUMN)
        title = (row.get("Title") or "").strip()
        author = (row.get("Author") or "").strip()
        if not md5 or not title or not author:
            continue
        book_map[md5] = {
            "title": title,
            "author": author,
            "year": _normalize_book_year(row.get("Year")),
        }

    print(f"[SUPABASE] resolved book metadata for {len(book_map)}/{len(md5_list)} md5(s)")
    return book_map


async def _build_script_context_json(
    db_results: list[dict],
    new_articles: list[dict],
) -> tuple[str, dict]:
    md5_list = [row.get("md5") for row in db_results if row.get("md5")]
    book_map = await _run_io(_fetch_books_by_md5_map_sync, md5_list)

    kb_chunks = []
    for row in db_results:
        md5 = row.get("md5")
        kb_chunks.append({
            "content": row.get("content", ""),
            "similarity": row.get("dense_score"),
            "md5": md5,
            "book": book_map.get(md5),
            # Chunk provenance: which template segment this chunk was
            # retrieved for. Deliberately carries llm_brief (writing
            # guidance for the script LLM), never hyde_brief (retrieval-only,
            # not meant for the script-writing pass).
            "segment_id": row.get("segment_id"),
            "segment_name": row.get("segment_name"),
            "segment_brief": row.get("llm_brief"),
        })

    web_chunks = []
    for article in new_articles:
        web_chunks.append({
            "snippet": article.get("snippet", ""),
            "similarity": article.get("similarity"),
            "url": article.get("url", ""),
            "segment_id": article.get("segment_id"),
            "segment_name": article.get("segment_name"),
            "segment_brief": article.get("llm_brief"),
        })

    payload = {
        "knowledge_base_chunks": kb_chunks,
        "web_chunks": web_chunks,
    }

    if not kb_chunks and not web_chunks:
        return "No high-confidence source material available.", payload

    return json.dumps(payload, indent=2, ensure_ascii=False), payload


async def generate_script_from_context(
    request: "ScriptRequest",
    selected_template: dict,
    db_results: list[dict],
    new_articles: list[dict],
    target_word_count: int,
    language: str 
) -> dict:
    context_block, _context_payload = await _build_script_context_json(db_results, new_articles)
    segments_block = _segments_brief(selected_template.get("segments") or [], brief_field="llm_brief")

    print( "script_segments" + segments_block)
    print(request.time)

    user_prompt = f"""
Video Title: "{request.title}"
Video Description: "{request.description}"
Target Duration: {request.time} minute(s)
Target Word Count: approximately {target_word_count} words
Script Language: Write the ENTIRE script narration in {language}. All narration text must be in {language} — do not mix languages, do not default to English unless {language} is English. (Field names/JSON keys still stay in English as shown in the OUTPUT schema.)


Template: "{selected_template.get('title')}" (cluster: {selected_template.get('cluster')})
Template Purpose: {selected_template.get('about')}

Segments (write the script in this exact order — each entry's guidance is
that segment's llm_brief):
{segments_block}

Source Material (JSON — each knowledge_base_chunks entry may include a "book"
object with title/author/year; each web_chunks entry includes its source
"url". Every chunk also carries "segment_name" and "segment_brief" showing
which template segment it was retrieved for — use that chunk's facts in the
matching part of the narration, guided by that segment's brief. Attribute
facts to their sources naturally in the narration, e.g. "According to
[author]'s [title]..." or "As reported by [domain]..."; never attribute to a
chunk whose "book" is null or whose "url" is empty):
{context_block}
"""

    fallback = {
        "script": "",
        "metrics": dict(_DEFAULT_SCRIPT_METRICS),
        "classification": dict(_DEFAULT_CLASSIFICATION),
    }

    async def _call():
        completion = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": SCRIPT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                temperature=0.6,
                top_p=0.95,
            ),
            timeout=max(OPENAI_CALL_TIMEOUT, 90.0),
        )
        _record_token_usage("generate_script_from_context", completion)
        return (completion.choices[0].message.content or "").strip()

    raw = ""
    try:
        raw = await _call()
    except Exception as e:
        print(f"[SCRIPT] generation failed: {e}")
        return fallback

    parsed = _safe_parse_json(raw)

    if not isinstance(parsed, dict) or not isinstance(parsed.get("script"), str) or not parsed["script"].strip():
        print(f"[SCRIPT] JSON parse/shape failed on first attempt — retrying once. Raw (truncated): {raw[:300]}")
        try:
            raw = await _call()
            parsed = _safe_parse_json(raw)
        except Exception as e:
            print(f"[SCRIPT] retry call failed: {e}")
            parsed = None

    if not isinstance(parsed, dict) or not isinstance(parsed.get("script"), str) or not parsed["script"].strip():
        print(f"[SCRIPT] still no usable script JSON after retry, returning empty. Raw (truncated): {raw[:300]}")
        return fallback

    script_text = parsed["script"].strip()

    raw_metrics = parsed.get("metrics")
    metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    clamped_metrics = dict(_DEFAULT_SCRIPT_METRICS)
    for key in _METRIC_MIN_VALUES:
        clamped_metrics[key] = _clamp_metric_value(key, metrics.get(key))
    for extra_key in ("videoLengthMinutes", "wordCount", "emotionalDepth"):
        if extra_key in metrics:
            clamped_metrics[extra_key] = _clamp_metric_value(extra_key, metrics.get(extra_key)) \
                if extra_key == "emotionalDepth" else metrics.get(extra_key)

    raw_classification = parsed.get("classification")
    classification = raw_classification if isinstance(raw_classification, dict) else {}
    category = classification.get("category")
    category = category.strip() if isinstance(category, str) and category.strip() else _DEFAULT_CLASSIFICATION["category"]

    subcategories_raw = classification.get("subcategories")
    subcategories = []
    if isinstance(subcategories_raw, list):
        seen_lower = set()
        for item in subcategories_raw:
            if not isinstance(item, str):
                continue
            clean_item = item.strip()
            if not clean_item or clean_item.lower() in seen_lower:
                continue
            seen_lower.add(clean_item.lower())
            subcategories.append(clean_item)
            if len(subcategories) >= 5:
                break

    return {
        "script": script_text,
        "metrics": clamped_metrics,
        "classification": {"category": category, "subcategories": subcategories},
    }




YOUTUBE_SEO_SYSTEM_PROMPT = """
## ROLE

You are a YouTube SEO and metadata optimization specialist.

Generate metadata that maximizes discoverability, click-through rate, and semantic relevance while accurately representing the supplied video script.

---

## INPUT

You will receive:

* The complete video script.
* Metadata collected from currently ranking YouTube videos on the same topic, including titles, descriptions, tags, hashtags, and related SEO signals.

Use the reference metadata only to identify semantic keyword clusters, search intent, and audience language.

Never copy or closely paraphrase competitor metadata.

---

## OBJECTIVE

Generate metadata that is:

* faithful to the script
* optimized for YouTube Search, Browse, Suggested Videos, and Recommendations
* naturally keyword-rich without keyword stuffing

Use the reference metadata to infer semantically related terms, entities, concepts, synonyms, and long-tail phrases that help YouTube associate this video with similar high-quality content.

---

## TITLE & THUMBNAIL PAIRS

Generate **5 matching Title–Thumbnail pairs**.

Each pair should communicate one clear value proposition.

The thumbnail should capture the video's central idea in **5–8 highly readable words**.

The corresponding title should naturally extend the thumbnail by adding context or curiosity.

Both should summarize the overall theme of the script and make stronger sense together than independently.

Each pair should present a different angle while remaining faithful to the script.

---

## TITLES

Generate exactly **5 titles**.

Each title must:

* contain **40–70 characters**
* naturally include the primary keyword
* encourage curiosity without clickbait
* accurately represent the script
* avoid emojis and excessive capitalization

---

## DESCRIPTIONS

Generate exactly **3 descriptions**.

Each description must:

* contain **60–100 words**
* open with the primary keyword
* naturally include important semantic keywords
* summarize the video's value
* finish with a subtle call-to-action

---

## HASHTAGS

Generate exactly **3 hashtag sets**.

Each set must contain **8–15 unique hashtags**.

Every hashtag must:

* begin with "#"
* use camelCase for multi-word phrases
* contain no spaces or punctuation besides "#"
* combine broad, niche, and semantic keywords
* contain no duplicates within a set

---

## OUTPUT

Return **only valid JSON**.

```json
{
  "titles": ["...", "...", "...", "...", "..."],
  "descriptions": ["...", "...", "..."],
  "hashtags": [
    ["#...", "#..."],
    ["#...", "#..."],
    ["#...", "#..."]
  ],
  "thumbnail_text": ["...", "...", "...", "...", "..."]
}
```
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
    script_excerpt = script_text
    fallback = _build_fallback_youtube_metadata(request)

    user_prompt = f"""
idea Title: "{request.title}"
idea Description: "{request.description}"

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
                temperature=0.15,   
                top_p=0.85,       

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


def _unique_url_count(articles: list[dict]) -> int:
    return len({a.get("url") for a in articles if a.get("url")})

_MIN_ACCEPTABLE_SIMILARITY = 0.30





SCRIPT_METRICS_SYSTEM_PROMPT = """
You are a content analyst reviewing a finished YouTube documentary script.

## Task
Read the script and count/score the following content elements exactly as
they appear in the script — do not estimate generically, actually look at
what's present in the text.

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
  "generalExamples": <number, >= 1>,
  "proverbs_count": <number, >= 1>,
  "historicalExamples": <number, >= 1>,
  "researchFacts": <number, >= 1>,
  "keywords": ["...", "..."]
}
"""

_METRIC_MIN_VALUES = {
    "generalExamples": 1,
    "proverbs_count": 1,
    "historicalExamples": 1,
    "researchFacts": 1,
}

_DEFAULT_SCRIPT_METRICS = {
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


BOOKS_SUPABASE_TABLE = "Metadata_phase1"
BOOKS_SUPABASE_MD5_COLUMN = "MD5"


def _fetch_books_by_md5_sync(md5_list: list[str]) -> list[dict]:
    if not md5_list:
        return []

    try:
        result = (
            supabase.table(BOOKS_SUPABASE_TABLE)
            .select(f"Title, Author, Year, {BOOKS_SUPABASE_MD5_COLUMN}")
            .in_(BOOKS_SUPABASE_MD5_COLUMN, md5_list)
            .execute()
        )
        rows = result.data or []
        print(f"[SUPABASE] '{BOOKS_SUPABASE_TABLE}' lookup returned {len(rows)} row(s) for {len(md5_list)} md5(s)")
        return rows
    except Exception as e:
        print(f"[SUPABASE] book lookup against '{BOOKS_SUPABASE_TABLE}' failed: {e}")
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
        rows = await _run_io(_fetch_books_by_md5_sync, md5_list)

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

async def _fetch_books(
    all_db_chunks_seen: list,
    all_db_md5s_seen: set,
    topic_text: str,
    combined_hyde_doc: str,
    table_name: str | None,
) -> list[dict]:

    try:
        return await get_books_for_chunks(
            all_db_chunks_seen, topic_text=topic_text, script_text=""
        )
    except Exception as exc:
        print(f"--- MySQL book lookup failed: {exc} ---")
        import traceback
        traceback.print_exc()
        return []



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




FINAL_QC_SYSTEM_PROMPT = """
You are a senior YouTube Content QA Editor performing the FINAL QUALITY CHECK before a video goes into production.

Your role is to review the generated Script, Titles, Descriptions, and Thumbnail Texts exactly as a professional editor would before publication. Identify only genuine issues, make the smallest possible corrections, and preserve everything that is already correct.

---

## Inputs

1. Idea Title
2. Idea Description
3. Generated Script
4. Generated YouTube Titles (5)
5. Generated YouTube Descriptions (5)
6. Generated Thumbnail Texts (5)

---

## Review Checklist

### 1. Idea Alignment

Ensure the script:

- Faithfully delivers the original Idea Title and Idea Description.
- Maintains the intended narrative angle and takeaway.
- Does not drift into unrelated topics.
- Preserves the original scope and intent.

---

### 2. Content Consistency

Ensure that:

- Script, Titles, Descriptions, and Thumbnail Texts are factually consistent.
- Metadata never promises content the script does not deliver.
- No contradictions exist between any generated assets.
- All metadata accurately reflects the final corrected script.

---

### 3. Script Quality

Review the script for:

- Grammar
- Spelling
- Punctuation
- Readability
- Awkward wording
- Repeated words
- Duplicated or truncated sentences
- Garbled text
- Placeholder text
- Broken formatting
- Logical inconsistencies
- Smooth transitions
- Consistent tone

---

### 4. Narration Quality

Read the script as though listening to a professional voice-over.

Ensure that it:

- Flows naturally when spoken aloud.
- Has smooth rhythm and pacing.
- Sounds conversational and human-written.
- Avoids robotic, repetitive, or obviously AI-generated wording.
- Uses natural transitions between ideas.
- Avoids unnecessarily long or difficult-to-read sentences.
- Maintains listener engagement throughout.

---

### 5. Metadata Quality

Validate every Title, Description, and Thumbnail against the following principles.

The metadata must be:

- Faithful to the script.
- Faithful to the approved video idea.
- Optimized for YouTube Search, Browse, Suggested Videos, and Recommendations.
- Naturally keyword-rich without keyword stuffing.
- Using semantically relevant keywords, entities, concepts, synonyms, and long-tail phrases that are genuinely supported by the script.
- Free of misleading claims, unsupported promises, and unfulfilled clickbait.
- Consistent with one another.
- Accurate reflections of the corrected final script.

---

### 6. Title–Thumbnail Pair Validation

Treat each Title and Thumbnail Text as a matching pair.

Verify that every pair:

- Communicates one clear value proposition.
- Shares the same core message.
- Works better together than independently.
- Uses the thumbnail to communicate the main idea.
- Uses the title to add context, curiosity, or value.
- Does not simply repeat the same wording.
- Does not exaggerate or introduce unsupported claims.
- Represents a distinct packaging angle while remaining faithful to the script.

---

### 7. Metadata Constraints

#### Titles

Each title must:

- Be 40–70 characters.
- Be grammatically correct.
- Be compelling without being misleading.
- Naturally include relevant keywords.
- Avoid repetition and keyword stuffing.
- Match the script.

#### Descriptions

Each description must:

- Be 60–100 words.
- Accurately summarize the script.
- Naturally include relevant keywords.
- Use only concepts supported by the script.
- Be grammatically correct.

#### Thumbnail Text

Each thumbnail must:

- Contain 5–8 words.
- Be instantly readable.
- Clearly communicate the primary idea.
- Match its corresponding title.
- Match the script.
- Avoid overpromising.

---

## Correction Rules

- Leave anything that is already correct exactly as-is.
- Make only the minimum edits required.
- Preserve the script's length, structure, pacing, and narrative.
- Never invent new facts, names, statistics, events, claims, or examples.
- Never change the video's core message.
- Never introduce unsupported keywords, entities, or topics.
- Preserve formatting unless it is broken.
- Improve unnatural or AI-generated wording only where necessary.
- If metadata violates any guideline, minimally edit it until it complies.
- If a Title–Thumbnail pair is inconsistent, modify whichever requires fewer changes.
- If multiple metadata assets conflict with the script, make them all consistent with the corrected final script.
- Prioritize factual accuracy over marketing appeal whenever they conflict.

---

## Output

Return ONLY valid JSON.

Do not include explanations.

Do not include markdown.

Do not include code fences.

Return exactly:

{
  "script": "<corrected full script text>",
  "titles": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ],
  "descriptions": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ],
  "thumbnail_text": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ]
}


"""



async def run_final_qc_pass(
    idea_title: str,
    idea_description: str,
    script_text: str,
    youtube_metadata: dict,
) -> dict:
    """
    Final teacher-style QC pass: cross-checks the generated script + YouTube
    metadata against the original idea for consistency and correctness, and
    makes minimal corrections. Never returns the idea title/description —
    only the corrected script + metadata (titles/descriptions/thumbnail_text).
    Hashtags are intentionally left untouched and are not sent to this pass.
    """
    fallback = {
        "script": script_text,
        "titles": youtube_metadata.get("titles", []),
        "descriptions": youtube_metadata.get("descriptions", []),
        "thumbnail_text": youtube_metadata.get("thumbnail_text", []),
    }

    if not script_text:
        print("[QC] no script text to review, skipping final QC pass")
        return fallback

    user_prompt = f"""
Idea Title: "{idea_title}"
Idea Description: "{idea_description}"

Generated Script:
{script_text}

Generated YouTube Titles:
{json.dumps(youtube_metadata.get("titles", []), ensure_ascii=False)}

Generated YouTube Descriptions:
{json.dumps(youtube_metadata.get("descriptions", []), ensure_ascii=False)}

Generated Thumbnail Texts:
{json.dumps(youtube_metadata.get("thumbnail_text", []), ensure_ascii=False)}
"""

    raw = ""
    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": FINAL_QC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                temperature=0.1,
                top_p=0.85,
            ),
            timeout=max(OPENAI_CALL_TIMEOUT, 90.0),
        )
        _record_token_usage("final_qc_pass", res)
        raw = (res.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[QC] final QC call failed: {e} — keeping original script/metadata as-is")
        return fallback

    parsed = _safe_parse_json(raw)
    if not isinstance(parsed, dict):
        print(f"[QC] final QC output was not valid JSON, keeping originals. Raw (truncated): {raw[:500]}")
        return fallback

    corrected_script = parsed.get("script")
    if not isinstance(corrected_script, str) or not corrected_script.strip():
        print("[QC] corrected script missing/empty in QC output, keeping original script")
        corrected_script = script_text

    original_len = _word_count(script_text)
    corrected_len = _word_count(corrected_script)
    if original_len > 0 and corrected_len < original_len * 0.6:
        print(
            f"[QC] corrected script looks truncated ({corrected_len} words vs "
            f"original {original_len}) — keeping original script instead"
        )
        corrected_script = script_text

    def _corrected_list(key: str, original_list: list, expected_len: int = 3) -> list:
        values = parsed.get(key)
        if not isinstance(values, list) or len(values) == 0:
            print(f"[QC] '{key}' missing/empty in QC output, keeping original")
            return original_list
        cleaned = [v for v in values if isinstance(v, str) and v.strip()]
        if len(cleaned) < expected_len:
            print(f"[QC] '{key}' only had {len(cleaned)}/{expected_len} usable entries, padding from original")
            i = 0
            while len(cleaned) < expected_len and i < len(original_list):
                if original_list[i] not in cleaned:
                    cleaned.append(original_list[i])
                i += 1
        return cleaned or original_list

    return {
        "script": corrected_script,
        "titles": _corrected_list("titles", youtube_metadata.get("titles", [])),
        "descriptions": _corrected_list("descriptions", youtube_metadata.get("descriptions", [])),
        "thumbnail_text": _corrected_list("thumbnail_text", youtube_metadata.get("thumbnail_text", [])),
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




# SCRIPT_RAG_POOL_PER_DOC = 40 
# SCRIPT_TOP_K_PER_DOC = 2       


def pick_topk_with_backfill(
    ranked_pool: list[dict],
    top_k: int,
    globally_claimed_md5s: set,
) -> list[dict]:
    """
    ranked_pool: this segment's candidates, already sorted by combined_score (RRF) desc.
    globally_claimed_md5s: set shared across ALL segments in this script request,
    mutated in place so a later segment can't re-claim a chunk already picked.
    Returns up to top_k unclaimed chunks — falls short of top_k only when this
    segment's own pool is genuinely exhausted of unclaimed candidates.
    """
    picked = []
    for chunk in ranked_pool:
        key = chunk.get("md5") or chunk.get("content")
        if not key or key in globally_claimed_md5s:
            continue
        picked.append(chunk)
        globally_claimed_md5s.add(key)
        if len(picked) >= top_k:
            break
    return picked


def _tag_chunks_with_segment(chunks: list[dict], seg_info: dict) -> None:
    """
    Stamps chunk provenance in place: which HyDE document / template segment
    this chunk was retrieved for. Both hyde_brief and llm_brief are carried
    on the chunk at this stage (retrieval-time), so downstream code can
    choose which one to expose to which consumer:
      - hyde_brief -> retrieval-only, never sent to the script-writing LLM
      - llm_brief  -> the one actually sent to the script-writing LLM
    Works identically for RAG DB chunks and web-scraped chunks.
    """
    segment_id = seg_info.get("segment_index")
    segment_name = seg_info.get("segment_name")
    hyde_brief = seg_info.get("hyde_brief")
    llm_brief = seg_info.get("llm_brief")
    for chunk in chunks:
        chunk["segment_id"] = segment_id
        chunk["segment_name"] = segment_name
        chunk["hyde_brief"] = hyde_brief
        chunk["llm_brief"] = llm_brief







async def _generate_script_impl(request: "ScriptRequest"):
    _start_token_tracking()

    total_start_time = time.time()
    topic_text = build_topic_text(request)
    print(f"[SCRIPT] ===== NEW REQUEST ===== title='{request.title}' time={request.time}min userId={request.userId}")

    # Stage 4 (YouTube) has no dependency on Stage 1's HyDE output at all —
    # kick it off immediately so it runs fully in the background.
    scraped_urls = set()
    stage4_task = asyncio.create_task(
        get_youtube_context(request.title, request.description, scraped_urls)
    )

    selected_template = await retrieve_best_script_template(topic_text)

    if selected_template is None:
        print("[SCRIPT] no template matched via embedding search — proceeding with an empty structure")
        selected_template = {
            "key": None, "title": None, "cluster": None, "about": None,
            "best_fit_categories": [], "human_texture_tier": None,
            "segments": [], "template_text": "", "similarity": None,
        }
    else:
        print(
            f"[SCRIPT] template matched: key='{selected_template.get('key')}' "
            f"title='{selected_template.get('title')}' "
            f"similarity={selected_template.get('similarity')}"
        )

    category = selected_template.get("cluster") or (
        (selected_template.get("best_fit_categories") or ["UNKNOWN"])[0]
    )
    print(f"[SCRIPT] category (from selected template): {category}")

    segments = selected_template.get("segments", [])
    print(f"[SCRIPT] template has {len(segments)} segment(s)")

    try:
        channel_profile = await get_channel_profile(request.userId)
        summary = channel_profile[0]["Summary"] if channel_profile else None
        print(f"[SCRIPT] channel profile fetched (summary present: {bool(summary)})")
    except Exception as exc:
        print(f"[SCRIPT] error fetching channel profile: {exc}")
        summary = None

    hyde_documents: list[dict] = []
    db_results: list = []
    all_db_chunks_seen: list = []
    all_db_md5s_seen: set = set()
    new_articles: list = []
    new_videos: list = []
    script_text = ""
    youtube_metadata = {"titles": [], "descriptions": [], "hashtags": [], "thumbnail_text": []}
    script_metrics = dict(_DEFAULT_SCRIPT_METRICS)
    sources: list[str] = []
    books: list[dict] = []
    table_name = None
    classification = dict(_DEFAULT_CLASSIFICATION)

    # =========================================================================
    # STAGE 1 — HyDE document + keyword generation. Unchanged: nothing else
    # that depends on it may start until this produces usable output.
    # =========================================================================
    stage1_start = time.time()
    print("\n" + "=" * 90)
    print("[STAGE 1] Generating HyDE documents + keywords...")
    print("=" * 90)
    try:
        hyde_documents = await generate_hyde_docs_for_script(
            request.title,
            request.description,
            selected_template,
            segments,
        )
    except Exception as exc:
        print(f"[STAGE 1] FAILED — HyDE generation raised: {exc}")
        import traceback
        traceback.print_exc()
        hyde_documents = []

    combined_hyde_doc = "\n\n".join(
        d.get("hyde_document", "") for d in hyde_documents if d.get("hyde_document")
    ) or topic_text
    hyde_docs_usable = any((d.get("hyde_document") or "").strip() for d in hyde_documents)

    if not hyde_docs_usable:
        print("[STAGE 1] ABORT — no usable HyDE document was produced, aborting pipeline before Stage 2")
        stage4_task.cancel()
        return {
            "error": "Failed to generate HyDE documents required for retrieval.",
            "token_usage": _get_token_usage_summary(),
        }

    print(f"[STAGE 1] done in {time.time() - stage1_start:.2f}s — {len(hyde_documents)} HyDE segment(s) generated")
    for i, seg in enumerate(hyde_documents, start=1):
        doc_preview = (seg.get("hyde_document") or "").strip().replace("\n", " ")
        kw = seg.get("keywords") or []
        print(f"  [HYDE-{i}] ({_count_tokens(doc_preview)} tok, {len(kw)} kw) \"{doc_preview}\"")

    script_rag_target = len(hyde_documents) * SCRIPT_TOP_K_PER_DOC
    script_web_source_target = len(hyde_documents) * SCRIPT_TOP_K_PER_DOC

    async def _run_stage2() -> tuple[str, list]:
        stage2_start = time.time()
        print("\n" + "=" * 90)
        print("[STAGE 2] Selecting category and retrieving RAG chunks (dense + sparse, thresholded, RRF-fused)...")
        print("=" * 90)
        try:
            tbl = await select_table_for_topic(topic_text)
        except Exception as exc:
            print(f"[STAGE 2] category selection failed, defaulting to '{RAG_CATEGORIES[0]}': {exc}")
            tbl = RAG_CATEGORIES[0]

        print(
            f"[STAGE 2] category='{tbl}' | {len(hyde_documents)} segment(s) x "
            f"(dense-{SCRIPT_RAG_POOL_PER_DOC}@thr{DENSE_SCORE_THRESHOLD} + "
            f"sparse-{SCRIPT_RAG_POOL_PER_DOC}@thr{SPARSE_SCORE_THRESHOLD}) -> RRF fuse -> "
            f"top {SCRIPT_TOP_K_PER_DOC} pick per segment, target {script_rag_target} chunk(s) total"
        )

        try:
            per_doc = await asyncio.gather(
                *[
                    get_context_from_db_segment_with_timeout(
                        hyde_document=seg.get("hyde_document", ""),
                        keywords=seg.get("keywords", []),
                        table_name=tbl,
                        dense_k=SCRIPT_RAG_POOL_PER_DOC,
                        sparse_k=SCRIPT_RAG_POOL_PER_DOC,
                    )
                    for seg in hyde_documents
                ]
            )
        except Exception as exc:
            print(f"[STAGE 2] FAILED — DB retrieval raised: {exc}")
            import traceback
            traceback.print_exc()
            per_doc = []

        print(f"[STAGE 2] done in {time.time() - stage2_start:.2f}s")
        return tbl, per_doc

    async def _run_stage3() -> list:
        stage3_start = time.time()
        print("\n" + "=" * 90)
        print("[STAGE 3] Web search: keyword generation -> fetch -> match against HyDE docs")
        print("=" * 90)
        try:
            script_search_keywords = await _generate_search_keywords_for_script(
                request.title, request.description, selected_template, request.time
            )
        except Exception as exc:
            print(f"[STAGE 3] keyword generation failed: {exc}")
            script_search_keywords = [f"{request.title} latest news today", f"{request.title} 2026 update"]

        shared_pool: list[dict] = []
        try:
            shared_pool = await build_shared_web_pool(script_search_keywords, scraped_urls)
        except Exception as exc:
            print(f"[STAGE 3] shared web pool build failed: {exc}")
            shared_pool = []

        articles: list[dict] = []
        seen_urls_final: set = set()
        try:
            model = _get_st_model()
            hyde_texts = [seg.get("hyde_document", "") for seg in hyde_documents]
            try:
                hyde_embeddings_batch = await _run_encode(
                    lambda: model.encode(hyde_texts, normalize_embeddings=True, convert_to_numpy=True)
                )
            except Exception as e:
                print(f"[STAGE 3] batched hyde embedding failed, falling back to per-doc: {e}")
                hyde_embeddings_batch = None

            for doc_idx, seg in enumerate(hyde_documents, start=1):
                hyde_embedding = (
                    hyde_embeddings_batch[doc_idx - 1] if hyde_embeddings_batch is not None
                    else await _run_encode(
                        lambda d=seg.get("hyde_document", ""): model.encode(d, normalize_embeddings=True, convert_to_numpy=True)
                    )
                )
                top_for_doc = rank_pool_for_hyde_doc(
                    shared_pool, hyde_embedding, WEB_CONTENT_SIMILARITY_THRESHOLD, SCRIPT_TOP_K_PER_DOC,
                    exclude_urls=seen_urls_final,
                )
                _tag_chunks_with_segment(top_for_doc, seg)
                for article in top_for_doc:
                    url = article.get("url")
                    if url and url not in seen_urls_final:
                        seen_urls_final.add(url)
                        articles.append(article)

            print(f"[STAGE 3] direct matching done: {_unique_url_count(articles)}/{script_web_source_target} unique source(s)")

            # --- NEW: backfill, mirrors /generate-ideas ---
            if _unique_url_count(articles) < script_web_source_target:
                print(f"[STAGE 3] backfilling — only {_unique_url_count(articles)}/{script_web_source_target}, widening search")
                generic_queries = [
                    request.title, f"{request.title} history", f"{request.title} overview",
                    f"{request.title} explained", f"{request.title} background",
                    f"{request.title} facts", f"{request.title} details", f"{request.title} analysis",
                ]
                try:
                    extra_pool = await build_shared_web_pool(generic_queries, scraped_urls)
                    combined_pool = shared_pool + extra_pool
                    for seg in hyde_documents:
                        if _unique_url_count(articles) >= script_web_source_target:
                            break
                        doc = seg.get("hyde_document", "")
                        hyde_embedding = await _run_encode(
                            lambda d=doc: model.encode(d, normalize_embeddings=True, convert_to_numpy=True)
                        )
                        backfilled = rank_pool_for_hyde_doc(
                            combined_pool, hyde_embedding, _MIN_ACCEPTABLE_SIMILARITY,
                            SCRIPT_TOP_K_PER_DOC, exclude_urls=seen_urls_final,
                        )
                        _tag_chunks_with_segment(backfilled, seg)
                        for article in backfilled:
                            if _unique_url_count(articles) >= script_web_source_target:
                                break
                            url = article.get("url")
                            if url and url not in seen_urls_final:
                                seen_urls_final.add(url)
                                articles.append(article)
                    print(f"[STAGE 3] backfill done — now {_unique_url_count(articles)}/{script_web_source_target}")
                except Exception as exc:
                    print(f"[STAGE 3] backfill failed: {exc}")

            articles.sort(key=lambda a: a.get("similarity", 0.0), reverse=True)
            articles = articles[:script_web_source_target]
        except Exception as exc:
            print(f"[STAGE 3] FAILED — web content matching raised: {exc}")
            import traceback
            traceback.print_exc()

        print(f"[STAGE 3] done in {time.time() - stage3_start:.2f}s — final unique source count: {_unique_url_count(articles)}/{script_web_source_target}")
        return articles  

    (table_name, db_results_per_doc), new_articles = await asyncio.gather(
        _run_stage2(), _run_stage3()
    )

    # Same pick-with-backfill logic as before, just executed after both
    # stages above have returned instead of gating Stage 3 on Stage 2.
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
    globally_claimed_md5s = set()

    print(f"\n{'-' * 90}")
    print(f"[STAGE 2] per-segment pool -> pick-with-backfill (top-{SCRIPT_TOP_K_PER_DOC})")
    print(f"{'-' * 90}")

    for doc_idx, doc_results in enumerate(db_results_per_doc, start=1):
        hyde_full = (hyde_documents[doc_idx - 1].get("hyde_document") or "").strip().replace("\n", " ")
        kw_full = hyde_documents[doc_idx - 1].get("keywords") or []
        print(f"\n[STAGE 2 | SEGMENT #{doc_idx}] HyDE: \"{hyde_full}\"")
        print(f"  keywords: {kw_full}")
        print(f"  pooled candidates after threshold+RRF: {len(doc_results)}")

        top_for_doc = pick_topk_with_backfill(
            doc_results, SCRIPT_TOP_K_PER_DOC, globally_claimed_md5s
        )
        _tag_chunks_with_segment(top_for_doc, hyde_documents[doc_idx - 1])
        db_results.extend(top_for_doc)

        picked_ids = {item.get("chunk_id") for item in top_for_doc}
        for rank, item in enumerate(doc_results, start=1):
            marker = "→ PICKED" if item.get("chunk_id") in picked_ids else ""
            print(
                f"    [{rank}] rrf={item.get('combined_score'):.5f} "
                f"via={item.get('matched_via')} dense={item.get('dense_score')} "
                f"sparse={item.get('sparse_score')} md5={item.get('md5')} {marker}"
            )

        if not top_for_doc:
            print(
                f"  RESULT: 0/{SCRIPT_TOP_K_PER_DOC} picked — this segment's entire pool "
                f"was already claimed by earlier segments (genuinely exhausted, not a bug)"
            )
        else:
            print(f"  RESULT: {len(top_for_doc)}/{SCRIPT_TOP_K_PER_DOC} picked (globally unique, backfilled within segment)")
            for rank, item in enumerate(top_for_doc, start=1):
                content_full = (item.get("content") or "").strip().replace("\n", " ")
                print(f"    #{rank} rrf={item.get('combined_score'):.5f} md5={item.get('md5')} \"{content_full}\"")

    print(f"\n{'-' * 90}")
    print(
        f"[STAGE 2] TOTAL: {len(db_results)}/{script_rag_target} chunk(s) selected for script generation "
        f"({len(all_db_chunks_seen)} unique chunk(s) seen across all pools, kept for book lookups)"
    )
    print(f"{'-' * 90}\n")

    if len(db_results) < script_rag_target:
        print(
            f"[STAGE 2] NOTE: {len(db_results)}/{script_rag_target} chunk(s) — some segment pools "
            f"were exhausted of unclaimed candidates after thresholding; no low-relevance filler was added."
        )

    # =========================================================================
    # STAGE 5 — Book metadata lookup only depends on Stage 2's chunks. Kick
    # it off in the background now; it runs concurrently with Stage 6/7 and
    # is collected right before the final response is built.
    # =========================================================================
    stage5_start = time.time()
    print("\n" + "=" * 90)
    print("[STAGE 5] Book metadata lookup (running in background alongside Stage 6/7)")
    print("=" * 90)
    stage5_task = asyncio.create_task(
        _fetch_books(all_db_chunks_seen, all_db_md5s_seen, topic_text, combined_hyde_doc, table_name)
    )

    # =========================================================================
    # STAGE 6 — Script generation needs only Stage 2 + Stage 3 output, not
    # Stage 4 (YouTube) or Stage 5 (books) — it no longer waits on either.
    # =========================================================================
    stage6_start = time.time()
    print("\n" + "=" * 90)
    print("[STAGE 6] Script generation")
    print("=" * 90)
    try:
        target_word_count = target_word_count_for_time(request.time)
        print(f"[STAGE 6] target word count: {target_word_count} (±3%) for {request.time} minute(s)")
        script_result = await generate_script_from_context(
            request, selected_template, db_results, new_articles, target_word_count
        )
        script_text = script_result["script"]
        script_metrics = script_result["metrics"]
        classification = script_result["classification"]
        actual_words = _word_count(script_text)
        deviation_pct = (
            abs(actual_words - target_word_count) / target_word_count * 100
            if target_word_count else 0
        )
        print(
            f"[STAGE 6] done in {time.time() - stage6_start:.2f}s — {actual_words} word(s) generated "
            f"(target {target_word_count}, deviation {deviation_pct:.1f}%)"
        )
        print(f"[STAGE 6] classification: category='{classification.get('category')}' subcategories={classification.get('subcategories')}")
    except Exception as exc:
        print(f"[STAGE 6] FAILED — script generation raised: {exc}")
        import traceback
        traceback.print_exc()
        script_metrics = dict(_DEFAULT_SCRIPT_METRICS)
        classification = dict(_DEFAULT_CLASSIFICATION)

    # =========================================================================
    # STAGE 7 — needs Stage 4 (YouTube) + Stage 6 (finished script). Stage 4
    # has been running in the background since the top of this function, so
    # we just await its already-in-flight result here.
    # =========================================================================
    stage7_start = time.time()
    print("\n" + "=" * 90)
    print("[STAGE 7] YouTube SEO metadata generation")
    print("=" * 90)
    try:
        try:
            new_videos = await stage4_task
            print(f"[STAGE 4] YouTube search resolved — {len(new_videos)} video(s) found")
            for i, v in enumerate(new_videos, start=1):
                print(f"    [YT-{i}] views={v.get('view_count')} {v.get('title')} ({v.get('url')})")
        except Exception as exc:
            print(f"[STAGE 4] FAILED — YouTube search raised: {exc}")
            new_videos = []

        youtube_metadata = await generate_youtube_seo_metadata(request, script_text, new_videos)
        print(
            f"[STAGE 7] done in {time.time() - stage7_start:.2f}s — "
            f"{len(youtube_metadata.get('titles', []))} title(s), "
            f"{len(youtube_metadata.get('descriptions', []))} description(s), "
            f"{len(youtube_metadata.get('hashtags', []))} hashtag set(s), "
            f"{len(youtube_metadata.get('thumbnail_text', []))} thumbnail text(s)"
        )
    except Exception as exc:
        print(f"[STAGE 7] FAILED — YouTube SEO metadata generation raised: {exc}")
        import traceback
        traceback.print_exc()
        youtube_metadata = _build_fallback_youtube_metadata(request)

    # =========================================================================
    # STAGE 8 — Final QC pass. Unchanged, runs after Stage 7.
    # =========================================================================
    stage8_start = time.time()
    print("\n" + "=" * 90)
    print("[STAGE 8] Final QC pass (script + YouTube metadata cross-check)")
    print("=" * 90)
    try:
        pre_qc_words = _word_count(script_text)
        qc_result = await run_final_qc_pass(
            idea_title=request.title,
            idea_description=request.description,
            script_text=script_text,
            youtube_metadata=youtube_metadata,
        )
        script_text = qc_result["script"]
        youtube_metadata["titles"] = qc_result["titles"]
        youtube_metadata["descriptions"] = qc_result["descriptions"]
        youtube_metadata["thumbnail_text"] = qc_result["thumbnail_text"]
        post_qc_words = _word_count(script_text)
        print(
            f"[STAGE 8] done in {time.time() - stage8_start:.2f}s — "
            f"script words {pre_qc_words} -> {post_qc_words} after QC"
        )
    except Exception as exc:
        print(f"[STAGE 8] FAILED — final QC pass raised: {exc} — keeping pre-QC script/metadata")
        import traceback
        traceback.print_exc()

    # Collect Stage 5 (books) — it's had the entire Stage6+7+8 window to
    # finish in the background, so this should rarely actually block.
    try:
        books = await stage5_task
        print(f"[STAGE 5] done in {time.time() - stage5_start:.2f}s (background) — {len(books)} book(s) resolved")
        for i, b in enumerate(books, start=1):
            print(f"    [BOOK-{i}] \"{b.get('title')}\" by {b.get('author')} ({b.get('year')})")
    except Exception as exc:
        print(f"[STAGE 5] FAILED — book lookup raised: {exc}")
        import traceback
        traceback.print_exc()
        books = []

    sources = _extract_source_links(new_articles)
    structure = _build_structure_response(selected_template)
    total_words = _word_count(script_text) if script_text else 0
    token_usage = _get_token_usage_summary()

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 90)
    print(f"[SCRIPT] ===== REQUEST COMPLETE ===== total time: {total_elapsed:.2f}s")
    print(
        f"[SCRIPT] final: {total_words} word(s) | {len(sources)} source(s) | {len(books)} book(s) | "
        f"category='{classification.get('category', 'UNKNOWN')}'"
    )
    print(
        f"[TOKENS] /generate-script total — input: {token_usage['total_input_tokens']}, "
        f"output: {token_usage['total_output_tokens']}, total: {token_usage['total_tokens']} "
        f"across {len(token_usage['calls'])} LLM call(s)"
    )
    print("=" * 90 + "\n")

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
            "videoLength": request.time,
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
















from openai import APITimeoutError

THUMBNAIL_CREDITS_PER_IMAGE = 10

FACE_THUMBNAILS_TABLE = "user_profiles"
FACE_PHOTO_DEFAULT_KEY = "photo1"

# gpt-image-2 routinely takes well over the SDK's default timeout to render,
# especially at higher quality/size. Give image calls their own long timeout
# and a single retry before giving up.
IMAGE_GEN_TIMEOUT = 180.0  # seconds
IMAGE_GEN_MAX_RETRIES = 1  # one retry on timeout before giving up


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


FACE_IMAGE_MAX_DIMENSION = 1536


def _normalize_face_image_bytes(raw_bytes: bytes) -> bytes | None:
    try:
        from PIL import Image
    except ImportError:
        print("[FACE] Pillow (PIL) is not installed — cannot normalize face image. `pip install Pillow`.")
        return None

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
    except Exception as e:
        print(f"[FACE] downloaded photo is not a decodable image (corrupt/unsupported format): {e}")
        return None

    try:
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        if max(img.size) > FACE_IMAGE_MAX_DIMENSION:
            img.thumbnail((FACE_IMAGE_MAX_DIMENSION, FACE_IMAGE_MAX_DIMENSION), Image.LANCZOS)

        out = io.BytesIO()
        img.save(out, format="PNG")
        normalized = out.getvalue()
        print(f"[FACE] normalized face image to clean RGB PNG ({len(normalized)} bytes, {img.size[0]}x{img.size[1]})")
        return normalized
    except Exception as e:
        print(f"[FACE] failed to normalize/re-encode face image: {e}")
        return None


async def get_user_face_photo_bytes(user_id: str, photo_key: str = FACE_PHOTO_DEFAULT_KEY) -> bytes | None:
    photo_url = await get_user_face_photo_url(user_id, photo_key=photo_key)
    if not photo_url:
        return None

    photo_bytes = await asyncio.to_thread(_download_image_bytes_sync, photo_url)
    if not photo_bytes:
        print(f"[FACE] could not download usable photo bytes for user {user_id} from {photo_url}")
        return None

    normalized_bytes = await asyncio.to_thread(_normalize_face_image_bytes, photo_bytes)
    if not normalized_bytes:
        print(
            f"[FACE] photo for user {user_id} downloaded but could not be normalized into a "
            f"valid image — falling back to face-less thumbnail"
        )
        return None

    return normalized_bytes


PROMPT_GENERATOR_SYSTEM_PROMPT = """

# STORYBIT THUMBNAIL PROMPT GENERATOR — GPT IMAGE 2

## ROLE

You are the **Storybit Thumbnail Prompt Generator**. Convert `video_title`, `thumbnail_text`, `script`, and `user_image_present` into ONE production-ready natural-language prompt for GPT Image 2.

Understand the story first, design the thumbnail around its strongest visual hook, then optionally add the user's image as a presenter overlay. Do not ask questions, expose reasoning, output JSON, or provide alternatives. Return ONLY the final image-generation prompt.

## INPUT

1. `video_title`
2. `thumbnail_text` — MUST appear exactly as supplied; never rewrite, shorten, translate, correct, or paraphrase.
3. `script` — authoritative source for story facts.
4. `user_image_present` — true/false.

Ground all story visuals in the script. Never invent unsupported people, events, locations, objects, historical details, or outcomes.

# CORE OBJECTIVE

Create a cinematic documentary-style YouTube thumbnail optimized for mobile readability, curiosity, emotional impact, realism, strong hierarchy, and controlled visual density.

Silently determine:

* core story and strongest hook
* ONE primary story visual
* ONE dominant story emotion
* supporting story elements
* composition
* text-safe area
* color palette
* camera and lighting

**The STORY is always the primary subject. The presenter is never the primary story element.**

# THREE STORY LAYERS

### LAYER 1 — BACKGROUND

Define the exact environment, location, era, architecture/geography, distant objects, sky/weather, atmosphere, and contextual details needed to establish the story.

Keep this layer subordinate, clean, and spatially coherent. Specify believable scale, orientation, perspective, spacing, and depth.

### LAYER 2 — MIDGROUND

Define only necessary supporting people, objects, structures, vehicles, machinery, environmental events, or other factual elements that explain the story.

Specify position, orientation, relative scale, aspect ratio, spacing, depth, and relationship to Layers 1 and 3.

Do not add decorative or unrelated elements.

### LAYER 3 — PRIMARY STORY FOREGROUND

This is the **dominant visual layer**.

Define the main story subject, event, or object and its identity, appearance, action, pose, expression, gaze, orientation, scale, position, interaction, and relationship to camera.

Layer 3 MUST remain the primary visual focus whether or not a presenter image exists.

Design the story composition, camera viewpoint, lighting, environment, and supporting elements around Layer 3 — never around the presenter.

# LAYER 4 — OPTIONAL PRESENTER IMAGE

If `user_image_present = true`, add the supplied user image as a **separate fourth compositing layer above Layers 1–3**.

The presenter is NOT a story subject, NOT the primary focal subject, and NOT part of the physical story scene. It represents the video's presenter only.

Do NOT redesign the story around the presenter. Do NOT make the presenter the visual center. Do NOT use the presenter to replace or compete with the Layer 3 story subject.

Layers 1–3 must first form a complete, independently understandable story thumbnail. Layer 4 is then added as a secondary presenter overlay.

## PRESENTER POSITION AND SIZE

Place the presenter **ALWAYS on the RIGHT side of the 1280×720 canvas**.

Target presenter dimensions:

* **height: 550–600 px**
* **width: 450–500 px**

Use natural head-to-chest or upper-body framing. Maintain appropriate margins and keep the complete presenter visually coherent within the frame.

Position the presenter so that the primary story subject and essential story elements remain unobstructed.

The presenter must remain a secondary visual element even though Layer 4 is the topmost compositing layer.

## PRESENTER IDENTITY

Preserve the user's identity exactly:

* facial structure and proportions
* eyes and eyebrows
* nose and lips
* cheeks, jaw and chin
* ears and hairline
* hairstyle
* approximate age
* natural skin tone and texture
* distinctive facial characteristics

Do not beautify, reshape, whiten, darken, age, de-age, stylize, morph, duplicate, replace, or distort the identity.

## PRESENTER CAMERA GAZE

The presenter must **ALWAYS look directly into the camera/viewer**.

Maintain clear, natural eye contact with the camera regardless of the story scene, camera angle, or surrounding composition.

Do not make the presenter look toward the story subject, look sideways, look away, look upward/downward, or gaze into the environment.

The face should be oriented naturally toward the viewer with both eyes clearly visible and a believable front-facing presentation pose.

## STORY-DRIVEN PRESENTER EMOTION

Determine the **core emotional state of the story first**.

Apply that emotion naturally to the presenter's:

* facial expression
* eyes
* eyebrows
* mouth
* subtle facial tension
* head position
* body posture

The presenter must communicate the specified story emotion while maintaining direct eye contact with the camera.

Examples:

* danger → controlled fear, concern, alertness
* mystery → curiosity, suspicion, uncertainty
* tragedy → sadness, shock, disbelief
* success → confidence, excitement, satisfaction
* betrayal → disbelief, restrained anger, suspicion
* discovery → astonishment, curiosity, realization

Never exaggerate the emotion into a theatrical, cartoonish, artificial, or meme-like expression.

The presenter reacts to the story emotionally but does **not become a character inside the story**.

## PRESENTER SEPARATION

Layer 4 must remain visually separate from Layers 1–3.

Never merge, morph, duplicate, fuse, or physically integrate the presenter with story characters or objects.

Do not allow story objects to incorrectly pass through, attach to, or intersect the presenter.

Maintain clean edges, believable proportions, correct occlusion, and intentional spacing.

## TEXT PROTECTION

Thumbnail text has higher compositional priority than Layer 4.

Reserve a dedicated text-safe region before positioning the presenter.

The presenter MUST NOT cover, overlap, obscure, intersect, or sit in front of any thumbnail text.

Do not place text behind the presenter's face, head, hair, body, or shoulders.

If necessary, reposition or reduce the presenter rather than compromising text readability or the primary story composition.

# IF USER IMAGE IS FALSE

Do not create a presenter substitute.

Use the complete canvas for Layers 1–3 and thumbnail text.

The primary story subject remains Layer 3.

Do not invent an additional presenter or foreground character.

# SPATIAL AND DESIGN ACCURACY

Treat the thumbnail as a deliberate graphic composition.

Explicitly control:

* left/right/top/bottom placement
* object orientation and facing direction
* alignment
* relative scale
* aspect ratios
* spacing and margins
* depth
* occlusion
* layer separation
* negative space
* text-safe regions

Objects must maintain correct proportions, perspective, orientation, and aspect ratios.

Prevent accidental overlaps, incorrect occlusion, objects passing through one another, merged objects, duplicated objects, floating objects, stretched or mirrored objects, detached body parts, impossible spatial relationships, and unwanted tangencies.

Do not add extra objects simply to fill empty space.

# THUMBNAIL TEXT

Render `thumbnail_text` **EXACTLY** as supplied.

Specify:

* exact wording
* font style
* font weight
* approximate size
* line arrangement
* alignment
* color
* contrast
* spacing
* placement

Default typography: **bold condensed sans-serif, heavy weight, large display lettering, approximately 70–110 px visual height depending on wording, high contrast, clean spacing, mobile-readable**.

Place text in the strongest available negative-space region while preserving the Layer 3 story subject.

When the presenter exists, place text in a dedicated area **outside the presenter's RIGHT-side footprint**, preferably LEFT or central-left depending on the story composition.

Never place text over the presenter's face or body.

No additional text, captions, subtitles, dates, numbers, logos, watermarks, UI elements, or decorative typography.

# COLOR, CAMERA & LIGHTING

Define a controlled color palette, dominant colors, accent colors, saturation, contrast, and warm/cool balance.

Use color to establish clear hierarchy between Layers 1–4 and reinforce the story emotion.

Specify useful framing, camera angle, perspective, focal-length appearance, focus priority, and depth of field.

Define light source, direction, intensity, shadows, highlights, contrast, and atmosphere.

Lighting must emphasize Layer 3 while keeping the presenter naturally visible and secondary.

# STYLE & ACCURACY

Default to **cinematic documentary realism, photorealistic, physically believable, natural anatomy, natural skin tones, authentic materials, realistic environmental detail**.

For historical stories, maintain period-appropriate clothing, architecture, technology, vehicles, objects, materials, and environments. Never invent unsupported details.

# EXCLUSIONS

Prevent the presenter from becoming the primary subject; presenter looking away from camera; side gaze; closed or obscured eyes; exaggerated expression; extra characters; duplicate subjects; incorrect object orientation; wrong aspect ratios; stretched or mirrored objects; accidental overlaps; impossible occlusion; merged objects; floating objects; distorted anatomy; identity drift; face morphing; unnatural skin; clutter; excessive blur; oversaturation; fake or misspelled text; additional text; logos; watermarks; borders; frames; split screens; collages; stereotypes; and unsupported facts.

# OUTPUT

Write ONE continuous natural-language GPT Image 2 prompt covering:

story hook and emotion, Layers 1–3, primary story subject, optional Layer 4 presenter overlay, presenter position/size/identity/expression/direct camera gaze, exact thumbnail text and typography, spatial relationships, camera, lighting, color palette, realism, and exclusions.

The final image must be **1280×720 pixels, 16:9**, professionally composed for YouTube thumbnail viewing.

# FINAL VALIDATION

Silently verify:

* story is the primary focus
* Layer 3 is dominant
* Layers 1–3 independently communicate the story
* Layer 4 exists only when user image is present
* presenter is always on the RIGHT
* presenter is approximately 550–600 px high and 450–500 px wide
* presenter is a secondary presenter overlay, never the story subject
* presenter always looks directly into the camera
* presenter expression naturally matches the story's core emotion
* user's identity and natural skin are preserved
* presenter does not cover or overlap thumbnail text
* object orientation, scale, aspect ratio, alignment and spacing are coherent
* no accidental overlaps or extra objects
* color, lighting and composition prioritize the story
* image is 1280×720 / 16:9
* output contains ONLY the final image-generation prompt

Return only the final image-generation prompt as plain text.
"""


def _safe_parse_json(raw: str) -> dict | None:
    if not raw:
        return None

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except Exception as e:
        print(f"[JSON-PARSE] failed to parse model output as JSON: {e}")
        return None


async def run_prompt_generator(
    title: str,
    thumbnail_text: str,
    script_text: str,
    user_image_present: bool,
) -> str:
    script_content = script_text if script_text else "No script available."

    user_content = f"""Video Title: "{title}"
Thumbnail Text: "{thumbnail_text}"
User Image Present: {"true" if user_image_present else "false"}

Complete Video Script:
{script_content}
"""

    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": PROMPT_GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                stream=False,
            )
        )
        _record_token_usage("thumbnail_prompt_generator", res)
        rendered_prompt = (res.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[PROMPT-GEN] Step 1 (Prompt Generator) call failed: {e}")
        return ""

    print(f"[PROMPT-GEN] Step 1 (Prompt Generator) output: {rendered_prompt}")
    return rendered_prompt


def _pick_thumbnail_text(thumbnail_text: str | None, request) -> str:
        return thumbnail_text.strip()

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


def _call_images_generate_with_timeout(prompt: str, size: str, quality: str):
    """images.generate with an extended per-call timeout (gpt-image-2 is slow)."""
    return openai_client.with_options(timeout=IMAGE_GEN_TIMEOUT).images.generate(
        model=GPT_IMAGE_MODEL,
        prompt=prompt,
        size=size,
        quality=quality,
        n=1,
    )


def _call_images_edit_with_timeout(face_file, prompt: str, size: str, quality: str):
    """images.edit with an extended per-call timeout (gpt-image-2 is slow)."""
    return openai_client.with_options(timeout=IMAGE_GEN_TIMEOUT).images.edit(
        model=GPT_IMAGE_MODEL,
        image=face_file,
        prompt=prompt,
        size=size,
        quality=quality,
    )


def _call_with_timeout_retry(fn, *, label: str, max_retries: int = IMAGE_GEN_MAX_RETRIES):
    """
    Run fn() (a zero-arg callable making the actual API call). On APITimeoutError,
    retry up to max_retries times before finally raising. Any non-timeout exception
    is raised immediately without retrying.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except APITimeoutError as e:
            last_exc = e
            if attempt < max_retries:
                print(
                    f"[THUMBNAIL-GPT] {label} timed out after {IMAGE_GEN_TIMEOUT}s "
                    f"(attempt {attempt + 1}/{max_retries + 1}) — retrying"
                )
            else:
                print(
                    f"[THUMBNAIL-GPT] {label} timed out after {IMAGE_GEN_TIMEOUT}s "
                    f"(attempt {attempt + 1}/{max_retries + 1}) — giving up"
                )
    raise last_exc


def _generate_thumbnail_image_gpt_image_sync(
    prompt: str,
    face_image_bytes: bytes | None = None,
    size: str = GPT_IMAGE_SIZE,
    quality: str = GPT_IMAGE_QUALITY,
) -> dict:
    used_face = bool(face_image_bytes)

    try:
        if face_image_bytes:
            print(f"[THUMBNAIL-GPT] editing WITH user face photo (image-to-image, model='{GPT_IMAGE_MODEL}', timeout={IMAGE_GEN_TIMEOUT}s)")
            face_file = io.BytesIO(face_image_bytes)
            face_file.name = "face.png"

            response = _call_with_timeout_retry(
                lambda: _call_images_edit_with_timeout(face_file, prompt, size, quality),
                label="images.edit (with face)",
            )
        else:
            print(f"[THUMBNAIL-GPT] generating text-to-image (model='{GPT_IMAGE_MODEL}', timeout={IMAGE_GEN_TIMEOUT}s)")

            response = _call_with_timeout_retry(
                lambda: _call_images_generate_with_timeout(prompt, size, quality),
                label="images.generate (text-to-image)",
            )
    except Exception as e:
        error_str = str(e)
        print(f"[THUMBNAIL-GPT] request to GPT Image 2 failed: {e}")
        if used_face and ("invalid_image_file" in error_str or "image_generation_user_error" in error_str):
            print("[THUMBNAIL-GPT] face photo was rejected by GPT Image 2 — retrying as text-to-image instead")
            try:
                response = _call_with_timeout_retry(
                    lambda: _call_images_generate_with_timeout(prompt, size, quality),
                    label="images.generate (fallback after rejected face)",
                )
                used_face = False
            except Exception as retry_e:
                print(f"[THUMBNAIL-GPT] text-to-image fallback also failed: {retry_e}")
                return {"image_base64": None, "error": f"request failed: {retry_e}"}
        else:
            return {"image_base64": None, "error": f"request failed: {e}"}

    try:
        image_base64 = response.data[0].b64_json
    except Exception as e:
        return {"image_base64": None, "error": f"failed to parse GPT Image 2 response: {e}"}

    if not image_base64:
        print("[THUMBNAIL-GPT] GPT Image 2 returned no image data (b64_json empty)")
        return {"image_base64": None, "error": "empty image data in response"}

    print(f"[THUMBNAIL-GPT] received image ({len(image_base64)} base64 chars, used_face={used_face})")
    return {"image_base64": image_base64, "error": None, "used_face": used_face}


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
    chosen_thumbnail_text: str,
) -> dict:
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

    has_reference_image = bool(face_image_bytes)

    # ---- STEP 1: title/thumbnail_text/script (+ whether a face photo will be
    # attached) go straight into one call that returns the final image prompt ----
    rendered_prompt = await run_prompt_generator(
        request.title,
        chosen_thumbnail_text,
        script_text,
        user_image_present=has_reference_image,
    )

    if not rendered_prompt:
        print("[PIPELINE] Step 1 returned empty output — using fallback prompt")
        rendered_prompt = _fallback_thumbnail_prompt(request, chosen_thumbnail_text)
    elif chosen_thumbnail_text.lower() not in rendered_prompt.lower():
        print("[PIPELINE] rendered prompt didn't mention the thumbnail text — appending it explicitly")
        rendered_prompt = (
            f'{rendered_prompt} Render the text "{chosen_thumbnail_text}" as bold, large, '
            f"high-contrast typography baked into the image, in a clear area that doesn't "
            f"overlap the main subject."
        )

    # ---- STEP 2: image generation ----
    result = await generate_thumbnail_image(rendered_prompt, face_image_bytes=face_image_bytes)
    return result


def _build_structure_response(selected_template: dict) -> list[dict]:
    segments = selected_template.get("segments") or []
    return [
        {"name": seg.get("segment_name"), "percentage": seg.get("percentage")}
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
    thumbnail_text: str | None = None
    # language: str = "English"


@app.post("/generate-thumbnail")
async def generate_thumbnail_endpoint(request: ThumbnailRequest):
    await require_valid_user(request.userId)

    async with _pipeline_semaphore:
        return await _generate_thumbnail_endpoint_impl(request)


FREE_TIER_LABELS = {"free", "free_tier", "free-tier", "trial", "none", ""}


async def _deduct_profile_credits(user_id: str, amount: int):
    try:
        result = (
            supabase.table("user_profiles")
            .select("credit_batches")
            .eq("id", user_id)
            .single()
            .execute()
        )
        batches = (result.data or {}).get("credit_batches") or []

        now = datetime.datetime.now(datetime.timezone.utc)
        active_batches = _expire_stale_batches(batches, now)

        updated_batches, deducted = _deduct_from_batches(active_batches, amount)
        if deducted == 0:
            print(f"[CREDITS] user {user_id} has insufficient credits for {amount}, skipping deduction")
            return

        new_total = _sum_batches(updated_batches)

        supabase.table("user_profiles").update({
            "credit_batches": updated_batches,
            "credits_remaining": new_total,
        }).eq("id", user_id).execute()

        print(
            f"[CREDITS] (batch-aware FIFO) Deducted {amount} credits from user {user_id}, "
            f"oldest-expiring batch spent first, new total={new_total}"
        )
    except Exception as exc:
        print(f"[CREDITS] Failed to deduct batch credits for user {user_id}: {exc}")
        import traceback
        traceback.print_exc()



async def _deduct_credits_for_action(user_id: str, amount: int, action_label: str = "credits"):
    if amount <= 0:
        print(f"[CREDITS] ({action_label}) amount <= 0 ({amount}), skipping deduction for user {user_id}")
        return

    await _deduct_profile_credits(user_id, amount)
    print(f"[CREDITS] ({action_label}) user {user_id} — deducted {amount} credits from user_profiles only")


async def _deduct_thumbnail_credits(user_id: str, amount: int = THUMBNAIL_CREDITS_PER_IMAGE):
    await _deduct_credits_for_action(user_id, amount, action_label="thumbnail")


async def _generate_thumbnail_endpoint_impl(request: "ThumbnailRequest"):
    _start_token_tracking()

    total_start_time = time.time()
    script_text = request.script or ""

    # target_language = _normalize_language(getattr(request, "language", None))
    chosen_thumbnail_text = _pick_thumbnail_text(request.thumbnail_text, request)

    # if target_language != "English" and chosen_thumbnail_text:
    #     try:
    #         print(f"[THUMBNAIL] Translating thumbnail text into {target_language}: '{chosen_thumbnail_text}'")
    #         chosen_thumbnail_text = await translate_text_full_pipeline(
    #             chosen_thumbnail_text, target_language
    #         )
    #         print(f"[THUMBNAIL] translated thumbnail text: '{chosen_thumbnail_text}'")
    #     except Exception as exc:
    #         print(f"--- thumbnail text translation failed, keeping English text: {exc} ---")
    print(f"[THUMBNAIL] chosen thumbnail text to render into image: '{chosen_thumbnail_text}'")

    thumbnail_result = {"image_base64": None, "prompt": None, "error": "not attempted"}

    try:
        print("[MAIN] Running 2-step thumbnail pipeline (Prompt Generation -> Image).")
        thumbnail_result = await generate_thumbnail_for_script(
            request, script_text, chosen_thumbnail_text
        )
        thumbnail_url = None
        if thumbnail_result.get("image_base64"):
            thumbnail_url = await save_thumbnail_to_supabase(
                request.userId, thumbnail_result["image_base64"]
            )
            # 1 thumbnail = 10 credits, deducted only after a successful save,
            # via the same batch-aware credit_batches FIFO logic used for voice generation.
            await _deduct_thumbnail_credits(request.userId, THUMBNAIL_CREDITS_PER_IMAGE)
        thumbnail_result["public_url"] = thumbnail_url
    except Exception as exc:
        print(f"--- thumbnail generation failed: {exc} ---")
        import traceback
        traceback.print_exc()

        thumbnail_result = {
            "image_base64": None,
            "prompt": _fallback_thumbnail_prompt(request, chosen_thumbnail_text),
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










class TranslateScriptRequest(BaseModel):
    userId: str
    script: str
    language: str


def _resolve_language(raw: str) -> tuple[str, str]:
    key = (raw or "English").strip().lower()

    if key in SUPPORTED_LANGUAGES:
        return key.title(), SUPPORTED_LANGUAGES[key]

    name = next((n for n, c in SUPPORTED_LANGUAGES.items() if c == key), None)
    if name:
        return name.title(), key

    raise HTTPException(status_code=400, detail=f"Unsupported language: {raw}")


def _build_translation_system_prompt(language_name: str, language_code: str) -> str:
    return f"""You are an expert native-level translator and scriptwriter for {language_name} ({language_code}).
Translate the given script into natural, conversational {language_name} as it is spoken in daily life by ordinary people in the region where {language_name} is spoken.

STYLE RULES (very important):
1. Use everyday spoken {language_name}, the way native speakers actually talk at home, in the market, with friends and family. Do NOT use the formal, bookish, literary, classical, or textbook register.
2. Prefer common, daily-use words over rare, heavy, or "pure" formal words. Avoid over-Sanskritized, over-Arabicized/Persianized, or otherwise archaic vocabulary. If a simple word and a formal word mean the same thing, always choose the simple one that people really say.
3. Use the natural spoken grammar, sentence flow, and casual expressions of {language_name}. It must sound like it was originally written in {language_name} by a native speaker, not like a literal or machine translation.
4. Words that people commonly use as-is in daily speech (e.g. phone, bus, doctor, hospital, mobile, online, TV, school, office, bank) may stay in English, written in the {language_name} script when natural. Do not force unnatural "pure" translations of such words.
5. Keep the tone, emotion, humor and pacing of the original. Keep sentences short, simple and speakable, since this script will be read aloud.
6. Make it easy to understand for the common viewer, not for scholars.

PRESERVE EXACTLY:
- Line breaks, paragraph breaks, bullet points and overall structure.
- Speaker labels, scene headings, timestamps, and any markup such as [SCENE 1], (pause), **bold**, etc. Translate only the human-readable words inside them.
- Numbers, dates, URLs, emails, hashtags, emojis.
- Proper nouns, brand names and product names (transliterate into {language_name} script only if that is how they are normally written).

OUTPUT RULES:
- Return ONLY the translated script. No explanations, notes, quotes, or code fences.
- Translate the ENTIRE script from start to finish. Do not add, remove, skip or summarize anything."""



async def translate_text_gpt_only(script: str, language_name: str, language_code: str) -> str:
    system_prompt = _build_translation_system_prompt(language_name, language_code)
    user_prompt = f"Translate this full script into everyday spoken {language_name}:\n\n{script}"

    last_exc: Exception | None = None
    for attempt in (1, 2):  # second attempt only if the first call errors
        try:
            completion = await _openai_create_with_timeout(
                lambda: openai_client.chat.completions.create(
                    model="gpt-5.4-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=False,
                ),
                timeout=max(OPENAI_CALL_TIMEOUT, 90.0),
            )
            _record_token_usage("translate_script", completion)

            choice = completion.choices[0]

            if choice.finish_reason == "length":
                raise RuntimeError("Translation was cut off; script is too long for a single call")

            text = (choice.message.content or "").strip()
            if not text:
                raise ValueError("Empty translation returned by GPT")
            return text

        except RuntimeError:
            raise  # cut off: retrying would give the same result
        except Exception as exc:
            last_exc = exc
            print(f"[TRANSLATE-SCRIPT] attempt {attempt}/2 failed: {exc}")
            if attempt == 1:
                await asyncio.sleep(1.5)

    raise RuntimeError(f"GPT translation failed: {last_exc}")


@app.post("/translate-script")
async def translate_script_endpoint(request: TranslateScriptRequest):
    await require_valid_user(request.userId)

    async with _pipeline_semaphore:
        return await _translate_script_impl(request)


async def _translate_script_impl(request: "TranslateScriptRequest"):
    _start_token_tracking()

    language_name, language_code = _resolve_language(request.language)

    if not request.script or not request.script.strip():
        return {
            "script": request.script,
            "language": language_name,
            "language_code": language_code,
            "translated": False,
            "token_usage": _get_token_usage_summary(),
        }

    if language_code == "en":
        return {
            "script": request.script,
            "language": language_name,
            "language_code": language_code,
            "translated": False,
            "token_usage": _get_token_usage_summary(),
        }

    translated = True
    try:
        print(f"[TRANSLATE-SCRIPT] Translating script into {language_name} (single call).")
        translated_script = await translate_text_gpt_only(request.script, language_name, language_code)
    except Exception as exc:
        print(f"--- /translate-script failed, returning original script: {exc} ---")
        translated_script = request.script
        translated = False

    return {
        "script": translated_script,
        "language": language_name,
        "language_code": language_code,
        "translated": translated,
        "token_usage": _get_token_usage_summary(),
    }





























import os
import requests
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

_http_session = requests.Session()
_http_adapter = requests.adapters.HTTPAdapter(
    pool_connections=20, pool_maxsize=20, max_retries=1
)
_http_session.mount("https://", _http_adapter)
_http_session.mount("http://", _http_adapter)


PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"
PEXELS_IMAGE_SEARCH_URL = "https://api.pexels.com/v1/search"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PexelsVideoSearchRequest(BaseModel):
    userId: str
    query: str = Field(..., description="Search term, e.g. 'ocean waves'")
    per_page: int = Field(50, ge=1, le=80, description="Results per page (max 80)")
    page: int = Field(1, ge=1, description="Page number")
    orientation: Optional[str] = Field(
        None, description="landscape | portrait | square (optional)"
    )
    size: Optional[str] = Field(
        None, description="large | medium | small (optional, min video resolution)"
    )


class PexelsImageSearchRequest(BaseModel):
    userId: str
    query: str = Field(..., description="Search term, e.g. 'ocean waves'")
    per_page: int = Field(50, ge=1, le=80, description="Results per page (max 80)")
    page: int = Field(1, ge=1, description="Page number")
    orientation: Optional[str] = Field(
        None, description="landscape | portrait | square (optional)"
    )
    size: Optional[str] = Field(
        None, description="large | medium | small (optional, min photo resolution)"
    )
    color: Optional[str] = Field(
        None,
        description="Desired photo color, e.g. 'red', 'blue', or hex like '#ffffff' (optional)",
    )


class PexelsMediaSearchRequest(BaseModel):
    """Used for the combined /search-pexels endpoint (videos + images)."""
    userId: str
    query: str = Field(..., description="Search term, e.g. 'ocean waves'")
    per_page: int = Field(50, ge=1, le=80, description="Results per page (max 80)")
    page: int = Field(1, ge=1, description="Page number")
    orientation: Optional[str] = Field(
        None, description="landscape | portrait | square (optional)"
    )
    size: Optional[str] = Field(
        None, description="large | medium | small (optional, min resolution)"
    )
    color: Optional[str] = Field(
        None, description="Photo color filter, e.g. 'red' or hex like '#ffffff' (images only)"
    )


# ---------------------------------------------------------------------------
# Sync HTTP calls (run via asyncio.to_thread)
# ---------------------------------------------------------------------------

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


def _pexels_search_images_sync(
    query: str,
    per_page: int,
    page: int,
    orientation: Optional[str],
    size: Optional[str],
    color: Optional[str],
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
    if color:
        params["color"] = color

    resp = _http_session.get(
        PEXELS_IMAGE_SEARCH_URL, headers=headers, params=params, timeout=15
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

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
        "type": "video",
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


def _format_image_result(photo: dict) -> dict:
    return {
        "type": "image",
        "id": photo.get("id"),
        "url": photo.get("url"),
        "width": photo.get("width"),
        "height": photo.get("height"),
        "photographer": {
            "name": photo.get("photographer"),
            "url": photo.get("photographer_url"),
        },
        "avg_color": photo.get("avg_color"),
        "alt": photo.get("alt"),
        "src": photo.get("src", {}),  # original, large2x, large, medium, small, portrait, landscape, tiny
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

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


@app.post("/search-pexels-images")
async def search_pexels_images(request: PexelsImageSearchRequest):
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
            _pexels_search_images_sync,
            query,
            request.per_page,
            request.page,
            request.orientation,
            request.size,
            request.color,
        )
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 502
        detail = e.response.text[:300] if e.response is not None else str(e)
        raise HTTPException(status_code=status, detail=f"Pexels API error: {detail}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pexels image search failed: {e}")

    photos = [_format_image_result(p) for p in (data.get("photos") or [])]

    return {
        "query": query,
        "page": data.get("page", request.page),
        "per_page": data.get("per_page", request.per_page),
        "total_results": data.get("total_results", 0),
        "photos": photos,
    }


@app.post("/search-pexels")
async def search_pexels_media(request: PexelsMediaSearchRequest):
    """Combined search — fires video and image search concurrently and
    returns both in one response."""
    await require_valid_user(request.userId)

    if not PEXELS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="PEXELS_API_KEY is not configured on the server.",
        )

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must be a non-empty string")

    async def _get_videos():
        try:
            return await asyncio.to_thread(
                _pexels_search_videos_sync,
                query,
                request.per_page,
                request.page,
                request.orientation,
                request.size,
            )
        except Exception as e:
            return {"error": str(e)}

    async def _get_images():
        try:
            return await asyncio.to_thread(
                _pexels_search_images_sync,
                query,
                request.per_page,
                request.page,
                request.orientation,
                request.size,
                request.color,
            )
        except Exception as e:
            return {"error": str(e)}

    video_data, image_data = await asyncio.gather(_get_videos(), _get_images())

    videos = []
    videos_error = None
    if "error" in video_data:
        videos_error = video_data["error"]
    else:
        videos = [_format_video_result(v) for v in (video_data.get("videos") or [])]

    photos = []
    images_error = None
    if "error" in image_data:
        images_error = image_data["error"]
    else:
        photos = [_format_image_result(p) for p in (image_data.get("photos") or [])]

    if videos_error and images_error:
        raise HTTPException(
            status_code=502,
            detail=f"Pexels search failed for both media types. "
                   f"videos: {videos_error} | images: {images_error}",
        )

    return {
        "query": query,
        "page": request.page,
        "per_page": request.per_page,
        "videos": {
            "total_results": video_data.get("total_results", 0) if not videos_error else 0,
            "results": videos,
            "error": videos_error,
        },
        "images": {
            "total_results": image_data.get("total_results", 0) if not images_error else 0,
            "results": photos,
            "error": images_error,
        },
    }












































































































import os
import time
import json
import datetime
import random
import string
import uuid as uuid_lib

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


CREDIT_CONFIG = {
    "monthly": {
        "plus": {"credits": 600,       "validity_days": 30},
        "pro":  {"credits": 1200,      "validity_days": 30},
    },
    "annual": {
        "plus": {"credits": 600 * 12,  "validity_days": 365},
        "pro":  {"credits": 1200 * 12, "validity_days": 365},
    },
}

CURRENCY_SYMBOL = {"INR": "Rs.", "USD": "$"}
SUPPORTED_CURRENCIES = ("INR", "USD")


async def get_plan_pricing(tier: str, currency: str, billing_cycle: str) -> dict:
    tier = (tier or "").lower()
    currency = (currency or "").upper()
    billing_cycle = (billing_cycle or "").lower()

    if currency not in SUPPORTED_CURRENCIES:
        raise HTTPException(status_code=400, detail="Unsupported currency. Supported: INR, USD.")
    if billing_cycle not in ("monthly", "annual"):
        raise HTTPException(status_code=400, detail="Invalid billing cycle. Must be 'monthly' or 'annual'.")

    try:
        resp = (
            supabase.table('subscriptions_plan')
            .select('plan_name, plan_amount, annual_amount, gst, usd_planamount, usd_annualamount, usd_gst')
            .ilike('plan_name', tier)
            .single()
            .execute()
        )
    except Exception as e:
        print(f"[PRICING] Supabase error fetching plan '{tier}': {e}")
        raise HTTPException(status_code=503, detail="Could not fetch plan pricing.")

    row = resp.data
    if not row:
        raise HTTPException(status_code=400, detail=f"No pricing found for tier '{tier}'.")

    if currency == "INR":
        amount = row.get('plan_amount') if billing_cycle == "monthly" else row.get('annual_amount')
        gst_rate = row.get('gst')
    else:  # USD
        amount = row.get('usd_planamount') if billing_cycle == "monthly" else row.get('usd_annualamount')
        gst_rate = row.get('usd_gst')

    if amount is None:
        raise HTTPException(
            status_code=400,
            detail=f"No {billing_cycle} price configured for tier '{tier}' in {currency}.",
        )

    return {
        "price": float(amount),
        "gst_rate": float(gst_rate) if gst_rate is not None else 0.0,
    }


def generate_invoice_pdf(
    invoice_no,
    customer_name,
    customer_address,
    customer_phone,
    item_name,
    amount,
    plan,
    currency="INR",
    gst_applicable=None,
    gst_rate=18.0,
    due_date=None,
    output_dir="invoices",
):
    """
    `amount` is the GST-EXCLUSIVE rate (pulled from Supabase via
    get_plan_pricing() by the caller). gst_rate is a percentage
    (e.g. 18.0 for 18%) fetched per-currency from `subscriptions_plan`
    (`gst` for INR, `usd_gst` for USD). GST is calculated ON TOP of
    `amount`:

        base_price  = amount
        gst_amount  = amount * gst_rate / 100   (if applicable)
        grand_total = base_price + gst_amount

    grand_total is the amount the customer actually pays / is invoiced for.

    `gst_applicable` defaults to True whenever `gst_rate` is > 0 — GST is
    no longer restricted to INR; USD orders get GST too when the table
    has a usd_gst rate configured for the tier.
    """
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
        [[Paragraph("<b>Storio AI</b>", brand_style), Paragraph("TAX INVOICE", invoice_label_style)]],
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
        "support@storio.tech",
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

    symbol = CURRENCY_SYMBOL.get(currency, f"{currency} ")

    # gst_applicable is decided by the caller. gst_rate is the live rate
    # pulled from Supabase for this tier/currency (percentage, e.g. 18.0) —
    # `gst` for INR, `usd_gst` for USD. GST is no longer INR-only: if this
    # tier/currency has a configured rate > 0, GST applies by default.
    if gst_applicable is None:
        gst_applicable = (gst_rate or 0.0) > 0

    rate_fraction = (gst_rate or 0.0) / 100

    # `amount` is GST-EXCLUSIVE. Add GST on top instead of backing it out.
    base_price = amount
    if gst_applicable and rate_fraction > 0:
        gst_amount = base_price * rate_fraction
    else:
        gst_amount = 0
    grand_total = base_price + gst_amount

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

    # Header + item row (always present, identical to before)
    table_data = [
        ['ITEM', 'PLAN', 'RATE', 'QTY', 'TOTAL'],
        [item_name, plan.title(), f"{symbol} {base_price:.2f}", "1", f"{symbol} {base_price:.2f}"],
    ]

    ts_commands = [
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
    ]

    # GST row — only added when GST actually applies for this invoice.
    if gst_applicable and rate_fraction > 0:
        gst_row_idx = len(table_data)
        table_data.append([
            Paragraph("", lp()),
            "",
            "",
            Paragraph(f"GST ({gst_rate:g}%)", rp(False, colors.HexColor('#555555'))),
            Paragraph(f"{symbol} {gst_amount:.2f}", rp(False, TEXT_DARK)),
        ])
        ts_commands += [
            ('SPAN',          (0,gst_row_idx),(2,gst_row_idx)),
            ('BACKGROUND',    (0,gst_row_idx),(-1,gst_row_idx), LIGHT_GRAY),
            ('TOPPADDING',    (0,gst_row_idx),(-1,gst_row_idx), 8),
            ('BOTTOMPADDING', (0,gst_row_idx),(-1,gst_row_idx), 8),
            ('LINEBELOW',     (0,gst_row_idx),(-1,gst_row_idx), 0.5, colors.HexColor('#dddddd')),
            ('LINEABOVE',     (0,gst_row_idx),(-1,gst_row_idx), 0.5, colors.HexColor('#dddddd')),
            ('VALIGN',        (0,gst_row_idx),(-1,gst_row_idx), 'MIDDLE'),
        ]

    # Grand total row (always present)
    total_row_idx = len(table_data)
    table_data.append([
        Paragraph("GRAND TOTAL", lp(True)),
        "",
        "",
        "",
        Paragraph(f"{symbol} {grand_total:.2f}", rp(True, TEXT_DARK)),
    ])
    ts_commands += [
        ('SPAN',          (0,total_row_idx),(3,total_row_idx)),
        ('BACKGROUND',    (0,total_row_idx),(-1,total_row_idx), MID_GRAY),
        ('TOPPADDING',    (0,total_row_idx),(-1,total_row_idx), 10),
        ('BOTTOMPADDING', (0,total_row_idx),(-1,total_row_idx), 10),
        ('LINEBELOW',     (0,total_row_idx),(-1,total_row_idx), 1.0, colors.HexColor('#cccccc')),
        ('VALIGN',        (0,total_row_idx),(-1,total_row_idx), 'MIDDLE'),
    ]

    ts_commands += [
        ('LEFTPADDING',   (0,0),(-1,-1), 8),
        ('RIGHTPADDING',  (0,0),(-1,-1), 8),
    ]

    ts = TableStyle(ts_commands)

    combined = Table(table_data, colWidths=CW)
    combined.setStyle(ts)
    elements.append(combined)
    elements.append(Spacer(1, 8*mm))

    doc.build(elements, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return file_path


def _expire_stale_batches(batches: list[dict], now: datetime.datetime) -> list[dict]:
    active = []
    for b in batches:
        try:
            expires_at = datetime.datetime.fromisoformat(b["expires_at"])
        except Exception:
            continue
        if expires_at > now:
            active.append(b)
    return active


def _sum_batches(batches: list[dict]) -> int:
    return sum(int(b.get("remaining", 0)) for b in batches)


def _add_credit_batch(
    batches: list[dict], credits: int, validity_days: int, tier: str, now: datetime.datetime,
) -> list[dict]:
    expires_at = now + datetime.timedelta(days=validity_days)
    new_batch = {
        "id": str(uuid_lib.uuid4()),
        "credits": credits,
        "remaining": credits,
        "granted_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "tier": tier,
    }
    return batches + [new_batch]


def _deduct_from_batches(batches: list[dict], amount: int) -> tuple[list[dict], int]:
    if _sum_batches(batches) < amount:
        return batches, 0

    sorted_batches = sorted(batches, key=lambda b: b["expires_at"])
    remaining_to_deduct = amount
    updated = []
    for b in sorted_batches:
        b = dict(b)
        if remaining_to_deduct > 0:
            take = min(b["remaining"], remaining_to_deduct)
            b["remaining"] -= take
            remaining_to_deduct -= take
        if b["remaining"] > 0:
            updated.append(b)

    return updated, amount


async def _get_legacy_batch_expiry(user_id: str, now: datetime.datetime) -> datetime.datetime:
    try:
        sub_res = (
            supabase.table('subscriptions')
            .select('validity, payment_status')
            .eq('userId', user_id)
            .order('purchased_date', desc=True)
            .limit(1)
            .execute()
        )
        rows = sub_res.data or []
        if rows and rows[0].get('validity') and rows[0].get('payment_status') == 'paid':
            validity_dt = datetime.datetime.fromisoformat(rows[0]['validity'])
            if validity_dt > now:
                return validity_dt
    except Exception as e:
        print(f"[CREDITS-MIGRATE] couldn't resolve legacy validity for {user_id}: {e}")
    return now + datetime.timedelta(days=30)


async def _get_active_batches_with_legacy_migration(
    user_id: str, current_batches: list[dict], credits_remaining: int, now: datetime.datetime,
) -> list[dict]:
    active = _expire_stale_batches(current_batches or [], now)

    already_migrated_total = _sum_batches(active)
    unaccounted = (credits_remaining or 0) - already_migrated_total

    if unaccounted > 0:
        legacy_expiry = await _get_legacy_batch_expiry(user_id, now)
        legacy_batch = {
            "id": str(uuid_lib.uuid4()),
            "credits": unaccounted,
            "remaining": unaccounted,
            "granted_at": now.isoformat(),
            "expires_at": legacy_expiry.isoformat(),
            "tier": "legacy",
        }
        print(
            f"[CREDITS-MIGRATE] user {user_id}: found {unaccounted} unaccounted legacy "
            f"credit(s) not in any batch — migrating into a batch expiring {legacy_expiry.isoformat()}"
        )
        active = active + [legacy_batch]

    return active


@app.post("/payments/create-order")
async def create_razorpay_order(
    request_data: CreateOrderRequest,
    current_user: User = Depends(get_current_user),
):
    if not razorpay_client:
        raise HTTPException(status_code=503, detail="Payment service unavailable.")

    user_id = current_user.id
    currency = (request_data.currency or "INR").upper()

    billing_cycle = (getattr(request_data, "billing_cycle", None) or "monthly").lower()
    target_tier = (request_data.target_tier or "").lower()

    if target_tier not in ('plus', 'pro'):
        raise HTTPException(status_code=400, detail="Invalid target tier.")
    if currency not in SUPPORTED_CURRENCIES:
        raise HTTPException(status_code=400, detail="Unsupported currency. Supported: INR, USD.")
    if billing_cycle not in ('monthly', 'annual'):
        raise HTTPException(status_code=400, detail="Invalid billing cycle. Must be 'monthly' or 'annual'.")

    
    pricing = await get_plan_pricing(target_tier, currency, billing_cycle)
    base_price = pricing["price"]
    gst_rate = pricing["gst_rate"]

    if base_price <= 0:
        raise HTTPException(status_code=400, detail="Invalid amount configured for this plan.")


    gst_applicable = gst_rate > 0
    gst_amount = base_price * (gst_rate / 100) if gst_applicable else 0.0
    charge_amount = base_price + gst_amount

    order_data = {
        "amount": int(round(charge_amount * 100)),
        "currency": currency,
        "receipt": request_data.receipt or f"rec_{int(time.time())}",
        "notes": {
            "user_id": str(user_id),
            "target_tier": target_tier,
            "billing_cycle": billing_cycle,
        },
    }
    try:
        order = razorpay_client.order.create(data=order_data)
        print(
            f"Created Razorpay order {order['id']} for user {user_id} "
            f"({currency}, {billing_cycle}): base={base_price}, gst_rate={gst_rate}%, "
            f"gst={gst_amount}, charge={charge_amount}"
        )
        return {
            "order_id": order['id'],
            "key_id": RAZORPAY_KEY_ID,
            "amount": charge_amount,
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
            amount_paid = order_entity.get('amount', 0) / 100  # this is base + GST (see create-order)
            currency    = (order_entity.get('currency') or 'INR').upper()

            notes         = order_entity.get('notes', {})
            user_id       = notes.get('user_id')
            target_tier   = notes.get('target_tier')
            billing_cycle = (notes.get('billing_cycle') or 'monthly').lower()

            if not user_id or not target_tier:
                print(f"ERROR: Missing notes in order {order_id}.")
                return {"status": "error", "message": "Missing required order notes."}

            target_tier = target_tier.lower()

            if currency not in SUPPORTED_CURRENCIES:
                print(f"ERROR: Unsupported currency '{currency}' in order {order_id}.")
                return {"status": "error", "message": "Unsupported currency."}

            credit_config = CREDIT_CONFIG.get(billing_cycle, {}).get(target_tier)
            if not credit_config:
                print(f"ERROR: Unknown tier '{target_tier}' or cycle '{billing_cycle}' in order {order_id}.")
                return {"status": "error", "message": "Unknown plan tier or billing cycle."}

            try:
                pricing = await get_plan_pricing(target_tier, currency, billing_cycle)
                gst_rate = pricing["gst_rate"]
                base_price = pricing["price"]  # GST-exclusive rate, for the invoice line item
            except HTTPException as e:
                print(f"ERROR: Could not fetch pricing for order {order_id}: {e.detail}")
                gst_rate = 18.0
                base_price = amount_paid  # best-effort fallback

            credits_to_add = credit_config['credits']
            validity_days  = credit_config['validity_days']
            now            = datetime.datetime.now(datetime.timezone.utc)
            validity_date  = now + datetime.timedelta(days=validity_days)

            try:
                profile_resp = (
                    supabase.table('user_profiles')
                    .select('credit_batches, credits_remaining')
                    .eq('id', user_id)
                    .single()
                    .execute()
                )
                existing_batches = (profile_resp.data or {}).get('credit_batches') or []
                existing_credits_remaining = (profile_resp.data or {}).get('credits_remaining') or 0

                active_batches = await _get_active_batches_with_legacy_migration(
                    user_id, existing_batches, existing_credits_remaining, now,
                )

                updated_batches = _add_credit_batch(
                    active_batches,
                    credits=credits_to_add,
                    validity_days=validity_days,
                    tier=target_tier,
                    now=now,
                )
                new_total_credits = _sum_batches(updated_batches)

                update_result = (
                    supabase.table('user_profiles')
                    .update({
                        'user_tier': target_tier,
                        'credit_batches': updated_batches,
                        'credits_remaining': new_total_credits,
                    })
                    .eq('id', user_id)
                    .execute()
                )

                if update_result.data:
                    updated_row = update_result.data[0]
                    if updated_row.get('credits_remaining') == new_total_credits:
                        print(
                            f"Confirmed: user {user_id} → tier '{target_tier}' ({billing_cycle}, {currency}), "
                            f"added batch of {credits_to_add} (expires {updated_batches[-1]['expires_at']}), "
                            f"new total={new_total_credits}"
                        )
                    else:
                        print(f"WARN: Update returned mismatched data for {user_id}: {updated_row}")
                else:
                    print(f"ERROR: Update affected 0 rows for user {user_id} (payment {payment_id}) — profile may not exist or RLS blocked it.")

            except APIError as e:
                print(f"ERROR: Supabase profiles error for {user_id}: {e}")
            except Exception as e:
                print(f"ERROR: Unexpected profiles error for {user_id}: {e}")

            try:
                subscription_row = {
                    "userId":               user_id,
                    "amount":               amount_paid,
                    "currency":             currency,
                    "plan":                 target_tier,
                    "billing_cycle":        billing_cycle,
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

                        # GST applies for BOTH INR and USD now, driven purely
                        # by the gst rate configured in `subscriptions_plan`
                        # (`gst` for INR, `usd_gst` for USD) — no more
                        # domestic-customer / payment-rail gating.
                        gst_applicable = gst_rate > 0
                        print(
                            f"[GST] user {user_id}: currency={currency}, "
                            f"method={payment_entity.get('method')}, gst_rate={gst_rate} -> "
                            f"gst_applicable={gst_applicable}"
                        )

                        # NOTE: amount_paid is base+GST (charged via Razorpay).
                        # generate_invoice_pdf expects the GST-EXCLUSIVE rate
                        # in `amount` and adds GST on top itself, so we pass
                        # base_price here, not amount_paid.
                        invoice_path = generate_invoice_pdf(
                            invoice_no=generate_invoice_number(),
                            customer_name=customer_name,
                            customer_address=customer_address,
                            customer_phone=customer_phone,
                            item_name=f"Storio AI {target_tier.title()} Plan ({billing_cycle.title()})",
                            amount=base_price,
                            plan=target_tier,
                            currency=currency,
                            gst_applicable=gst_applicable,
                            gst_rate=gst_rate,
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

            notes         = payment_entity.get('notes', {})
            user_id       = notes.get('user_id')
            target_tier   = notes.get('target_tier')
            billing_cycle = (notes.get('billing_cycle') or 'monthly').lower()
            amount_paid   = payment_entity.get('amount', 0) / 100
            failed_currency = (payment_entity.get('currency') or 'INR').upper()

            if user_id:
                try:
                    failed_row = {
                        "userId":               user_id,
                        "amount":               amount_paid,
                        "currency":             failed_currency,
                        "plan":                 (target_tier or 'unknown').lower(),
                        "billing_cycle":        billing_cycle,
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








































































































from fastapi import UploadFile, File, Form
from typing import List

AUDIO_BUCKET = "user-audio"
AUDIO_TABLE = "user_audio"

ALLOWED_AUDIO_CONTENT_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/x-m4a",
    "audio/aac",
}

MAX_AUDIO_SIZE_BYTES = int(os.getenv("MAX_AUDIO_SIZE_BYTES", str(200 * 1024 * 1024)))
SIGNED_URL_EXPIRY_SECONDS = int(os.getenv("AUDIO_SIGNED_URL_EXPIRY_SECONDS", str(60 * 60 * 24 * 7)))

# Language code is provided by the frontend and isn't restricted to a fixed
# list. It's just validated for shape (short, alphanumeric/hyphen/underscore)
# so it's safe to use as a JSON key and storage path segment.
LANGUAGE_CODE_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,20}$")


def _sanitize_filename(filename: str) -> str:
    filename = filename or "audio"
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    return filename[:150] or "audio"


def _guess_extension(content_type: str, filename: str) -> str:
    ext_from_name = os.path.splitext(filename or "")[1]
    if ext_from_name:
        return ext_from_name
    mapping = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/aac": ".aac",
    }
    return mapping.get(content_type, ".bin")


def _upload_audio_to_storage_sync(
    bucket: str,
    storage_path: str,
    file_bytes: bytes,
    content_type: str,
) -> None:
    supabase.storage.from_(bucket).upload(
        path=storage_path,
        file=file_bytes,
        file_options={
            "content-type": content_type or "application/octet-stream",
            "upsert": "false",
        },
    )


def _create_signed_url_sync(bucket: str, storage_path: str, expires_in: int) -> str | None:
    try:
        result = supabase.storage.from_(bucket).create_signed_url(storage_path, expires_in)
        return result.get("signedURL") or result.get("signedUrl")
    except Exception as e:
        print(f"[AUDIO] failed to create signed URL for '{storage_path}': {e}")
        return None


def _get_current_audio_url_array_sync(user_id: str) -> list:
    """Fetch the existing audio_url jsonb array for this user (defaults to [])."""
    result = (
        supabase.table("user_profiles")
        .select("audio_url")
        .eq("id", user_id)
        .single()
        .execute()
    )
    data = result.data or {}
    existing = data.get("audio_url")
    return existing if isinstance(existing, list) else []


def _merge_language_entry(existing_array: list, language: str, url: str) -> list:
    """
    Replace the entry for `language` if present, otherwise append a new one.
    Result shape: [{"en": "..."}, {"te": "..."}, ...]
    """
    updated = []
    replaced = False
    for entry in existing_array:
        if isinstance(entry, dict) and language in entry:
            updated.append({language: url})
            replaced = True
        else:
            updated.append(entry)
    if not replaced:
        updated.append({language: url})
    return updated


@app.post("/save-audio")
async def save_audio(
    userId: str = Form(...),
    languages: List[str] = Form(...),
    audios: List[UploadFile] = File(...),
):
    """
    Upload one or more audio files at once, each tagged with a language code.
    `languages` and `audios` are matched up by position, so send them in the
    same order, e.g.:

        languages = ["en", "te"]
        audios    = [en_recording.m4a, te_recording.m4a]
    """
    await require_valid_user(userId)

    if len(languages) != len(audios):
        raise HTTPException(
            status_code=400,
            detail=f"Got {len(languages)} language codes but {len(audios)} audio files; they must match 1:1.",
        )

    if len(languages) == 0:
        raise HTTPException(status_code=400, detail="No audio files provided")

    cleaned_languages = [lang.strip().lower() for lang in languages]

    # Reject duplicate language codes within the same request up front.
    seen = set()
    for lang in cleaned_languages:
        if lang in seen:
            raise HTTPException(status_code=400, detail=f"Duplicate language code in request: {lang}")
        seen.add(lang)

    # Validate everything before uploading anything, so a bad entry doesn't
    # leave a partial set of files sitting in storage.
    for lang, audio in zip(cleaned_languages, audios):
        if not LANGUAGE_CODE_PATTERN.match(lang):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid language code: {lang!r} (must be 1-20 chars, letters/numbers/-/_ only)",
            )
        if audio.content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio content type for '{lang}': {audio.content_type}",
            )

    results = []  # list of (language, file_url)

    for lang, audio in zip(cleaned_languages, audios):
        file_bytes = await audio.read()
        size_bytes = len(file_bytes)

        if size_bytes == 0:
            raise HTTPException(status_code=400, detail=f"Uploaded audio file for '{lang}' is empty")

        if size_bytes > MAX_AUDIO_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file for '{lang}' too large ({size_bytes} bytes, max {MAX_AUDIO_SIZE_BYTES} bytes)",
            )

        safe_name = _sanitize_filename(audio.filename)
        extension = _guess_extension(audio.content_type, safe_name)
        unique_name = f"{uuid.uuid4().hex}{extension}"
        storage_path = f"{userId}/{lang}/{unique_name}"

        print(
            f"[AUDIO] uploading '{safe_name}' ({size_bytes} bytes, {audio.content_type}) "
            f"for userId={userId}, language={lang} -> {AUDIO_BUCKET}/{storage_path}"
        )

        try:
            await asyncio.to_thread(
                _upload_audio_to_storage_sync,
                AUDIO_BUCKET,
                storage_path,
                file_bytes,
                audio.content_type,
            )
        except Exception as e:
            print(f"[AUDIO] storage upload FAILED for '{lang}': {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to upload audio for '{lang}' to storage: {e}")

        file_url = await asyncio.to_thread(
            _create_signed_url_sync, AUDIO_BUCKET, storage_path, SIGNED_URL_EXPIRY_SECONDS
        )

        if not file_url:
            raise HTTPException(status_code=500, detail=f"Audio for '{lang}' uploaded but failed to generate URL")

        results.append((lang, file_url))

    try:
        current_array = await asyncio.to_thread(_get_current_audio_url_array_sync, userId)
        updated_array = current_array
        for lang, file_url in results:
            updated_array = _merge_language_entry(updated_array, lang, file_url)

        await asyncio.to_thread(
            lambda: supabase.table("user_profiles")
            .update({"audio_url": updated_array})
            .eq("id", userId)
            .execute()
        )
    except Exception as e:
        print(f"[AUDIO] failed to save audio_url on user_profiles: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Audio uploaded but failed to save URLs to profile: {e}")

    return {
        "message": "Audio uploaded successfully",
        "userId": userId,
        "uploaded": [{"language": lang, "url": url} for lang, url in results],
        "url_expires_in_seconds": SIGNED_URL_EXPIRY_SECONDS,
        "audio_url": updated_array,
    }









































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

















import math

FISH_AUDIO_API_KEY = os.getenv("FISH_AUDIO_API_KEY")
FISH_AUDIO_TTS_URL = "https://api.fish.audio/v1/tts"

import httpx
import base64
from fish_audio_sdk import Session, TTSRequest, Prosody  # FIX (voice too fast): added Prosody import
from langdetect import detect, LangDetectException  # FIX (lang skip): script-language detection

fish_session = Session(FISH_AUDIO_API_KEY)

GENERATED_AUDIO_BUCKET = "generated-audio"

# FIX (voice too fast): _run_fish_tts_sync below never set any prosody/speed
# option, so Fish Audio used its own default speed of 1.0 (normal) — nothing
# was ever making it fast, nothing was ever slowing it down either. This is
# the global default speed applied whenever a request doesn't specify its
# own `speed`. Range 0.5-2.0, lower = slower. Kept as an env var so it can be
# tuned without a redeploy.
TTS_SPEECH_SPEED = float(os.getenv("TTS_SPEECH_SPEED", "0.95"))

# ---- Credit pricing for voice generation ----
# 1 minute of generated audio = 5 credits.
#
# NOTE: despite the field name, `durationMinutes` sent by the client is
# actually a whole number of MINUTES (1, 2, 3, ...), not seconds. We deduct
# credits directly as minutes * VOICE_CREDITS_PER_MINUTE — no /60 conversion.
VOICE_CREDITS_PER_MINUTE = 5


_LANG_CODE_TO_NAME = {v.lower(): k for k, v in SUPPORTED_LANGUAGES.items()}


def _lang_name_from_code(lang_code: str) -> str:
    if not lang_code or not lang_code.strip():
        return DEFAULT_LANGUAGE
    name = _LANG_CODE_TO_NAME.get(lang_code.strip().lower())
    if not name:
        print(f"[TTS] unrecognized langCode '{lang_code}', defaulting to English")
        return DEFAULT_LANGUAGE
    return _normalize_language(name)


# FIX (lang skip): Fish Audio's TTSRequest (see _run_fish_tts_sync below)
# has NO separate "language" field at all — reference_id picks the voice,
# but the actual language spoken is determined ENTIRELY by what script
# text is in `text=script`. That means whether the voiceover comes out in
# the right language depends completely on whether `script` genuinely IS
# in that language by the time it reaches TTS — there's no fallback
# parameter to correct it. This detector is what lets the endpoint tell
# "script is already Telugu" apart from "script is English and needs
# translating", instead of assuming every input is English.
def _detect_script_lang_code(text: str) -> str | None:
    """Best-effort ISO 639-1 detection of the script's actual language.
    Checked against a few hundred chars — enough for langdetect to be
    reliable without scanning a whole long script.

    Returns None when detection genuinely could not produce an answer
    (langdetect RAISES LangDetectException on short/low-feature text —
    a handful of words, mostly numbers/punctuation, etc — this is not a
    "detected as some other language" result, it is "no answer at all".
    Callers MUST treat None as "inconclusive, assume the input is already
    correct" — never as a signal to translate. Conflating the two was a
    real production bug: a short second-chunk fragment (TTS calls get
    split for provider character limits) raised this exception, came back
    as None, and a caller comparing `detected == lang_code` treated that
    the same as a genuine mismatch and tried to translate an already-
    correct Telugu script.
    """
    sample = (text or "").strip()[:500]
    if not sample:
        return None
    try:
        return detect(sample)
    except LangDetectException:
        return None


class GenerateSpeechRequest(BaseModel):
    userId: str
    script: str
    voice: str
    langCode: str = "en"
    durationMinutes: int = 0
    speed: float | None = None  # FIX (voice too fast): 0.5-2.0, lower = slower; None = use TTS_SPEECH_SPEED
    volume: float | None = None  # Fish Audio prosody.volume — dB adjustment, roughly -20..20; None = Fish Audio's own default (0)
    loudnessNormalization: bool | None = None  # Fish Audio prosody.normalize_loudness; None = Fish Audio's own default
    textNormalization: bool | None = None  # Fish Audio top-level "normalize"; None = Fish Audio's own default (true)


async def _download_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


def _create_fish_model_sync(ref_audio_bytes: bytes, title: str) -> str:
    model = fish_session.create_model(
        title=title,
        description="Auto-created voice clone",
        voices=[ref_audio_bytes],
        visibility="private",
    )
    return model.id


def _run_fish_tts_sync(
    script: str,
    reference_id: str,
    speed: float = TTS_SPEECH_SPEED,
    volume: float | None = None,
    loudness_normalization: bool | None = None,
    text_normalization: bool | None = None,
) -> bytes:
    prosody_kwargs = {"speed": speed}
    if volume is not None:
        prosody_kwargs["volume"] = volume
    if loudness_normalization is not None:
        prosody_kwargs["normalize_loudness"] = loudness_normalization

    tts_request = TTSRequest(
        text=script,
        reference_id=reference_id,
        temperature=0.5,
        top_p=0.7,
        repetition_penalty=1.2,
        chunk_length=300,
        latency="normal",
        normalize=text_normalization if text_normalization is not None else True,
        format="mp3",
        mp3_bitrate=192,
        condition_on_previous_chunks=True,
        prosody=Prosody(**prosody_kwargs),
    )
    audio_chunks = []
    for chunk in fish_session.tts(tts_request):
        audio_chunks.append(chunk)
    return b"".join(audio_chunks)

def _credits_for_voice_minutes(duration_minutes: float) -> int:
    if duration_minutes <= 0:
        return 0
    credits = math.ceil(duration_minutes * VOICE_CREDITS_PER_MINUTE)
    return max(credits, 1)


async def _deduct_voice_credits(user_id: str, duration_minutes: float):
    credits_to_deduct = _credits_for_voice_minutes(duration_minutes)
    if credits_to_deduct <= 0:
        print(f"[CREDITS] (voice_generation) nothing to deduct for user {user_id} (duration={duration_minutes:.2f} min)")
        return
    print(
        f"[CREDITS] (voice_generation) user {user_id} — {duration_minutes:.2f} min of audio "
        f"→ {credits_to_deduct} credits (rate: {VOICE_CREDITS_PER_MINUTE}/min)"
    )
    # Reuses the same batch-aware FIFO deduction (credit_batches) already used
    # for thumbnail credits: _deduct_credits_for_action -> _deduct_profile_credits
    # -> _expire_stale_batches / _deduct_from_batches / _sum_batches.
    await _deduct_credits_for_action(user_id, credits_to_deduct, action_label="voice_generation")


def _get_public_url_sync(bucket: str, path: str) -> str:
    res = supabase.storage.from_(bucket).get_public_url(path)
    if isinstance(res, dict):
        return res.get("publicUrl") or res.get("public_url")
    return res


# FIX (per-language reference audio): audio_url used to be a single plain
# URL string on user_profiles. It's now an array of { langCode: url }
# entries — one recorded reference clip per language, e.g.:
#   [{ "en": "https://.../en/xxx.m4a?..." }, { "mr": "https://.../mr/yyy.m4a?..." }]
# This picks the entry matching lang_code. Also tolerates two edge cases
# so it doesn't hard-break on data that predates the format change:
#   - the column coming back as a raw JSON string instead of an
#     already-parsed list (depends on how the client/driver handles
#     jsonb — parse it defensively rather than assume)
#   - a legacy row that's still a single plain string URL (the old
#     one-URL-per-user format) — since that shape carries no language
#     info, it's only used when the request is for English, which is
#     the only thing a single legacy URL could ever have meant.
# Returns None if there's no reference audio for that language at all.
def _find_reference_audio_url_for_lang(audio_url_data, lang_code: str) -> str | None:
    if audio_url_data is None:
        return None

    if isinstance(audio_url_data, str):
        stripped = audio_url_data.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                audio_url_data = json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                return stripped if lang_code == "en" else None
        else:
            return stripped if lang_code == "en" else None

    if isinstance(audio_url_data, dict):
        return audio_url_data.get(lang_code)

    if isinstance(audio_url_data, list):
        for entry in audio_url_data:
            if isinstance(entry, dict) and lang_code in entry:
                return entry[lang_code]

    return None


@app.post("/generate-speech")
async def generate_speech(body: GenerateSpeechRequest):
    userId = body.userId
    script = body.script
    voice = body.voice.strip() if body.voice else ""
    lang_code = (body.langCode or "en").strip()
    speed = body.speed if body.speed is not None else TTS_SPEECH_SPEED

    await require_valid_user(userId)

    if not script.strip():
        raise HTTPException(status_code=400, detail="script cannot be empty")

    if not voice:
        raise HTTPException(status_code=400, detail="voice cannot be empty")

    if not FISH_AUDIO_API_KEY:
        raise HTTPException(status_code=500, detail="Fish Audio API key not configured")

    target_language = _lang_name_from_code(lang_code)
    if target_language != "English":
        # FIX (lang skip): previously translated unconditionally whenever
        # langCode != "en", assuming `script` was always English-authored.
        # For /edit-video specifically, the payload contract sends script
        # already written in the target language — so this was
        # re-translating already-correct text every time.
        #
        # FIX 2 (None-detection fail-safe): _detect_script_lang_code
        # returns None when langdetect could not produce ANY answer (it
        # raises on short/low-feature text — a few words, a chunk
        # boundary landing on mostly numbers, etc — that is not the same
        # thing as "detected as a different language"). The original
        # `if detected == lang_code` treated None and a genuine mismatch
        # identically, so an inconclusive detection on a short TTS chunk
        # (narrations get split into multiple calls for provider length
        # limits) fell through to "translate it" — corrupting an
        # already-correct Telugu chunk. Since this endpoint's payload
        # contract already guarantees script arrives in the target
        # language, an inconclusive detection should side with trusting
        # that contract, not overriding it.
        #
        # FIX 3 (this message): translate_text_full_pipeline never
        # actually existed in this codebase (see chat) — the real,
        # already-built translation function is translate_text_gpt_only
        # (single LLM call, native-spoken-register prompt, no Google
        # Translate dependency), the same one /translate-script uses.
        # Calling that directly here instead.
        detected = _detect_script_lang_code(script)
        if detected is None or detected == lang_code:
            print(
                f"[TTS] script already assumed '{lang_code}' "
                f"(detector said: {detected!r}) — skipping translation"
            )
        else:
            try:
                print(f"[TTS] translating script into {target_language} (langCode='{lang_code}') before TTS")
                script = await translate_text_gpt_only(script, target_language, lang_code)
            except Exception as e:
                print(f"[TTS] translation to {target_language} failed, using original script as-is: {e}")

    if voice.lower() == "user":
        try:
            result = await asyncio.to_thread(
                lambda: supabase.table("user_profiles")
                .select("audio_url")
                .eq("id", userId)
                .maybe_single()
                .execute()
            )
        except Exception as e:
            print(f"[TTS] failed to fetch user_profiles row: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch user profile: {e}")

        row = result.data if result else None
        audio_url_data = row.get("audio_url") if row else None

        # FIX (per-language reference audio): audio_url is now an array of
        # { langCode: url } entries — one recorded reference clip per
        # language — instead of a single URL. Pick the one matching this
        # request's langCode so the voice clone is actually built from
        # audio spoken in that language, not whatever the user's default/
        # first-recorded clip happened to be.
        audio_url = _find_reference_audio_url_for_lang(audio_url_data, lang_code.lower())

        if not audio_url:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No reference audio on file for this user in language "
                    f"'{lang_code}'. Upload one for this language via "
                    f"/save-audio first."
                ),
            )

        try:
            ref_audio_bytes = await _download_bytes(audio_url)
        except Exception as e:
            print(f"[TTS] failed to download user reference audio: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download user reference audio: {e}")

        try:
            reference_id = await asyncio.to_thread(
                _create_fish_model_sync, ref_audio_bytes, f"user-{userId}"
            )
        except Exception as e:
            print(f"[TTS] failed to create Fish Audio model: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=502, detail=f"Failed to create voice model: {e}")
    else:
        reference_id = voice

    try:
        audio_bytes = await asyncio.to_thread(_run_fish_tts_sync, script, reference_id,speed,body.volume,
            body.loudnessNormalization,
            body.textNormalization,
)
    except Exception as e:
        print(f"[TTS] Fish Audio TTS failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Fish Audio TTS failed: {e}")

    if not audio_bytes:
        raise HTTPException(status_code=502, detail="Fish Audio returned empty audio")

    storage_path = f"{userId}/{uuid.uuid4().hex}.mp3"

    try:
        await asyncio.to_thread(
            _upload_audio_to_storage_sync,
            GENERATED_AUDIO_BUCKET,
            storage_path,
            audio_bytes,
            "audio/mpeg",
        )
    except Exception as e:
        print(f"[TTS] failed to upload generated audio to storage: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save generated audio: {e}")

    try:
        public_url = await asyncio.to_thread(
            _get_public_url_sync, GENERATED_AUDIO_BUCKET, storage_path
        )
    except Exception as e:
        print(f"[TTS] failed to build public URL: {e}")
        raise HTTPException(status_code=500, detail="Generated audio saved but failed to create URL")

    if not public_url:
        raise HTTPException(status_code=500, detail="Generated audio saved but failed to create URL")

    try:
        duration_minutes = body.durationMinutes or 0
        await _deduct_voice_credits(userId, duration_minutes)
    except Exception as e:
        print(f"[TTS] credit deduction failed for user {userId}: {e}")
        import traceback
        traceback.print_exc()

    return {
        "message": "Speech generated successfully",
        "userId": userId,
        "voice": voice,
        "langCode": lang_code,
        "speed": speed,
        "reference_id": reference_id,
        "storage_path": storage_path,
        "url": public_url,
    }















class AddScriptTagsRequest(BaseModel):
    userId: str
    script: str



SCRIPT_TAG_SYSTEM_PROMPT = f"""

# STORYBIT — FISH AUDIO S2 HUMAN PERFORMANCE ANNOTATION ENGINE

## ROLE

You are the **Storybit Voice Performance Annotation Agent**.

The input is the user's **complete final script**. Convert it into a **Fish Audio S2/S2-Pro-ready performance script** by inserting carefully selected performance tags so the voice sounds natural, emotionally believable, human-performed, and professionally narrated.

You are **not** a writer, editor, summarizer, or rewriter.

## CORE RULE

Preserve the script exactly. You may **ONLY INSERT performance tags**.

Never:

* add, remove, paraphrase, reorder, or rewrite words
* change names, numbers, facts, dialogue, quotations, or meaning
* invent narration, dialogue, emotions, or sounds
* add explanations or commentary

Preserve original punctuation and structure unless a minimal punctuation adjustment is essential for natural speech.

---

# FISH AUDIO S2/S2-PRO TAG VOCABULARY

Use square brackets `[ ]` for performance instructions.

### BASIC EMOTIONS

`[happy]` `[sad]` `[angry]` `[excited]` `[calm]` `[nervous]` `[confident]` `[surprised]` `[satisfied]` `[delighted]` `[scared]` `[worried]` `[upset]` `[frustrated]` `[depressed]` `[empathetic]` `[embarrassed]` `[disgusted]` `[moved]` `[proud]` `[relaxed]` `[grateful]` `[curious]` `[sarcastic]`

### ADVANCED EMOTIONS

`[disdainful]` `[unhappy]` `[anxious]` `[hysterical]` `[indifferent]` `[uncertain]` `[doubtful]` `[confused]` `[disappointed]` `[regretful]` `[guilty]` `[ashamed]` `[jealous]` `[envious]` `[hopeful]` `[optimistic]` `[pessimistic]` `[nostalgic]` `[lonely]` `[bored]` `[contemptuous]` `[sympathetic]` `[compassionate]` `[determined]` `[resigned]`

### TONE MARKERS

`[in a hurry tone]` `[shouting]` `[screaming]` `[whispering]` `[soft tone]` `[emphasis]`

### AUDIO EFFECTS

`[laughing]` `[chuckling]` `[sobbing]` `[crying loudly]` `[sighing]` `[groaning]` `[panting]` `[gasping]` `[yawning]` `[snoring]` `[clear throat]`

### CUSTOM PERFORMANCE TAGS

Concise natural-language descriptions may be used when they express the intended delivery more precisely.

Examples:

`[short pause]` `[pause]` `[long dramatic pause]` `[quietly tense]` `[restrained emotional tone]` `[professional broadcast tone]` `[quiet realization]` `[building suspense]` `[serious documentary tone]` `[warm conversational tone]`

Keep custom tags concise. Never create long instructions inside brackets.

---

# CRITICAL TAG INTERPRETATION & PLACEMENT

**Every sentence MUST have at least one appropriate sentence-level performance tag.**

The purpose is to create **emotional continuity throughout the entire voice-over**, so the narration never feels mechanically neutral or emotionally disconnected from the story.

However, this does **not** mean every sentence should sound highly emotional or dramatically performed. The selected tag must represent the most natural emotional or delivery state of that sentence within the surrounding narrative.

### EMOTION TAGS — SENTENCE LEVEL

Emotion tags operate at **sentence level**, not within a fixed word window.

A sentence-level emotion tag at the beginning of a sentence colors the **whole sentence**.

Therefore:

* Place the primary emotion tag **immediately before the sentence it controls**.
* Select the emotion according to the sentence's meaning and the surrounding emotional arc.
* Maintain continuity between neighboring sentences.
* Change emotion when the narrative, meaning, or emotional state genuinely changes.
* Do not force dramatic emotions onto neutral information.
* Calm, confident, curious, uncertain, reflective, hopeful, sympathetic, or other subtle states may be used when appropriate.

Correct:

`[curious] But why did nobody notice the warning?`

`[calm] The investigation continued for several months.`

`[concerned] Then the first unexpected result appeared.`

Do not place the emotion tag after the sentence it controls.

### EMPHASIS — WORD/PHRASE LEVEL

`[emphasis]` is **word/phrase-level**, not sentence-level.

Place it immediately before the exact word or phrase requiring emphasis.

Example:

`[confident] The [emphasis]real reason was never revealed.`

Do not use `[emphasis]` as a substitute for the sentence-level emotion tag.

### TONE MARKERS — POINT OF ACTIVATION

Tone markers such as `[shouting]`, `[whispering]`, `[soft tone]`, and `[in a hurry tone]` act at the point where they occur.

Place them immediately before the text where that delivery begins.

### AUDIO EFFECTS — POINT OF OCCURRENCE

Audio effects act at the point where they occur.

Place them immediately before the associated vocal event or text.

Use them sparingly and only when contextually justified.

### PAUSES

`[short pause]`, `[pause]`, and `[long dramatic pause]` act at their point of occurrence.

Place them immediately before the text following the intended pause.

---

# SENTENCE TAGGING & EMOTIONAL CONTINUITY

**Tag every sentence, but do not over-perform every sentence.**

For each sentence, silently determine its most appropriate emotional/delivery state based on:

1. the sentence's meaning
2. the preceding sentence
3. the following sentence
4. the current narrative beat
5. the overall emotional arc

Then place **one primary sentence-level emotion or delivery tag immediately before that sentence**.

The goal is a **continuous emotional performance**, not constant emotional switching.

Guidelines:

* Every sentence: **at least 1 appropriate sentence-level tag**
* Default: **1 primary emotion per sentence**
* Maximum: **3 combined emotion cues per sentence recommended**
* Change emotion only when the emotional state or delivery meaningfully changes
* Maintain the same or closely related emotional state across consecutive sentences when appropriate
* Use subtle emotions for ordinary informational sentences
* Do not use intense emotions merely to satisfy the tagging requirement
* Space major emotional changes naturally
* Do not overuse emotion tags in short sentences
* Additional tone, emphasis, pause, or sound-effect tags are optional and should be used only when genuinely useful

**Emotional continuity is more important than emotional variety.**

---

# STORY ANALYSIS

Before annotating, silently understand the complete script:

* genre and narrator style
* emotional arc
* scene and idea transitions
* suspense and tension
* revelations and climaxes
* important facts and emphasis
* questions
* dialogue
* emotional moments
* pacing
* conclusion

Annotate according to the **complete story context**, not isolated sentences.

---

# HUMAN PERFORMANCE

The performance should feel like a skilled narrator continuously telling one coherent story.

Create natural variation through purposeful changes in emotion, tone, emphasis, pauses, intensity, and occasional vocal reactions.

Do not make every sentence dramatic.

Avoid unnecessary tag stacking.

Bad:

`[excited] [happy] [shouting] [emphasis]`

Prefer:

`[excited]`

Never combine contradictory directions such as:

`[whispering] [screaming]`

---

# DIALOGUE

Preserve dialogue exactly.

Place the primary delivery/emotion tag immediately before the dialogue.

Examples:

`[angry] "You knew."`

`[uncertain] "I... I don't know."`

`[whispering] "Don't tell anyone."`

Do not invent speaker labels, character voices, or dialogue.

---

# SUSPENSE & REVELATIONS

Build performance progressively:

**normal/appropriate emotion → subtle tension → [short pause] → reveal → [emphasis]/[quiet realization] → appropriate emotional continuity**

Use strong emotions, shouting, screaming, or dramatic effects only when genuinely justified.

---

# DO'S

* Tag **every sentence** with an appropriate primary sentence-level emotion/delivery tag.
* Make the tags create emotional continuity across the complete voice-over.
* Match emotions to context rather than keywords.
* Keep related emotional states consistent across consecutive sentences.
* Change emotional state when the story genuinely changes.
* Place sentence-level tags immediately before their sentence.
* Place `[emphasis]` immediately before its target word/phrase.
* Place tone/effect tags at their point of activation.
* Use subtle emotions for neutral or informational material.
* Use a maximum of 3 combined emotions per sentence when genuinely necessary.
* Space major emotional changes naturally.

# DON'TS

* Don't leave any sentence without an appropriate sentence-level tag.
* Don't treat emotion tags as fixed word-window controls.
* Don't place emotion tags after the sentence they control.
* Don't use `[emphasis]` as a sentence-level emotion.
* Don't force intense emotions onto ordinary information.
* Don't change emotions unnecessarily from sentence to sentence.
* Don't overuse emotion tags in short text.
* Don't mix conflicting emotions.
* Don't stack redundant or contradictory tags.
* Don't add unjustified audio effects.
* Don't alter the original script.

---

# OUTPUT

Return **ONLY the fully annotated script**.

No analysis, explanations, headings, JSON, code fences, notes, summaries, or introductory text.

# FINAL VALIDATION

Before output, silently verify:

1. Every original word remains unchanged.
2. Nothing was added, deleted, paraphrased, or reordered except performance tags.
3. Facts, names, numbers, quotations, dialogue, and meaning remain unchanged.
4. Every sentence has **at least one appropriate sentence-level performance tag**.
5. Every sentence-level emotion tag appears **immediately before the sentence it controls**.
6. Emotion tags are interpreted as sentence-level controls, not fixed word windows.
7. `[emphasis]` appears immediately before its specific word/phrase.
8. Tone and audio-effect tags occur at their point of activation.
9. Emotional continuity exists across neighboring sentences.
10. Emotional changes occur only when context justifies them.
11. One primary emotion is used per sentence by default.
12. No more than 3 combined emotions are used per sentence unless genuinely necessary.
13. Short sentences are not unnecessarily over-performed.
14. Intense emotions are reserved for moments that justify them.
15. Pauses and audio effects remain purposeful and natural.
16. No contradictory or redundant tags exist.
17. The complete voice-over feels emotionally continuous and human-performed.
18. Output contains **only the annotated script**.

## INPUT

The attached/input content is the **complete final user-generated script**.

Analyze the entire script first, then return the same script with **only the necessary Fish Audio S2/S2-Pro performance tags inserted according to the tag-specific interpretation, placement, frequency, and emotional-continuity rules above**.

""".strip()


_TAG_PATTERN = re.compile(r"\[([^\[\]]{1,40})\]")


def _validate_tagged_script(original: str, tagged: str, min_tags: int = 3) -> bool:
    if not tagged or not tagged.strip():
        return False

    tag_matches = _TAG_PATTERN.findall(tagged)

    if len(tag_matches) < min_tags:
        print(
            f"[TAG-SCRIPT] validation failed: only {len(tag_matches)} tag(s) found "
            f"(minimum required: {min_tags})"
        )
        return False

    stripped = _TAG_PATTERN.sub("", tagged)
    stripped_words = stripped.split()
    original_words = original.split()

    if not original_words:
        return False

    len_ratio = len(stripped_words) / len(original_words)
    if len_ratio < 0.9 or len_ratio > 1.1:
        print(
            f"[TAG-SCRIPT] validation failed: word count ratio {len_ratio:.3f} "
            f"outside [0.9, 1.1] ({len(stripped_words)} vs {len(original_words)} words)"
        )
        return False

    return True


def _min_expected_tags(word_count: int) -> int:
    return max(2, word_count // 100)


def _insert_fallback_tags(script_text: str, min_tags: int) -> str:
    paragraphs = script_text.split("\n\n")
    if len(paragraphs) < 2:
        sentences = re.split(r"(?<=[.!?])\s+", script_text.strip())
        tagged_sentences = []
        tags_used = 0
        interval = max(2, len(sentences) // max(min_tags, 1))
        for i, sentence in enumerate(sentences):
            if i > 0 and i % interval == 0 and tags_used < min_tags:
                tagged_sentences.append("[pause] " + sentence)
                tags_used += 1
            else:
                tagged_sentences.append(sentence)
        return " ".join(tagged_sentences)

    tagged_paragraphs = [paragraphs[0]]
    tags_used = 0
    for para in paragraphs[1:]:
        if tags_used < min_tags and para.strip():
            tagged_paragraphs.append("[short pause] " + para)
            tags_used += 1
        else:
            tagged_paragraphs.append(para)
    return "\n\n".join(tagged_paragraphs)


async def generate_tagged_script(script_text: str) -> str:
    word_count = _word_count(script_text)
    min_tags = _min_expected_tags(word_count)

    user_prompt = f"""Script to tag (this script contains {word_count} words — you MUST insert at least {min_tags} tags total, distributed naturally throughout, not clustered):{script_text}
"""

    async def _call(extra_instruction: str = ""):
        messages = [
            {"role": "system", "content": SCRIPT_TAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt + extra_instruction},
        ]
        completion = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=messages,
                stream=False,
                temperature=0.5,
                top_p=0.9,
            ),
            timeout=max(OPENAI_CALL_TIMEOUT, 90.0),
        )
        _record_token_usage("generate_tagged_script", completion)
        return (completion.choices[0].message.content or "").strip()

    raw = ""
    try:
        raw = await _call()
    except Exception as e:
        print(f"[TAG-SCRIPT] generation failed: {e}")
        raw = ""

    if raw and not _validate_tagged_script(script_text, raw, min_tags=min_tags):
        tag_count_found = len(_TAG_PATTERN.findall(raw))
        print(
            f"[TAG-SCRIPT] attempt 1 invalid ({tag_count_found} tag(s) found, "
            f"need >= {min_tags}) — retrying with explicit correction. "
            f"Raw (truncated): {raw[:300]}"
        )
        try:
            raw = await _call(
                extra_instruction=(
                    f"\n\nIMPORTANT CORRECTION: Your previous attempt returned "
                    f"{tag_count_found} tag(s), which is BELOW the required minimum "
                    f"of {min_tags}. This is not optional — you must scan the entire "
                    f"script from beginning to end and insert at least {min_tags} tags "
                    f"from the supported list at genuine emotional or rhythmic turning "
                    f"points (paragraph openings, contrasts introduced by 'but'/'yet', "
                    f"rhetorical questions, reveals, transitions). Do not return the "
                    f"script with fewer tags than required. Preserve every original "
                    f"word exactly — only insert bracketed tags."
                )
            )
        except Exception as e:
            print(f"[TAG-SCRIPT] retry call failed: {e}")
            raw = ""

    if raw and _validate_tagged_script(script_text, raw, min_tags=min_tags):
        return raw

    tag_count_found = len(_TAG_PATTERN.findall(raw)) if raw else 0
    print(
        f"[TAG-SCRIPT] still invalid after retry ({tag_count_found} tag(s)) — "
        f"applying deterministic fallback tagging instead of returning "
        f"a fully untagged script"
    )
    return _insert_fallback_tags(script_text, min_tags)


@app.post("/add-script-tags")
async def add_script_tags(request: AddScriptTagsRequest):
    await require_valid_user(request.userId)

    _start_token_tracking()

    script_text = (request.script or "").strip()
    if not script_text:
        raise HTTPException(status_code=400, detail="script must be a non-empty string")

    print(f"\n[TAG-SCRIPT] tagging script ({_word_count(script_text)} word(s)) for userId={request.userId}")

    try:
        tagged_script = await generate_tagged_script(script_text)
    except Exception as exc:
        print(f"--- /add-script-tags failed: {exc} ---")
        import traceback
        traceback.print_exc()
        return {
            "error": "Failed to generate tagged script.",
            "detail": str(exc),
            "script": script_text,
            "token_usage": _get_token_usage_summary(),
        }

    tag_count = len(_TAG_PATTERN.findall(tagged_script))
    token_usage = _get_token_usage_summary()

    print(f"[TAG-SCRIPT] done — {tag_count} tag(s) inserted")

    return {
        "tagged_script": tagged_script,
        "tag_count": tag_count,
        "word_count": _word_count(script_text),
        "token_usage": token_usage,
    }














































































































# editing 


TEMPLATE_METADATA = {
  "purpose": "Few-shot examples for the LLM that writes an ideal template spec for an editing beat. Embed output.match for semantic search; use output.editing, output.timing and output.requirements as filters and scoring. Remove _maps_to before using in the prompt.",
  "examples": [
    {
      "_maps_to": "bar_chart",
      "input": {
        "narration": "In January the channel got 42 thousand views, and by June it crossed 96 thousand.",
        "beat_role": "data_comparison",
        "duration_sec": 5,
        "available_images": 0,
        "fps": 30,
        "beat_start_sec": 42.0,
        "beat_end_sec": 47.0,
        "sync_keyword": "96 thousand",
        "keyword_offset_sec": 1.5,
        "prev_beat_role": "hook",
        "next_beat_role": "emphasis"
      },
      "output": {
        "match": {
          "category": "data_visualization",
          "motion": "bars_rise_staggered",
          "description": "Vertical bar chart in a card over the footage comparing monthly views, bars rising one after another with value labels.",
          "use_when": "Comparing 3-8 numeric values across months or categories.",
          "tags": [
            "chart",
            "bar",
            "growth",
            "comparison"
          ]
        },
        "editing": {
          "beat_role": "data_comparison",
          "energy": "medium",
          "tone": [
            "neutral",
            "educational"
          ],
          "layer": "overlay",
          "sync_target": "keyword",
          "exit": "hard_cut"
        },
        "timing": {
          "fps": 30,
          "clip_start_sec": 42.3,
          "clip_end_sec": 47.0,
          "duration_frames": 141,
          "entrance_frames": 36,
          "sync_frame": 36,
          "sync_lands_at_sec": 43.5
        },
        "requirements": {
          "covers_footage": False,
          "placement": "center",
          "item_count": 6,
          "image_count": 0,
          "transition_role": "null"
        },
        "props": {
          "title": "Monthly Views (K)",
          "data": [
            {
              "label": "Jan",
              "value": 42
            },
            {
              "label": "Feb",
              "value": 58
            },
            {
              "label": "Mar",
              "value": 51
            },
            {
              "label": "Apr",
              "value": 73
            },
            {
              "label": "May",
              "value": 88
            },
            {
              "label": "Jun",
              "value": 96
            }
          ]
        }
      }
    },
    {
      "_maps_to": "animated_list",
      "input": {
        "narration": "Four money habits that changed my life: an emergency fund, automatic SIPs, zero credit card debt, and reviewing insurance every year.",
        "beat_role": "list",
        "duration_sec": 6,
        "available_images": 0,
        "fps": 30,
        "beat_start_sec": 118.5,
        "beat_end_sec": 124.5,
        "sync_keyword": "Four money habits",
        "keyword_offset_sec": 0.0,
        "prev_beat_role": "section_start",
        "next_beat_role": "list"
      },
      "output": {
        "match": {
          "category": "list_and_steps",
          "motion": "items_slide_in_staggered",
          "description": "Bullet list in a card over the footage, each row sliding in with a matching icon.",
          "use_when": "Narration lists 3-6 tips, habits or features with no fixed order.",
          "tags": [
            "list",
            "tips",
            "icons",
            "points"
          ]
        },
        "editing": {
          "beat_role": "list",
          "energy": "medium",
          "tone": [
            "neutral",
            "educational"
          ],
          "layer": "overlay",
          "sync_target": "keyword",
          "exit": "hard_cut"
        },
        "timing": {
          "fps": 30,
          "clip_start_sec": 118.5,
          "clip_end_sec": 124.5,
          "duration_frames": 180,
          "entrance_frames": 35,
          "sync_frame": 35,
          "sync_lands_at_sec": 119.67
        },
        "requirements": {
          "covers_footage": False,
          "placement": "center",
          "item_count": 4,
          "image_count": 0,
          "transition_role": "null"
        },
        "props": {
          "items": [
            "Emergency fund",
            "Automatic SIPs",
            "Zero credit card debt",
            "Yearly insurance review"
          ],
          "icon_name": [
            "piggy-bank",
            "repeat",
            "credit-card",
            "shield"
          ]
        }
      }
    },
    {
      "_maps_to": "progress_steps",
      "input": {
        "narration": "The loan process is simple. You apply, complete KYC, get approval, and the money is disbursed.",
        "beat_role": "process",
        "duration_sec": 5,
        "available_images": 0,
        "fps": 30,
        "beat_start_sec": 205.0,
        "beat_end_sec": 210.0,
        "sync_keyword": "The loan process",
        "keyword_offset_sec": 0.0,
        "prev_beat_role": "emphasis",
        "next_beat_role": "key_stat"
      },
      "output": {
        "match": {
          "category": "list_and_steps",
          "motion": "steps_activate_in_sequence",
          "description": "Horizontal row of numbered steps connected by lines, lighting up one after another to show progress through a process.",
          "use_when": "An ordered process of 3-6 stages where sequence matters.",
          "tags": [
            "process",
            "steps",
            "workflow",
            "timeline"
          ]
        },
        "editing": {
          "beat_role": "process",
          "energy": "medium",
          "tone": [
            "educational",
            "neutral"
          ],
          "layer": "overlay",
          "sync_target": "beat_start",
          "exit": "hard_cut"
        },
        "timing": {
          "fps": 30,
          "clip_start_sec": 205.0,
          "clip_end_sec": 210.0,
          "duration_frames": 150,
          "entrance_frames": 150,
          "sync_frame": 0,
          "sync_lands_at_sec": 205.0
        },
        "requirements": {
          "covers_footage": False,
          "placement": "center",
          "item_count": 4,
          "image_count": 0,
          "transition_role": "null"
        },
        "props": {
          "items": [
            "Apply",
            "KYC",
            "Approval",
            "Disbursal"
          ]
        }
      }
    },
    {
      "_maps_to": "text_highlight",
      "input": {
        "narration": "Here's the truth: most people quit right before it starts working.",
        "beat_role": "emphasis",
        "duration_sec": 4,
        "available_images": 0,
        "fps": 30,
        "beat_start_sec": 61.3,
        "beat_end_sec": 65.3,
        "sync_keyword": "right before it starts working",
        "keyword_offset_sec": 2.1,
        "prev_beat_role": "key_stat",
        "next_beat_role": "list"
      },
      "output": {
        "match": {
          "category": "text_animation",
          "motion": "marker_sweeps_over_phrase",
          "description": "Sentence over the footage with a highlighter stroke sweeping across the most important phrase.",
          "use_when": "One part of a spoken sentence needs extra emphasis.",
          "tags": [
            "text",
            "highlight",
            "emphasis",
            "keyword"
          ]
        },
        "editing": {
          "beat_role": "emphasis",
          "energy": "medium",
          "tone": [
            "educational",
            "neutral"
          ],
          "layer": "overlay",
          "sync_target": "keyword",
          "exit": "hard_cut"
        },
        "timing": {
          "fps": 30,
          "clip_start_sec": 62.2,
          "clip_end_sec": 65.3,
          "duration_frames": 93,
          "entrance_frames": 36,
          "sync_frame": 36,
          "sync_lands_at_sec": 63.4
        },
        "requirements": {
          "covers_footage": False,
          "placement": "center",
          "item_count": "null",
          "image_count": 0,
          "transition_role": "null"
        },
        "props": {
          "text": "Most people quit right before it starts working",
          "highlight": "right before it starts working"
        }
      }
    },
    {
      "_maps_to": "ken_burns",
      "input": {
        "narration": "Built in 1653, the Taj Mahal took over twenty thousand workers and twenty-two years to complete.",
        "beat_role": "visual_broll",
        "duration_sec": 7,
        "available_images": 1,
        "fps": 30,
        "beat_start_sec": 12.0,
        "beat_end_sec": 19.0,
        "sync_keyword": "Taj Mahal",
        "keyword_offset_sec": 1.4,
        "prev_beat_role": "intro",
        "next_beat_role": "key_stat"
      },
      "output": {
        "match": {
          "category": "media_image",
          "motion": "slow_zoom_pan",
          "description": "A single historical photo filling the screen with a slow documentary-style zoom and pan.",
          "use_when": "Narration describes a place, person or object and one still photo is available.",
          "tags": [
            "photo",
            "documentary",
            "zoom",
            "history"
          ]
        },
        "editing": {
          "beat_role": "visual_broll",
          "energy": "low",
          "tone": [
            "cinematic",
            "serious"
          ],
          "layer": "full_screen",
          "sync_target": "beat_start",
          "exit": "hard_cut"
        },
        "timing": {
          "fps": 30,
          "clip_start_sec": 12.0,
          "clip_end_sec": 19.0,
          "duration_frames": 210,
          "entrance_frames": 0,
          "sync_frame": 0,
          "sync_lands_at_sec": 12.0
        },
        "requirements": {
          "covers_footage": True,
          "placement": "full_frame",
          "item_count": "null",
          "image_count": 1,
          "transition_role": "null"
        },
        "props": {
          "image": "<image_url_1>"
        }
      }
    },
    {
      "_maps_to": "image_split_screen",
      "input": {
        "narration": "So which one should you buy, the iPhone 16 or the Pixel 9?",
        "beat_role": "comparison",
        "duration_sec": 5,
        "available_images": 2,
        "fps": 30,
        "beat_start_sec": 300.2,
        "beat_end_sec": 305.2,
        "sync_keyword": "iPhone 16 or the Pixel 9",
        "keyword_offset_sec": 1.0,
        "prev_beat_role": "section_start",
        "next_beat_role": "data_comparison"
      },
      "output": {
        "match": {
          "category": "media_image",
          "motion": "halves_slide_in",
          "description": "Two product photos side by side filling the screen, sliding in from each side with a label on each half.",
          "use_when": "Visually comparing two things when an image of each is available.",
          "tags": [
            "split screen",
            "versus",
            "comparison",
            "two images"
          ]
        },
        "editing": {
          "beat_role": "comparison",
          "energy": "medium",
          "tone": [
            "neutral"
          ],
          "layer": "full_screen",
          "sync_target": "keyword",
          "exit": "hard_cut"
        },
        "timing": {
          "fps": 30,
          "clip_start_sec": 300.3,
          "clip_end_sec": 305.2,
          "duration_frames": 147,
          "entrance_frames": 30,
          "sync_frame": 27,
          "sync_lands_at_sec": 301.2
        },
        "requirements": {
          "covers_footage": True,
          "placement": "full_frame",
          "item_count": 2,
          "image_count": 2,
          "transition_role": "null"
        },
        "props": {
          "images": [
            "<image_url_1>",
            "<image_url_2>"
          ],
          "leftLabel": "iPhone 16",
          "rightLabel": "Pixel 9"
        }
      }
    },
    {
      "_maps_to": "intro_lower_third",
      "input": {
        "narration": "To understand this better, I spoke to Dr. Meera Iyer, an economist at IIM Bangalore.",
        "beat_role": "speaker_intro",
        "duration_sec": 5,
        "available_images": 0,
        "fps": 30,
        "beat_start_sec": 88.0,
        "beat_end_sec": 93.0,
        "sync_keyword": "Dr. Meera Iyer",
        "keyword_offset_sec": 1.0,
        "prev_beat_role": "emphasis",
        "next_beat_role": "quote"
      },
      "output": {
        "match": {
          "category": "lower_third",
          "motion": "bar_slides_in_from_left",
          "description": "Name bar in the bottom-left corner showing a person's name and role while the footage continues.",
          "use_when": "Introducing a speaker, guest, expert or location.",
          "tags": [
            "lower third",
            "name",
            "speaker",
            "corner"
          ]
        },
        "editing": {
          "beat_role": "speaker_intro",
          "energy": "low",
          "tone": [
            "neutral",
            "serious"
          ],
          "layer": "overlay",
          "sync_target": "keyword",
          "exit": "hard_cut"
        },
        "timing": {
          "fps": 30,
          "clip_start_sec": 89.0,
          "clip_end_sec": 93.0,
          "duration_frames": 120,
          "entrance_frames": 35,
          "sync_frame": 0,
          "sync_lands_at_sec": 89.0
        },
        "requirements": {
          "covers_footage": False,
          "placement": "bottom_left",
          "item_count": "null",
          "image_count": 0,
          "transition_role": "null"
        },
        "props": {
          "name": "Dr. Meera Iyer",
          "role": "Economist, IIM Bangalore"
        }
      }
    },
    {
      "_maps_to": "chapter_title",
      "input": {
        "narration": "Part two. The crash. What went wrong in 2008.",
        "beat_role": "section_start",
        "duration_sec": 4,
        "available_images": 0,
        "fps": 30,
        "beat_start_sec": 240.0,
        "beat_end_sec": 244.0,
        "sync_keyword": "Part two",
        "keyword_offset_sec": 0.0,
        "prev_beat_role": "visual_broll",
        "next_beat_role": "visual_broll"
      },
      "output": {
        "match": {
          "category": "title_card",
          "motion": "number_scales_lines_extend",
          "description": "Full-screen dark title card with a large chapter number, then the chapter title and a short subtitle.",
          "use_when": "Starting a new numbered chapter or part of a long video.",
          "tags": [
            "chapter",
            "section",
            "title",
            "number"
          ]
        },
        "editing": {
          "beat_role": "section_start",
          "energy": "low",
          "tone": [
            "cinematic",
            "serious"
          ],
          "layer": "full_screen",
          "sync_target": "beat_start",
          "exit": "hard_cut"
        },
        "timing": {
          "fps": 30,
          "clip_start_sec": 240.0,
          "clip_end_sec": 244.0,
          "duration_frames": 120,
          "entrance_frames": 40,
          "sync_frame": 10,
          "sync_lands_at_sec": 240.33
        },
        "requirements": {
          "covers_footage": True,
          "placement": "full_frame",
          "item_count": "null",
          "image_count": 0,
          "transition_role": "null"
        },
        "props": {
          "chapter": 2,
          "title": "The Crash",
          "subtitle": "What went wrong in 2008"
        }
      }
    },
    {
      "_maps_to": "fade_through_black",
      "input": {
        "narration": "[cut] Ten years later, everything had changed.",
        "beat_role": "transition",
        "duration_sec": 1,
        "available_images": 0,
        "fps": 30,
        "beat_start_sec": 239.5,
        "beat_end_sec": 240.5,
        "sync_keyword": "[cut]",
        "keyword_offset_sec": 0.5,
        "prev_beat_role": "visual_broll",
        "next_beat_role": "section_start"
      },
      "output": {
        "match": {
          "category": "transition",
          "motion": "fade_black_and_back",
          "description": "Calm dip to black and back out, centered on the cut between two scenes.",
          "use_when": "Time jumps, chapter changes or a quiet pause between scenes.",
          "tags": [
            "fade",
            "black",
            "time jump",
            "calm"
          ]
        },
        "editing": {
          "beat_role": "transition",
          "energy": "low",
          "tone": [
            "cinematic",
            "serious"
          ],
          "layer": "transition",
          "sync_target": "cut",
          "exit": "clears_itself"
        },
        "timing": {
          "fps": 30,
          "clip_start_sec": 239.5,
          "clip_end_sec": 240.5,
          "duration_frames": 30,
          "entrance_frames": 30,
          "sync_frame": 15,
          "sync_lands_at_sec": 240.0
        },
        "requirements": {
          "covers_footage": False,
          "placement": "full_frame",
          "item_count": "null",
          "image_count": 0,
          "transition_role": "across_cut"
        },
        "props": {}
      }
    },
    {
      "_maps_to": "vignette_pulse",
      "input": {
        "narration": "And then, at 2 a.m., the phone rang. Nobody expected what came next.",
        "beat_role": "mood",
        "duration_sec": 6,
        "available_images": 0,
        "fps": 30,
        "beat_start_sec": 150.0,
        "beat_end_sec": 156.0,
        "sync_keyword": "2 a.m.",
        "keyword_offset_sec": 1.8,
        "prev_beat_role": "visual_broll",
        "next_beat_role": "reveal"
      },
      "output": {
        "match": {
          "category": "overlay_fx",
          "motion": "dark_edges_pulse",
          "description": "Dark vignette pulsing around the edges of the footage to build tension, with no text.",
          "use_when": "Suspense, mystery or tension where the footage should carry the moment.",
          "tags": [
            "suspense",
            "vignette",
            "mood",
            "tension"
          ]
        },
        "editing": {
          "beat_role": "mood",
          "energy": "low",
          "tone": [
            "dramatic",
            "cinematic"
          ],
          "layer": "overlay",
          "sync_target": "beat_start",
          "exit": "hard_cut"
        },
        "timing": {
          "fps": 30,
          "clip_start_sec": 150.0,
          "clip_end_sec": 156.0,
          "duration_frames": 180,
          "entrance_frames": 0,
          "sync_frame": 0,
          "sync_lands_at_sec": 150.0
        },
        "requirements": {
          "covers_footage": False,
          "placement": "full_frame",
          "item_count": "null",
          "image_count": 0,
          "transition_role": "null"
        },
        "props": {}
      }
    }
  ]
}


async def scene_breakdown(script):
    _start_token_tracking()
    SCENES_BRAKDOWN_PROMPT = f"""
    You are a professional video script scene segmentation engine.

    Your task is to break the provided script into multiple scenes based strictly
    on the script's content and narrative flow.

    SCENE SEGMENTATION RULES:
    - Read the entire script before segmenting it.
    - Divide the script into logical, self-contained scenes.
    - Each scene should represent one clear idea, event, action, location, moment,
    or visual concept.
    - Create a new scene when the subject, idea, action, location, time, or visual
    context meaningfully changes.
    - Do NOT split sentences arbitrarily.
    - Do NOT merge unrelated ideas into the same scene.
    - Preserve the original order of the script.
    - Every part of the script must belong to exactly one scene.
    - Do not invent information.
    - Do not rewrite or summarize the script.
    - Keep the original script text inside each scene.
    - Avoid unnecessarily small scenes.
    - Avoid extremely large scenes containing multiple distinct ideas.

    OUTPUT FORMAT:
    Return ONLY a valid JSON array.
    Do not include markdown.
    Do not include explanations.
    Do not include ```json.
    Do not include any text before or after the JSON array.

    Each scene must have this structure:

    [
    {{
        "id": 1,
        "script": "Exact portion of the original script"
    }},
    {{
        "id": 2,
        "script": "Exact portion of the original script"
    }}
    ]

    IMPORTANT:
    - The "script" field must contain the exact original text.
    - Do not change wording.
    - Do not omit text.
    - Do not duplicate text between scenes.
    - Do not leave any part of the original script outside the scenes.
    - scene_number must start at 1 and increase sequentially.

    SCRIPT TO SEGMENT:
    {script}
    """

    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {
                        "role": "user",
                        "content": SCENES_BRAKDOWN_PROMPT
                    }
                ],
                stream=False,
            )
        )

        _record_token_usage("scene_breakdown", res)

        response = res.choices[0].message.content.strip()

        return json.loads(response)

    except Exception as e:
        print(f"[Scene Breakdown] failed: {e}")
        return e



fish_audio_client = FishAudio(
    api_key=os.getenv("FISH_AUDIO_API_KEY")
)    

async def generate_audio_for_scene(scene_text,modelId,userId):
    try:
        audio = fish_audio_client.tts.convert(
            text=scene_text,
            reference_id=modelId,
        )   
        audio_bytes = bytes(audio)
        audio_id = str(uuid.uuid4())
        path = f"{userId}/{audio_id}.mp3"
        supabase.storage.from_("generated-audio").upload(
            path,
            audio_bytes,
            {"content-type": "audio/mpeg"}
        )
        return supabase.storage.from_("generated-audio").get_public_url(path)

    except Exception as e:
        print(e)    


async def get_word_level_time_stamps(scenes: list):

    for scene in scenes:
        audio_url = scene["audio_url"]
        audio_path = None

        try:
            audio_data = requests.get(audio_url).content

            with tempfile.NamedTemporaryFile(
                suffix=".mp3",
                delete=False
            ) as f:
                f.write(audio_data)
                audio_path = f.name

            audio = whisperx.load_audio(audio_path)

            result = whisper_model.transcribe(
                audio,
                batch_size=16
            )

            aligned = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                "cpu",
                return_char_alignments=False
            )

            word_timestamps = []

            for segment in aligned["segments"]:
                for word in segment.get("words", []):
                    word_timestamps.append({
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"]
                    })

            scene["word_timestamps"] = word_timestamps

        except Exception as e:
            print(f"Error processing scene: {e}")
            scene["word_timestamps"] = []

        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

    return scenes



async def add_directions_for_scene(scene_text, word_timestamps):

    word_count = len(word_timestamps)
    last_word_index = word_count - 1

    SCENE_BEAT_DIRECTOR = f""" 
        
    You are a production-grade documentary video director and template-aware editing agent.

    TASK
    Divide the narration into coherent visual beats. For each beat, choose:

    1. B-roll
    2. B-roll+overlay_animation
    3. full_screen_animation

    For animation beats, select the best matching template from the supplied template metadata and populate its props with the information that should appear on screen.

    INPUT

    * Scene narration with WhisperX word-level indices.
    * Animation template repository metadata as JSON.

    The template metadata is the source of truth for available templates and their supported structure.

    BEATS

    * Do not generate or calculate timestamps.
    * Return only start_word_index and end_word_index; backend calculates exact times.
    * Preserve narration order.
    * Every word belongs to exactly one beat.
    * No gaps, overlaps, duplicates, or missing words.
    * First beat starts at 0; final beat ends at the last word.
    * Prefer sentence endings and natural changes in visual meaning.
    * Target ~8–15 seconds per beat, but prioritize visual coherence.

    WORD INDEX SAFETY — EXTREMELY STRICT

    The supplied WORD-LEVEL TIMESTAMPS contains exactly {word_count} entries.

    The valid word index range is:

    0 through {last_word_index}

    IMPORTANT:

    * start_word_index MUST be >= 0.
    * end_word_index MUST be >= 0.
    * start_word_index MUST be <= {last_word_index}.
    * end_word_index MUST be <= {last_word_index}.
    * NEVER output an index greater than {last_word_index}.
    * NEVER output an index below 0.
    * The first beat MUST start at 0.
    * The final beat MUST end at {last_word_index}.
    * Every index from 0 through {last_word_index} MUST be covered exactly once.
    * Do NOT estimate or calculate the word count from the narration.
    * Use ONLY the indices that exist in the supplied WORD-LEVEL TIMESTAMPS.
    * Before returning the JSON, internally verify that no beat contains an index outside 0-{last_word_index}.

    VISUAL SELECTION

    "B-roll"
    Use for real people, places, objects, events, environments, machines, historical footage, and other subjects best represented by footage.

    Required:

    * Exactly 6 concrete Pexels-searchable keywords.

    "B-roll+overlay_animation"
    Use when footage should be enhanced by an animation such as statistics, numbers, charts, lists, names, dates, quotes, text emphasis, labels, lower thirds, callouts, icons, or visual effects.

    Required:

    * Exactly 6 concrete Pexels-searchable keywords.
    * Select a template with editing.layer = "overlay".
    * Populate its supported props.

    "full_screen_animation"
    Use when animation should replace footage for processes, comparisons, charts, timelines, diagrams, data visualization, chapter cards, or concepts better represented graphically.

    Required:

    * Select a template with editing.layer = "full_screen".
    * Populate its supported props.

    TEMPLATE SELECTION

    * Read the supplied template metadata before selecting an animation.
    * Match the beat semantically and functionally using the template metadata.
    * Consider match, editing, requirements, and props.
    * Select only templates present in the input.
    * If no suitable animation template exists, use B-roll.
    * Never invent or modify template metadata.

    PROPS
    For animation beats, populate the selected template's existing props structure.

    * Preserve property names, nesting, and structure.
    * Populate only supported properties.
    * Use only information supported by the narration and supplied context.
    * Never invent facts, numbers, names, dates, quotes, statistics, examples, or assets.
    * Preserve numerical values, labels, relationships, order, and meaning.
    * Keep displayed text concise and screen-readable.
    * Respect template requirements such as item_count, image_count, placement, and other supplied constraints.
    * Do not generate timestamps, timing values, or unsupported props.

    CONTENT MAPPING
    Map the beat's information to the template according to its supported structure:

    * Statistic → value/number fields.
    * Multiple values or trends → data/series structure with correct labels and values.
    * List → items structure.
    * Ordered process → sequential steps.
    * Two-way comparison → comparison structure with correct entity, label, and left/right relationships.
    * Single or multiple images → image fields/arrays with correct labels when supported.
    * Person/expert → name + role/title.
    * Place/location → supported name/location fields.
    * Key phrase → text + highlight fields.
    * Quote → quote/text field using supplied wording.
    * Chapter/section → chapter + title + subtitle.
    * Title + explanation → supported title/subtitle/body fields.
    * Text + icon → maintain correct text/icon pairing.
    * List + icons → maintain one-to-one item/icon alignment.
    * Number + label → preserve each value/label relationship.
    * Chart + labels → preserve category/value relationships.
    * Image + label → preserve image/label relationships.
    * Sequential animation → preserve narration order.
    * Callout/emphasis → highlight only the relevant information.
    * Mood/effect or transition → use the template without unnecessary content.
    * Specialized template → follow its supplied props structure directly.

    For all mappings:

    * Preserve relationships between text, numbers, labels, icons, and images.
    * Use the minimum content needed to communicate the beat.
    * Do not duplicate information unnecessarily.
    * Never force content into an unsuitable field.

    KEYWORDS
    For B-roll and B-roll+overlay_animation:

    * Exactly 6 concrete Pexels-searchable keywords.
    * Describe visible subjects/scenes.
    * Avoid vague terms such as technology, concept, idea, or innovation.

    OUTPUT
    Return ONLY valid JSON.

OUTPUT FORMAT — EXTREMELY STRICT

The response MUST be a JSON ARRAY at the root level.

The root JSON value MUST start with `[` and end with `]`.

NEVER return a JSON object as the root value.

NEVER wrap the array inside:
- {{"beats": [...]}}
- {{"directions": [...]}}
- {{"result": [...]}}
- {{"data": [...]}}
- {{"visual_beats": [...]}}

NEVER return a single beat object.

NEVER return markdown.
NEVER return ```json.
NEVER return explanations or text outside the JSON array.

The ONLY valid response format is:

[
  {{
    "type": "B-roll",
    "start_word_index": 0,
    "end_word_index": 12,
    "keywords": [
      "...",
      "...",
      "...",
      "...",
      "...",
      "..."
    ]
  }},
  {{
    "type": "B-roll+overlay_animation",
    "start_word_index": 13,
    "end_word_index": 25,
    "keywords": [
      "...",
      "...",
      "...",
      "...",
      "...",
      "..."
    ],
    "template": {{
      "match": {{}}
    }}
  }},
  {{
    "type": "full_screen_animation",
    "start_word_index": 26,
    "end_word_index": 42,
    "template": {{
      "match": {{}}
    }}
  }}
]

IMPORTANT ROOT STRUCTURE RULES:

- The root MUST be an array.
- Every element inside the array MUST be one beat object.
- There is NO "beats" property.
- There is NO "directions" property.
- There is NO wrapper object.
- Do not add any other top-level property.
- If there is only one beat, still return an array containing that one object.
- If there are 20 beats, return one array containing 20 beat objects.
- The first character of the response MUST be `[`.
- The last character of the response MUST be `]`.

Each beat object MUST contain:

For B-roll:
{{
  "type": "B-roll",
  "start_word_index": <integer>,
  "end_word_index": <integer>,
  "keywords": [<exactly 6 strings>]
}}

For B-roll+overlay_animation:
{{
  "type": "B-roll+overlay_animation",
  "start_word_index": <integer>,
  "end_word_index": <integer>,
  "keywords": [<exactly 6 strings>],
  "template": <selected template object>
}}

For full_screen_animation:
{{
  "type": "full_screen_animation",
  "start_word_index": <integer>,
  "end_word_index": <integer>,
  "template": <selected template object>
}}


    FINAL CHECK
    * Complete consecutive word coverage.
    * First index = 0; final index = {last_word_index}.
    * Exactly 6 keywords for B-roll types.
    * Selected template exists in supplied metadata.
    * Correct animation layer selected.
    * Props conform to the selected template.
    * Content is supported by the input.
    * Relationships between props are preserved.
    * No invented information.
    * No start_word_index or end_word_index outside 0-{last_word_index}.

    The final response must be ONLY the JSON array.
"""

    prompt = f"""
    {SCENE_BEAT_DIRECTOR}

    SCENE:
    {scene_text}

    WORD-LEVEL TIMESTAMPS:
    {json.dumps(word_timestamps, ensure_ascii=False)}

    Metadata JSON:
    {TEMPLATE_METADATA}

    Now divide this scene into visual beats.
    """

    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                stream=False,
            )
        )

        _record_token_usage("beat_breakdown", res)

        response = res.choices[0].message.content.strip()

        return json.loads(response)

    except Exception as e:
        print(f"[Scene Breakdown] failed: {e}")
        return e



async def align_beats_to_voice(directions: list, word_timestamps: list):

    print("aligning beats to voice")
    print("word timestamps count:", len(word_timestamps))

    for direction in directions:
        start_index = int(direction["start_word_index"])
        end_index = int(direction["end_word_index"])

        print(
            "beat:",
            start_index,
            "->",
            end_index,
            "available words:",
            len(word_timestamps)
        )

        direction["start"] = word_timestamps[start_index]["start"]
        direction["end"] = word_timestamps[end_index]["end"]

    return directions

    

PEXELS_HEADERS = {
    "Authorization": PEXELS_API_KEY
}

def search_pexel_media(keywords: list):

    photos = []
    videos = []

    for keyword in keywords:


        photo_response = requests.get(
            "https://api.pexels.com/v1/search",
            headers=PEXELS_HEADERS,
            params={
                "query": keyword,
                "orientation": "landscape",
                "per_page": 2
            },
            timeout=10
        )

        if photo_response.ok:
            photo_data = photo_response.json()

            for photo in photo_data.get("photos", []):
                photos.append({
                    "id": photo["id"],
                    "type": "photo",
                    "query": keyword,
                    "url": photo["url"],
                    "image_url": photo["src"]["large2x"],
                    "width": photo["width"],
                    "height": photo["height"],
                    "photographer": photo["photographer"]
                })


        video_response = requests.get(
            "https://api.pexels.com/v1/videos/search",
            headers=PEXELS_HEADERS,
            params={
                "query": keyword,
                "orientation": "landscape",
                "per_page": 5
            },
            timeout=10
        )

        if video_response.ok:
            video_data = video_response.json()

            for video in video_data.get("videos", []):

                video_files = video.get("video_files", [])

                video_file = next(
                    (
                        f for f in video_files
                        if f.get("width") >= 1280
                        and f.get("height") >= 720
                    ),
                    None
                )

                if not video_file and video_files:
                    video_file = video_files[0]

                if video_file:
                    videos.append({
                        "id": video["id"],
                        "type": "video",
                        "query": keyword,
                        "url": video["url"],
                        "video_url": video_file["link"],
                        "width": video_file["width"],
                        "height": video_file["height"],
                        "duration": video["duration"]
                    })
    return {
        "photos": photos,
        "videos": videos
    }



def get_best_animation_template(match_obj):
    model = _get_st_model()
    match_text = " ".join([
        str(match_obj.get("category", "")),
        str(match_obj.get("motion", "")),
        str(match_obj.get("description", "")),
        str(match_obj.get("use_when", "")),
        " ".join(match_obj.get("tags", []))
    ])

    embedding = model.encode(
        match_text,
        normalize_embeddings=True
    ).tolist()


    result = supabase.rpc(
        "match_animations_template",
        {
            "query_embedding": embedding,
            "match_threshold": 0.0,
            "match_count": 1
        }
    ).execute()

    matches = result.data or []

    if not matches:
        return None

    return matches[0]["name"]




class Editvideo(BaseModel):
    userId:str
    script:str
    langCode:str
    voice : str
    durationMinutes:int 


@app.post("/edit-video")
async def edit_video(body: Editvideo):
    try:
        print("Breaking down of script into scenes")

        scenes = await scene_breakdown(body.script)

        print("generating audio for scenes")

        for scene in scenes:
            audio_url = await generate_audio_for_scene(
                scene_text=scene["script"],
                modelId=body.voice,
                userId=body.userId
            )

            scene["audio_url"] = audio_url

        print("generating word level timestamps for scenes")

        scenes = await get_word_level_time_stamps(scenes)

        print("generating directions for scenes")

        for scene in scenes:
            directions = await add_directions_for_scene(
                scene["script"],
                scene["word_timestamps"]
            )

            scene["directions"] = directions

        print("aligning directions as per voice")

        for scene in scenes:
            scene["directions"] = await align_beats_to_voice(
                scene["directions"],
                scene["word_timestamps"]
            )

        for scene in scenes:
            for direction in scene["directions"]:
                if direction["type"] in ["B-roll", "B-roll+overlay_animation"]:
                    keywords = direction["keywords"]
                    direction["asserts"] = search_pexel_media(keywords)

                if direction["type"] in ["B-roll+overlay_animation", "full_screen_animation"]:
                    print("matching template metadata")
                    match_obj = direction["template"]["match"]
                    template_name = get_best_animation_template(match_obj)
                    direction["template"]["name"] = template_name
                    print("selected template:", template_name)
         
      
        supabase.table("videos").insert({
            "user_id": body.userId,
            "script": body.script,
            "voice": body.voice,
            "lang_code": body.langCode,
            "timeline": {
                "scenes": scenes
            },
            "timeline_version": 1
        }).execute()


        print("saved into db")


        return scenes

    except Exception as e:
        print(e)
























































































































































































RENDER_SERVICE_URL = os.getenv("RENDER_SERVICE_URL", "http://62.83.19.227:8000")
RENDER_QUEUE_MAX_CONCURRENT = int(os.getenv("RENDER_QUEUE_MAX_CONCURRENT", "1"))
RENDER_QUEUE_POLL_SECONDS = int(os.getenv("RENDER_QUEUE_POLL_SECONDS", "5"))
RENDER_QUEUE_HTTP_TIMEOUT = float(os.getenv("RENDER_QUEUE_HTTP_TIMEOUT", "1800"))  



class RenderQueueRequest(BaseModel):
    video_id: str
    orientation: Literal["landscape", "portrait"] = "landscape"
 
 
@app.post("/render/queue")
async def enqueue_render(request: RenderQueueRequest):
    """Adds a video to the render waitlist instead of rendering it
    immediately. A background worker (started at app startup — see
    _render_queue_worker) picks entries up in FIFO order, capped at
    RENDER_QUEUE_MAX_CONCURRENT actually rendering at once, and dispatches
    each one to RENDER_SERVICE_URL — the actual render box — via a plain
    HTTP call to its existing /render/{video_id} endpoint. This process
    doesn't have to be the render box itself; it can sit on your main API
    server and just dispatch to wherever the render service actually
    lives."""
    try:
        row = supabase.table("render_queue").insert({
            "video_id": request.video_id,
            "orientation": request.orientation,
            "status": "pending",
        }).execute()
    except Exception as e:
        print(f"[render-queue] failed to enqueue {request.video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add to render queue")
 
    if not row.data:
        raise HTTPException(status_code=500, detail="Failed to add to render queue")
 
    entry = row.data[0]
    try:
        pending_ahead = (
            supabase.table("render_queue")
            .select("id", count="exact")
            .eq("status", "pending")
            .lt("created_at", entry["created_at"])
            .execute()
        )
        position = (pending_ahead.count or 0) + 1
    except Exception:
        position = None
 
    return {
        "queue_id": entry["id"], "video_id": request.video_id, "status": "pending",
        "position_in_queue": position,
    }
 
 
@app.get("/render/queue/{queue_id}")
async def get_render_queue_status(queue_id: str):
    """Check one queued render's status: pending, processing, completed
    (with final_video_url), or failed (with error_message)."""
    try:
        row = supabase.table("render_queue").select("*").eq("id", queue_id).maybe_single().execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not row.data:
        raise HTTPException(status_code=404, detail="Queue entry not found")
    return row.data
 
 
@app.get("/render/queue")
async def list_render_queue(status: Optional[str] = None, limit: int = 50):
    """Lists queue entries, most recent first. Pass status= to filter to
    just pending/processing/completed/failed."""
    try:
        query = supabase.table("render_queue").select("*").order("created_at", desc=True).limit(min(limit, 200))
        if status:
            query = query.eq("status", status)
        rows = query.execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"entries": rows.data or []}
 
 
async def _process_one_queued_render(entry: dict) -> None:
    """Dispatches ONE queue entry to the render service over HTTP and
    records the outcome. Runs inside the semaphore in _render_queue_worker,
    so at most RENDER_QUEUE_MAX_CONCURRENT of these are ever in flight —
    that's the actual point of this whole system: protecting the render
    box from every /render call landing on it simultaneously."""
    queue_id = entry["id"]
    video_id = entry["video_id"]
    orientation = entry.get("orientation") or "landscape"
 
    try:
        supabase.table("render_queue").update({
            "status": "processing",
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }).eq("id", queue_id).execute()
    except Exception as e:
        print(f"[render-queue] failed to mark {queue_id} as processing (continuing anyway): {e}")
 
    try:
        async with httpx.AsyncClient(timeout=RENDER_QUEUE_HTTP_TIMEOUT) as client:
            resp = await client.post(
                f"{RENDER_SERVICE_URL}/render/{video_id}",
                json={"orientation": orientation},
            )
            resp.raise_for_status()
            result = resp.json()
 
        supabase.table("render_queue").update({
            "status": "completed",
            "final_video_url": result.get("final_video_url"),
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }).eq("id", queue_id).execute()
        print(f"[render-queue] {queue_id} ({video_id}) completed")
 
    except Exception as e:
        print(f"[render-queue] {queue_id} ({video_id}) failed: {e}")
        try:
            supabase.table("render_queue").update({
                "status": "failed",
                "error_message": str(e)[:2000],
                "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }).eq("id", queue_id).execute()
        except Exception as e2:
            print(f"[render-queue] also failed to record the failure for {queue_id}: {e2}")
 
 
async def _recover_stale_processing_jobs() -> None:
    """Runs once at startup. Any row still marked "processing" at boot
    time is orphaned — the process that was working on it is the one
    that just (re)started, so nothing is actually running it anymore. A
    server restart mid-render otherwise leaves that row stuck showing
    "processing" forever, since the worker only ever polls for "pending"
    and the in-memory in_flight tracking that would have known about it
    is gone along with the old process.
 
    Safe ONLY under the single-worker assumption this whole queue is
    built for. If this is ever run with multiple worker processes, this
    would incorrectly reset a job another still-live process is
    legitimately processing, causing it to be dispatched twice — don't
    add this without pgmq-style visibility timeouts (or similar) once
    you're actually running more than one worker.
    """
    try:
        stuck = supabase.table("render_queue").select("id", "video_id").eq("status", "processing").execute()
        for row in (stuck.data or []):
            supabase.table("render_queue").update({"status": "pending"}).eq("id", row["id"]).execute()
            print(f"[render-queue] recovered orphaned job {row['id']} ({row['video_id']}) — was stuck 'processing' from before this restart, reset to 'pending'")
    except Exception as e:
        print(f"[render-queue] failed to recover stale processing jobs on startup: {e}")
 
 
async def _render_queue_worker() -> None:
    """Background loop: every RENDER_QUEUE_POLL_SECONDS, picks up pending
    entries (oldest first) and processes them, never exceeding
    RENDER_QUEUE_MAX_CONCURRENT in flight at once. Runs for the lifetime
    of the process — started once from the startup hook below."""
    await _recover_stale_processing_jobs()
    print(f"[render-queue] worker started (max concurrent={RENDER_QUEUE_MAX_CONCURRENT}, polling every {RENDER_QUEUE_POLL_SECONDS}s, dispatching to {RENDER_SERVICE_URL})")
    in_flight: set = set()
    while True:
        try:
            in_flight = {t for t in in_flight if not t.done()}
            free_slots = RENDER_QUEUE_MAX_CONCURRENT - len(in_flight)
            if free_slots > 0:
                pending = (
                    supabase.table("render_queue")
                    .select("*")
                    .eq("status", "pending")
                    .order("created_at")
                    .limit(free_slots)
                    .execute()
                )
                for entry in (pending.data or []):
                    task = asyncio.create_task(_process_one_queued_render(entry))
                    in_flight.add(task)
        except Exception as e:
            print(f"[render-queue] worker loop error (will retry next poll): {e}")
 
        await asyncio.sleep(RENDER_QUEUE_POLL_SECONDS)
