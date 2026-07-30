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
from deep_translator import GoogleTranslator


import requests
import numpy as np
from sqlalchemy import create_engine, text, bindparam
from sklearn.feature_extraction.text import HashingVectorizer
from fastapi import HTTPException
from openai import OpenAI
import trafilatura



SCRIPTS_UNIVERSAL_TABLE = "scripts_universal"
IDEAS_HYDE_DOC_COUNT = 5

SUPPORTED_LANGUAGES = {
    "english": "en",
    "hindi": "hi",
    "gujarati": "gu",
    "kannada": "kn",
    "bengali": "bn",
    "malayalam": "ml",
    "telugu": "te",
    "tamil": "ta",
    "marathi": "mr",
    "odia": "or",
    "punjabi": "pa",
}
 
DEFAULT_LANGUAGE = "English"
 
TRANSLATE_CHUNK_MAX_CHARS = 4000

_TRANSLATE_ARRAY_SEP_LINE = "\u2021\u2021\u2021ITEM\u2021\u2021\u2021"  
_LEADING_NUMBERING_RE = re.compile(r"^\s*(?:[\-\*\u2022]|\d+[\.\)])\s*")
 
TRANSLATION_QC_ARRAY_SYSTEM_PROMPT = """
You are a professional {language} language editor and translation QC specialist
working on YouTube metadata (titles, descriptions, or hashtag phrases).
 
You will be given:
1. The ORIGINAL English text — exactly {item_count} item(s), each separated
   by a line that contains exactly the token: {sep_token}
2. A DRAFT machine translation of the same text into {language}, using the
   same separator token
 
## Task
- Fix grammar, spelling, and word order so each item reads naturally to a
  native {language} speaker
- Keep each item short and punchy, suitable for a YouTube title, description,
  or hashtag phrase
- Translate EVERY item independently. Do NOT merge, deduplicate, reorder,
  drop, summarize, or combine items — even if two items look similar or
  redundant to you, keep them as separate items in the same position
- Do NOT add numbering, bullets, or any new formatting
- Preserve names, numbers, and proper nouns accurately
 
## CRITICAL OUTPUT REQUIREMENT
Your output MUST contain EXACTLY {item_count} item(s) separated by the exact
token "{sep_token}" on its own line — the same count as the input, no more,
no fewer. This is a hard structural requirement, not a style suggestion.
 
## Output
Return ONLY the corrected {language} text, items separated by "{sep_token}"
on their own line, same order, same count — no preamble, no notes, no
markdown, no explanations.
"""



async def refine_array_translation_with_llm(
    original_text: str, draft_translation: str, target_language: str, item_count: int
) -> str:
    if not draft_translation:
        return draft_translation
 
    system_prompt = TRANSLATION_QC_ARRAY_SYSTEM_PROMPT.format(
        language=target_language,
        item_count=item_count,
        sep_token=_TRANSLATE_ARRAY_SEP_LINE,
    )
    user_prompt = f"""ORIGINAL (English):
{original_text}
 
DRAFT TRANSLATION ({target_language}):
{draft_translation}
"""
 
    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                temperature=0.15,   
                top_p=0.85,        

            )
        )
        _record_token_usage(f"translation_qc_array_{target_language.lower()}", res)
        refined = (res.choices[0].message.content or "").strip()
        return refined or draft_translation
    except Exception as e:
        print(f"[TRANSLATE] array LLM QC pass failed for {target_language}: {e} — using library draft as-is")
        return draft_translation



_PURE_DIGIT_TAG_RE = re.compile(r"^#?\d+$")
_MIN_HASHTAG_LEN = 3  

async def translate_hashtag_sets(
    hashtag_sets: list[list[str]], target_language: str
) -> list[list[str]]:
    target_language = _normalize_language(target_language)
    if target_language == "English" or not hashtag_sets:
        return hashtag_sets
 
    flat_phrases: list[str] = []
    set_sizes: list[int] = []
    for tag_set in hashtag_sets:
        set_sizes.append(len(tag_set))
        flat_phrases.extend(_hashtag_to_phrase(tag) for tag in tag_set)
 
    if not flat_phrases:
        return hashtag_sets
 
    try:
        translated_phrases = await translate_array_full_pipeline(flat_phrases, target_language)
    except Exception as e:
        print(f"[TRANSLATE] hashtag translation failed, keeping English hashtags: {e}")
        return hashtag_sets
 
    if len(translated_phrases) != len(flat_phrases):
        print("[TRANSLATE] hashtag translation count mismatch, keeping English hashtags")
        return hashtag_sets
 
    rebuilt: list[list[str]] = []
    cursor = 0
    for tag_set, size in zip(hashtag_sets, set_sizes):
        translated_slice = translated_phrases[cursor: cursor + size]
        cursor += size
 
        tags: list[str] = []
        seen: set[str] = set()
 
        for original_tag, phrase in zip(tag_set, translated_slice):
            candidate = _keyword_to_hashtag(phrase)
 
            is_invalid = (
                not candidate
                or _PURE_DIGIT_TAG_RE.match(candidate)
                or len(candidate.lstrip("#")) < _MIN_HASHTAG_LEN
                or candidate.lower() in seen
            )
 
            if is_invalid:
                candidate = original_tag if original_tag.lower() not in seen else None
 
            if candidate and candidate.lower() not in seen:
                seen.add(candidate.lower())
                tags.append(candidate)
        if len(tags) < size:
            for original_tag in tag_set:
                if len(tags) >= size:
                    break
                if original_tag.lower() not in seen:
                    seen.add(original_tag.lower())
                    tags.append(original_tag)
 
        rebuilt.append(tags)
 
    return rebuilt


async def translate_array_full_pipeline(items: list[str], target_language: str) -> list[str]:
    if not items:
        return items
 
    target_language = _normalize_language(target_language)
    if target_language == "English":
        return items
 
    joined = f"\n{_TRANSLATE_ARRAY_SEP_LINE}\n".join(items)
 
    parts: list[str] = []
    try:
        draft = await translate_with_library(joined, target_language)
        refined = await refine_array_translation_with_llm(
            joined, draft, target_language, item_count=len(items)
        )
        raw_parts = [
            p.strip()
            for p in re.split(re.escape(_TRANSLATE_ARRAY_SEP_LINE), refined)
        ]
        # Strip any stray numbering/bullets the model might have added despite
        # instructions not to (this is what caused artifacts like a lone "#3").
        parts = [_LEADING_NUMBERING_RE.sub("", p).strip() for p in raw_parts if p.strip()]
    except Exception as e:
        print(f"[TRANSLATE] array batch translation failed: {e}")
        parts = []
 
    if len(parts) == len(items):
        return parts
 
    print(
        f"[TRANSLATE] batch array translation returned {len(parts)} part(s), "
        f"expected {len(items)} — falling back to per-item translation "
        f"(never drops items)"
    )
 
    results = []
    for item in items:
        try:
            translated = await translate_text_full_pipeline(item, target_language)
            results.append(translated or item)
        except Exception as e:
            print(f"[TRANSLATE] per-item fallback failed for '{item[:40]}...': {e} — keeping original")
            results.append(item)
    return results


_CAMEL_SPLIT_RE = re.compile(r'(?<!^)(?=[A-Z])')
_HASHTAG_WORD_RE = re.compile(r"[^\s#]+", re.UNICODE)
 
def _hashtag_to_phrase(hashtag: str) -> str:
        """Reverses '#artificialIntelligence' -> 'artificial Intelligence' so it
        can be sent through translation like normal text."""
        word = (hashtag or "").lstrip("#")
        spaced = _CAMEL_SPLIT_RE.sub(" ", word)
        return spaced.strip()



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
    """Validate/normalize whatever the client sent. Falls back to English on
    anything unrecognized so a bad value never breaks the pipeline."""
    if not language or not language.strip():
        return DEFAULT_LANGUAGE
    key = language.strip().lower()
    if key not in SUPPORTED_LANGUAGES:
        print(f"[LANG] unrecognized language '{language}', defaulting to English")
        return DEFAULT_LANGUAGE
    return "Odia" if key == "odia" else language.strip().title()
 
 
def _lang_code(language: str) -> str:
    return SUPPORTED_LANGUAGES.get(language.strip().lower(), "en")
 


def _chunk_text_for_translation(text_value: str, max_chars: int = TRANSLATE_CHUNK_MAX_CHARS) -> list[str]:
    """Splits on paragraph boundaries so we never cut a sentence mid-way and
    never exceed the translator's per-request character limit."""
    if len(text_value) <= max_chars:
        return [text_value]
 
    paragraphs = text_value.split("\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n" + para) if current else para
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
 



def _translate_with_library_sync(text_value: str, target_lang_code: str) -> str:
    chunks = _chunk_text_for_translation(text_value)
    translated_chunks = []
    for chunk in chunks:
        if not chunk.strip():
            translated_chunks.append(chunk)
            continue
        try:
            translated = GoogleTranslator(source="en", target=target_lang_code).translate(chunk)
            translated_chunks.append(translated or chunk)
        except Exception as e:
            print(f"[TRANSLATE] library translation failed for a chunk ({len(chunk)} chars): {e}")
            translated_chunks.append(chunk)  
    return "\n".join(translated_chunks)




async def translate_with_library(text_value: str, target_language: str) -> str:
    if not text_value:
        return text_value
    target_lang_code = _lang_code(target_language)
    return await asyncio.to_thread(_translate_with_library_sync, text_value, target_lang_code)
 


TRANSLATION_QC_SYSTEM_PROMPT = """
You are a professional {language} language editor and translation QC specialist.
 
You will be given:
1. The ORIGINAL English text
2. A DRAFT machine translation of that text into {language}
 
## Task
Produce a corrected, publication-ready {language} version of the text by:
- Fixing grammar, spelling, conjugation, gender agreement, and word order errors
- Making the phrasing sound natural and fluent to a native {language} speaker
  — not a literal/awkward machine translation
- Preserving the original meaning, tone, and factual content exactly
- Preserving names, numbers, statistics, proper nouns, and technical terms
  accurately (transliterate names naturally, never mistranslate them)
- Preserving any structural markers exactly as they appear in the original,
  such as segment labels in square brackets (e.g. "[Hook]", "[Climax]") —
  keep these bracketed labels in English exactly as-is, untranslated
- Keeping paragraph/line structure consistent with the original
 
## Output
Return ONLY the corrected {language} text. No preamble, no explanations, no
notes, no markdown, no side-by-side comparison — just the final text.
"""




async def generate_ideas_hyde_documents(topic: str, num_docs: int = IDEAS_HYDE_DOC_COUNT) -> list[str]:
    async def _one(idx: int) -> str:
        hyde_prompt = f"""
        # HYPOTHETICAL DOCUMENT GENERATOR (HDG v2)

## ROLE

You are a **Hypothetical Document Generator** for a Retrieval-Augmented Generation (RAG) system.

Given a user query, generate **exactly five hypothetical documents** that maximize semantic similarity with authoritative source material likely to exist in a large corpus of books.

These documents are **retrieval anchors**, not answers. They will be embedded and used to retrieve the most relevant passages from a vector database.

Your objective is to maximize semantic recall while maintaining high precision.

---

## INPUT

Topic: "{topic}"
---

## TASK

First infer the query's primary knowledge domain(s). A query may span multiple domains such as business, history, technology, psychology, philosophy, religion, science, politics, law, biography, economics, medicine, sociology, geography, finance, sports, or other academic disciplines.

Then generate **five complementary hypothetical documents**, each representing a distinct perspective naturally suited to the query. Examples of perspectives include historical context, conceptual foundations, chronology, mechanisms, research, technical implementation, stakeholders, economic impact, legal framework, scientific evidence, philosophical interpretation, practical applications, comparative analysis, controversies, or future developments.

Select the perspectives dynamically according to the topic rather than using a fixed template.

---

## DOCUMENT REQUIREMENTS

Each document should resemble an excerpt from a high-quality book, textbook, encyclopedia, scholarly work, or authoritative reference.

Each document must:

* focus on a unique semantic perspective
* naturally include relevant domain terminology
* include related concepts, synonyms, alternate terminology, broader and narrower concepts
* naturally reference important entities when they are strongly implied by the query (people, organizations, technologies, locations, historical events, theories, frameworks, institutions)
* emphasize conceptual relationships instead of isolated keywords
* avoid conversational language
* avoid directly answering the user's question
* avoid repetition across documents

---

## FACTUAL DISCIPLINE

The purpose is semantic retrieval, not factual completion.

Do not invent specific dates, statistics, quotations, study results, citations, named publications, financial figures, researcher names, or organizations unless they are explicitly present in the user's query or are universally inseparable from the topic.

When uncertain, describe concepts generically using the vocabulary expected in authoritative books.

Prioritize semantic relevance over fabricated specificity.

---

## STYLE

Write in an objective, information-dense academic style.

The documents should read like genuine reference material rather than generated summaries.

Avoid:

* opinions
* recommendations
* storytelling
* introductions
* conclusions
* speculative language

Prefer precise terminology, conceptual depth, and natural academic phrasing.

---

## LENGTH

Generate approximately **70–100 words** per document.

---

## OUTPUT

Generate exactly five documents.

Format:

Document 1: <text>

Document 2: <text>

Document 3: <text>

Document 4: <text>

Document 5: <text>

Return only the five documents.
        """
        try:
            res = await _openai_create_with_timeout(
                lambda: openai_client.chat.completions.create(
                    model="gpt-5.4-mini",
                    messages=[{"role": "user", "content": hyde_prompt}],
                    max_completion_tokens=400,
                    stream=False,
                    temperature=0.25, 
                    top_p=0.9,       

                )
            )
            _record_token_usage(f"generate-ideas HYDE #{idx + 1}", res)
            raw_doc = (res.choices[0].message.content or "").strip()

            if not raw_doc:
                try:
                    retry_res = await _openai_create_with_timeout(
                        lambda: openai_client.chat.completions.create(
                            model="gpt-5.4-mini",
                            messages=[{"role": "user", "content": hyde_prompt}],
                            max_completion_tokens=1500,
                            stream=False,
                        )
                    )
                    _record_token_usage(f"generate-ideas HYDE #{idx + 1} (retry)", retry_res)
                    raw_doc = (retry_res.choices[0].message.content or "").strip()
                except Exception as retry_exc:
                    print(f"[IDEAS-HYDE] doc #{idx + 1} retry failed: {retry_exc}")
                    raw_doc = ""

            return _cap_hyde_doc_tokens(raw_doc) if raw_doc else topic
        except Exception as exc:
            print(f"[IDEAS-HYDE] doc #{idx + 1} generation failed: {exc}")
            return topic

    docs = await asyncio.gather(*[_one(i) for i in range(num_docs)])
    return list(docs)



async def refine_translation_with_llm(
    original_text: str, draft_translation: str, target_language: str
) -> str:
    if not draft_translation:
        return draft_translation
 
    system_prompt = TRANSLATION_QC_SYSTEM_PROMPT.format(language=target_language)
    user_prompt = f"""ORIGINAL (English):
{original_text}
 
DRAFT TRANSLATION ({target_language}):
{draft_translation}
"""
 
    try:
        res = await _openai_create_with_timeout(
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
        _record_token_usage(f"translation_qc_{target_language.lower()}", res)
        refined = (res.choices[0].message.content or "").strip()
        return refined or draft_translation
    except Exception as e:
        print(f"[TRANSLATE] LLM grammar/QC pass failed for {target_language}: {e} — using library draft as-is")
        return draft_translation
 




async def translate_text_full_pipeline(text_value: str, target_language: str) -> str:
    if not text_value:
        return text_value
 
    target_language = _normalize_language(target_language)
    if target_language == "English":
        return text_value
 
    print(f"[TRANSLATE] translating text into {target_language} ({len(text_value)} chars)")
    try:
        draft = await translate_with_library(text_value, target_language)
        refined = await refine_translation_with_llm(text_value, draft, target_language)
        print(f"[TRANSLATE] translation into {target_language} complete ({len(refined)} chars)")
        return refined
    except Exception as e:
        print(f"[TRANSLATE] full pipeline failed for {target_language}, returning original English: {e}")
        return text_value




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

IDEAS_DB_CHUNKS_TO_LLM = IDEAS_HYDE_DOC_COUNT * 2      
IDEAS_WEB_SOURCES_TO_LLM = IDEAS_HYDE_DOC_COUNT * 2   
IDEAS_SEARCH_KEYWORD_COUNT = 10
SCRIPT_SEARCH_KEYWORD_COUNT = 10


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
SCRIPT_CREDITS_PER_MINUTE = 3
THUMBNAIL_CREDITS_PER_IMAGE = 20


def to_pgvector(embedding) -> str:
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"



_bge_model = None

from typing import List, Dict, Any

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
    print(f"[DB] Starting retrieval for topic: '{hyde_doc}'")

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
            for r in ddgs.text(keyword, max_results=max_results * 2, backend="html"):
                url = r.get("href") or r.get("url")
                snippet = r.get("body", "") or r.get("title", "")
                if not url or _is_blocked_source_url(url):
                    continue
                results.append((url, snippet))
                if len(results) >= max_results:
                    break
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
    keywords: list[str] | None = None,
    target_count: int = MAX_WEB_SOURCES,
) -> list[dict]:

    print(f"[DDGS] Starting news search for topic: '{topic}' (target_count={target_count})")

    if keywords is None:
        keywords = await _generate_web_search_keywords(topic)
    else:
        print(f"[DDGS] reusing {len(keywords)} previously generated keyword(s) — skipping keyword regeneration")

    model = _get_st_model()

    hyde_embedding = await _run_encode(
        lambda: model.encode(hyde_doc, normalize_embeddings=True, convert_to_numpy=True)
    )

    articles = []
    for keyword in keywords:
        if len(articles) >= target_count:
            print(f"[DDGS] Reached cap of {target_count} sources, stopping further keyword searches")
            break

        try:
            pairs = await _run_scrape(_ddgs_search_for_ideas, keyword, max_results)
            print(f"[DDGS] keyword '{keyword}' returned {len(pairs)} results")
        except Exception as e:
            print(f"[DDGS] thread failed for '{keyword}': {e}")
            pairs = []

        for url, snippet in pairs:
            if len(articles) >= target_count:
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
Generate exactly 10 high-quality search engine keyword combinations that
maximize information retrieval from Google, Bing, academic search engines,
and news websites, to gather source material for writing this script. Use
the segment structure and template purpose above to make sure the keywords
collectively cover what each part of the script will need, not just the
idea title in isolation.

Requirements:
- Every phrase must incorporate the topic's core subject/entities naturally —
  do NOT output the raw title string verbatim as one of the 10 lines by
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
- Return ONLY the 10 keyword combinations, nothing else — no preamble, no
  restating the title on its own line.
- Number each result.
"""


def _ddgs_search_for_script(keyword: str, max_results: int) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    try:
        with DDGS() as ddgs:
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


async def get_ddgs_news_context_for_script(
    topic: str,
    scraped_urls: set,
    hyde_doc: str,
    max_results: int = 10,
    similarity_threshold: float = WEB_CONTENT_SIMILARITY_THRESHOLD,
    keywords: list[str] | None = None,
    target_count: int = MAX_WEB_SOURCES,
) -> list[dict]:

    print(f"[DDGS-SCRIPT] Starting news search for topic: '{topic}' (similarity_threshold={similarity_threshold}, target_count={target_count})")

    if keywords is None:
        keywords = await _generate_search_keywords_for_script(topic, "", {}, 0)
    else:
        print(f"[DDGS-SCRIPT] reusing {len(keywords)} previously generated keyword(s) — skipping keyword regeneration")

    model = _get_st_model()

    hyde_embedding = await _run_encode(
        lambda: model.encode(hyde_doc, normalize_embeddings=True, convert_to_numpy=True)
    )

    articles = []
    for keyword in keywords:
        if len(articles) >= target_count:
            print(f"[DDGS-SCRIPT] Reached cap of {target_count} sources, stopping further keyword searches")
            break

        try:
            pairs = await _run_scrape(_ddgs_search_for_script, keyword, max_results)
            print(f"[DDGS-SCRIPT] keyword '{keyword}' returned {len(pairs)} results")
        except Exception as e:
            print(f"[DDGS-SCRIPT] thread failed for '{keyword}': {e}")
            pairs = []

        for url, snippet in pairs:
            if len(articles) >= target_count:
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
            print(f"[MAIN] table selection failed, defaulting to {TABLES[0]}: {exc}")
            table_name = TABLES[0]

        similar_task = asyncio.create_task(get_similar_saved_ideas(topic, combined_hyde_doc))

        db_results_per_doc = []
        try:
            db_results_per_doc = await asyncio.gather(
                *[get_context_from_db(topic, doc, table_name=table_name) for doc in hyde_documents]
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

        db_results = []
        seen_md5_context = set()
        max_len = max((len(d) for d in db_results_per_doc), default=0)
        round_idx = 0
        while len(db_results) < IDEAS_DB_CHUNKS_TO_LLM and round_idx < max_len:
            for doc_results in db_results_per_doc:
                if len(db_results) >= IDEAS_DB_CHUNKS_TO_LLM:
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
            f"[MAIN] Combined DB context (ideas): {len(db_results)} unique chunk(s) "
            f"(round-robin across {len(hyde_documents)} HyDE doc(s), capped at "
            f"{IDEAS_DB_CHUNKS_TO_LLM})."
        )

        scraped_urls = set()
        try:
            ideas_search_keywords = await _generate_web_search_keywords(topic)
        except Exception as exc:
            print(f"[MAIN] ideas keyword generation failed: {exc}")
            ideas_search_keywords = [f"{topic} latest news today", f"{topic} 2026 update"]

        new_articles = []
        try:
            new_articles = await get_ddgs_news_context(
                topic, scraped_urls, combined_hyde_doc,
                keywords=ideas_search_keywords,
                target_count=IDEAS_WEB_SOURCES_TO_LLM,
            )

            if _unique_url_count(new_articles) < IDEAS_WEB_SOURCES_TO_LLM:
                print(
                    f"[MAIN] Only {_unique_url_count(new_articles)} unique source URL(s) found "
                    f"(ideas), running multi-round backfill to reach {IDEAS_WEB_SOURCES_TO_LLM}."
                )
                new_articles = await _backfill_sources_to_target(
                    new_articles, scraped_urls, topic, combined_hyde_doc,
                    keywords=ideas_search_keywords,
                    target_count=IDEAS_WEB_SOURCES_TO_LLM,
                )
        except Exception as exc:
            print(f"[MAIN] web search (DDGS) failed: {exc}")
            new_articles = []

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

from pydantic import BaseModel
from fastapi import HTTPException


class UnlockRequest(BaseModel):
    userId: str
    duration: float 

CREDITS_PER_MINUTE = 3


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
    topic : str

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


async def generate_hyde_doc_for_segments(
    title: str,
    description: str,
    template: dict,
    segment_group: list[dict],
    time_minutes: float,
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
                temperature=0.3,   
                top_p=0.9
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
                temperature=0.6,   
                top_p=0.95       
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

# Never let ANY code path (primary or backfill) accept a source below this,
# no matter how much it "relaxes" to hit a target count.
_MIN_ACCEPTABLE_SIMILARITY = 0.30


async def _score_and_filter_url(
    url: str,
    fallback_snippet: str,
    hyde_embedding,
    model,
    similarity_threshold: float,
) -> dict | None:
    """The ONLY way a URL is allowed into `sources`/`new_articles`: fetch its
    real content, chunk it, embed the chunks, and check that at least one
    chunk is semantically close to the topic's HyDE embedding. Keyword match
    from a search engine is never sufficient on its own — this is what
    stops things like an unrelated org's homepage or a same-name-different-
    topic Wikipedia page from being reported as a source."""
    similarity_threshold = max(similarity_threshold, _MIN_ACCEPTABLE_SIMILARITY)

    full_text = await _fetch_full_article_text_with_timeout(url)
    used_source = "full" if full_text else "fallback"
    content = full_text if full_text else fallback_snippet
    if not content:
        print(f"[SCORE] SKIP (no content at all) {url}")
        return None

    content = _truncate_words(content, max_words=600)
    chunks = _split_into_chunks(content, max_words_per_chunk=40)
    if not chunks:
        print(f"[SCORE] SKIP (no chunks) {url}")
        return None

    try:
        chunk_embeddings = await _run_encode(
            lambda c=chunks: model.encode(c, normalize_embeddings=True, convert_to_numpy=True)
        )
    except Exception as e:
        print(f"[SCORE] SKIP (embedding failed: {e}) {url}")
        return None

    chunk_similarities = np.dot(chunk_embeddings, hyde_embedding)
    picked = [
        (chunk, float(sim))
        for chunk, sim in zip(chunks, chunk_similarities)
        if sim >= similarity_threshold
    ]

    if not picked:
        best_sim = float(np.max(chunk_similarities)) if len(chunk_similarities) else 0.0
        print(
            f"[SCORE] REJECT (best_sim={best_sim:.4f} < required {similarity_threshold:.4f}, "
            f"{len(chunks)} passage(s) checked, used_source={used_source}) {url}"
        )
        return None

    picked.sort(key=lambda p: p[1], reverse=True)
    picked_text = _truncate_words(" ".join(c for c, _ in picked), max_words=200)
    overall_similarity = picked[0][1]

    print(f"[SCORE] ACCEPT (sim={overall_similarity:.4f}, {len(picked)}/{len(chunks)} passages matched) {url}")

    return {
        "url": url,
        "snippet": picked_text,
        "source": used_source,
        "similarity": overall_similarity,
        "picked_passage_count": len(picked),
        "total_passage_count": len(chunks),
    }


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

    model = _get_st_model()
    hyde_embedding = await _run_encode(
        lambda: model.encode(hyde_doc, normalize_embeddings=True, convert_to_numpy=True)
    )

    relax_schedule = [
        WEB_CONTENT_SIMILARITY_THRESHOLD,
        max(WEB_CONTENT_SIMILARITY_THRESHOLD * 0.75, _MIN_ACCEPTABLE_SIMILARITY),  
    ]

    round_num = 0
    while _unique_url_count(new_articles) < target_count and round_num < max_rounds:
        round_num += 1
        added_this_round = 0
        current_threshold = relax_schedule[min(round_num - 1, len(relax_schedule) - 1)]

        if round_num == 1:
            candidate_pairs = []
            for kw in (keywords or [title]):
                try:
                    pairs = await _run_scrape(_ddgs_search_for_script, kw, 20)
                    candidate_pairs.extend(pairs)
                except Exception as e:
                    print(f"[MAIN] backfill round {round_num} search failed for '{kw}': {e}")
        else:
            generic_queries = [
                title, f"{title} history", f"{title} overview", f"{title} explained",
                f"{title} background", f"{title} facts", f"{title} details",
                f"{title} analysis", f"{title} biography", f"{title} encyclopedia",
            ]
            candidate_pairs = []
            for query in generic_queries:
                if _unique_url_count(new_articles) >= target_count:
                    break
                try:
                    pairs = await _run_scrape(_ddgs_search_for_script, query, 20)
                    candidate_pairs.extend(pairs)
                except Exception as e:
                    print(f"[MAIN] backfill generic query '{query}' failed: {e}")

        existing_urls = _existing_urls()
        for url, snippet in candidate_pairs:
            if _unique_url_count(new_articles) >= target_count:
                break
            if not url or url in scraped_urls or url in existing_urls or _is_blocked_source_url(url):
                continue
            scraped_urls.add(url)

            scored = await _score_and_filter_url(url, snippet, hyde_embedding, model, current_threshold)
            if scored is None:
                continue

            existing_urls.add(url)
            new_articles.append(scored)
            added_this_round += 1

        if added_this_round == 0:
            print(f"[MAIN] backfill round {round_num} added 0 new relevant source(s), stopping this round type")

    final_count = _unique_url_count(new_articles)
    if final_count < target_count:
        print(
            f"[MAIN] Only {final_count}/{target_count} unique SEMANTICALLY RELEVANT source(s) found — "
            f"the web genuinely doesn't have more on-topic results for this query. "
            f"(Not filled with junk to hit the number.)"
        )
    else:
        print(f"[MAIN] backfill reached target with {final_count}/{target_count} semantically relevant source(s)")

    return new_articles


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


STORY_ANALYST_SYSTEM_PROMPT = """
# STORYBIT STORY ANALYST (SSL v1)

## ROLE

You are **Storybit Story Analyst**, the first stage of the Storybit AI Pipeline. Your responsibility is to convert a YouTube documentary script into a compact **Story Semantic Language (SSL v1)**. You are **not** a script writer, thumbnail designer, or prompt engineer. You are a deterministic semantic compiler. Your output will be consumed only by the Storybit Thumbnail Intelligence Agent. Optimize for machine communication, not human readability.

## INPUT

You receive three inputs: (1) Video Title, (2) Thumbnail Text, (3) Complete Video Script. Treat the complete script as the source of truth. Use the title and thumbnail text only to strengthen confidence or resolve ambiguity. If they conflict with the script, follow the script.

## OBJECTIVE

Extract only the semantic information required for thumbnail generation. Ignore information that does not influence visual storytelling. Every output token must carry semantic value.

## RULES

Return valid JSON only. No markdown, explanations, reasoning, summaries, opinions, recommendations, or natural-language paragraphs. No duplicate information. Omit null values, empty arrays, and default values. Keep arrays ordered by importance. Never invent facts. Infer only when strongly supported by the script.

## OUTPUT

```json
{"v":1,"core":{},"sub":[],"rel":[],"evt":[],"emo":{},"conf":{},"vis":{},"sig":{}}
```

## core

Story metadata.

**Allowed fields:** `id cat era plot stage`

**Definitions:** `id` = canonical story identifier. `cat` = story category (`business, history, war, technology, biography, finance, politics, science, crime, sports`). `era` = relevant historical period (e.g., `1990s, 2010-2014, Cold War, Modern`). `plot` = ordered plot stages using concise keywords (`rise, growth, success, dominance, conflict, betrayal, crisis, decline, collapse, failure, recovery, innovation, sale, merger, victory, defeat`). `stage` = overall narrative state (`beginning, middle, ending`).

## sub

Ordered subjects.

```json
{"id":"","type":"","role":"","rank":1}
```

**Allowed types:** `person, company, country, organization, technology, product, place, event`

**Maximum:** 8 subjects.

## rel

Relationships.

```json
[source, relation, target]
```

**Relations:** `leads, owns, acquires, competes, defeats, supports, replaces, creates, invests, criticizes, wins, loses`

**Maximum:** 20 relationships.

## evt

Only visually important events.

```json
{"id":"","type":"","rank":1}
```

**Event types:** `launch, bankruptcy, acquisition, war, speech, protest, accident, crash, announcement, lawsuit, election`

**Maximum:** 10 events.

## emo

Dominant emotional signals.

**Fields:** `primary secondary viewer`

**Allowed values:** `curiosity, fear, trust, hope, success, failure, anger, shock, nostalgia, excitement, uncertainty`

**Maximum:** One value per field.

## conf

Primary conflict.

**Fields:** `type a b winner loser`

**Conflict types:** `market, political, military, technology, legal, social, economic, personal`

## vis

Visual candidates.

**Fields:** `hero support objects symbols locations environment`

**Limits:** Hero (2), Support (4), Objects (8), Symbols (5), Locations (5), Environment (1). Include only visually useful items. Ignore non-visual concepts.

## sig

Thumbnail signals.

**Fields:** `hook contrast focus risk impact`

**Hook:** `why, how, secret, mistake, collapse, truth, inside`

**Contrast:** `past_present, winner_loser, before_after, success_failure, small_big, old_new`

**Definitions:** `focus` = primary visual anchor, `risk` = main perceived danger, `impact` = main consequence. Maximum one value for each field.

## VALIDATION

Before returning, verify: script is authoritative; plot stages are chronological; subjects are ranked by visual importance; relationships are unique; events are visually significant; exactly one primary emotion; exactly one primary conflict; hero subject exists whenever possible; thumbnail signals are consistent with the script; JSON is valid; no redundant fields remain.

Return only the JSON.
"""


THUMBNAIL_INTELLIGENCE_SYSTEM_PROMPT = """
# STORYBIT THUMBNAIL INTELLIGENCE AGENT (TSL v1)

## ROLE

You are Storybit Thumbnail Intelligence Agent, Stage 2 of the Storybit Pipeline. Convert Story Semantic Language (SSL v1) into Thumbnail Specification Language (TSL v1). Produce only machine-readable visual specifications. Do not generate image prompts, explanations, reasoning, summaries, or creative writing. You are a deterministic compiler that converts story semantics into visual planning. The Prompt Renderer expands your output into a GPT Image prompt.

## INPUT

Receive:

* SSL v1
* `user_image` (true|false)

SSL is the only story source of truth. Infer only when strongly supported. Omit uncertain information.

If `user_image=true`, plan how the reference person should integrate into the thumbnail. Do not replace story subjects unless the story naturally requires it.

## OBJECTIVE

Generate the smallest possible TSL while preserving all important visual decisions. Optimize for high CTR, curiosity, emotional clarity, documentary realism, mobile readability, visual simplicity, clear hierarchy, and one dominant focal point.

## RULES

Return JSON only. No markdown. No explanations. No comments. No reasoning. No prose. Omit nulls, defaults, unsupported fields, and duplicates. Use compact keys, enums, and application IDs whenever available. Keep arrays ranked by importance. Never invent story facts.

## OUTPUT

```json
{
  "v":1,
  "cs":{},
  "vb":{},
  "rs":{}
}
```

If `user_image=true`, include:

```json
"idn":{}
```

between `vb` and `rs`.

## cs (Creative Strategy)

Keys:

`goal promise emo psy hook style tone simp urg shock myst trust focus`

Populate only non-default values.

## vb (Visual Blueprint)

Describe **what appears**, never **how it is rendered**.

Keys:

`sub obj sym loc env era expr pose fg bg focus layers layout text maxs maxo`

Subjects always represent the planned thumbnail composition, not the reference image.

## idn (Identity Integration)

Generate only when `user_image=true`.

Purpose: Describe how the user's reference image integrates into the planned composition.

Keys:

`mode slot expr pose gaze scale interact blend occ`

Definitions:

* mode → `replace insert foreground background observer group`
* slot → `hero left right center foreground background`
* expr → expression enum
* pose → pose enum
* gaze → `camera left right subject object up down`
* scale → `primary secondary background`
* interact → `none looking pointing holding shaking arguing celebrating`
* blend → `match auto`
* occ → `front partial behind`

The `idn` section defines composition only. Never describe rendering.

## rs (Rendering Specification)

Populate only non-default values.

Groups:

`comp cam light clr txt fx neg`

comp → `rule balance depth crop`

cam → `shot angle lens dist dof`

light → `style dir temp contrast`

clr → `pal accent`

txt → `pos size weight`

fx → ordered rendering effect IDs

neg → ordered negative constraint IDs

## ENUMS

Always use predefined application enums.

Emotion → `fear hope anger trust curiosity surprise success failure`

Style → `doc biz tech hist war editorial minimal`

Layout → `hero left right center split triangle diagonal`

Shot → `close medium wide`

Angle → `low eye high`

Lighting → `dramatic soft studio natural rim`

Palette → `warm cool mono neutral`

All subjects, objects, locations, symbols, typography, effects, constraints, poses, and expressions come from application dictionaries. Never invent enum values.

## VALIDATION

Before returning:

* Preserve SSL story facts.
* Maintain one dominant subject.
* Maintain one dominant emotion.
* Maintain one dominant curiosity hook.
* Preserve subject ranking.
* Preserve composition hierarchy.
* Emit `idn` only when `user_image=true`.
* Ensure identity integration supports the story composition.
* Remove defaults and redundancy.
* Return valid JSON only.
"""


PROMPT_RENDERER_SYSTEM_PROMPT = """
# STORYBIT PROMPT RENDERER (PR v2)
## ROLE
Stage 3 of the Storybit Pipeline. Compile TSL v1 into an optimized prompt for the selected Image Model. Preserve all TSL decisions. Do not redesign, reinterpret, optimize, summarize or invent content. Deterministic compiler only.
## INPUT
* TSL v1
* Thumbnail Text
* Image Model
* Reference Image (optional)
TSL is the only planning source. If Reference Image exists, apply `idn` while preserving facial identity.
## OBJECTIVE
Generate the shortest high-fidelity prompt compatible with the selected Image Model while preserving composition, hierarchy, rendering intent and realism.
## RULES
Return plain text only. No JSON, markdown, comments, explanations or reasoning. Expand TSL IDs using built-in dictionaries. Never expose internal fields. Remove duplicates. Merge compatible descriptors. Omit missing/default values. Never invent story facts. Never modify Thumbnail Text.
## MODEL
Adapt descriptor ordering, syntax and rendering terms for the selected Image Model. Use only supported descriptors. Minimize prompt length without reducing fidelity.
## OUTPUT ORDER
Style → Primary Subject → Supporting Subjects → Expression → Pose → Interaction → Objects → Symbols → Environment → Composition → Camera → Lighting → Color Palette → Foreground → Background → Thumbnail Text → Rendering Quality → Negative Constraints.
## COMPOSITION
Preserve hierarchy, layout and focal point. Keep composition simple, clear and mobile-readable.
## REFERENCE IMAGE
Ignore if absent. If present: preserve facial identity, proportions, approximate age, hairstyle and skin tone; apply `idn` pose, expression, gaze, placement and scale; match lighting, perspective and color; blend naturally; never duplicate user; `replace` replaces only hero subject; `insert` preserves story subjects.
## THUMBNAIL TEXT
Render exactly as supplied. Never rewrite, translate or shorten. Follow TSL placement. Large bold typography, high contrast, safe margins, mobile readable.
## QUALITY
Emit model-appropriate quality descriptors once only.
## NEGATIVE
Expand negative IDs, merge duplicates, keep concise.
## VALIDATION
Verify: TSL preserved; composition preserved; Thumbnail Text preserved; identity rules applied only when Reference Image exists; IDs expanded; no internal metadata; prompt optimized for selected Image Model.
Return only the compiled prompt.
"""


def _safe_parse_json(raw: str) -> dict | None:
    """Strip optional markdown code fences and parse JSON defensively."""
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


async def run_story_analyst(title: str, thumbnail_text: str, script_text: str) -> dict:
    script_excerpt = _truncate_words(script_text, max_words=1200) if script_text else "No script available."

    user_content = f"""Video Title: "{title}"
Thumbnail Text: "{thumbnail_text}"

Complete Video Script:
{script_excerpt}
"""

    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": STORY_ANALYST_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                stream=False,
            )
        )
        _record_token_usage("story_analyst_ssl", res)
        raw = (res.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[SSL] Step 1 (Story Analyst) call failed: {e}")
        return {}

    ssl_json = _safe_parse_json(raw)
    if ssl_json is None:
        print(f"[SSL] Step 1 output was not valid JSON. Raw output: {raw[:800]}")
        return {}

    print(f"[SSL] Step 1 (Story Analyst) output: {json.dumps(ssl_json, ensure_ascii=False)[:1000]}")
    return ssl_json



async def run_thumbnail_intelligence(ssl_json: dict, user_image: bool) -> dict:
    user_content = json.dumps(
        {
            "ssl": ssl_json,
            "user_image": bool(user_image),
        },
        ensure_ascii=False,
    )

    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": THUMBNAIL_INTELLIGENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                stream=False,
            )
        )
        _record_token_usage("thumbnail_intelligence_tsl", res)
        raw = (res.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[TSL] Step 2 (Thumbnail Intelligence) call failed: {e}")
        return {}

    tsl_json = _safe_parse_json(raw)
    if tsl_json is None:
        print(f"[TSL] Step 2 output was not valid JSON. Raw output: {raw[:800]}")
        return {}

    print(f"[TSL] Step 2 (Thumbnail Intelligence) output: {json.dumps(tsl_json, ensure_ascii=False)[:1000]}")
    return tsl_json


async def run_prompt_renderer(
    tsl_json: dict,
    thumbnail_text: str,
    image_model: str,
    has_reference_image: bool,
) -> str:
    user_content = f"""TSL v1:
{json.dumps(tsl_json, ensure_ascii=False)}

Thumbnail Text: "{thumbnail_text}"

Image Model: {image_model}

Reference Image: {"provided" if has_reference_image else "none"}
"""

    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": PROMPT_RENDERER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                stream=False,
            )
        )
        _record_token_usage("prompt_renderer", res)
        rendered_prompt = (res.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[PR] Step 3 (Prompt Renderer) call failed: {e}")
        return ""

    print(f"[PR] Step 3 (Prompt Renderer) output: {rendered_prompt}")
    return rendered_prompt


def _pick_thumbnail_text(thumbnail_text: str | None, request) -> str:
    if isinstance(thumbnail_text, str) and thumbnail_text.strip():
        return thumbnail_text.strip()

    base = (request.title or "Watch Now").strip()
    return base[:28] if base else "Watch Now"


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



def _generate_thumbnail_image_gpt_image_sync(
    prompt: str,
    face_image_bytes: bytes | None = None,
    size: str = GPT_IMAGE_SIZE,
    quality: str = GPT_IMAGE_QUALITY,
) -> dict:
    used_face = bool(face_image_bytes)

    try:
        if face_image_bytes:
            print(f"[THUMBNAIL-GPT] editing WITH user face photo (image-to-image, model='{GPT_IMAGE_MODEL}')")
            face_file = io.BytesIO(face_image_bytes)
            face_file.name = "face.png"

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
        error_str = str(e)
        print(f"[THUMBNAIL-GPT] request to GPT Image 2 failed: {e}")
        if used_face and ("invalid_image_file" in error_str or "image_generation_user_error" in error_str):
            print("[THUMBNAIL-GPT] face photo was rejected by GPT Image 2 — retrying as text-to-image instead")
            try:
                response = openai_client.images.generate(
                    model=GPT_IMAGE_MODEL,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=1,
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
    image_model = getattr(request, "image_model", None) or GPT_IMAGE_MODEL

    ssl_json = await run_story_analyst(request.title, chosen_thumbnail_text, script_text)

    tsl_json = await run_thumbnail_intelligence(ssl_json, user_image=has_reference_image)

    rendered_prompt = await run_prompt_renderer(
        tsl_json,
        chosen_thumbnail_text,
        image_model,
        has_reference_image,
    )

    if not rendered_prompt:
        print("[PIPELINE] Step 3 returned empty output — using fallback prompt")
        rendered_prompt = _fallback_thumbnail_prompt(request, chosen_thumbnail_text)
    elif chosen_thumbnail_text.lower() not in rendered_prompt.lower():
        print("[PIPELINE] rendered prompt didn't mention the thumbnail text — appending it explicitly")
        rendered_prompt = (
            f'{rendered_prompt} Render the text "{chosen_thumbnail_text}" as bold, large, '
            f"high-contrast typography baked into the image, in a clear area that doesn't "
            f"overlap the main subject."
        )

    # ---- STEP 4: Image generation ----
    result = await generate_thumbnail_image(rendered_prompt, face_image_bytes=face_image_bytes)
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
    thumbnail_text: str | None = None
    language: str = "English"  


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


async def _deduct_subscription_credits(user_id: str, amount: int):
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
            print(f"[CREDITS] No subscription rows found for user {user_id} (non-free tier) — skipping subscriptions deduction.")
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
            f"[CREDITS] (subscriptions, most recent by created_at) "
            f"Deducted {amount} credits from subscription {subscription_id} "
            f"(user {user_id}): {current_credits} -> {new_credits}"
        )
    except Exception as exc:
        print(f"[CREDITS] Failed to deduct subscription credits for user {user_id}: {exc}")
        import traceback
        traceback.print_exc()


async def _deduct_credits_for_action(user_id: str, amount: int, action_label: str = "credits"):
    if amount <= 0:
        print(f"[CREDITS] ({action_label}) amount <= 0 ({amount}), skipping deduction for user {user_id}")
        return

    tier = await _get_user_tier(user_id)
    is_free = tier in FREE_TIER_LABELS

    await _deduct_profile_credits(user_id, amount)

    if not is_free:
        print(
            f"[CREDITS] ({action_label}) user {user_id} tier='{tier}' (non-free) — "
            f"also deducting {amount} credits from subscriptions"
        )
        await _deduct_subscription_credits(user_id, amount)
    else:
        print(
            f"[CREDITS] ({action_label}) user {user_id} tier='{tier or 'free (default)'}' — "
            f"free tier, only user_profiles was deducted"
        )


async def _deduct_thumbnail_credits(user_id: str, amount: int = THUMBNAIL_CREDITS_PER_IMAGE):
    await _deduct_credits_for_action(user_id, amount, action_label="thumbnail")


async def _generate_thumbnail_endpoint_impl(request: "ThumbnailRequest"):
    _start_token_tracking()

    total_start_time = time.time()
    script_text = request.script or ""

    target_language = _normalize_language(getattr(request, "language", None))
    chosen_thumbnail_text = _pick_thumbnail_text(request.thumbnail_text, request)
 
    if target_language != "English" and chosen_thumbnail_text:
        try:
            print(f"[THUMBNAIL] Translating thumbnail text into {target_language}: '{chosen_thumbnail_text}'")
            chosen_thumbnail_text = await translate_text_full_pipeline(
                chosen_thumbnail_text, target_language
            )
            print(f"[THUMBNAIL] translated thumbnail text: '{chosen_thumbnail_text}'")
        except Exception as exc:
            print(f"--- thumbnail text translation failed, keeping English text: {exc} ---")
    print(f"[THUMBNAIL] chosen thumbnail text to render into image: '{chosen_thumbnail_text}'")

    thumbnail_result = {"image_base64": None, "prompt": None, "error": "not attempted"}

    try:
        print("[MAIN] Running 4-step thumbnail pipeline (SSL -> TSL -> Prompt -> Image).")
        thumbnail_result = await generate_thumbnail_for_script(
            request, script_text, chosen_thumbnail_text
        )
        thumbnail_url = None
        if thumbnail_result.get("image_base64"):
            thumbnail_url = await save_thumbnail_to_supabase(
                request.userId, thumbnail_result["image_base64"]
            )
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



FINAL_QC_SYSTEM_PROMPT = """
You are a senior YouTube content editor doing a FINAL QUALITY CHECK before a
video goes into production. Think of yourself as a strict teacher grading
and correcting a student's finished work.

## Inputs you will receive
1. Idea Title — the original idea this video was greenlit from
2. Idea Description — the original idea's description
3. Generated Script — the full narration script written for this idea
4. Generated YouTube Titles — 3 candidate titles
5. Generated YouTube Descriptions — 3 candidate descriptions
6. Generated Thumbnail Texts — 3 candidate thumbnail texts

## Your job
Check whether the Script, Titles, Descriptions, and Thumbnail Texts are:
- Factually consistent with each other (no contradictions between what the
  script says and what the title/description promises)
- Faithful to the original Idea Title and Idea Description — the video
  should deliver on the angle/premise the idea promised, not drift into an
  unrelated angle
- Free of grammar, spelling, punctuation, and awkward-phrasing errors
- Free of duplicated, garbled, or truncated sentences
- Internally coherent — segment markers like "[Hook]" preserved exactly as
  they appear, narration reads naturally aloud, no leftover placeholder
  text, no broken formatting
- Titles are 40-70 characters, Descriptions are 60-100 words, Thumbnail
  Texts are 4-8 words, punchy and readable at a glance — fix any that
  violate these bounds without changing their core message

## Correction rules
- If something is ALREADY correct, leave it EXACTLY as-is. Do not rewrite
  for style preference — only correct genuine errors or inconsistencies.
- If you fix something, make the MINIMAL edit needed to correct it. Do not
  rewrite whole paragraphs when only a sentence or phrase is wrong.
- Never shorten or pad the script's length materially — preserve its
  approximate word count and structure.
- Never invent new facts, statistics, names, or events not already present
  in the script.
- Preserve all segment markers (e.g. "[Hook]", "[Climax]") exactly, in the
  same order, untranslated.
- Do NOT return or reference the Idea Title or Idea Description in your
  output — they are provided only as context for your check.

## Output Format
Respond with ONLY valid JSON, no markdown fences, no preamble, in exactly
this shape:

{
  "script": "<corrected full script text>",
  "titles": ["...", "...", "..."],
  "descriptions": ["...", "...", "..."],
  "thumbnail_text": ["...", "...", "..."]
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

    # Sanity guard: never accept a QC "correction" that gutted the script
    # (e.g. a truncated or empty rewrite from a flaky completion).
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



async def _generate_script_impl(request: "ScriptRequest"):
    _start_token_tracking()

    total_start_time = time.time()
    topic_text = build_topic_text(request)
    print(f"SCRIPT GENERATION: Received request for title: '{request.title}'")
    english_script_text = ""
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
    script_context_target = num_docs * 2
    script_web_source_target = num_docs * 2
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
        while len(db_results) < script_context_target and round_idx < max_len:
            for doc_results in db_results_per_doc:
                if len(db_results) >= script_context_target:
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
            f"{script_context_target}, target >= {DB_SIMILARITY_THRESHOLD} similarity). "
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
            request.title, scraped_urls, combined_hyde_doc,
            keywords=script_search_keywords,
            target_count=script_web_source_target,
        )

        unique_source_count = _unique_url_count(new_articles)
        if unique_source_count < script_web_source_target:
            print(
                f"[MAIN] Only {unique_source_count} unique source URL(s) found, "
                f"running multi-round backfill to reach the target {script_web_source_target}."
            )
            try:
                new_articles = await _backfill_sources_to_target(
                    new_articles, scraped_urls, request.title, combined_hyde_doc,
                    keywords=script_search_keywords,
                    target_count=script_web_source_target,
                )
            except Exception as backfill_exc:
                print(f"[MAIN] sources backfill failed: {backfill_exc}")

        print(f"[MAIN] Final unique source count: {_unique_url_count(new_articles)}/{script_web_source_target}")
    except Exception as exc:
        print(f"--- web search (DDGS) failed: {exc} ---")
        import traceback
        traceback.print_exc()

    try:
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
        youtube_metadata = await generate_youtube_seo_metadata(request, english_script_text, new_videos)
    except Exception as exc:
        print(f"--- YouTube SEO metadata generation failed: {exc} ---")
        import traceback
        traceback.print_exc()
        youtube_metadata = _build_fallback_youtube_metadata(request)

    try:
        print("[MAIN] Running final QC pass (script + YouTube metadata cross-check).")
        qc_result = await run_final_qc_pass(
            idea_title=request.title,
            idea_description=request.description,
            script_text=script_text,
            youtube_metadata=youtube_metadata,
        )
        script_text = qc_result["script"]
        english_script_text = script_text 
        youtube_metadata["titles"] = qc_result["titles"]
        youtube_metadata["descriptions"] = qc_result["descriptions"]
        youtube_metadata["thumbnail_text"] = qc_result["thumbnail_text"]
    except Exception as exc:
        print(f"--- final QC pass failed, keeping pre-QC script/metadata: {exc} ---")
        import traceback
        traceback.print_exc()

    try:
        print("[MAIN] Generating script content metrics.")
        script_metrics = await generate_script_metrics(english_script_text, topic_text=topic_text)
    except Exception as exc:
        print(f"--- script metrics generation failed: {exc} ---")
        import traceback
        traceback.print_exc()

    try:
        print("[MAIN] Generating category and subcategory classification.")
        classification = await generate_category_and_subcategory(
            request.title, request.description, english_script_text
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
    structure = _build_structure_response(selected_template)

    total_words = _word_count(script_text) if script_text else 0

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













































class TranslateScriptRequest(BaseModel):
    userId: str
    script: str
    language: str = "English"


@app.post("/translate-script")
async def translate_script_endpoint(request: TranslateScriptRequest):
    await require_valid_user(request.userId)

    async with _pipeline_semaphore:
        return await _translate_script_impl(request)


async def _translate_script_impl(request: "TranslateScriptRequest"):
    _start_token_tracking()

    target_language = _normalize_language(request.language)

    if not request.script:
        return {
            "script": request.script,
            "language": target_language,
            "token_usage": _get_token_usage_summary(),
        }

    if target_language == "English":
        return {
            "script": request.script,
            "language": target_language,
            "token_usage": _get_token_usage_summary(),
        }

    try:
        print(f"[TRANSLATE-SCRIPT] Translating script into {target_language}.")
        translated_script = await translate_text_full_pipeline(request.script, target_language)
    except Exception as exc:
        print(f"--- /translate-script failed, returning original script: {exc} ---")
        translated_script = request.script

    token_usage = _get_token_usage_summary()

    return {
        "script": translated_script,
        "language": target_language,
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

    # grand_total = round(amount, 2)
    # base_price  = round(amount / 1.18, 2)
    # gst_amount  = round(grand_total - base_price, 2)

    base_price  = amount / 1.18
    gst_amount  = base_price * 0.18
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





import uuid as uuid_lib

def _expire_stale_batches(batches: list[dict], now: datetime.datetime) -> list[dict]:
    """Drop any batch past its validity. This is the literal 'old credits
    become 0 once expired' rule — an expired batch is removed from the pool
    entirely and can never be spent again, regardless of what's in it."""
    active = []
    for b in batches:
        try:
            expires_at = datetime.datetime.fromisoformat(b["expires_at"])
        except Exception:
            continue  # malformed/missing expiry — treat as already expired
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
    return batches + [new_batch]  # old batch(es) stay, new one just appended


def _deduct_from_batches(batches: list[dict], amount: int) -> tuple[list[dict], int]:
    """FIFO by expires_at — oldest-expiring batch spent first. Returns
    (updated_batches, amount_actually_deducted); 0 means insufficient
    credits and NOTHING was deducted (all-or-nothing)."""
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
                'plus': {'credits': 500, 'validity_days': 30},
                'pro':   {'credits': 1200, 'validity_days': 30},
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
                    .select('credit_batches')
                    .eq('id', user_id)
                    .single()
                    .execute()
                )
                existing_batches = (profile_resp.data or {}).get('credit_batches') or []
                active_batches = _expire_stale_batches(existing_batches, now)

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
                            f"Confirmed: user {user_id} → tier '{target_tier}', "
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
                            item_name=f"Storio AI {target_tier.title()} Plan",
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