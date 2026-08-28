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
    "https://www.testing.storio.tech",
    "https://testing.storio.tech",
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


# class CreateOrderRequest(BaseModel):
#     amount: float
#     currency: str = "INR"
#     receipt: str | None = None
#     target_tier: str

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
import uuid
import hashlib
import contextvars
import concurrent.futures
from urllib.parse import urlparse
from deep_translator import GoogleTranslator


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
    "aymara": "ay",
    "azerbaijani": "az",
    "bambara": "bm",
    "basque": "eu",
    "belarusian": "be",
    "bengali": "bn",
    "bhojpuri": "bho",
    "bosnian": "bs",
    "bulgarian": "bg",
    "catalan": "ca",
    "cebuano": "ceb",
    "chichewa": "ny",
    "chinese simplified": "zh-CN",
    "chinese traditional": "zh-TW",
    "corsican": "co",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dhivehi": "dv",
    "dogri": "doi",
    "dutch": "nl",
    "english": "en",
    "esperanto": "eo",
    "estonian": "et",
    "ewe": "ee",
    "filipino": "tl",
    "finnish": "fi",
    "french": "fr",
    "frisian": "fy",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "guarani": "gn",
    "gujarati": "gu",
    "haitian creole": "ht",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "iw",
    "hindi": "hi",
    "hmong": "hmn",
    "hungarian": "hu",
    "icelandic": "is",
    "igbo": "ig",
    "ilocano": "ilo",
    "indonesian": "id",
    "irish": "ga",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "kannada": "kn",
    "kazakh": "kk",
    "khmer": "km",
    "kinyarwanda": "rw",
    "konkani": "gom",
    "korean": "ko",
    "krio": "kri",
    "kurdish kurmanji": "ku",
    "kurdish sorani": "ckb",
    "kyrgyz": "ky",
    "lao": "lo",
    "latin": "la",
    "latvian": "lv",
    "lingala": "ln",
    "lithuanian": "lt",
    "luganda": "lg",
    "luxembourgish": "lb",
    "macedonian": "mk",
    "maithili": "mai",
    "malagasy": "mg",
    "malay": "ms",
    "malayalam": "ml",
    "maltese": "mt",
    "maori": "mi",
    "marathi": "mr",
    "meiteilon/manipuri": "mni-Mtei",
    "mizo": "lus",
    "mongolian": "mn",
    "myanmar": "my",
    "nepali": "ne",
    "norwegian": "no",
    "odia": "or",
    "oromo": "om",
    "pashto": "ps",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "quechua": "qu",
    "romanian": "ro",
    "russian": "ru",
    "samoan": "sm",
    "sanskrit": "sa",
    "scots gaelic": "gd",
    "sepedi": "nso",
    "serbian": "sr",
    "sesotho": "st",
    "shona": "sn",
    "sindhi": "sd",
    "sinhala": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "spanish": "es",
    "sundanese": "su",
    "swahili": "sw",
    "swedish": "sv",
    "tajik": "tg",
    "tamil": "ta",
    "tatar": "tt",
    "telugu": "te",
    "thai": "th",
    "tigrinya": "ti",
    "tsonga": "ts",
    "turkish": "tr",
    "turkmen": "tk",
    "twi": "ak",
    "ukrainian": "uk",
    "urdu": "ur",
    "uyghur": "ug",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
    "xhosa": "xh",
    "yiddish": "yi",
    "yoruba": "yo",
    "zulu": "zu",
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
You are a professional native {language} linguist, literary translator, localization specialist, and translation quality reviewer.

You will be given:

1. The ORIGINAL English script.
2. A DRAFT translation of that script into {language}.

Your task is NOT to translate from scratch.

Your task is to perform a professional Translation Quality Check (Translation QC), correcting only what is necessary so the final script reads as though it was originally written in {language}, while preserving the author's intent exactly.

────────────────────────
PRIMARY OBJECTIVE
────────────────────────

Produce a publication-ready {language} script that is:

• Semantically identical to the English original.
• Completely natural to native speakers.
• Fluent, idiomatic, and engaging.
• Optimized for spoken narration.
• Emotionally equivalent to the original.
• Free from machine translation artifacts.

Never:
- add information
- remove information
- change facts
- rewrite for personal preference

────────────────────────
QUALITY REQUIREMENTS
────────────────────────

1. Semantic Fidelity
- Preserve explicit and implicit meaning.
- Preserve logical relationships.
- Preserve chronology.
- Preserve cause and effect.
- Preserve emphasis.
- Preserve comparisons and negations.
- Never hallucinate.
- Never omit information.

2. Tone & Style
Preserve:
- writing style
- narrative voice
- documentary storytelling style
- emotional intensity
- suspense
- curiosity
- humor
- irony
- persuasion
- inspiration

3. Native Fluency
Rewrite unnatural machine-translated text into language that sounds completely native.

Ensure:
- natural grammar
- natural sentence structure
- natural word order
- natural punctuation
- natural vocabulary
- natural collocations
- smooth transitions
- idiomatic phrasing where appropriate

The reader should never feel the text was translated.

4. Spoken Narration
Since this script will be narrated:

- Prefer conversational but professional language.
- Improve rhythm and readability.
- Avoid awkward literal translations.
- Preserve dramatic pacing.
- Preserve emotional flow.

5. Context
Interpret every sentence using surrounding context.

Ensure:
- correct pronoun references
- correct meaning of ambiguous words
- correct domain terminology
- consistent terminology throughout the script

6. Cultural Localization
Where appropriate:

- Localize idioms naturally.
- Localize metaphors naturally.
- Replace literal expressions with native equivalents.

Do NOT localize:
- historical facts
- company names
- organization names
- government names
- product names
- official titles
unless an officially established equivalent exists.

7. Grammar
Correct:
- grammar
- spelling
- punctuation
- agreement
- tense consistency
- gender agreement
- plurality
- syntax

────────────────────────
PRESERVE EXACTLY
────────────────────────

Keep exactly as written unless an official localized form is universally used.

This includes:

- numbers
- dates
- statistics
- URLs
- email addresses
- product names
- brand names
- company names
- organization names
- government bodies
- ministry names
- official department names
- common English abbreviations
- well-known English acronyms
- programming languages
- APIs
- file names
- code
- technical standards

Examples include:

NASA
ISRO
WHO
UN
NATO
GDP
AI
ML
GPU
CPU
USB
Wi-Fi
Google
Microsoft
Windows
Linux
Android
iPhone

Keep these in English exactly as written.

Personal names should only be transliterated if that is the accepted convention in {language}.

────────────────────────
FORMATTING
────────────────────────

Preserve:

- paragraph breaks
- line breaks
- dialogue formatting
- bullet lists
- numbering
- quotation marks
- sentence boundaries

Do not merge or split paragraphs unless required for grammatically correct {language}.

────────────────────────
OUTPUT
────────────────────────

Return ONLY the corrected {language} script.

Do NOT include:

- explanations
- comments
- notes
- markdown
- analysis
- confidence scores
- comparisons
- introductory text
- closing text

Output only the final corrected translation.

"""



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
FETCH_TIMEOUT_SECONDS = float(os.getenv("FETCH_TIMEOUT_SECONDS", "6"))   # was 15

def to_pgvector(embedding) -> str:
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"



_bge_model = None

from typing import List, Dict, Any

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
    overall_timeout: float = 20.0,   # NEW — hard cap for the whole pool-build stage
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
            print(f"[MAIN] table selection failed, defaulting to {TABLES[0]}: {exc}")
            table_name = TABLES[0]

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


def _strip_json_fences(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


RRF_K = 60  

SCRIPT_RAG_POOL_PER_DOC = 40
SCRIPT_TOP_K_PER_DOC = 2       

DENSE_SCORE_THRESHOLD = 0.30
SPARSE_SCORE_THRESHOLD = 0.20

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
    timeout: float = 20.0,
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
    Returns a list of {"hyde_document": str, "keywords": list[str]} — one
    entry per template segment, in order.
    """
    segment_briefs = "\n".join(
        f"- {seg.get('name', 'segment')} ({seg.get('percentage', 0)}%): {seg.get('brief', '')}"
        for seg in segments
    )

    fallback_text = f"{title}\n\n{description}".strip()
    fallback_docs = lambda: [{"hyde_document": fallback_text, "keywords": []} for _ in segments]

    if not segments:
        return [{"hyde_document": fallback_text, "keywords": []}]

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
        for entry in raw_docs:
            if not isinstance(entry, dict):
                continue
            text_value = (entry.get("hyde_document") or "").strip()
            raw_keywords = entry.get("keywords") or []
            if not isinstance(raw_keywords, list):
                raw_keywords = []
            kw_clean = [str(k).strip() for k in raw_keywords if str(k).strip()]
            if text_value:
                docs.append({"hyde_document": text_value, "keywords": kw_clean})

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
## ROLE

You are a professional YouTube documentary script writer for long-form educational videos.

Transform the supplied source material into a compelling, narration-ready documentary script that is factually accurate, engaging, and optimized for human voice-over. Never invent information beyond the supplied sources.

---

## INPUT

You will receive:

* Idea Title
* Idea Description
* Target Duration (minutes)
* Script Template (title, cluster, purpose, ordered segments)
* Retrieved Knowledge Chunks with sources (Book name, author, published year)
* Recent Web/News Chunks with source details (with link of web article)


All retrieved chunks have already passed semantic relevance filtering and should be treated as the trusted knowledge base.

---

## OBJECTIVE

Produce one complete documentary narration that:

* internally follows the supplied template in the exact order
* fulfills every template segment's purpose
* flows naturally as one continuous story
* remains engaging from beginning to end
* sounds conversational when spoken aloud
* is informative, emotionally engaging, and easy to understand
* stays completely grounded in the supplied sources

The audience should never notice the underlying template structure.

---

## SCRIPT REQUIREMENTS

### Template

Internally follow every template segment.

Do not skip, merge, reorder, or invent segments.

The final narration must not expose segment boundaries, template details, runtime percentages, or metadata.

Distribute the narration approximately according to each segment's runtime percentage.

### Word Count

The final script length **must equal:**

**Target Duration × 130 words**

Maintain approximately **±3%** of the calculated target.

### Factual Integrity

Use only supported information from the supplied source material.

Never invent facts, statistics, quotations, dates, events, research findings, financial figures, historical claims, or scientific conclusions.

When multiple sources discuss the same subject, synthesize them into one coherent explanation.

Whenever an important fact, statistic, study, report, policy, discovery, historical conclusion, or expert opinion is presented, naturally attribute it within the narration.

Examples:

* According to the World Health Organization...
* Research published in Nature suggests...
* NASA reports...
* A World Bank study found...

Blend attribution seamlessly into the script without citations, hyperlinks, footnotes, or reference sections.

---

## NARRATION GUIDELINES

Write for **viewer retention**, not just information delivery.

The narration should feel like a professionally produced documentary.

Throughout the script, naturally apply:

* a compelling opening hook
* periodic re-hooks before attention declines
* curiosity gaps
* setup and payoff
* callbacks to earlier ideas
* foreshadowing where appropriate
* escalating insights
* emotional progression suited to the topic
* smooth transitions between ideas

These techniques should feel invisible, varied, and never repetitive.

Never sacrifice factual accuracy for dramatic effect.

Write using:

* conversational narration
* cinematic documentary storytelling
* natural human rhythm
* varied sentence lengths
* vivid but factual language
* logical progression

Structure the script into natural paragraphs.

Where appropriate, naturally include relevant quotations, proverbs, sayings, analogies, or comparisons only if they genuinely strengthen the narration.

The final narration must be a continuous paragraph-based documentary with no headings, visible sections, markdown, template information, or segment names.

---

## SCRIPT ANALYTICS

After generating the script, evaluate it.

Every numeric value must be **at least 1**.

Generate:

* **videoLengthMinutes**: Copy the Target Duration (minutes) exactly as provided in the input.
* **wordCount**: Total number of words in the generated script.
* **emotionalDepth** (1–10): Emotional engagement, storytelling quality, tension, vivid imagery, and human resonance.
* **generalExamples**: Number of illustrative examples, analogies, comparisons, or hypothetical scenarios that are not historical examples.
* **proverbs_count**: Number of proverbs, sayings, quotations, aphorisms, or memorable quotes used.
* **historicalExamples**: Number of historical events, figures, discoveries, civilizations, companies, or eras referenced.
* **researchFacts**: Number of distinct research-backed facts, reports, studies, surveys, or statistics referenced.

---

## CONTENT CLASSIFICATION

Classify the completed script.

Generate:

* **category** — one primary category (1–3 words)
* **subcategories** — up to five concise subcategories (1–3 words each)

Examples:

* Business → Marketing, Finance, Strategy
* Technology → Artificial Intelligence, Robotics
* History → Ancient History, Empires
* Science → Physics, Biology
* Psychology → Human Behavior, Cognitive Bias

Choose the category and subcategories that best represent the completed script.

Do not repeat the category within the subcategories.

---

## OUTPUT

Return **only valid JSON**.

```json
{
  "script": "Complete documentary narration in continuous paragraphs.",

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
```

Requirements:

* The `script` must begin immediately with the narration and end naturally.
* Use only continuous paragraphs.
* Do not include headings, markdown, segment names, template details, notes, or explanations.
* Ensure the JSON is syntactically valid.
* Return nothing except the JSON object.


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
        })

    web_chunks = []
    for article in new_articles:
        web_chunks.append({
            "snippet": article.get("snippet", ""),
            "similarity": article.get("similarity"),
            "url": article.get("url", ""),
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
) -> dict:
    context_block, _context_payload = await _build_script_context_json(db_results, new_articles)
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

Source Material (JSON — each knowledge_base_chunks entry may include a "book"
object with title/author/year; each web_chunks entry includes its source
"url". Attribute facts to these sources naturally in the narration, e.g.
"According to [author]'s [title]..." or "As reported by [domain]..."; never
attribute to a chunk whose "book" is null or whose "url" is empty):
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

    # =========================================================================
    # STAGE 2 (RAG retrieval) and STAGE 3 (web search) both only depend on
    # Stage 1's hyde_documents — not on each other — so they now run
    # concurrently instead of sequentially. All retrieval math/thresholds/
    # RRF fusion logic inside each is byte-for-byte unchanged.
    # =========================================================================
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
            print(f"[STAGE 3] {len(script_search_keywords)} search keyword(s) generated:")
            for i, kw in enumerate(script_search_keywords, start=1):
                print(f"    [KW-{i}] {kw}")
        except Exception as exc:
            print(f"[STAGE 3] keyword generation failed: {exc}")
            script_search_keywords = [f"{request.title} latest news today", f"{request.title} 2026 update"]

        shared_pool: list[dict] = []
        try:
            shared_pool = await build_shared_web_pool(script_search_keywords, scraped_urls)
            print(f"[STAGE 3] shared web pool built: {len(shared_pool)} article(s) fetched/chunked/embedded")
        except Exception as exc:
            print(f"[STAGE 3] shared web pool build failed: {exc}")
            import traceback
            traceback.print_exc()
            shared_pool = []

        articles: list[dict] = []
        try:
            model = _get_st_model()
            seen_urls_final: set = set()

            # Batch all segment HyDE embeddings in a single encode() call
            # instead of one call per segment — same vectors, far fewer
            # sequential model invocations on CPU.
            hyde_texts = [seg.get("hyde_document", "") for seg in hyde_documents]
            try:
                hyde_embeddings_batch = await _run_encode(
                    lambda: model.encode(hyde_texts, normalize_embeddings=True, convert_to_numpy=True)
                )
            except Exception as e:
                print(f"[STAGE 3] batched hyde embedding failed, falling back to per-doc: {e}")
                hyde_embeddings_batch = None

            for doc_idx, seg in enumerate(hyde_documents, start=1):
                if hyde_embeddings_batch is not None:
                    hyde_embedding = hyde_embeddings_batch[doc_idx - 1]
                else:
                    doc = seg.get("hyde_document", "")
                    hyde_embedding = await _run_encode(
                        lambda d=doc: model.encode(d, normalize_embeddings=True, convert_to_numpy=True)
                    )

                top_for_doc = rank_pool_for_hyde_doc(
                    shared_pool, hyde_embedding, WEB_CONTENT_SIMILARITY_THRESHOLD, SCRIPT_TOP_K_PER_DOC
                )
                newly_added = 0
                for article in top_for_doc:
                    url = article.get("url")
                    if url and url not in seen_urls_final:
                        seen_urls_final.add(url)
                        articles.append(article)
                        newly_added += 1
                print(
                    f"[STAGE 3 | SEGMENT #{doc_idx}] matched {newly_added} new source(s) from the "
                    f"pool of {len(shared_pool)} (running unique total: {len(seen_urls_final)})"
                )

            unique_source_count = _unique_url_count(articles)
            print(f"[STAGE 3] direct matching done: {unique_source_count}/{script_web_source_target} unique source(s)")

            articles.sort(key=lambda a: a.get("similarity", 0.0), reverse=True)
            articles = articles[:script_web_source_target]
            print(
                f"[STAGE 3] {_unique_url_count(articles)}/{script_web_source_target} "
                f"unique semantically relevant web source(s) available (no backfill)."
            )
        except Exception as exc:
            print(f"[STAGE 3] FAILED — web content matching raised: {exc}")
            import traceback
            traceback.print_exc()

        print(f"[STAGE 3] done in {time.time() - stage3_start:.2f}s — final unique source count: {_unique_url_count(articles)}/{script_web_source_target}")
        for i, a in enumerate(articles, start=1):
            print(f"    [SRC-{i}] sim={a.get('similarity'):.4f} {a.get('url')}")
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


@app.post("/save-audio")
async def save_audio(
    userId: str = Form(...),
    audio: UploadFile = File(...),
):
    await require_valid_user(userId)

    if audio.content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio content type: {audio.content_type}",
        )

    file_bytes = await audio.read()
    size_bytes = len(file_bytes)

    if size_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    if size_bytes > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({size_bytes} bytes, max {MAX_AUDIO_SIZE_BYTES} bytes)",
        )

    safe_name = _sanitize_filename(audio.filename)
    extension = _guess_extension(audio.content_type, safe_name)
    unique_name = f"{uuid.uuid4().hex}{extension}"
    storage_path = f"{userId}/{unique_name}"

    print(
        f"[AUDIO] uploading '{safe_name}' ({size_bytes} bytes, {audio.content_type}) "
        f"for userId={userId} -> {AUDIO_BUCKET}/{storage_path}"
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
        print(f"[AUDIO] storage upload FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to upload audio to storage: {e}")

    file_url = await asyncio.to_thread(
        _create_signed_url_sync, AUDIO_BUCKET, storage_path, SIGNED_URL_EXPIRY_SECONDS
    )

    if not file_url:
        raise HTTPException(status_code=500, detail="Audio uploaded but failed to generate URL")

    try:
        await asyncio.to_thread(
            lambda: supabase.table("user_profiles")
            .update({"audio-url": file_url})
            .eq("id", userId)
            .execute()
        )
    except Exception as e:
        print(f"[AUDIO] failed to save audio-url on user_profiles: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Audio uploaded but failed to save URL to profile: {e}")

    return {
        "message": "Audio uploaded successfully",
        "userId": userId,
        "url": file_url,
        "url_expires_in_seconds": SIGNED_URL_EXPIRY_SECONDS,
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

fish_session = Session(FISH_AUDIO_API_KEY)

GENERATED_AUDIO_BUCKET = "generated-audio"

# FIX (voice too fast): _run_fish_tts_sync below never set any prosody/speed
# option, so Fish Audio used its own default speed of 1.0 (normal) — nothing
# was ever making it fast, nothing was ever slowing it down either. This is
# the global default speed applied whenever a request doesn't specify its
# own `speed`. Range 0.5-2.0, lower = slower. Kept as an env var so it can be
# tuned without a redeploy.
TTS_SPEECH_SPEED = float(os.getenv("TTS_SPEECH_SPEED", "0.85"))

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


class GenerateSpeechRequest(BaseModel):
    userId: str
    script: str
    voice: str
    langCode: str = "en"
    durationMinutes: int = 0
    speed: float | None = None  # FIX (voice too fast): 0.5-2.0, lower = slower; None = use TTS_SPEECH_SPEED


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


def _run_fish_tts_sync(script: str, reference_id: str, speed: float = TTS_SPEECH_SPEED) -> bytes:
    tts_request = TTSRequest(
        text=script,
        reference_id=reference_id,
        temperature=0.5,              
        top_p=0.7,                    
        repetition_penalty=1.2,     
        chunk_length=300,           
        latency="normal",             
        normalize=True,              
        format="mp3",             
        mp3_bitrate=192,               
        condition_on_previous_chunks=True,
        prosody=Prosody(speed=speed),  # FIX (voice too fast): the actual speed control
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

@app.post("/generate-speech")
async def generate_speech(body: GenerateSpeechRequest):
    userId = body.userId
    script = body.script
    voice = body.voice.strip() if body.voice else ""
    lang_code = (body.langCode or "en").strip()
    # FIX (voice too fast): resolve the speed to use — request-specified if
    # given, otherwise the new global default.
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
        try:
            print(f"[TTS] translating script into {target_language} (langCode='{lang_code}') before TTS")
            script = await translate_text_full_pipeline(script, target_language)
        except Exception as e:
            print(f"[TTS] translation to {target_language} failed, using original script as-is: {e}")

    if voice.lower() == "user":
        try:
            result = await asyncio.to_thread(
                lambda: supabase.table("user_profiles")
                .select("audio-url")
                .eq("id", userId)
                .maybe_single()
                .execute()
            )
        except Exception as e:
            print(f"[TTS] failed to fetch user_profiles row: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch user profile: {e}")

        row = result.data if result else None
        audio_url = row.get("audio-url") if row else None

        if not audio_url:
            raise HTTPException(
                status_code=400,
                detail="No reference audio on file for this user. Upload one via /save-audio first.",
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
        audio_bytes = await asyncio.to_thread(_run_fish_tts_sync, script, reference_id, speed)
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

You are the **Storybit Voice Performance Annotation Agent**. The input is the user's **complete final script**. Convert it into a **Fish Audio S2/S2-Pro-ready performance script** by inserting carefully chosen inline performance tags so the audio sounds naturally narrated, emotionally believable, human-performed, and professionally dubbed.

You are **not** a writer, editor, summarizer, or rewriter.

## CORE RULE

Preserve the script exactly. You may **ONLY INSERT performance tags**.

Never:

* add, remove, paraphrase, reorder, or rewrite words
* change names, numbers, facts, dialogue, quotations, or meaning
* add narration or explanations
* invent emotions, sounds, or dialogue

Preserve original punctuation and structure unless a minimal punctuation change is essential for natural speech.

## FISH AUDIO S2/S2-PRO SYNTAX

Use **square brackets `[ ]`** for performance instructions.

Fish S2/S2-Pro supports **inline/localized control** and **free-form natural-language performance descriptions**, not merely a fixed tag dictionary.

Examples include:
`[pause] [short pause] [emphasis] [dramatic emphasis] [laughing] [chuckle] [inhale] [exhale] [sigh] [gasp] [whisper] [low voice] [low volume] [shouting] [screaming] [volume up] [volume down] [pitch up] [pitch down] [excited] [sad] [angry] [shocked] [surprised] [clearing throat] [tsk] [audience laughter]`

Concise custom descriptions are allowed when appropriate, e.g.:
`[quietly tense]`
`[restrained emotional tone]`
`[professional broadcast tone]`
`[quiet realization]`
`[building suspense]`

Do not create long instructions inside tags.

## ANALYZE THE ENTIRE SCRIPT FIRST

Silently identify:

* genre and narrator style
* emotional arc
* scene/idea transitions
* suspense and tension
* revelations and climaxes
* important facts and emphasis
* questions
* dialogue
* emotional moments
* pacing and conclusion

Then annotate according to the **context of the complete story**, not sentence-by-sentence isolation.

## HUMAN PERFORMANCE PRINCIPLE

**Natural/neutral narration is the default.**

Do not tag every sentence. Add a tag only when it meaningfully improves:

* emotion
* emphasis
* pacing
* tension
* realism
* conversational delivery
* narrative clarity

A human narrator does not constantly perform. Most ordinary sentences should remain untagged.

Prefer roughly:

* 0 tags for ordinary sentences
* 1 tag for meaningful delivery changes
* 1–2 tags around major moments
* occasional pauses at important boundaries

If removing a tag would sound equally natural, remove it.

## TAG PLACEMENT

Place each tag exactly where the vocal behavior should change.

Good:
`Nobody knew what would happen. [short pause] Then the door opened.`

Good:
`And then he [quiet realization] understood the truth.`

Good:
`[whisper] Nobody was supposed to know.`

Use inline placement rather than automatically placing tags at paragraph beginnings.

## PAUSES

Use pauses to reproduce natural thought and narrative timing:

`[short pause]` — brief separation
`[pause]` — meaningful pause
`[long dramatic pause]` — rare, major moment only

Use pauses for revelations, suspense, transitions, rhetorical questions, emotional realization, important statements, or before major payoffs.

Do not pause after every sentence.

## EMOTION & DELIVERY

Choose emotion from context, not keywords.

Useful controls:
`[serious] [solemn] [reflective] [authoritative] [excited] [sad] [angry] [shocked] [surprised] [fearful] [nervous] [mysterious] [ominous] [melancholic] [nostalgic] [urgent]`

For nuanced delivery use concise descriptions such as:
`[quietly tense]`
`[restrained emotion]`
`[serious documentary tone]`
`[warm conversational tone]`

Do not repeatedly restate a tone that naturally continues.

## EMPHASIS, INTENSITY & PITCH

Use `[emphasis]` or `[strong emphasis]` only on genuinely important words/phrases.

Use:
`[whisper] [soft voice] [low voice] [low volume] [loud] [shouting] [screaming] [volume up] [volume down]`

and, sparingly:
`[pitch up] [pitch down]`

Strong intensity is justified only by context. **Strongest does not mean loudest.**

## BREATH & PARALINGUISTICS

Use sparingly and only when context requires them:

`[inhale] [exhale] [deep breath] [sharp inhale] [sigh] [gasp] [chuckle] [laughing] [clearing throat] [tsk] [panting]`

Appropriate for shock, fear, exhaustion, physical action, emotional strain, or genuine conversational behavior.

Never add sounds merely to make audio seem "human."

## SUSPENSE & REVELATIONS

Build performance progressively rather than tagging everything dramatically.

Typical pattern:
normal narration → subtle tension → `[short pause]` → reveal → `[emphasis]`/`[quiet realization]` → normal narration.

Use `[dramatic]` or equivalent sparingly.

## DIALOGUE

Preserve dialogue exactly. Add delivery tags only when the dialogue clearly requires them.

Examples:
`[whisper] "Don't tell anyone."`
`[angry] "You knew."`
`[hesitant] "I... I don't know."`

Do not invent character voices or speaker labels.

## TAG COMBINATIONS

Avoid unnecessary stacking.

Bad:
`[excited] [dramatic] [loud] [pitch up] [emphasis]`

Prefer one precise instruction:
`[excited]`

Never use contradictory tags such as `[whisper] [shouting]` at the same location.

## HUMAN-NATURALNESS

The goal is **human performance, not maximum tagging**.

Create natural variation through purposeful changes in:

* pauses
* emphasis
* emotion
* intensity
* pace
* occasional breaths/reactions

Do not manufacture imperfections randomly.

## OUTPUT

Return **ONLY the fully annotated script**.

No analysis, explanation, headings, JSON, code fences, notes, summaries, or introductory text.

## FINAL VALIDATION

Before output, silently verify:

1. Every original word remains unchanged.
2. Nothing was invented, deleted, paraphrased, or reordered.
3. Tags use `[ ]`.
4. Tags are contextually justified and correctly placed.
5. Ordinary narration remains mostly untagged.
6. Pauses are natural.
7. Breaths/laughter/paralinguistic sounds are rare and justified.
8. No contradictory or excessive tags exist.
9. Major narrative beats receive appropriate performance treatment.
10. The result sounds like a skilled human narrator, not an over-directed TTS demo.
11. Output contains only the annotated script.

## INPUT

The attached/input content is the **complete final user-generated script**. Analyze the entire script first, then return the same script with only the necessary Fish Audio S2/S2-Pro performance tags inserted.
""".strip()


# Tags are free-form and come directly from the model per the system prompt
# above (it explicitly allows concise custom descriptions, not just a fixed
# dictionary) — so there is no whitelist here. We only check that the output
# contains bracketed tags at all and that the underlying words are untouched.
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














































import re
import os
import json
import math
import uuid
import shutil
import tempfile
import asyncio
import subprocess
from pathlib import Path
from typing import Any, Optional, Literal

import httpx
import whisperx
from fastapi import HTTPException
from pydantic import BaseModel


# =============================================================================
# B-ROLL ORIENTATION HELPERS  (FIX A)
# =============================================================================

def _looks_like_playable_media_url(url: Optional[str]) -> bool:
    if not url:
        return False
    if "videos.pexels.com" in url or "images.pexels.com" in url:
        return True
    if re.search(r"\.(mp4|mov|webm|jpg|jpeg|png|webp)(\?|$)", url, re.IGNORECASE):
        return True
    return False


def _is_landscape_dimensions(width, height) -> bool:
    """
    True only if width clearly exceeds height (landscape/horizontal).

    FIX A: requires a minimum 1.2:1 ratio, not just width > height. A clip
    that's only marginally wider than it is tall (e.g. 1100x1000) still
    crops very badly onto a 1080x1920 canvas and reads as "basically
    vertical" once cropped — the old `w > h` check let those through.
    """
    try:
        w = float(width)
        h = float(height)
    except (TypeError, ValueError):
        return False
    return w > 0 and h > 0 and w >= h * 1.2


def _resolve_broll_file_url(candidate: Optional[dict], source: Optional[str]) -> Optional[str]:
    """
    Pick the best-quality file_url for a b-roll candidate.

    Pexels frequently returns video_files where every entry's "quality"
    field is null (seen on many assets in this dataset, e.g. 34730966,
    36767561, 26833619). A naive "match quality=='hd', else take
    video_files[0]" falls through to whatever Pexels happened to list
    first — which is often the smallest/lowest-resolution file, not the
    best one. That silently produced blurry/near-unusable broll (asset
    34730966 resolved to a 640x360 clip instead of its available
    3840x2160 version).

    Selection order now is:
      1. Images: just use the highest-resolution src variant available.
      2. Videos: prefer an entry with an explicit "hd"/"uhd" quality
         label if one exists (skips low-res "sd" entries when better
         is labeled and available). Otherwise — including when every
         entry has quality=None — pick the single largest file by
         pixel area (width * height). This guarantees we never favor
         a smaller file over a larger one just because of list order.
    """
    if not candidate:
        return None

    if source == "image":
        src = candidate.get("src") or {}
        for key in ("original", "large2x", "large", "portrait", "landscape", "medium", "small", "tiny"):
            url = src.get(key)
            if url:
                return url
        return None

    if source == "video":
        video_files = candidate.get("video_files") or []
        if not video_files:
            return None

        def _area(f: dict) -> int:
            w = f.get("width") or 0
            h = f.get("height") or 0
            return w * h

        labeled_hd = [f for f in video_files if (f.get("quality") or "").lower() in ("hd", "uhd")]
        if labeled_hd:
            best = max(labeled_hd, key=_area)
            if best.get("link"):
                return best["link"]

        candidates_with_links = [f for f in video_files if f.get("link")]
        if not candidates_with_links:
            return None
        best = max(candidates_with_links, key=_area)
        return best["link"]

    return None


def _resolve_broll_file_url_any_orientation(candidate: dict, source: str) -> Optional[str]:
    """
    FIX A (blur-pad fallback support): unlike `_resolve_broll_file_url`,
    this ignores orientation entirely and just returns the best playable
    URL for the candidate, whatever shape it is. Used ONLY as input to the
    blur-pad composite in `_prepare_broll_clip` — never used to select what
    gets shown full-bleed.
    """
    if not candidate:
        return None

    existing = candidate.get("file_url")
    if _looks_like_playable_media_url(existing):
        return existing

    if source == "video":
        if candidate.get("video_url") and _looks_like_playable_media_url(candidate["video_url"]):
            return candidate["video_url"]
        video_files = candidate.get("video_files") or []
        if video_files:
            for vf in video_files:
                if vf.get("quality") == "hd" and vf.get("file_type") == "video/mp4":
                    return vf.get("link")
            return video_files[0].get("link")
        return candidate.get("url") if _looks_like_playable_media_url(candidate.get("url")) else None

    src = candidate.get("src") or {}
    for key in ("large2x", "large", "original", "medium"):
        if src.get(key):
            return src[key]
    return candidate.get("url") if _looks_like_playable_media_url(candidate.get("url")) else None


def _video_is_landscape(v: dict) -> bool:
    return _resolve_broll_file_url(v, "video") is not None


def _image_is_landscape(p: dict) -> bool:
    return _resolve_broll_file_url(p, "image") is not None


# =============================================================================
# SCENE PLANNING PROMPTS  (unchanged from your original)
# =============================================================================

SCRIPT_SCENE_PROMPT = f""" 
System Prompt

You are Storybit's Scene Planner, an AI that converts documentary-style narration into a structured scene manifest for an automated video editing pipeline.

Your output is consumed directly by backend services, so it must be valid JSON only with no markdown, explanations, comments, or code fences.

Objective

Transform a narration script into a sequence of visually coherent scenes while preserving the original narration exactly.

The output must contain no timestampsw. Timing will be generated later from voiceover alignment.

Scene Segmentation Rules

Return AT MOST 5 scenes total, regardless of script length. This is a hard limit — never exceed 5.

If the narration would naturally split into more than 5 scenes, merge related sentences/ideas together until you have 5 or fewer scenes, rather than dropping any narration text.

Split whenever the spoken idea or visual changes, but stay within the 5-scene limit.

Keep scene lengths balanced; avoid overly long scenes.

Preserve the narration verbatim inside vo_text. Every word of the original script must appear in exactly one scene's vo_text — do not drop or paraphrase any narration when merging scenes to fit the limit.

Output Schema

Return a JSON array (5 objects or fewer) where every object contains exactly these fields:

{{
  "scene_id": "s1",
  "vo_text": "Exact narration for this scene.",
  "visual_intent": "Concise documentary-style description of what should be shown.",
  "on_screen_text": "Short text overlay or empty string.",
  "requires_animation": true/false,
  "broll_keywords": ["query one", "query two", "query three", "query four", "query five"]
}}
Field Guidelines

scene_id

Sequential: s1, s2, s3, s4, s5 (never more than 5).

vo_text

Copy the narration exactly.

Do not paraphrase or rewrite.

visual_intent

Write a concise documentary-style search query suitable for B-roll retrieval.

Prefer real-world imagery.

Mention important subjects, locations, time periods, or events.

Avoid cinematic adjectives like "epic" or "dramatic" unless explicitly stated.

on_screen_text

Use only when helpful for viewers, such as:

Years

Dates

Locations

People's names

Statistics

Short titles

Otherwise return "".

requires_animation

Return true only if the scene benefits from:

Kinetic typography

Lower-third text

Maps

Charts

Timelines

Infographics

Otherwise return false.

broll_keywords

Return a list of 5-6 distinct stock-footage search phrases for this scene,
suitable for searching a stock video/photo API (e.g. Pexels).

- Each phrase should be 2-6 words, concrete, and searchable (real-world
  nouns: places, objects, actions, eras — not abstract concepts).
- Each phrase should target a DIFFERENT visual angle on the same scene, so
  together they widen the pool of usable B-roll rather than repeating the
  same query.
- Always return at least 5 phrases, and up to 6.
- Do not invent specific names, dates, or facts not present in the narration.
- Do not use cinematic adjectives like "epic" or "dramatic".
- Prefer real, photographable subjects over metaphors.

Note: these scene-level keywords are only a seed/fallback. The actual B-roll
shown to viewers rotates every ~10 seconds within a scene, driven by a
second, beat-level keyword pass (see BEAT_KEYWORDS_PROMPT) that looks at
exactly what's being said in each smaller window of narration.

Constraints

Do not invent facts.

Do not create timestamps.

Do not include camera directions unless they improve B-roll retrieval (e.g., "aerial view", "satellite map", "close-up").

Keep visual_intent under roughly 15 words.

Never output more than 5 scenes.

Ensure the output is valid, parseable JSON.
"""


# =============================================================================
# BEAT-LEVEL (~10s) B-ROLL KEYWORD PROMPT
#
# NEW: this prompt now also drives image MOTION selection (Ken Burns-style
# pan/zoom/tilt). The LLM is the only place that picks a motion_type — the
# render layer (_build_image_motion_filter, see RENDER SERVICE section) just
# turns whatever motion_type it's handed into the matching FFmpeg zoompan
# expression. No motion is ever chosen by render-time code except as a
# last-resort default when the LLM call itself fails (see
# _generate_beat_keywords' except block) or returns something invalid.
# =============================================================================

BEAT_KEYWORDS_PROMPT = """
System Prompt

You are Storybit's B-roll Director for a single ~10 second beat (roughly
8-12 seconds) inside a longer scene of a documentary-style video.

You are given, as JSON:
- scene_visual_intent: the overall visual direction for the whole scene.
- scene_full_narration: the full narration spoken across the whole scene.
- beat_narration: the exact narration spoken during THIS beat only.
- beat_number / total_beats: this beat's position among the scene's beats.
- previous_media_type: the media type ("video" or "image") actually used
  for the immediately preceding beat in this scene, or null if this is the
  first beat.
- previous_motion_type: the image motion effect used for the most recent
  IMAGE beat in this scene (one of the six values listed below), or null
  if no image beat has occurred yet in this scene. This is only relevant
  when THIS beat also ends up being an image.

Your job has three parts:

1. Return fresh stock-footage search phrases for THIS beat specifically, so
   the B-roll on screen visibly changes every beat instead of repeating the
   same shot for the whole scene.
2. Decide whether THIS beat should be filled with a VIDEO clip or a STILL
   IMAGE, based on what the beat_narration actually calls for.
3. Choose a motion_type — the camera-style pan/zoom/tilt effect that will
   animate the still image if (and only if) this beat ends up using an
   image. You must always return a valid motion_type even when media_type
   is "video" (it will simply be ignored in that case) — never omit it or
   return null.

Output Schema

Return exactly one JSON object with these three fields, and nothing else —
no markdown, no explanations, no code fences:

{
  "keywords": ["query one", "query two", "query three", "query four", "query five"],
  "media_type": "video",
  "motion_type": "zoom_in"
}

Keyword Rules (accuracy is critical — bad keywords produce wrong footage)

- Each phrase: 2-6 words, concrete, real-world, photographable (places,
  objects, actions, eras) — never an abstract concept, emotion, or theme
  on its own (e.g. "hope" or "progress" is not searchable; "person walking
  toward sunrise" is).
- Every phrase must be grounded in a concrete noun or subject that is
  literally present, or very directly implied, in beat_narration. If
  beat_narration names a specific person, place, object, or event, at
  least one phrase must center on that exact subject — do not substitute a
  generic stand-in when the specific one is stated.
- Do not invent facts, names, dates, or subjects not present in the text.
- If this beat's narration doesn't introduce new visual content beyond the
  scene's overall visual intent, still produce phrases that give a
  DIFFERENT concrete angle on the same subject (wide shot vs. close-up,
  a related real-world subject, a different moment in the same activity)
  — never just repeat the scene-level search generically.
- No cinematic adjectives ("epic", "dramatic", "breathtaking").
- Prefer real, photographable subjects over metaphors.
- Always return 5-6 phrases.

Media Type Rules (this decision drives what actually gets shown — get it right)

Choose "image" when the beat's narration is about:
- A specific static subject best shown as a single frame: a portrait, a
  named person, a document, a building exterior, a product, a screenshot-
  like reference, a statistic or data point, a historical/archival moment.
- A quote, a definition, or a reflective/explanatory beat with no motion
  implied by the narration.

Choose "video" when the beat's narration is about:
- Motion, action, process, or change over time: someone doing something,
  a place being walked through, a sequence of events, anything where
  seeing movement adds meaning that a still frame would lose.

Do not default to "video" out of habit — a documentary rotates between
motion and stills deliberately, and stills are often the stronger choice
for factual/explanatory beats. Avoid choosing the same media_type as
previous_media_type when the narration content would support either
choice equally well, so consecutive beats visually vary (motion, then a
still, then motion again) rather than showing video back-to-back for an
entire scene by default. Only repeat the previous media_type when THIS
beat's content clearly calls for it regardless of variety.

Motion Type Rules (only applied if media_type is "image" — pick it anyway)

Choose exactly one of these six values for motion_type:

- "zoom_in" — slowly pushes in toward the center of the image. Best for a
  single clear subject you want the viewer's attention drawn into
  (a portrait, a document, a product, a specific detail).
- "zoom_out" — starts closer in and pulls back to reveal the full frame.
  Best for a reveal moment, or establishing a wider scene after a close
  detail was just implied by the narration.
- "pan_left" — fixed zoom, drifts left-to-right across the frame. Best for
  wide scenes with a horizontal layout (a skyline, a crowd, a landscape,
  a row of objects) where sweeping across adds information.
- "pan_right" — fixed zoom, drifts right-to-left across the frame. Same
  use case as pan_left; alternate direction for variety rather than
  always defaulting to one direction.
- "tilt_up" — fixed zoom, drifts bottom-to-top. Best for tall subjects:
  a building, a monument, a standing person, anything with vertical
  scale worth revealing.
- "tilt_down" — fixed zoom, drifts top-to-bottom. Best for a "descending
  reveal" feel — starting on a wide/high vantage and settling on the
  actual subject below.

Pick based on the image subject implied by beat_narration and
scene_visual_intent (tall subject -> tilt_up/tilt_down, wide subject ->
pan_left/pan_right, single focal subject -> zoom_in/zoom_out). When the
content doesn't clearly favor one, still choose deliberately for variety:
avoid returning the same motion_type as previous_motion_type back-to-back
across consecutive image beats in this scene unless the subject genuinely
calls for repeating it.
"""


ANIMATION_TAXONOMY = {
    "full_screen": [
        "full_screen_broll",
        "full_screen_title_card",
        "full_screen_data_viz",
        "full_screen_transition",
        "full_screen_color_wash",
        "full_screen_quote_card",
    ],
    "overlay_text": [
        "lower_third",
        "kinetic_caption",
        "bullet_list_reveal",
        "callout_textbox",
        "stat_counter_overlay",
    ],
    "overlay_graphic": [
        "icon_pop_in",
        "icon_sequence",
        "logo_watermark",
        "emoji_reaction",
        "arrow_highlight",
        "badge_sticker",
    ],
    "pip": [
        "pip_video",
        "split_screen",
        "multi_panel_grid",
    ],
    "branding": [
        "avatar_overlay",
        "mascot_animation",
    ],
    "transition": [
        "ken_burns_pan_zoom",
        "parallax_layering",
        "shake_impact",
        "speed_ramp_indicator",
    ],
}

_ANIMATION_TYPE_TO_CATEGORY = {
    anim_type: category
    for category, anim_types in ANIMATION_TAXONOMY.items()
    for anim_type in anim_types
}
_VALID_ANIMATION_TYPES = set(_ANIMATION_TYPE_TO_CATEGORY.keys())
_VALID_PLACEMENTS = {
    "top_left", "top_center", "top_right",
    "center_left", "center", "center_right",
    "bottom_left", "bottom_center", "bottom_right",
    "full_frame",
}
_VALID_Z_LAYERS = {"background", "midground", "foreground"}
_VALID_TRIGGERS = {"time_offset", "on_keyword", "on_beat", "scene_start"}
_VALID_RENDER_HINTS = {"remotion", "ffmpeg"}

ANIMATION_PLANNER_PROMPT = f"""
System Prompt

You are Storybit's Animation Director — the creative director of this video.
You alone decide WHICH animation treatment plays for a scene and EXACTLY
WHERE on screen it is placed. The rendering pipeline (hybrid Remotion +
FFmpeg) draws the animation at the literal placement/z-index/trigger you
return here — there is no other layout logic downstream, so your placement
choice IS the final on-screen position.

Your output must be valid JSON only — no markdown, explanations, comments,
or code fences.

ALLOWED animation_type VALUES (grouped by category):

FULL_SCREEN (category: "full_screen")
- full_screen_broll, full_screen_title_card, full_screen_data_viz,
  full_screen_transition, full_screen_color_wash, full_screen_quote_card

OVERLAY_TEXT (category: "overlay_text")
- lower_third, kinetic_caption, bullet_list_reveal, callout_textbox,
  stat_counter_overlay

OVERLAY_GRAPHIC (category: "overlay_graphic")
- icon_pop_in, icon_sequence, logo_watermark, emoji_reaction,
  arrow_highlight, badge_sticker

PICTURE_IN_PICTURE (category: "pip")
- pip_video, split_screen, multi_panel_grid

CHARACTER (category: "branding")
- avatar_overlay, mascot_animation

MOTION_EFFECT (category: "transition")
- ken_burns_pan_zoom, parallax_layering, shake_impact, speed_ramp_indicator

Placement values you may choose from (choose the one that best composes
against the B-roll and on-screen text for this scene):
top_left, top_center, top_right, center_left, center, center_right,
bottom_left, bottom_center, bottom_right, full_frame

IMPORTANT — caption safe zone: word-by-word captions are always burned in
along the very bottom edge of the frame (the lowest ~15% of the canvas).
For any overlay_text or overlay_graphic animation_type, do NOT choose
bottom_left, bottom_center, or bottom_right — that band is reserved for
captions and an overlay placed there will visually collide with them.
Prefer top_left/top_center/top_right, center_left/center/center_right
instead for those categories. Bottom placements remain fine only for
full_screen or transition category animation_types, which own the whole
frame anyway.

Output Schema

Return exactly one JSON object with these fields:

{{
  "animation_type": "icon_pop_in",
  "category": "overlay_graphic",
  "placement": "top_right",
  "z_index_layer": "foreground",
  "trigger": "on_keyword",
  "duration_frames": 45,
  "content_binding": "icon:lightbulb",
  "render_engine_hint": "remotion"
}}

Constraints

Return ONLY the JSON object. No markdown, no code fences, no commentary.
Every field is required. Do not omit any field.
Choose `placement` deliberately based on where it will read best given the
scene's on_screen_text and visual_intent — this is a real creative/staging
decision, not a formality.
"""

class EditVideo(BaseModel):
    userId: str
    script: str
    voice: str
    langCode: str = "en"
    durationMinutes: int = 0


class TrackPatch(BaseModel):
    track_id: str
    updates: dict[str, Any]


_VALID_CAPTION_ANIMATION_TYPES = {
    "kinetic_caption",
    "static_line",
    "typewriter",
    "word_pop",
}


class SceneVoiceUpdate(BaseModel):
    voice: str


class SceneStyleUpdate(BaseModel):
    font_size: Optional[int] = None
    text_color: Optional[str] = None
    outline_color: Optional[str] = None
    animation_type: Optional[str] = None
    background_color: Optional[str] = None
    vertical_position: Optional[Literal["top", "bottom"]] = None   # NEW
    margin_bottom_percent: Optional[float] = None                  # NEW


class SceneTrimUpdate(BaseModel):
    start: float
    end: float


class SceneInfographicUpdate(BaseModel):
    animation_type: Optional[str] = None
    props: Optional[dict[str, Any]] = None
    duration_frames: Optional[int] = None


class SceneBrollSelectUpdate(BaseModel):
    asset_id: Any
    source: str
    # NEW: a scene can now have several rotating B-roll "beats" (one every
    # ~10s). beat_id selects which beat's B-roll this override applies
    # to. If omitted, defaults to the scene's first beat for backward
    # compatibility with single-beat (<=12s) scenes.
    beat_id: Optional[str] = None
    # NEW: optional manual override of the image motion effect (Ken Burns
    # pan/zoom/tilt) for this beat. Only meaningful when the selected
    # candidate's source is "image" — ignored otherwise. When omitted, the
    # beat keeps whichever motion_type the LLM already chose for it in
    # BEAT_KEYWORDS_PROMPT (see _resolve_beat_motion_type).
    motion_type: Optional[str] = None


def _hex_to_ass_color(hex_color: str, alpha_hex: str = "00") -> str:
    h = (hex_color or "").strip().lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha_hex}{b}{g}{r}"


def _normalize_ffmpeg_color(hex_color: str) -> str:
    h = (hex_color or "").strip().lstrip("#")
    if len(h) != 6:
        h = "111827"
    return f"0x{h}"


_HEX_COLOR_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


def _validate_hex_color(value: Optional[str], field_name: str) -> None:
    if value is None:
        return
    if not _HEX_COLOR_RE.match(value.strip()):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be a 6-digit hex color like '#111827', got {value!r}",
        )


WHISPERX_MODEL_SIZE = os.getenv("WHISPERX_MODEL_SIZE", "small")
WHISPERX_DEVICE = os.getenv("WHISPERX_DEVICE", "cpu")
WHISPERX_COMPUTE_TYPE = os.getenv("WHISPERX_COMPUTE_TYPE", "int8")

PEXELS_SCENE_RESULT_LIMIT = int(os.getenv("PEXELS_SCENE_RESULT_LIMIT", "6"))
PEXELS_SCENE_PER_KEYWORD_PER_PAGE = int(os.getenv("PEXELS_SCENE_PER_KEYWORD_PER_PAGE", "4"))
PEXELS_SCENE_PAGE = 1

PEXELS_SCENE_VIDEO_ORIENTATION = "landscape"
PEXELS_SCENE_IMAGE_ORIENTATION = os.getenv("PEXELS_SCENE_IMAGE_ORIENTATION", "landscape")
PEXELS_SCENE_SIZE = os.getenv("PEXELS_SCENE_SIZE", None)
PEXELS_SCENE_COLOR = os.getenv("PEXELS_SCENE_COLOR", None)

# FIX (blank/color-fallback beats): a scene with N beats fires up to
# N * len(keywords) * 2 (video+image) Pexels calls all at once via
# asyncio.gather, and a full multi-scene /edit-video request can add up to
# several hundred concurrent requests. Pexels' rate limit is easy to hit at
# that volume; a throttled call is caught and logged per-keyword but still
# leaves that beat with fewer (or zero) candidates — indistinguishable from
# Pexels genuinely having no footage for the query. Capping concurrency
# smooths the request burst out over time instead of firing it all at once,
# which cuts down on throttling-caused empty results.
PEXELS_MAX_CONCURRENT_REQUESTS = int(os.getenv("PEXELS_MAX_CONCURRENT_REQUESTS", "8"))
_pexels_semaphore = asyncio.Semaphore(PEXELS_MAX_CONCURRENT_REQUESTS)

BROLL_KEYWORDS_MAX = int(os.getenv("BROLL_KEYWORDS_MAX", "6"))
BROLL_KEYWORDS_MIN = int(os.getenv("BROLL_KEYWORDS_MIN", "5"))

# NEW: how long each rotating B-roll "beat" is allowed to be. Any scene
# whose spoken duration exceeds BEAT_MAX_SECONDS gets split into multiple
# beats, each with its own LLM-generated keywords and its own Pexels
# search, so the visible B-roll changes roughly every 10 seconds.
#
# We keep a narrow [MIN, MAX] band rather than a single fixed "10" because
# `_compute_beat_boundaries` divides a scene's total duration evenly across
# whole beats — a hard-coded exact 10s target would leave a tiny, ugly
# remainder beat on scenes that aren't exact multiples of 10s (e.g. a 34s
# scene would produce a 4s last beat). An 8-12s band lets the splitter
# round each beat's length to something close to 10s while still landing
# exactly on the scene's start/end.
BEAT_MIN_SECONDS = int(os.getenv("BEAT_MIN_SECONDS", "8"))
BEAT_MAX_SECONDS = int(os.getenv("BEAT_MAX_SECONDS", "12"))

TIMELINE_FPS = int(os.getenv("TIMELINE_FPS", "30"))
TIMELINE_WIDTH = int(os.getenv("TIMELINE_WIDTH", "1080"))
TIMELINE_HEIGHT = int(os.getenv("TIMELINE_HEIGHT", "1920"))

# NEW: captions default to a very low, near-bottom-edge position on the
# 9:16 canvas. `vertical_position`/`margin_bottom_percent` are read by the
# downstream ASS/Remotion caption renderer (outside this file) to place the
# karaoke caption block. Expressed as a percentage of TIMELINE_HEIGHT rather
# than a fixed pixel margin so it stays "very low" consistently across any
# canvas size. margin_bottom_percent is kept small (close to the edge) per
# explicit request — if platform UI (like/comment/share buttons on
# YouTube Shorts/TikTok/Reels) ends up overlapping captions on a specific
# export target, raise this a few points rather than removing it outright.
# Any per-scene `caption_style` the user/editor sets is merged on top of
# this default — it only fills in fields the caller didn't specify.
DEFAULT_CAPTION_STYLE = {
    "vertical_position": "bottom",
    "margin_bottom_percent": 3,
}

TTS_MAX_CHARS_PER_CALL = int(os.getenv("TTS_MAX_CHARS_PER_CALL", "900"))
TTS_AUDIO_BUCKET = os.getenv("TTS_AUDIO_BUCKET", "generated-audio")

_whisperx_model = None
_whisperx_align_cache = {}
_whisperx_lock = asyncio.Lock()


def _get_whisperx_model():
    global _whisperx_model
    if _whisperx_model is None:
        print(f"[WHISPERX] loading model '{WHISPERX_MODEL_SIZE}' on {WHISPERX_DEVICE}")
        _whisperx_model = whisperx.load_model(
            WHISPERX_MODEL_SIZE,
            WHISPERX_DEVICE,
            compute_type=WHISPERX_COMPUTE_TYPE,
        )
    return _whisperx_model


def _preload_whisperx_align_model(language_code: str = "en"):
    if language_code not in _whisperx_align_cache:
        print(f"[WHISPERX] preloading alignment model for language '{language_code}'")
        align_model, metadata = whisperx.load_align_model(
            language_code=language_code, device=WHISPERX_DEVICE
        )
        _whisperx_align_cache[language_code] = (align_model, metadata)


def _run_whisperx_sync(audio_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        model = _get_whisperx_model()
        audio = whisperx.load_audio(tmp.name)

        result = model.transcribe(audio, batch_size=16)
        language = result["language"]

        if language not in _whisperx_align_cache:
            print(f"[WHISPERX] loading alignment model for language '{language}'")
            align_model, metadata = whisperx.load_align_model(
                language_code=language, device=WHISPERX_DEVICE
            )
            _whisperx_align_cache[language] = (align_model, metadata)

        align_model, metadata = _whisperx_align_cache[language]

        aligned_result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            WHISPERX_DEVICE,
            return_char_alignments=False,
        )

        return {
            "language": language,
            "segments": aligned_result["segments"],
            "word_segments": aligned_result.get("word_segments", []),
        }


async def _generate_word_timestamps(audio_url: str) -> dict:
    audio_bytes = await _download_bytes(audio_url)
    async with _whisperx_lock:
        return await asyncio.to_thread(_run_whisperx_sync, audio_bytes)


# ---------------------------------------------------------------------------
# TTS chunking helpers
# ---------------------------------------------------------------------------

def _split_text_for_tts(text: str, max_chars: int = TTS_MAX_CHARS_PER_CALL) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        candidate = f"{current} {sent}".strip() if current else sent
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(sent) <= max_chars:
            current = sent
        else:
            for i in range(0, len(sent), max_chars):
                chunks.append(sent[i:i + max_chars])
    if current:
        chunks.append(current)
    return chunks


async def _upload_audio_to_storage(local_path: str, user_id: str) -> str:
    storage_path = f"{user_id}/{uuid.uuid4().hex}.mp3"
    with open(local_path, "rb") as f:
        supabase.storage.from_(TTS_AUDIO_BUCKET).upload(
            storage_path, f, {"content-type": "audio/mpeg", "upsert": "true"}
        )
    return supabase.storage.from_(TTS_AUDIO_BUCKET).get_public_url(storage_path)


async def _generate_speech_possibly_chunked(
    user_id: str, tagged_text: str, voice: str, lang_code: str,
) -> dict:
    chunks = _split_text_for_tts(tagged_text)

    if len(chunks) <= 1:
        speech_request = GenerateSpeechRequest(
            userId=user_id, script=tagged_text, voice=voice,
            langCode=lang_code, durationMinutes=0,
        )
        return await generate_speech(speech_request)

    print(
        f"[tts] narration is {len(tagged_text)} chars — splitting into "
        f"{len(chunks)} TTS calls to avoid provider truncation"
    )

    work_dir = tempfile.mkdtemp(prefix="tts_chunks_")
    try:
        chunk_paths = []
        for idx, chunk_text in enumerate(chunks):
            speech_request = GenerateSpeechRequest(
                userId=user_id, script=chunk_text, voice=voice,
                langCode=lang_code, durationMinutes=0,
            )
            chunk_result = await generate_speech(speech_request)
            chunk_bytes = await _download_bytes(chunk_result["url"])
            chunk_path = os.path.join(work_dir, f"chunk_{idx:03d}.mp3")
            with open(chunk_path, "wb") as f:
                f.write(chunk_bytes)
            chunk_paths.append(chunk_path)

        list_path = os.path.join(work_dir, "concat_list.txt")
        with open(list_path, "w") as f:
            for p in chunk_paths:
                f.write(f"file '{p}'\n")

        combined_path = os.path.join(work_dir, "combined.mp3")
        # Re-encode rather than stream-copy — chunk calls can come back
        # with slightly different internal formats, and concat demuxer
        # stream-copy is fragile across those; re-encoding guarantees a
        # single consistent output file.
        cmd = [
            FFMPEG_BIN, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c:a", "libmp3lame", "-b:a", "192k",
            combined_path,
        ]
        await _run(cmd)

        combined_url = await _upload_audio_to_storage(combined_path, user_id)
        return {"url": combined_url}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _get_scene_broll_keywords(scene: dict) -> list:
    raw_keywords = scene.get("broll_keywords")

    keywords = []
    if isinstance(raw_keywords, list):
        keywords = [k.strip() for k in raw_keywords if isinstance(k, str) and k.strip()]
        seen = set()
        deduped = []
        for k in keywords:
            key = k.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(k)
        keywords = deduped

    if len(keywords) >= BROLL_KEYWORDS_MIN:
        return keywords[:BROLL_KEYWORDS_MAX]

    fallback_extras = []
    visual_intent = (scene.get("visual_intent") or "").strip()
    if visual_intent and visual_intent.lower() not in [k.lower() for k in keywords]:
        fallback_extras.append(visual_intent)

    vo_snippet = (scene.get("vo_text") or "").strip()[:60]
    if vo_snippet and vo_snippet.lower() not in [k.lower() for k in keywords]:
        fallback_extras.append(vo_snippet)

    keywords = (keywords + fallback_extras)[:BROLL_KEYWORDS_MAX]

    if not keywords:
        print(f"[edit-video] scene {scene.get('scene_id')} has no usable broll_keywords or fallback text")

    return keywords


def _dedupe_and_trim(items: list, limit: int) -> list:
    def _identity(item: dict):
        for key in ("id", "video_id", "photo_id", "asset_id"):
            if key in item:
                return item[key]
        for key in ("url", "src", "image", "video_url", "link"):
            if key in item:
                return item[key]
        return None

    seen = set()
    deduped = []
    for item in items:
        item_id = _identity(item)
        if item_id is not None:
            if item_id in seen:
                continue
            seen.add(item_id)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


async def _fetch_media_for_keywords(keywords: list, label: str) -> dict:
    """
    NEW: shared Pexels-search core, factored out of what used to be
    `_fetch_scene_media` so both whole-scene lookups (silent scenes with no
    narration, used as a single "beat") and per-beat lookups (the normal,
    rotating-every-~10s path) share identical search/filter/dedupe logic.

    `label` is only used for log lines (e.g. "s2" or "s2:s2_b3") so failures
    are traceable to the exact scene/beat.
    """
    empty_result = {
        "videos": {"total_results": 0, "results": [], "error": None},
        "images": {"total_results": 0, "results": [], "error": None},
    }

    if not keywords:
        return empty_result

    if not PEXELS_API_KEY:
        empty_result["videos"]["error"] = "PEXELS_API_KEY is not configured on the server."
        empty_result["images"]["error"] = "PEXELS_API_KEY is not configured on the server."
        return empty_result

    async def _get_videos_for(keyword: str):
        try:
            async with _pexels_semaphore:
                data = await asyncio.to_thread(
                    _pexels_search_videos_sync,
                    keyword,
                    PEXELS_SCENE_PER_KEYWORD_PER_PAGE,
                    PEXELS_SCENE_PAGE,
                    PEXELS_SCENE_VIDEO_ORIENTATION,
                    PEXELS_SCENE_SIZE,
                )
            return {"keyword": keyword, "data": data, "error": None}
        except Exception as e:
            return {"keyword": keyword, "data": None, "error": str(e)}

    async def _get_images_for(keyword: str):
        try:
            async with _pexels_semaphore:
                data = await asyncio.to_thread(
                    _pexels_search_images_sync,
                    keyword,
                    PEXELS_SCENE_PER_KEYWORD_PER_PAGE,
                    PEXELS_SCENE_PAGE,
                    PEXELS_SCENE_IMAGE_ORIENTATION,
                    PEXELS_SCENE_SIZE,
                    PEXELS_SCENE_COLOR,
                )
            return {"keyword": keyword, "data": data, "error": None}
        except Exception as e:
            return {"keyword": keyword, "data": None, "error": str(e)}

    video_tasks = [_get_videos_for(k) for k in keywords]
    image_tasks = [_get_images_for(k) for k in keywords]
    video_results, image_results = await asyncio.gather(
        asyncio.gather(*video_tasks),
        asyncio.gather(*image_tasks),
    )

    video_errors = [r["error"] for r in video_results if r["error"]]
    videos_pool = []
    for r in video_results:
        if r["error"]:
            continue
        videos_pool.extend(_format_video_result(v) for v in (r["data"].get("videos") or []))

    pre_filter_count = len(videos_pool)
    videos_pool = [v for v in videos_pool if _video_is_landscape(v)]
    dropped = pre_filter_count - len(videos_pool)
    if dropped:
        print(f"[edit-video] {label}: dropped {dropped} non-landscape video result(s) at fetch time")

    image_errors = [r["error"] for r in image_results if r["error"]]
    images_pool = []
    for r in image_results:
        if r["error"]:
            continue
        images_pool.extend(_format_image_result(p) for p in (r["data"].get("photos") or []))

    pre_filter_img_count = len(images_pool)
    images_pool = [p for p in images_pool if _image_is_landscape(p)]
    dropped_img = pre_filter_img_count - len(images_pool)
    if dropped_img:
        print(f"[edit-video] {label}: dropped {dropped_img} non-landscape image result(s) at fetch time")

    videos = _dedupe_and_trim(videos_pool, limit=PEXELS_SCENE_RESULT_LIMIT)
    photos = _dedupe_and_trim(images_pool, limit=PEXELS_SCENE_RESULT_LIMIT)

    videos_error = "; ".join(video_errors) if video_errors and not videos else None
    images_error = "; ".join(image_errors) if image_errors and not photos else None

    return {
        "videos": {"total_results": len(videos), "results": videos, "error": videos_error},
        "images": {"total_results": len(photos), "results": photos, "error": images_error},
    }


async def _fetch_scene_media(scene: dict) -> dict:
    """
    Whole-scene Pexels lookup. After the beats refactor this is now used
    ONLY as the fallback path for scenes with no narration (so there's no
    voiceover timing to split into beats) — it produces a single implicit
    beat covering the whole scene. Normal narrated scenes go through
    `_build_scene_beats` -> `_fetch_media_for_keywords` per beat instead.
    """
    keywords = _get_scene_broll_keywords(scene)
    media = await _fetch_media_for_keywords(keywords, str(scene.get("scene_id")))
    return {"keywords": keywords, "videos": media["videos"], "images": media["images"]}


# =============================================================================
# BEAT SPLITTING — rotate B-roll every ~10 seconds within a scene, and
# choose an image motion effect (LLM-driven) for any beat that becomes a
# still image.
# =============================================================================

def _compute_beat_boundaries(
    scene_start: float,
    scene_end: float,
    min_sec: int = BEAT_MIN_SECONDS,
    max_sec: int = BEAT_MAX_SECONDS,
) -> list[tuple[float, float]]:
    """
    Splits [scene_start, scene_end] into consecutive windows, each between
    `min_sec` and `max_sec` long (the last window absorbs any remainder so
    boundaries land exactly on scene_end). Scenes shorter than `max_sec`
    are returned as a single window — no need to rotate B-roll on a scene
    that's already short.
    """
    total = scene_end - scene_start
    if total <= 0:
        return [(scene_start, scene_end)]
    if total <= max_sec:
        return [(scene_start, scene_end)]

    num_beats = max(2, math.ceil(total / max_sec))
    beat_len = total / num_beats
    while beat_len < min_sec and num_beats > 1:
        num_beats -= 1
        beat_len = total / num_beats

    boundaries = []
    for i in range(num_beats):
        b_start = scene_start + i * beat_len
        b_end = scene_end if i == num_beats - 1 else scene_start + (i + 1) * beat_len
        boundaries.append((b_start, b_end))
    return boundaries


def _words_in_range(timed_words: list, start: float, end: float) -> str:
    words = [
        (w.get("word") or "").strip()
        for w in timed_words
        if w.get("start", -1) >= start - 1e-6 and w.get("start", -1) < end + 1e-6
    ]
    return " ".join(w for w in words if w)


_VALID_MEDIA_TYPES = {"video", "image"}

# NEW: the six LLM-selectable image motion effects (Ken Burns-style
# pan/zoom/tilt). Kept as an explicit ordered list (not just a set) so the
# diversity fallback below can deterministically cycle through them.
_MOTION_TYPE_LIST = ["zoom_in", "zoom_out", "pan_left", "pan_right", "tilt_up", "tilt_down"]
_VALID_MOTION_TYPES = set(_MOTION_TYPE_LIST)
_DEFAULT_MOTION_TYPE = "zoom_in"


def _pick_diversified_motion_type(previous_motion_type: Optional[str], beat_index: int) -> str:
    """
    Used only when the LLM call for a beat fails outright, or returns a
    motion_type outside the six valid values — never used to override a
    valid LLM choice. Cycles through the motion types that aren't the same
    as the previous image beat's motion, so even the failure path avoids
    showing the exact same pan/zoom back-to-back.
    """
    candidates = [m for m in _MOTION_TYPE_LIST if m != previous_motion_type] or list(_MOTION_TYPE_LIST)
    return candidates[beat_index % len(candidates)]


async def _generate_beat_keywords(
    scene: dict, beat_vo_text: str, beat_index: int, total_beats: int, fallback_keywords: list,
    previous_media_type: Optional[str] = None,
    previous_motion_type: Optional[str] = None,
) -> tuple[list, str, str]:
    """
    LLM call #2 for B-roll: given the scene's overall visual intent plus the
    exact narration spoken in THIS beat, ask the model for a fresh set of
    search phrases, which media type (video vs. still image) best fits this
    beat, AND — new — which image motion effect (pan/zoom/tilt) to use if
    this beat ends up as a still image. The LLM is the only place any of
    these three decisions get made; render-time code (see
    _build_image_motion_filter) only ever translates whatever motion_type
    it's handed into the matching FFmpeg filter, and only substitutes a
    diversified default when this call fails or returns something invalid.

    Returns (keywords, media_type, motion_type) where media_type is "video"
    or "image", and motion_type is always one of the six values in
    _VALID_MOTION_TYPES (even when media_type is "video", in which case
    it's simply unused downstream).
    """
    context = {
        "scene_visual_intent": scene.get("visual_intent", ""),
        "scene_full_narration": scene.get("vo_text", ""),
        "beat_narration": beat_vo_text,
        "beat_number": beat_index + 1,
        "total_beats": total_beats,
        "previous_media_type": previous_media_type,
        "previous_motion_type": previous_motion_type,
    }
    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": BEAT_KEYWORDS_PROMPT},
                    {"role": "user", "content": json.dumps(context)},
                ],
                stream=False,
            )
        )
        _record_token_usage("edit video - beat keywords", res)

        content = (res.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()

        raw = json.loads(content)
        if not isinstance(raw, dict):
            raise ValueError(f"expected a JSON object, got {type(raw)}")

        raw_keywords = raw.get("keywords")
        if not isinstance(raw_keywords, list):
            raise ValueError(f"expected 'keywords' to be a list, got {type(raw_keywords)}")

        keywords = [k.strip() for k in raw_keywords if isinstance(k, str) and k.strip()]
        seen = set()
        deduped = []
        for k in keywords:
            key = k.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(k)

        if len(deduped) < BROLL_KEYWORDS_MIN:
            raise ValueError(f"model returned only {len(deduped)} usable keyword(s)")

        media_type = str(raw.get("media_type") or "").strip().lower()
        if media_type not in _VALID_MEDIA_TYPES:
            print(
                f"[edit-video] scene {scene.get('scene_id')} beat {beat_index + 1}/{total_beats} "
                f"returned invalid media_type {raw.get('media_type')!r} — defaulting to 'video'"
            )
            media_type = "video"

        raw_motion = str(raw.get("motion_type") or "").strip().lower()
        if raw_motion not in _VALID_MOTION_TYPES:
            print(
                f"[edit-video] scene {scene.get('scene_id')} beat {beat_index + 1}/{total_beats} "
                f"returned invalid motion_type {raw.get('motion_type')!r} — picking a diversified default"
            )
            motion_type = _pick_diversified_motion_type(previous_motion_type, beat_index)
        else:
            motion_type = raw_motion

        return deduped[:BROLL_KEYWORDS_MAX], media_type, motion_type

    except Exception as e:
        print(
            f"[edit-video] scene {scene.get('scene_id')} beat {beat_index + 1}/{total_beats} "
            f"keyword generation failed, using fallback: {e}"
        )
        extra = [beat_vo_text.strip()[:60]] if beat_vo_text.strip() else []
        combined = (fallback_keywords or []) + extra
        seen = set()
        out = []
        for k in combined:
            key = (k or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(k.strip())
        fallback_motion = _pick_diversified_motion_type(previous_motion_type, beat_index)
        return out[:BROLL_KEYWORDS_MAX], "video", fallback_motion


def _dedupe_beats_media_across_scene(beats: list) -> None:
    """
    FIX (rotation bug): each beat's Pexels search runs independently, so
    two adjacent beats with similar keywords very often come back with the
    same top-ranked asset as their #1 candidate. Since beat selection
    (`_resolve_beat_broll_selection`) always takes each beat's own
    candidates[0], that made consecutive beats silently pick the *same*
    clip — which is exactly why the rendered video showed one piece of
    footage holding for far longer than one beat's window (multiple beats
    in a row, all "different" beats, all showing identical footage).

    This walks a scene's beats in order and, for each beat's video/image
    candidate list, moves any asset id already selected by an earlier beat
    in this scene to the back of the list — so the default pick
    (candidates[0]) is a fresh asset whenever the beat's own pool has one.
    If a beat's entire pool was already used elsewhere in the scene, it
    falls back to reusing an asset rather than showing nothing — repetition
    only happens when there's truly no alternative footage available.

    Mutates `beats` in place (reorders each beat's media.videos/images
    results list); does not change which beats exist or their timing.
    """
    used_ids: set = set()
    for beat in beats:
        media = beat.get("media") or {}
        for pool_key in ("videos", "images"):
            pool = media.get(pool_key) or {}
            results = pool.get("results") or []
            if not results:
                continue
            fresh = [r for r in results if r.get("id") not in used_ids]
            reused = [r for r in results if r.get("id") in used_ids]
            pool["results"] = fresh + reused
        beat["media"] = media

        # Mark whichever asset this beat will actually default to (per its
        # AI-decided preferred_media_type — see _resolve_beat_broll_selection)
        # as used, so the *next* beat avoids picking the same one.
        default_asset, _source = _resolve_beat_broll_selection(beat)
        if default_asset and default_asset.get("id") is not None:
            used_ids.add(default_asset["id"])


def _beat_media_is_empty(beat: dict) -> bool:
    media = beat.get("media") or {}
    return not (media.get("videos") or {}).get("results") and not (media.get("images") or {}).get("results")


# FIX (blank beats): a handful of broad, near-universally-available search
# terms used only as an absolute last resort when nothing scene- or
# beat-specific returned anything — real photographic footage instead of a
# flat color card, even in the worst case.
_LAST_RESORT_BROLL_TERMS = [
    "abstract background", "soft light texture", "historical documents",
    "old map parchment", "clouds sky timelapse", "city skyline aerial",
]


async def _fill_empty_beats(scene: dict, beats: list, fallback_keywords: list) -> None:
    """
    A beat whose own keywords return zero results in BOTH the video and
    image pools falls straight through to a flat color card at render time
    (`_prepare_broll_clip`'s last-resort fallback) — visually indistinguishable
    from a bug, and confirmed happening in real renders (exact RGB match to
    the default fallback color 0x111827). This repair chain runs right after
    the initial per-beat Pexels fetch and tries, in order, to replace an
    empty beat's media with something real before ever letting it stay empty:

      1. Retry with the scene-level broll_keywords — broader/more generic
         than one beat's narrow phrasing, so it often has hits the beat's
         own search didn't.
      2. Borrow the candidate pool from the nearest sibling beat in this
         SAME scene that has real candidates (closest by beat_index first).
         Reused footage for one beat beats a flat color card.
      3. If literally every beat in the scene is empty (usually a sign of
         an API/config problem, not a content problem), fetch a small set
         of generic, near-universally-available terms as an absolute last
         resort so the scene shows *something* photographic.

    Mutates `beats` in place. Marks any beat that had to fall back to (2)
    or (3) with `_media_fallback_reason` for traceability in the timeline.
    """
    scene_id = scene.get("scene_id")

    empty_beats = [b for b in beats if _beat_media_is_empty(b)]
    if not empty_beats:
        return

    retry_results = await asyncio.gather(*[
        _fetch_media_for_keywords(fallback_keywords, f"{scene_id}:{b['beat_id']}:retry-scene-keywords")
        for b in empty_beats
    ])
    for b, media in zip(empty_beats, retry_results):
        if (media.get("videos") or {}).get("results") or (media.get("images") or {}).get("results"):
            b["media"] = media
            print(f"[edit-video] beat {b['beat_id']} was empty, filled via scene-level keyword retry")

    still_empty = [b for b in beats if _beat_media_is_empty(b)]
    if not still_empty:
        return

    non_empty = [b for b in beats if not _beat_media_is_empty(b)]
    if non_empty:
        for b in still_empty:
            idx = b["beat_index"]
            nearest = min(non_empty, key=lambda o: abs(o["beat_index"] - idx))
            # Shallow-copy the outer dict so this beat and the one it
            # borrowed from don't share the exact same mutable container —
            # `_dedupe_beats_media_across_scene` reorders pool result lists
            # in place, and giving each beat its own outer dict (while still
            # sharing the underlying candidate objects, which is fine/cheap)
            # keeps that reordering from having surprising cross-beat effects.
            b["media"] = {
                "videos": dict(nearest["media"].get("videos") or {}),
                "images": dict(nearest["media"].get("images") or {}),
            }
            b["_media_fallback_reason"] = f"borrowed from {nearest['beat_id']}"
            print(f"[edit-video] beat {b['beat_id']} was empty, borrowed media from beat {nearest['beat_id']}")
        return

    print(f"[edit-video][WARN] scene {scene_id}: EVERY beat came back empty — trying generic last-resort terms")
    last_resort_media = await _fetch_media_for_keywords(_LAST_RESORT_BROLL_TERMS, f"{scene_id}:last-resort")
    if (last_resort_media.get("videos") or {}).get("results") or (last_resort_media.get("images") or {}).get("results"):
        for b in still_empty:
            b["media"] = last_resort_media
            b["_media_fallback_reason"] = "generic last-resort terms"
    else:
        print(
            f"[edit-video][ERROR] scene {scene_id}: last-resort generic search ALSO returned "
            f"nothing — check PEXELS_API_KEY / Pexels reachability, this is no longer a content issue"
        )


async def _build_scene_beats(scene: dict) -> list:
    """
    Splits a scene's aligned narration into ~10s beats (8-12s each, see
    BEAT_MIN_SECONDS/BEAT_MAX_SECONDS) and, for each beat, generates its own
    B-roll keywords, media type (video/image), and — new — image motion
    effect (LLM, all three in one call — see _generate_beat_keywords /
    BEAT_KEYWORDS_PROMPT). This is what makes the B-roll rotate mid-scene
    instead of one static clip stretched across the whole voiceover, AND
    what makes still-image beats pan/zoom/tilt instead of sitting frozen.

    After fetching each beat's candidates, `_dedupe_beats_media_across_scene`
    reorders each beat's candidate pool so consecutive beats don't default
    to the exact same Pexels asset (see its docstring) — the actual fix for
    "one clip plays for way longer than one beat".

    Falls back to a single implicit beat (using the scene-level
    broll_keywords / whole-scene Pexels search) when there's no timed
    narration to split — e.g. silent scenes, or scenes whose voice/alignment
    step failed. That implicit beat gets the default motion_type since no
    LLM call runs for it.
    """
    scene_id = scene.get("scene_id")
    word_segments = scene.get("word_segments") or []
    timed_words = [w for w in word_segments if "start" in w and "end" in w]

    if not timed_words:
        media = await _fetch_scene_media(scene)
        if not media["videos"]["results"] and not media["images"]["results"]:
            print(
                f"[edit-video][WARN] scene {scene_id}: single implicit beat came back "
                f"empty — trying generic last-resort terms"
            )
            last_resort_media = await _fetch_media_for_keywords(
                _LAST_RESORT_BROLL_TERMS, f"{scene_id}:last-resort"
            )
            media["videos"] = last_resort_media["videos"]
            media["images"] = last_resort_media["images"]
        return [{
            "beat_id": f"{scene_id}_b1",
            "beat_index": 0,
            "start": None,
            "end": None,
            "vo_text": scene.get("vo_text", ""),
            "keywords": media["keywords"],
            "motion_type": _DEFAULT_MOTION_TYPE,
            "media": {"videos": media["videos"], "images": media["images"]},
        }]

    scene_start = timed_words[0]["start"]
    scene_end = timed_words[-1]["end"]
    boundaries = _compute_beat_boundaries(scene_start, scene_end)
    fallback_keywords = _get_scene_broll_keywords(scene)

    beats = []
    for i, (b_start, b_end) in enumerate(boundaries):
        beats.append({
            "beat_id": f"{scene_id}_b{i + 1}",
            "beat_index": i,
            "start": b_start,
            "end": b_end,
            "vo_text": _words_in_range(timed_words, b_start, b_end),
        })

    # NOTE: this loop runs sequentially (not asyncio.gather) rather than
    # in parallel, specifically so each beat's media_type/motion_type
    # decision can be told what the PREVIOUS beat actually got
    # (previous_media_type / previous_motion_type) — that is what lets the
    # prompt alternate video/image and vary pan/zoom/tilt deliberately
    # across a scene instead of every beat guessing independently and, by
    # chance, all landing on the same choice. This trades a little latency
    # for beats that visually vary on purpose.
    previous_media_type = None
    previous_motion_type = None
    for b in beats:
        kws, media_type, motion_type = await _generate_beat_keywords(
            scene, b["vo_text"], b["beat_index"], len(beats), fallback_keywords,
            previous_media_type=previous_media_type,
            previous_motion_type=previous_motion_type,
        )
        b["keywords"] = kws
        b["preferred_media_type"] = media_type
        b["motion_type"] = motion_type
        previous_media_type = media_type
        # Only update the "previous image motion" memory when this beat
        # actually was an image — a video beat in between two image beats
        # shouldn't reset the diversity tracking back to null.
        if media_type == "image":
            previous_motion_type = motion_type

    media_results = await asyncio.gather(*[
        _fetch_media_for_keywords(b["keywords"], f"{scene_id}:{b['beat_id']}")
        for b in beats
    ])
    for b, media in zip(beats, media_results):
        b["media"] = media

    await _fill_empty_beats(scene, beats, fallback_keywords)

    _dedupe_beats_media_across_scene(beats)

    expected_min_beats = math.ceil((scene_end - scene_start) / BEAT_MAX_SECONDS)
    if len(beats) < expected_min_beats:
        # Defensive tripwire: if this ever fires, `_compute_beat_boundaries`
        # itself under-split the scene relative to BEAT_MAX_SECONDS — flag
        # it loudly rather than silently shipping a scene that will show
        # one clip for way longer than BEAT_MAX_SECONDS on screen.
        print(
            f"[edit-video][WARN] scene {scene_id}: {scene_end - scene_start:.1f}s of narration "
            f"only produced {len(beats)} beat(s), expected at least {expected_min_beats} "
            f"given BEAT_MAX_SECONDS={BEAT_MAX_SECONDS} — B-roll may hold too long on screen"
        )

    print(
        f"[edit-video] scene {scene_id}: {scene_end - scene_start:.1f}s of narration "
        f"split into {len(beats)} rotating B-roll beat(s)"
    )
    return beats


def _aggregate_beats_media(beats: list) -> dict:
    """Merges every beat's candidate pool into one scene-level view, kept
    on scene['media'] purely for backward-compatible UIs that expect a
    single 'all candidates for this scene' list."""
    all_videos, all_images = [], []
    for b in beats:
        m = b.get("media") or {}
        all_videos.extend((m.get("videos") or {}).get("results") or [])
        all_images.extend((m.get("images") or {}).get("results") or [])
    limit = PEXELS_SCENE_RESULT_LIMIT * max(len(beats), 1)
    videos = _dedupe_and_trim(all_videos, limit=limit)
    images = _dedupe_and_trim(all_images, limit=limit)
    return {
        "videos": {"total_results": len(videos), "results": videos, "error": None},
        "images": {"total_results": len(images), "results": images, "error": None},
    }


def _default_animation(scene: dict) -> dict:
    on_screen_text = (scene.get("on_screen_text") or "").strip()

    if on_screen_text:
        # Placement defaults to top_center, not bottom_center: captions now
        # sit in the very-bottom safe zone (see DEFAULT_CAPTION_STYLE /
        # ANIMATION_PLANNER_PROMPT), so an overlay_text fallback needs to
        # stay out of that band to avoid overlapping the caption line.
        return {
            "animation_type": "lower_third",
            "category": "overlay_text",
            "placement": "top_center",
            "z_index_layer": "foreground",
            "trigger": "scene_start",
            "duration_frames": 60,
            "content_binding": "text:on_screen_text",
            "render_engine_hint": "remotion",
        }

    return {
        "animation_type": "full_screen_broll",
        "category": "full_screen",
        "placement": "full_frame",
        "z_index_layer": "background",
        "trigger": "scene_start",
        "duration_frames": 150,
        "content_binding": "broll:visual_intent",
        "render_engine_hint": "ffmpeg",
    }


def _validate_animation(raw: dict, scene: dict) -> dict:
    fallback = _default_animation(scene)

    if not isinstance(raw, dict):
        return fallback

    animation_type = raw.get("animation_type")
    if animation_type not in _VALID_ANIMATION_TYPES:
        return fallback

    category = _ANIMATION_TYPE_TO_CATEGORY[animation_type]

    placement = raw.get("placement")
    if placement not in _VALID_PLACEMENTS:
        placement = "full_frame" if category in ("full_screen", "transition") else "top_center"

    # Caption safe zone: word-by-word captions are always burned in along
    # the very bottom of the frame (see DEFAULT_CAPTION_STYLE). If the LLM
    # nonetheless returned a bottom_* placement for an overlay_text/
    # overlay_graphic animation, remap it up and out of that band rather
    # than trusting it verbatim — full_screen/transition categories are
    # exempt since they own the whole frame anyway.
    if category in ("overlay_text", "overlay_graphic") and placement in (
        "bottom_left", "bottom_center", "bottom_right",
    ):
        placement = "top_center"

    z_index_layer = raw.get("z_index_layer")
    if z_index_layer not in _VALID_Z_LAYERS:
        z_index_layer = "background" if category == "full_screen" else "foreground"

    trigger = raw.get("trigger")
    if trigger not in _VALID_TRIGGERS:
        trigger = "scene_start"

    duration_frames = raw.get("duration_frames")
    try:
        duration_frames = int(duration_frames)
        if duration_frames <= 0 or duration_frames > 900:
            raise ValueError
    except (TypeError, ValueError):
        duration_frames = fallback["duration_frames"]

    content_binding = raw.get("content_binding")
    if not isinstance(content_binding, str) or not content_binding.strip():
        content_binding = fallback["content_binding"]

    render_engine_hint = raw.get("render_engine_hint")
    if render_engine_hint not in _VALID_RENDER_HINTS:
        render_engine_hint = "remotion" if category in ("overlay_text", "overlay_graphic", "pip", "branding") else "ffmpeg"

    return {
        "animation_type": animation_type,
        "category": category,
        "placement": placement,
        "z_index_layer": z_index_layer,
        "trigger": trigger,
        "duration_frames": duration_frames,
        "content_binding": content_binding,
        "render_engine_hint": render_engine_hint,
    }


async def _plan_scene_animation(scene: dict) -> dict:
    scene_context = {
        "scene_id": scene.get("scene_id"),
        "vo_text": scene.get("vo_text", ""),
        "visual_intent": scene.get("visual_intent", ""),
        "on_screen_text": scene.get("on_screen_text", ""),
        "requires_animation": scene.get("requires_animation", False),
    }

    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": ANIMATION_PLANNER_PROMPT},
                    {"role": "user", "content": json.dumps(scene_context)},
                ],
                stream=False,
            )
        )
        _record_token_usage("edit video - animation plan", res)

        content = (res.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()

        raw = json.loads(content)

    except Exception as e:
        print(f"[edit-video] scene {scene.get('scene_id')} animation planning failed, using fallback: {e}")
        return _default_animation(scene)

    return _validate_animation(raw, scene)


def _props_title_card(scene: dict, animation: dict) -> dict:
    return {
        "title": (scene.get("on_screen_text") or "").strip() or (scene.get("scene_id") or ""),
        "subtitle": "",
    }


def _props_quote_card(scene: dict, animation: dict) -> dict:
    vo_text = (scene.get("vo_text") or "").strip()
    quote = vo_text.split(". ")[0].strip()
    if quote and not quote.endswith((".", "!", "?")):
        quote += "."
    return {
        "quote": quote,
        "attribution": "",
    }


def _props_stat_counter(scene: dict, animation: dict) -> dict:
    on_screen_text = (scene.get("on_screen_text") or "").strip()
    match = re.search(r"[\d,]+(?:\.\d+)?", on_screen_text)
    value = match.group(0) if match else on_screen_text
    label = on_screen_text.replace(value, "").strip(" •-") if match else ""
    return {
        "value": value,
        "label": label or on_screen_text,
    }


def _props_bullet_list(scene: dict, animation: dict) -> dict:
    on_screen_text = (scene.get("on_screen_text") or "").strip()
    if "•" in on_screen_text:
        items = [item.strip() for item in on_screen_text.split("•") if item.strip()]
    elif on_screen_text:
        items = [on_screen_text]
    else:
        vo_text = (scene.get("vo_text") or "").strip()
        items = [s.strip() for s in re.split(r"[.;]", vo_text) if s.strip()][:4]
    return {
        "title": "",
        "items": items,
    }


def _props_data_viz(scene: dict, animation: dict) -> dict:
    return {
        "label": (scene.get("on_screen_text") or "").strip(),
        "caption": (scene.get("visual_intent") or "").strip(),
    }


def _props_icon_sequence(scene: dict, animation: dict) -> dict:
    binding = animation.get("content_binding", "")
    icon = binding.split(":", 1)[1] if ":" in binding else "circle"
    return {
        "icons": [icon],
        "label": (scene.get("on_screen_text") or "").strip(),
    }


REMOTION_INFOGRAPHIC_LIBRARY = {
    "full_screen_title_card": {
        "composition_id": "TitleCard",
        "props_builder": _props_title_card,
    },
    "full_screen_quote_card": {
        "composition_id": "QuoteCard",
        "props_builder": _props_quote_card,
    },
    "full_screen_data_viz": {
        "composition_id": "DataVizFullScreen",
        "props_builder": _props_data_viz,
    },
    "stat_counter_overlay": {
        "composition_id": "StatCounterOverlay",
        "props_builder": _props_stat_counter,
    },
    "bullet_list_reveal": {
        "composition_id": "BulletListReveal",
        "props_builder": _props_bullet_list,
    },
    "icon_sequence": {
        "composition_id": "IconSequenceOverlay",
        "props_builder": _props_icon_sequence,
    },
    "icon_pop_in": {
        "composition_id": "IconPopIn",
        "props_builder": _props_icon_sequence,
    },
}


def _get_scene_infographic(scene: dict, animation: dict) -> dict:
    animation_type = animation.get("animation_type")
    template = REMOTION_INFOGRAPHIC_LIBRARY.get(animation_type)

    if not template:
        return None

    try:
        props = template["props_builder"](scene, animation)
    except Exception as e:
        print(f"[edit-video] scene {scene.get('scene_id')} infographic prop-building failed: {e}")
        props = {}

    return {
        "composition_id": template["composition_id"],
        "animation_type": animation_type,
        "props": props,
        "duration_frames": animation.get("duration_frames"),
        "trigger": animation.get("trigger"),
        "placement": animation.get("placement"),
        "render_engine_hint": "remotion",
    }


async def _get_or_create_tagged_text(scene: dict, scene_id, user_id: str, vo_text: str) -> str:
    """
    Voice tags are mandatory before any TTS call — narration must always be
    run through add_script_tags first, whether that happens at initial scene
    creation or later when a scene's voice is regenerated. This raises on
    failure rather than ever silently falling back to the raw, untagged
    script, so callers must handle the exception (see _process_scene and
    update_scene_voice for how each path surfaces that failure).
    """
    tags_request = AddScriptTagsRequest(
        userId=user_id,
        script=vo_text,
    )
    tags_result = await add_script_tags(tags_request)
    return tags_result["tagged_script"]


async def _process_scene(scene: dict, request: EditVideo) -> dict:
    scene_out = dict(scene)
    vo_text = scene.get("vo_text", "")

    # Animation/placement planning has no dependency on audio timing, so it
    # can run fully in parallel with everything below.
    animation_task = asyncio.create_task(_plan_scene_animation(scene))

    async def _finalize() -> dict:
        # Beats depend on scene_out["word_segments"]/["start"]/["end"]
        # already being set by the caller before _finalize() runs — this is
        # why beat-building happens AFTER voice + alignment instead of in
        # parallel with them like the old scene-level media fetch did.
        scene_out["beats"] = await _build_scene_beats(scene_out)
        scene_out["media"] = _aggregate_beats_media(scene_out["beats"])
        scene_out["animation"] = await animation_task
        scene_out["infographics"] = _get_scene_infographic(scene, scene_out["animation"])
        if scene_out.get("start") is not None and scene_out.get("end") is not None:
            scene_out["duration_seconds"] = round(scene_out["end"] - scene_out["start"], 3)
        else:
            scene_out["duration_seconds"] = None
        return scene_out

    if not vo_text.strip():
        scene_out["voiceover"] = None
        scene_out["start"] = None
        scene_out["end"] = None
        scene_out["word_segments"] = []
        scene_out["error"] = None
        return await _finalize()

    # Voice tags are mandatory before any voiceover is generated — narration
    # must always go through add_script_tags first. If tagging fails, the
    # scene fails its voiceover step outright (same as a voice-gen or
    # alignment failure) instead of silently speaking the raw, untagged
    # script.
    try:
        tagged_text = await _get_or_create_tagged_text(scene, scene.get("scene_id"), request.userId, vo_text)
    except Exception as e:
        print(f"[edit-video] scene {scene.get('scene_id')} tagging failed: {e}")
        scene_out["tagged_vo_text"] = None
        scene_out["voiceover"] = None
        scene_out["start"] = None
        scene_out["end"] = None
        scene_out["word_segments"] = []
        scene_out["error"] = f"voice tagging failed: {e}"
        return await _finalize()

    try:
        speech_result = await _generate_speech_possibly_chunked(
            user_id=request.userId,
            tagged_text=tagged_text,
            voice=request.voice,
            lang_code=request.langCode,
        )
    except Exception as e:
        print(f"[edit-video] scene {scene.get('scene_id')} voice generation failed: {e}")
        scene_out["tagged_vo_text"] = tagged_text
        scene_out["voiceover"] = None
        scene_out["start"] = None
        scene_out["end"] = None
        scene_out["word_segments"] = []
        scene_out["error"] = f"voice generation failed: {e}"
        return await _finalize()

    try:
        scene_timestamps = await _generate_word_timestamps(speech_result["url"])
    except Exception as e:
        print(f"[edit-video] scene {scene.get('scene_id')} whisperx alignment failed: {e}")
        scene_out["tagged_vo_text"] = tagged_text
        scene_out["voiceover"] = speech_result
        scene_out["start"] = None
        scene_out["end"] = None
        scene_out["word_segments"] = []
        scene_out["error"] = f"timestamp alignment failed: {e}"
        return await _finalize()

    word_segments = scene_timestamps.get("word_segments", [])
    timed_words = [w for w in word_segments if "start" in w and "end" in w]

    scene_out["tagged_vo_text"] = tagged_text
    scene_out["voiceover"] = speech_result
    scene_out["start"] = timed_words[0]["start"] if timed_words else None
    scene_out["end"] = timed_words[-1]["end"] if timed_words else None
    scene_out["word_segments"] = word_segments
    scene_out["error"] = None

    return await _finalize()


def _enforce_scene_limit(scenes: list, max_scenes: int = 5) -> list:
    if len(scenes) <= max_scenes:
        return scenes

    print(f"[edit-video] model returned {len(scenes)} scenes, merging down to {max_scenes}")

    kept = scenes[: max_scenes - 1]
    overflow = scenes[max_scenes - 1:]

    merged_vo_text = " ".join(s.get("vo_text", "").strip() for s in overflow if s.get("vo_text"))
    last_scene = dict(overflow[0]) if overflow else {}
    last_scene["scene_id"] = f"s{max_scenes}"
    last_scene["vo_text"] = merged_vo_text

    return kept + [last_scene]


# =============================================================================
# BEAT-LEVEL B-ROLL SELECTION  (replaces the old scene-level resolver)
# =============================================================================

def _resolve_beat_broll_selection(beat: dict) -> tuple:
    """Per-beat equivalent of the old `_resolve_scene_broll_selection`.
    Each beat carries its own candidate pool and its own optional manual
    override, so B-roll selection/override now happens beat by beat."""
    override = beat.get("broll_override")
    if override and override.get("asset_id") is not None:
        source = override.get("source")
        file_url = _resolve_broll_file_url(override, source)
        if file_url:
            return (
                {
                    "id": override.get("asset_id"),
                    "file_url": file_url,
                    "source": source,
                    **{k: v for k, v in override.items() if k not in ("asset_id", "source", "file_url")},
                },
                source,
            )
        print(
            f"[timeline] beat {beat.get('beat_id')} has a broll_override "
            f"that couldn't be resolved to a landscape file — falling back to default candidate"
        )

    media = beat.get("media") or {}
    video_candidates = (media.get("videos") or {}).get("results") or []
    image_candidates = (media.get("images") or {}).get("results") or []

    # FIX (images almost never used): this used to always try video first
    # and only fall back to images when zero video candidates existed —
    # which meant "use an image here" was never a real creative choice, it
    # only ever happened by accident of Pexels having no video results.
    # `preferred_media_type` is now decided per-beat by the LLM in
    # `_generate_beat_keywords` (see BEAT_KEYWORDS_PROMPT), so respect that
    # choice first and only fall through to the other pool if the
    # AI-preferred type genuinely has no candidates available.
    preferred = beat.get("preferred_media_type")
    if preferred == "image":
        if image_candidates:
            return image_candidates[0], "image"
        if video_candidates:
            return video_candidates[0], "video"
        return None, None

    # preferred == "video", or no preference recorded (legacy beats) —
    # keep the original video-first behavior.
    if video_candidates:
        return video_candidates[0], "video"
    if image_candidates:
        return image_candidates[0], "image"
    return None, None


def _resolve_beat_motion_type(beat: dict) -> str:
    """
    NEW: resolves which image motion effect (pan/zoom/tilt) a beat should
    use if it renders as a still image. Precedence:

      1. A manual override's motion_type (set via
         PATCH /timeline/{video_id}/scene/{scene_id}/broll with
         motion_type provided) — an explicit user choice always wins.
      2. The motion_type the LLM chose for this beat in
         `_generate_beat_keywords` (see BEAT_KEYWORDS_PROMPT).
      3. `_DEFAULT_MOTION_TYPE`, only ever reached for legacy beats saved
         before this feature existed.

    This is purely a lookup — no new motion selection happens here. The
    only two places any motion_type value is ever actually CHOSEN are the
    LLM call (_generate_beat_keywords) and its failure-path diversified
    default (_pick_diversified_motion_type).
    """
    override = beat.get("broll_override") or {}
    motion = override.get("motion_type")
    if motion in _VALID_MOTION_TYPES:
        return motion

    motion = beat.get("motion_type")
    if motion in _VALID_MOTION_TYPES:
        return motion

    return _DEFAULT_MOTION_TYPE


def _find_beat_broll_candidate(beat: dict, asset_id: Any, source: str) -> Optional[dict]:
    media = beat.get("media") or {}
    pool_key = "videos" if source == "video" else "images"
    candidates = (media.get(pool_key) or {}).get("results") or []
    for c in candidates:
        if str(c.get("id")) == str(asset_id):
            return c
    return None


def _seconds_to_frames(seconds: float, fps: int = TIMELINE_FPS) -> int:
    return max(round((seconds or 0.0) * fps), 0)


def build_timeline_from_scenes(scenes: list, fps: int = TIMELINE_FPS) -> dict:
    tracks = []
    cumulative_frames = 0

    for scene in scenes:
        scene_id = scene.get("scene_id")
        start_sec = scene.get("start") or 0.0
        end_sec = scene.get("end") or 0.0
        scene_duration_frames = max(_seconds_to_frames(end_sec - start_sec, fps), fps)

        scene_start_frame = cumulative_frames
        scene_end_frame = cumulative_frames + scene_duration_frames

        voiceover = scene.get("voiceover")
        if voiceover and voiceover.get("url"):
            tracks.append({
                "track_id": f"audio_{scene_id}",
                "scene_id": scene_id,
                "type": "audio",
                "file_url": voiceover["url"],
                "vo_text": scene.get("vo_text"),
                "startFrame": scene_start_frame,
                "endFrame": scene_end_frame,
                # explicit scene timing, in seconds, alongside the frame
                # numbers — persisted verbatim into timeline_json/raw_scenes.
                "scene_start_sec": start_sec,
                "scene_end_sec": end_sec,
            })

        word_segments = scene.get("word_segments") or []
        words = []
        for w in word_segments:
            if "start" not in w or "end" not in w:
                continue
            words.append({
                "word": w.get("word", ""),
                "startFrame": scene_start_frame + _seconds_to_frames(w["start"] - start_sec, fps),
                "endFrame": scene_start_frame + _seconds_to_frames(w["end"] - start_sec, fps),
            })
        if words:
            caption_track = {
                "track_id": f"caption_{scene_id}",
                "scene_id": scene_id,
                "type": "caption_word",
                "words": words,
            }
            # Always attach a style — merge the scene's custom caption_style
            # (if any) on top of DEFAULT_CAPTION_STYLE, so captions render
            # very low on screen unless a scene explicitly overrides the
            # position, while still keeping any other custom style fields
            # (font_size, text_color, etc.) intact.
            caption_track["style"] = {**DEFAULT_CAPTION_STYLE, **(scene.get("caption_style") or {})}
            tracks.append(caption_track)

        # ---------------------------------------------------------------
        # ROTATING B-ROLL: one track per beat (~10s each), instead of a
        # single broll track spanning the whole scene. This is what makes
        # the visible footage change mid-scene. Each track also now carries
        # `motion_type` — the LLM-chosen (or manually overridden) pan/zoom/
        # tilt effect to apply if this beat's selected_asset is an image.
        # ---------------------------------------------------------------
        beats = scene.get("beats") or []
        if not beats:
            # Legacy fallback for rows saved before the beats refactor —
            # synthesize a single beat from the old scene-level shape so
            # old videos keep rendering.
            beats = [{
                "beat_id": f"{scene_id}_b1",
                "start": start_sec,
                "end": end_sec,
                "media": scene.get("media") or {},
                "broll_override": scene.get("broll_override"),
            }]

        for beat in beats:
            b_start = beat.get("start")
            b_end = beat.get("end")
            if b_start is None or b_end is None:
                beat_start_frame = scene_start_frame
                beat_end_frame = scene_end_frame
            else:
                beat_start_frame = scene_start_frame + _seconds_to_frames(b_start - start_sec, fps)
                beat_end_frame = scene_start_frame + _seconds_to_frames(b_end - start_sec, fps)
                beat_end_frame = min(beat_end_frame, scene_end_frame)
                beat_start_frame = max(scene_start_frame, min(beat_start_frame, beat_end_frame))

            default_asset, default_source = _resolve_beat_broll_selection(beat)
            media = beat.get("media") or {}
            video_candidates = (media.get("videos") or {}).get("results") or []
            image_candidates = (media.get("images") or {}).get("results") or []

            broll_track = {
                "track_id": f"broll_{scene_id}_{beat.get('beat_id')}",
                "scene_id": scene_id,
                "beat_id": beat.get("beat_id"),
                "type": "broll",
                "layer": "background",
                "startFrame": beat_start_frame,
                "endFrame": beat_end_frame,
                "beat_start_sec": b_start,
                "beat_end_sec": b_end,
                "keywords": beat.get("keywords"),
                "preferred_media_type": beat.get("preferred_media_type"),
                # NEW: resolved motion effect for this beat (LLM choice,
                # or manual override if one was set). The render step reads
                # this directly — see _render_scene / _prepare_broll_clip.
                "motion_type": _resolve_beat_motion_type(beat),
                "selected_asset": {
                    "asset_id": (default_asset or {}).get("id"),
                    "file_url": _resolve_broll_file_url(default_asset, default_source) if default_asset else None,
                    "source": default_source,
                    "width": (default_asset or {}).get("width"),
                    "height": (default_asset or {}).get("height"),
                    "video_files": (default_asset or {}).get("video_files"),
                    "src": (default_asset or {}).get("src"),
                } if default_asset else None,
                "candidates": {
                    "videos": video_candidates,
                    "images": image_candidates,
                },
            }

            background_color = scene.get("background_color")
            if background_color:
                broll_track["background_color"] = background_color

            tracks.append(broll_track)

        # ---------------------------------------------------------------
        # ANIMATION / INFOGRAPHIC — placement is entirely LLM-directed
        # (see ANIMATION_PLANNER_PROMPT). The renderer draws it at exactly
        # animation["placement"] / z_index_layer / trigger below; nothing
        # else in this pipeline decides where it goes.
        # ---------------------------------------------------------------
        animation = scene.get("animation") or {}
        infographic = scene.get("infographics")

        overlay_duration = animation.get("duration_frames") or (scene_end_frame - scene_start_frame)
        overlay_start = scene_start_frame
        overlay_end = min(scene_start_frame + overlay_duration, scene_end_frame)

        if infographic:
            tracks.append({
                "track_id": f"infographic_{scene_id}",
                "scene_id": scene_id,
                "type": "infographic",
                "layer": "foreground",
                "composition_id": infographic.get("composition_id"),
                "animation_type": infographic.get("animation_type"),
                "placement": infographic.get("placement"),
                "props": infographic.get("props", {}),
                "startFrame": overlay_start,
                "endFrame": overlay_end,
                "duration_frames": overlay_end - overlay_start,
                "status": "pending_render",
                "asset_url": None,
                "render_engine_hint": "remotion",
            })
        elif animation:
            tracks.append({
                "track_id": f"overlay_{scene_id}",
                "scene_id": scene_id,
                "type": "animation",
                "layer": animation.get("z_index_layer", "foreground"),
                "animation_type": animation.get("animation_type"),
                "placement": animation.get("placement"),
                "content_binding": animation.get("content_binding"),
                "startFrame": overlay_start,
                "endFrame": overlay_end,
                "render_engine_hint": animation.get("render_engine_hint"),
            })

        cumulative_frames = scene_end_frame

    return {
        "fps": fps,
        "total_frames": cumulative_frames,
        "resolution": {"width": TIMELINE_WIDTH, "height": TIMELINE_HEIGHT},
        "tracks": tracks,
    }


@app.post("/edit-video")
async def edit_video(request: EditVideo):
    try:
        res = await _openai_create_with_timeout(
            lambda: openai_client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": SCRIPT_SCENE_PROMPT},
                    {"role": "user", "content": request.script},
                ],
                stream=False,
            )
        )
        _record_token_usage("edit video", res)

        content = (res.choices[0].message.content or "").strip()

        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()

    except HTTPException:
        raise
    except Exception as e:
        print(f"[edit-video] scene generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate scenes")

    try:
        parsed = json.loads(content)

        if isinstance(parsed, dict) and "scenes" in parsed:
            scenes = parsed["scenes"]
        elif isinstance(parsed, list):
            scenes = parsed
        else:
            raise ValueError(f"Unexpected JSON shape: {type(parsed)}")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[edit-video] JSON parse failed: {e} | raw content: {content[:500]}")
        raise HTTPException(status_code=502, detail="Model did not return valid JSON")

    scenes = _enforce_scene_limit(scenes, max_scenes=5)

    scenes_with_voice_and_timestamps = []
    for scene in scenes:
        scene_result = await _process_scene(scene, request)
        scenes_with_voice_and_timestamps.append(scene_result)

    failed_scenes = [s["scene_id"] for s in scenes_with_voice_and_timestamps if s.get("error")]
    if failed_scenes:
        print(f"[edit-video] completed with {len(failed_scenes)} failed scene(s): {failed_scenes}")

    timeline_json = build_timeline_from_scenes(scenes_with_voice_and_timestamps)

    # Explicit start/end (and duration) summary per scene, independent of
    # the deeper raw_scenes/timeline_json shapes — convenient for clients
    # that just want "when does each scene start/end" without walking
    # tracks, and persisted into the DB row below via scene_timings.
    scene_timings = [
        {
            "scene_id": s.get("scene_id"),
            "start": s.get("start"),
            "end": s.get("end"),
            "duration_seconds": s.get("duration_seconds"),
            "beats": [
                {"beat_id": b.get("beat_id"), "start": b.get("start"), "end": b.get("end")}
                for b in (s.get("beats") or [])
            ],
        }
        for s in scenes_with_voice_and_timestamps
    ]

    video_id = str(uuid.uuid4())
    try:
        supabase.table("videos").insert({
            "id": video_id,
            "user_id": request.userId,
            "script": request.script,
            "voice": request.voice,
            "lang_code": request.langCode,
            "timeline_json": timeline_json,
            "timeline_version": 1,
            "raw_scenes": scenes_with_voice_and_timestamps,
            "scene_timings": scene_timings,
        }).execute()
    except Exception as e:
        print(f"[edit-video] failed to persist video row: {e}")
        raise HTTPException(status_code=500, detail="Failed to save video")

    return {
        "video_id": video_id,
        "timeline": timeline_json,
        "scenes": scenes_with_voice_and_timestamps,
        "scene_timings": scene_timings,
        "failed_scene_ids": failed_scenes,
    }


@app.get("/timeline/{video_id}")
async def get_timeline(video_id: str):
    try:
        row = (
            supabase.table("videos")
            .select("timeline_json, timeline_version, scene_timings")
            .eq("id", video_id)
            .single()
            .execute()
        )
    except Exception as e:
        print(f"[get-timeline] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    return row.data


@app.patch("/timeline/{video_id}")
async def patch_timeline(video_id: str, patch: TrackPatch):
    try:
        row = (
            supabase.table("videos")
            .select("timeline_json, timeline_version, raw_scenes")
            .eq("id", video_id)
            .single()
            .execute()
        )
    except Exception as e:
        print(f"[patch-timeline] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    timeline = row.data["timeline_json"]
    current_version = row.data.get("timeline_version", 1)
    raw_scenes = row.data.get("raw_scenes") or []

    track_found = None
    for track in timeline.get("tracks", []):
        if track.get("track_id") == patch.track_id:
            track.update(patch.updates)
            track_found = track
            break

    if not track_found:
        raise HTTPException(status_code=404, detail=f"Track {patch.track_id} not found in timeline")

    scene_id = track_found.get("scene_id")
    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)

    visual_change = False
    if scene_index is not None:
        scene = dict(raw_scenes[scene_index])

        if track_found.get("type") == "broll":
            beat_id = track_found.get("beat_id")
            beats = scene.get("beats") or []
            beat_idx = next((i for i, b in enumerate(beats) if b.get("beat_id") == beat_id), None)

            if "selected_asset" in patch.updates and patch.updates["selected_asset"]:
                sel = patch.updates["selected_asset"]
                override = {
                    "asset_id": sel.get("asset_id") or sel.get("id"),
                    "source": sel.get("source"),
                    "file_url": sel.get("file_url"),
                    "width": sel.get("width"),
                    "height": sel.get("height"),
                    "video_files": sel.get("video_files"),
                    "src": sel.get("src"),
                    # NEW: allow a motion_type to travel along with a manual
                    # asset override applied via generic track patching too,
                    # not just the dedicated /broll endpoint below.
                    "motion_type": sel.get("motion_type"),
                }
                if beat_idx is not None:
                    beats[beat_idx] = {**beats[beat_idx], "broll_override": override}
                    scene["beats"] = beats
                else:
                    # legacy rows without beats: keep the old scene-level override behavior
                    scene["broll_override"] = override
                visual_change = True

            if "background_color" in patch.updates:
                if patch.updates["background_color"]:
                    _validate_hex_color(patch.updates["background_color"], "background_color")
                scene["background_color"] = patch.updates["background_color"]
                visual_change = True

        elif track_found.get("type") == "caption_word":
            if "style" in patch.updates and isinstance(patch.updates["style"], dict):
                existing_style = scene.get("caption_style") or {}
                scene["caption_style"] = {**existing_style, **patch.updates["style"]}
                visual_change = True

        raw_scenes[scene_index] = scene

    new_version = current_version + 1

    update_payload = {
        "timeline_json": timeline,
        "timeline_version": new_version,
    }
    if scene_index is not None:
        update_payload["raw_scenes"] = raw_scenes
    if visual_change:
        update_payload["final_video_url"] = None
        update_payload["render_status"] = "stale_needs_render"

    try:
        supabase.table("videos").update(update_payload).eq("id", video_id).execute()
    except Exception as e:
        print(f"[patch-timeline] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save edit")

    return {
        "timeline_version": new_version,
        "track_id": patch.track_id,
        "needs_render": visual_change,
    }


@app.patch("/timeline/{video_id}/scene/{scene_id}/style")
async def update_scene_style(video_id: str, scene_id: str, update: SceneStyleUpdate):
    if update.animation_type is not None and update.animation_type not in _VALID_CAPTION_ANIMATION_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"animation_type must be one of {sorted(_VALID_CAPTION_ANIMATION_TYPES)}",
        )
    _validate_hex_color(update.text_color, "text_color")
    _validate_hex_color(update.outline_color, "outline_color")
    _validate_hex_color(update.background_color, "background_color")

    if (
        update.font_size is None and update.text_color is None
        and update.outline_color is None and update.animation_type is None
        and update.background_color is None
    ):
        raise HTTPException(status_code=422, detail="Provide at least one field to update")

    try:
        row = (
            supabase.table("videos")
            .select("raw_scenes, timeline_version")
            .eq("id", video_id)
            .single()
            .execute()
        )
    except Exception as e:
        print(f"[update-scene-style] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)
    if scene_index is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    scene = dict(raw_scenes[scene_index])

    style_fields = {
        "font_size": update.font_size,
        "text_color": update.text_color,
        "outline_color": update.outline_color,
        "animation_type": update.animation_type,
    }
    style_fields = {k: v for k, v in style_fields.items() if v is not None}

    if style_fields:
        existing_style = scene.get("caption_style") or {}
        scene["caption_style"] = {**existing_style, **style_fields}

    if update.background_color is not None:
        scene["background_color"] = update.background_color

    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes,
            "timeline_json": timeline_json,
            "timeline_version": new_version,
            "final_video_url": None,
            "render_status": "stale_needs_render",
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[update-scene-style] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save style edit")

    return {
        "video_id": video_id,
        "scene_id": scene_id,
        "timeline_version": new_version,
        "caption_style": scene.get("caption_style"),
        "background_color": scene.get("background_color"),
        "timeline": timeline_json,
        "needs_render": True,
    }


@app.post("/timeline/{video_id}/scene/{scene_id}/voice")
async def update_scene_voice(video_id: str, scene_id: str, update: SceneVoiceUpdate):
    try:
        row = (
            supabase.table("videos")
            .select("raw_scenes, user_id, lang_code, timeline_version")
            .eq("id", video_id)
            .single()
            .execute()
        )
    except Exception as e:
        print(f"[update-scene-voice] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    user_id = row.data.get("user_id")
    lang_code = row.data.get("lang_code", "en")
    current_version = row.data.get("timeline_version", 1)

    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)
    if scene_index is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    scene = dict(raw_scenes[scene_index])

    # Voice tags are mandatory before any TTS call — never speak raw vo_text.
    # Reuse an existing tagged version if we have one, otherwise (re-)tag
    # from the raw script and persist the tagged text back onto the scene.
    # If tagging fails, the whole request fails instead of speaking
    # untagged text.
    raw_text = scene.get("vo_text", "")
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail=f"Scene {scene_id} has no narration text to regenerate")

    tagged_text = scene.get("tagged_vo_text")
    if not tagged_text or not tagged_text.strip():
        try:
            tagged_text = await _get_or_create_tagged_text(scene, scene_id, user_id, raw_text)
        except Exception as e:
            print(f"[update-scene-voice] voice tagging failed for scene {scene_id}: {e}")
            raise HTTPException(status_code=502, detail=f"Voice tagging failed: {e}")
        scene["tagged_vo_text"] = tagged_text

    try:
        speech_result = await _generate_speech_possibly_chunked(
            user_id=user_id,
            tagged_text=tagged_text,
            voice=update.voice,
            lang_code=lang_code,
        )
    except Exception as e:
        print(f"[update-scene-voice] voice generation failed for scene {scene_id}: {e}")
        raise HTTPException(status_code=502, detail=f"Voice generation failed: {e}")

    try:
        scene_timestamps = await _generate_word_timestamps(speech_result["url"])
    except Exception as e:
        print(f"[update-scene-voice] whisperx alignment failed for scene {scene_id}: {e}")
        raise HTTPException(status_code=502, detail=f"Timestamp alignment failed: {e}")

    word_segments = scene_timestamps.get("word_segments", [])
    timed_words = [w for w in word_segments if "start" in w and "end" in w]

    scene["voice"] = update.voice
    scene["voiceover"] = speech_result
    scene["start"] = timed_words[0]["start"] if timed_words else None
    scene["end"] = timed_words[-1]["end"] if timed_words else None
    scene["word_segments"] = word_segments
    scene.pop("trim", None)
    scene.pop("word_segments_full", None)
    scene["error"] = None

    # Timing just changed completely — rebuild the rotating ~10s B-roll
    # beats (new keywords + new media_type + new motion_type + new Pexels
    # search per beat) so B-roll windows line up with the new voiceover
    # instead of stale ones.
    scene["beats"] = await _build_scene_beats(scene)
    scene["media"] = _aggregate_beats_media(scene["beats"])
    scene["duration_seconds"] = (
        round(scene["end"] - scene["start"], 3)
        if scene.get("start") is not None and scene.get("end") is not None
        else None
    )

    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes,
            "timeline_json": timeline_json,
            "timeline_version": new_version,
            "final_video_url": None,
            "render_status": "stale_needs_render",
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[update-scene-voice] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save voice edit")

    return {
        "video_id": video_id,
        "scene_id": scene_id,
        "voice": update.voice,
        "start": scene["start"],
        "end": scene["end"],
        "beats": [{"beat_id": b["beat_id"], "start": b["start"], "end": b["end"]} for b in scene["beats"]],
        "timeline_version": new_version,
        "timeline": timeline_json,
        "needs_render": True,
    }


@app.patch("/timeline/{video_id}/scene/{scene_id}/trim")
async def update_scene_trim(video_id: str, scene_id: str, update: SceneTrimUpdate):
    if update.start < 0 or update.end <= update.start:
        raise HTTPException(
            status_code=422,
            detail="`end` must be greater than `start`, and `start` must be >= 0",
        )

    try:
        row = (
            supabase.table("videos")
            .select("raw_scenes, timeline_version")
            .eq("id", video_id)
            .single()
            .execute()
        )
    except Exception as e:
        print(f"[update-scene-trim] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)
    if scene_index is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    scene = dict(raw_scenes[scene_index])

    full_word_segments = scene.get("word_segments_full")
    if full_word_segments is None:
        full_word_segments = scene.get("word_segments") or []
        scene["word_segments_full"] = full_word_segments

    if not full_word_segments:
        raise HTTPException(
            status_code=400,
            detail=f"Scene {scene_id} has no word-level timestamps to trim against",
        )

    clip_start = full_word_segments[0].get("start", 0.0)
    clip_end = full_word_segments[-1].get("end", 0.0)

    if update.start < clip_start - 1e-3 or update.end > clip_end + 1e-3:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Trim range [{update.start}, {update.end}] is outside this scene's "
                f"original audio bounds [{clip_start}, {clip_end}]"
            ),
        )

    trimmed_words = [
        w for w in full_word_segments
        if "start" in w and "end" in w and w["start"] >= update.start and w["end"] <= update.end
    ]

    scene["trim"] = {"start": update.start, "end": update.end}
    scene["word_segments"] = trimmed_words
    scene["start"] = trimmed_words[0]["start"] if trimmed_words else update.start
    scene["end"] = trimmed_words[-1]["end"] if trimmed_words else update.end
    scene["error"] = None

    # Trimming changes the scene's effective duration, so the rotating
    # ~10s B-roll beats need to be recomputed against the trimmed window
    # too (fewer/shorter beats if the trim shortened the scene a lot).
    scene["beats"] = await _build_scene_beats(scene)
    scene["media"] = _aggregate_beats_media(scene["beats"])
    scene["duration_seconds"] = round(scene["end"] - scene["start"], 3)

    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes,
            "timeline_json": timeline_json,
            "timeline_version": new_version,
            "final_video_url": None,
            "render_status": "stale_needs_render",
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[update-scene-trim] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save trim edit")

    return {
        "video_id": video_id,
        "scene_id": scene_id,
        "trim": scene["trim"],
        "timeline_version": new_version,
        "timeline": timeline_json,
        "needs_render": True,
    }


@app.patch("/timeline/{video_id}/scene/{scene_id}/infographic")
async def update_scene_infographic(video_id: str, scene_id: str, update: SceneInfographicUpdate):
    if update.animation_type is None and update.props is None and update.duration_frames is None:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of animation_type, props, duration_frames",
        )

    if update.props is not None and update.animation_type is None:
        raise HTTPException(
            status_code=422,
            detail="animation_type is required whenever props is provided, so they don't end up mismatched",
        )

    if update.animation_type is not None and update.animation_type not in REMOTION_INFOGRAPHIC_LIBRARY:
        raise HTTPException(
            status_code=422,
            detail=f"animation_type must be one of {sorted(REMOTION_INFOGRAPHIC_LIBRARY.keys())}",
        )

    if update.duration_frames is not None:
        if update.duration_frames <= 0 or update.duration_frames > 900:
            raise HTTPException(
                status_code=422,
                detail="duration_frames must be between 1 and 900 (30s @30fps)",
            )

    try:
        row = (
            supabase.table("videos")
            .select("raw_scenes, timeline_version")
            .eq("id", video_id)
            .single()
            .execute()
        )
    except Exception as e:
        print(f"[update-scene-infographic] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)
    if scene_index is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    scene = dict(raw_scenes[scene_index])
    animation = dict(scene.get("animation") or _default_animation(scene))

    if update.animation_type is not None:
        animation["animation_type"] = update.animation_type
        animation["category"] = _ANIMATION_TYPE_TO_CATEGORY[update.animation_type]

    if update.duration_frames is not None:
        animation["duration_frames"] = update.duration_frames

    animation = _validate_animation(animation, scene)

    infographic = _get_scene_infographic(scene, animation)
    if infographic is None:
        raise HTTPException(
            status_code=422,
            detail=f"'{animation['animation_type']}' is not an infographic composition",
        )

    if update.props is not None:
        infographic["props"] = {**infographic.get("props", {}), **update.props}

    scene["animation"] = animation
    scene["infographics"] = infographic
    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes,
            "timeline_json": timeline_json,
            "timeline_version": new_version,
            "final_video_url": None,
            "render_status": "stale_needs_render",
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[update-scene-infographic] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save infographic edit")

    return {
        "video_id": video_id,
        "scene_id": scene_id,
        "animation": animation,
        "infographic": infographic,
        "timeline_version": new_version,
        "timeline": timeline_json,
        "needs_render": True,
    }


@app.patch("/timeline/{video_id}/scene/{scene_id}/broll")
async def update_scene_broll(video_id: str, scene_id: str, update: SceneBrollSelectUpdate):
    if update.source not in ("video", "image"):
        raise HTTPException(status_code=422, detail="source must be 'video' or 'image'")

    # NEW: validate a manually-provided motion_type up front. Omitted
    # (None) is fine — the beat's existing (LLM-chosen) motion_type is kept
    # in that case, see _resolve_beat_motion_type.
    if update.motion_type is not None and update.motion_type not in _VALID_MOTION_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"motion_type must be one of {sorted(_VALID_MOTION_TYPES)}",
        )

    try:
        row = (
            supabase.table("videos")
            .select("raw_scenes, timeline_version")
            .eq("id", video_id)
            .single()
            .execute()
        )
    except Exception as e:
        print(f"[update-scene-broll] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")

    raw_scenes = row.data.get("raw_scenes") or []
    current_version = row.data.get("timeline_version", 1)

    scene_index = next((i for i, s in enumerate(raw_scenes) if s.get("scene_id") == scene_id), None)
    if scene_index is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    scene = dict(raw_scenes[scene_index])
    beats = scene.get("beats") or []

    if not beats:
        raise HTTPException(
            status_code=422,
            detail=f"Scene {scene_id} has no B-roll beats to select from yet",
        )

    if update.beat_id:
        beat_idx = next((i for i, b in enumerate(beats) if b.get("beat_id") == update.beat_id), None)
        if beat_idx is None:
            raise HTTPException(
                status_code=404,
                detail=f"Beat {update.beat_id} not found in scene {scene_id} (available: "
                       f"{[b.get('beat_id') for b in beats]})",
            )
    else:
        # No beat specified: default to the first beat, which is correct
        # for the common case of a single-beat (<=12s) scene.
        beat_idx = 0

    beat = dict(beats[beat_idx])

    candidate = _find_beat_broll_candidate(beat, update.asset_id, update.source)
    if candidate is None:
        raise HTTPException(
            status_code=404,
            detail=f"No {update.source} candidate with id {update.asset_id} in beat "
                   f"{beat.get('beat_id')}'s media pool",
        )

    file_url = _resolve_broll_file_url(candidate, update.source)
    if not file_url:
        raise HTTPException(
            status_code=422,
            detail=f"Candidate {update.asset_id} has no landscape/horizontal file available — only horizontal videos are allowed",
        )

    # Previously: scene.pop("background_color", None) ran unconditionally,
    # deleting the scene's fallback base layer as soon as an override was
    # set — even though nothing here confirms the renderer can actually
    # draw this particular file_url. If the render step fails/skips this
    # asset for any reason (unsupported source, fetch failure, orientation
    # mismatch, etc.), there is now nothing underneath it, which is a
    # plausible cause of a fully black frame instead of at least a
    # fallback color. Keep it stashed instead of discarding it, so a
    # renderer-side fallback (if one exists) still has something to use.
    if "background_color" in scene:
        scene["_previous_background_color"] = scene["background_color"]
        scene.pop("background_color", None)

    beat["broll_override"] = {
        "asset_id": candidate.get("id"),
        "source": update.source,
        "file_url": file_url,
        "width": candidate.get("width"),
        "height": candidate.get("height"),
        "video_files": candidate.get("video_files"),
        "src": candidate.get("src"),
        # NEW: only set when the caller explicitly provided one; otherwise
        # left as None so _resolve_beat_motion_type falls through to the
        # beat's own LLM-chosen motion_type instead of silently resetting
        # it to the default on every manual asset pick.
        "motion_type": update.motion_type,
    }
    beats[beat_idx] = beat
    scene["beats"] = beats

    raw_scenes[scene_index] = scene

    timeline_json = build_timeline_from_scenes(raw_scenes)
    new_version = current_version + 1

    try:
        supabase.table("videos").update({
            "raw_scenes": raw_scenes,
            "timeline_json": timeline_json,
            "timeline_version": new_version,
            "final_video_url": None,
            "render_status": "stale_needs_render",
        }).eq("id", video_id).execute()
    except Exception as e:
        print(f"[update-scene-broll] failed to save video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save broll selection")

    return {
        "video_id": video_id,
        "scene_id": scene_id,
        "beat_id": beat.get("beat_id"),
        "selected_asset": beat["broll_override"],
        "resolved_motion_type": _resolve_beat_motion_type(beat),
        "timeline_version": new_version,
        "timeline": timeline_json,
        "needs_render": True,
    }




# =============================================================================
# RENDER SERVICE
# =============================================================================

from fastapi import HTTPException, BackgroundTasks, Response, status


RENDER_TMP_ROOT = os.getenv("RENDER_TMP_ROOT", "/tmp/storybit-render")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
 
REMOTION_PROJECT_DIR = os.getenv("REMOTION_PROJECT_DIR", "")
 
# SPEED: default concurrency now scales with available CPUs instead of a
# hardcoded "2" — each scene render is CPU-bound (multiple ffmpeg encodes),
# so on an 8-core box this alone roughly quadruples scene-level parallelism.
# Override with RENDER_CONCURRENCY env var if you need to cap it (e.g. to
# limit memory use on a shared box).
RENDER_CONCURRENCY = int(os.getenv("RENDER_CONCURRENCY", str(max(os.cpu_count() or 2, 2))))
 
# SPEED: x264 defaults to "-preset medium" with no explicit thread count when
# neither is passed, which is needlessly slow for what is essentially
# throwaway intermediate encoding (broll fit, caption burn, etc.) followed by
# a final encode. "veryfast" trades a small amount of compression efficiency
# (slightly larger files at the same CRF) for a large speed win — typically
# 3-6x faster than "medium" on the same hardware. CRF 23 keeps visual quality
# effectively unchanged from the previous unset default (23 is libx264's own
# default CRF). "-threads 0" lets x264 use all available cores per encode.
FFMPEG_X264_PRESET = os.getenv("FFMPEG_X264_PRESET", "veryfast")
FFMPEG_X264_CRF = os.getenv("FFMPEG_X264_CRF", "23")
FFMPEG_X264_FLAGS = [
    "-c:v", "libx264",
    "-preset", FFMPEG_X264_PRESET,
    "-crf", FFMPEG_X264_CRF,
    "-pix_fmt", "yuv420p",
    "-threads", "0",
]
 
SILENT_AUDIO_SAMPLE_RATE = int(os.getenv("SILENT_AUDIO_SAMPLE_RATE", "48000"))
SILENT_AUDIO_CHANNEL_LAYOUT = os.getenv("SILENT_AUDIO_CHANNEL_LAYOUT", "stereo")
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RENDER_OUTPUT_DIR = os.environ.get("RENDER_OUTPUT_DIR", os.path.join(BASE_DIR, "rendered_videos"))
os.makedirs(RENDER_OUTPUT_DIR, exist_ok=True)
 
LANDSCAPE_RESOLUTION = {"width": 1920, "height": 1080}
PORTRAIT_RESOLUTION = {"width": 1080, "height": 1920}
 
SUPABASE_RENDERED_VIDEOS_BUCKET = os.getenv("SUPABASE_RENDERED_VIDEOS_BUCKET", "rendered-videos")


# =============================================================================
# IMAGE MOTION EFFECTS  (NEW)
#
# Pure translation layer from an LLM-chosen motion_type (one of
# _VALID_MOTION_TYPES, always decided upstream in _generate_beat_keywords /
# BEAT_KEYWORDS_PROMPT — see that section) into the matching FFmpeg
# zoompan filter expression. Nothing in this section makes a creative
# choice; it only implements whichever choice it's handed. If motion_type
# is missing or invalid by the time it reaches here, it falls back to
# _DEFAULT_MOTION_TYPE rather than failing the whole render.
# =============================================================================

def _ease_expr(duration_frames: int) -> str:
    """
    Cosine ease-in/ease-out progress curve, 0 -> 1 across the clip's frames,
    referencing zoompan's built-in `on` (output frame number) variable.
    Applying this instead of a raw linear on/d ramp is what keeps the
    pan/zoom/tilt from looking like it's moving at constant speed — it
    starts slow, speeds up through the middle, and eases out at the end,
    which reads as far more polished ("Ken Burns with easing").
    """
    d = max(duration_frames - 1, 1)
    return f"(1-cos(PI*min(on/{d},1)))/2"


def _build_image_motion_filter(
    motion_type: str, duration_frames: int, fps: int, width: int, height: int,
) -> str:
    """
    Builds the full `-vf` filter string for animating a still image with
    the given motion_type. Always pre-scales the source far larger than
    the output (scale=8000:-1) before zoompan — without this, zoompan
    re-samples a static-resolution source every frame and produces visible
    judder instead of a smooth pan/zoom.

    motion_type must be one of _VALID_MOTION_TYPES; anything else falls
    through to a subtle "breathing" zoom as a safe default rather than
    raising, since this only ever runs after LLM selection has already
    happened — by the time we're here, a bad value means "render it
    anyway with something reasonable," not "fail the beat."
    """
    ease = _ease_expr(duration_frames)
    zoom_base = 1.2
    zoom_amount = 0.3

    if motion_type == "zoom_in":
        z_expr = f"1+{zoom_amount}*({ease})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif motion_type == "zoom_out":
        z_expr = f"{1 + zoom_amount}-{zoom_amount}*({ease})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif motion_type == "pan_left":
        # left -> right drift at a fixed zoom level
        z_expr = f"{zoom_base}"
        x_expr = f"({ease})*(iw-iw/zoom)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif motion_type == "pan_right":
        # right -> left drift at a fixed zoom level
        z_expr = f"{zoom_base}"
        x_expr = f"(1-({ease}))*(iw-iw/zoom)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif motion_type == "tilt_up":
        # bottom -> top drift at a fixed zoom level
        z_expr = f"{zoom_base}"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = f"(1-({ease}))*(ih-ih/zoom)"
    elif motion_type == "tilt_down":
        # top -> bottom drift at a fixed zoom level
        z_expr = f"{zoom_base}"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = f"({ease})*(ih-ih/zoom)"
    else:
        print(f"[render] unrecognized motion_type {motion_type!r} — using a subtle default zoom")
        z_expr = f"1+0.05*({ease})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

    return (
        f"scale=8000:-1,"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
        f"d={duration_frames}:s={width}x{height}:fps={fps}"
    )
 
 
def _upload_rendered_video_to_supabase(local_path: str, video_id: str, filename: Optional[str] = None) -> str:
    # FIX C (stale/cached video served after re-render): every render used
    # to overwrite the exact same object at the exact same path
    # ("{video_id}/final.mp4"), which means every render also produced the
    # exact same public URL. Overwriting the object does NOT invalidate any
    # cache — browser cache, a CDN in front of Supabase Storage, or any
    # proxy in between — that already has that URL cached, so a fresh,
    # correctly-rendered file can silently keep being shadowed by old
    # cached bytes at the same URL forever.
    #
    # Giving each render its own filename means each render gets its own
    # URL, so there is never a stale cache entry to collide with.
    if filename is None:
        filename = f"final_{uuid.uuid4().hex}.mp4"
    dest_path = f"{video_id}/{filename}"
 
    with open(local_path, "rb") as f:
        file_bytes = f.read()
 
    supabase.storage.from_(SUPABASE_RENDERED_VIDEOS_BUCKET).upload(
        path=dest_path,
        file=file_bytes,
        file_options={
            "content-type": "video/mp4",
            "upsert": "true",  # overwrite if this video_id was rendered before
        },
    )
 
    public_url = supabase.storage.from_(SUPABASE_RENDERED_VIDEOS_BUCKET).get_public_url(dest_path)
    return public_url
 
 
class RenderVideoRequest(BaseModel):
    force: bool = False
    # FIX: explicit orientation, defaults to landscape (YouTube long-form).
    orientation: Literal["landscape", "portrait"] = "landscape"
 
 
RUN_SUBPROCESS_TIMEOUT_SECONDS = int(os.getenv("RUN_SUBPROCESS_TIMEOUT_SECONDS", "300"))


async def _run(cmd: list[str], cwd: Optional[str] = None, timeout: Optional[float] = None) -> None:
    # FIX D (silent hang risk): stdin=DEVNULL means npx can never block
    # forever on an interactive "install this package? (y)" prompt it will
    # never get an answer to in a server process. The timeout is a backstop
    # for any other subprocess that wedges (network stall, browser launch
    # failure, etc.) — turns an indefinite hang into a clear, loggable
    # error instead.
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout or RUN_SUBPROCESS_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(
            f"Command timed out after {timeout or RUN_SUBPROCESS_TIMEOUT_SECONDS}s "
            f"and was killed: {' '.join(cmd)}"
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"--- stderr ---\n{stderr.decode(errors='replace')[-4000:]}"
        )
 
 
async def _probe_duration_seconds(path: str) -> float:
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {stderr.decode(errors='replace')}")
    try:
        return float(stdout.decode().strip())
    except ValueError:
        raise RuntimeError(f"ffprobe returned unparsable duration for {path}: {stdout!r}")
 
 
async def _probe_video_dimensions(path: str) -> Optional[tuple[int, int]]:
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
        "-of", "json",
        path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        print(f"[render] ffprobe dimension check failed on {path}: {stderr.decode(errors='replace')[-500:]}")
        return None
 
    try:
        data = json.loads(stdout.decode())
        streams = data.get("streams") or []
        if not streams:
            return None
        stream = streams[0]
 
        width = int(stream.get("width", 0) or 0)
        height = int(stream.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            return None
 
        rotation = 0
        tags_rotate = (stream.get("tags") or {}).get("rotate")
        if tags_rotate is not None:
            try:
                rotation = int(float(tags_rotate))
            except (TypeError, ValueError):
                rotation = 0
 
        for sd in (stream.get("side_data_list") or []):
            if "rotation" in sd:
                try:
                    rotation = int(float(sd["rotation"]))
                except (TypeError, ValueError):
                    pass
 
        rotation = abs(rotation) % 360
        if rotation in (90, 270):
            width, height = height, width
 
        return width, height
    except (ValueError, KeyError, IndexError, json.JSONDecodeError):
        return None
 
 
async def _download(url: str, dest: str, client: httpx.AsyncClient) -> str:
    resp = await client.get(url, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    return dest
 
 
async def _make_color_fallback(
    duration_frames: int, fps: int, width: int, height: int, tmp_dir: str,
    color: str = "0x111827",
) -> str:
    out_path = os.path.join(tmp_dir, "broll_fallback.mp4")
    seconds = duration_frames / fps
    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "lavfi",
        "-i", f"color=c={color}:s={width}x{height}:d={seconds:.3f}:r={fps}",
        *FFMPEG_X264_FLAGS,
        out_path,
    ]
    await _run(cmd)
    return out_path
 
 
async def _make_blurpad_fallback(
    local_src: str,
    source_kind: str,
    duration_frames: int,
    fps: int,
    width: int,
    height: int,
    tmp_dir: str,
) -> Optional[str]:
    target_seconds = duration_frames / fps
    out_path = os.path.join(tmp_dir, "broll_blurpad.mp4")
 
    filter_complex = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma=30,eq=brightness=-0.05[bg];"
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps}[outv]"
    )
 
    if source_kind == "image":
        cmd = [
            FFMPEG_BIN, "-y",
            "-loop", "1", "-i", local_src,
            "-t", f"{target_seconds:.3f}",
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-an",
            *FFMPEG_X264_FLAGS,
            out_path,
        ]
        try:
            await _run(cmd)
            return out_path
        except Exception as e:
            print(f"[render] blur-pad image composite failed: {e}")
            return None
 
    try:
        source_seconds = await _probe_duration_seconds(local_src)
    except Exception:
        source_seconds = target_seconds
 
    if source_seconds >= target_seconds:
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", local_src,
            "-t", f"{target_seconds:.3f}",
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-an",
            *FFMPEG_X264_FLAGS,
            out_path,
        ]
    else:
        loops_needed = int(target_seconds // max(source_seconds, 0.1)) + 1
        cmd = [
            FFMPEG_BIN, "-y",
            "-stream_loop", str(loops_needed),
            "-i", local_src,
            "-t", f"{target_seconds:.3f}",
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-an",
            *FFMPEG_X264_FLAGS,
            out_path,
        ]
 
    try:
        await _run(cmd)
        return out_path
    except Exception as e:
        print(f"[render] blur-pad video composite failed: {e}")
        return None
 
 
async def _prepare_broll_clip(
    broll_track: dict,
    scene_duration_frames: int,
    fps: int,
    width: int,
    height: int,
    tmp_dir: str,
    client: httpx.AsyncClient,
    background_color_override: Optional[str] = None,
    motion_type: Optional[str] = None,
) -> str:
    if background_color_override:
        return await _make_color_fallback(
            scene_duration_frames, fps, width, height, tmp_dir,
            color=_normalize_ffmpeg_color(background_color_override),
        )
 
    selected = broll_track.get("selected_asset")
 
    if not selected:
        print("[render] no b-roll asset selected for this scene at all — using color fallback")
        return await _make_color_fallback(scene_duration_frames, fps, width, height, tmp_dir)
 
    source = selected.get("source", "video")
 
    # First choice: strict landscape-only URL.
    landscape_url = _resolve_broll_file_url(selected, source)
 
    if landscape_url:
        target_seconds = scene_duration_frames / fps
        ext = ".mp4" if source == "video" else ".jpg"
        local_src = os.path.join(tmp_dir, f"broll_src{ext}")
 
        try:
            await _download(landscape_url, local_src, client)
        except Exception as e:
            print(f"[render] broll download failed ({landscape_url}): {e} — trying blur-pad fallback")
            landscape_url = None  # fall through to the any-orientation path below
 
        if landscape_url:
            if source == "video":
                dims = await _probe_video_dimensions(local_src)
                if dims is not None and not _is_landscape_dimensions(dims[0], dims[1]):
                    print(
                        f"[render] downloaded broll video DISPLAYS as portrait/square "
                        f"(effective {dims[0]}x{dims[1]} after rotation) despite metadata "
                        f"claiming landscape — using blur-pad composite instead of stretching it"
                    )
                    blurpad = await _make_blurpad_fallback(
                        local_src, "video", scene_duration_frames, fps, width, height, tmp_dir
                    )
                    if blurpad:
                        return blurpad
                    return await _make_color_fallback(scene_duration_frames, fps, width, height, tmp_dir)
 
            out_path = os.path.join(tmp_dir, "broll_fit.mp4")

            if source == "image":
                # NEW: still images now get an LLM-chosen Ken Burns-style
                # pan/zoom/tilt via zoompan instead of a static scale+crop.
                # motion_type comes from the timeline's broll track
                # (already resolved through _resolve_beat_motion_type at
                # timeline-build time) — this call never picks the motion
                # itself, only maps it to a filter expression.
                resolved_motion = motion_type if motion_type in _VALID_MOTION_TYPES else _DEFAULT_MOTION_TYPE
                motion_filter = _build_image_motion_filter(
                    resolved_motion, duration_frames=scene_duration_frames,
                    fps=fps, width=width, height=height,
                )
                cmd = [
                    FFMPEG_BIN, "-y",
                    "-loop", "1", "-i", local_src,
                    "-t", f"{target_seconds:.3f}",
                    "-vf", motion_filter,
                    "-an",
                    *FFMPEG_X264_FLAGS,
                    out_path,
                ]
                await _run(cmd)
                return out_path

            scale_crop = (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},setsar=1,fps={fps}"
            )

            source_seconds = await _probe_duration_seconds(local_src)
            if source_seconds >= target_seconds:
                cmd = [
                    FFMPEG_BIN, "-y",
                    "-i", local_src,
                    "-t", f"{target_seconds:.3f}",
                    "-vf", scale_crop,
                    "-an",
                    *FFMPEG_X264_FLAGS,
                    out_path,
                ]
            else:
                loops_needed = int(target_seconds // source_seconds) + 1
                cmd = [
                    FFMPEG_BIN, "-y",
                    "-stream_loop", str(loops_needed),
                    "-i", local_src,
                    "-t", f"{target_seconds:.3f}",
                    "-vf", scale_crop,
                    "-an",
                    *FFMPEG_X264_FLAGS,
                    out_path,
                ]
            await _run(cmd)
            return out_path
 
    # FIX A: no landscape file for this candidate — try blur-pad with the
    # ORIGINAL asset (any orientation) rather than showing a blank card.
    # (Motion is intentionally not applied on this fallback path — it's
    # already a degraded-quality fallback, and keeping it simple/static
    # avoids compounding two different failure-recovery filters at once.)
    any_url = _resolve_broll_file_url_any_orientation(selected, source)
    if not any_url:
        print(
            f"[render] no downloadable file at all for asset "
            f"{selected.get('asset_id') or selected.get('id')} — using color fallback"
        )
        return await _make_color_fallback(scene_duration_frames, fps, width, height, tmp_dir)
 
    ext = ".mp4" if source == "video" else ".jpg"
    local_src = os.path.join(tmp_dir, f"broll_anyorient{ext}")
    try:
        await _download(any_url, local_src, client)
    except Exception as e:
        print(f"[render] any-orientation broll download failed ({any_url}): {e} — using color fallback")
        return await _make_color_fallback(scene_duration_frames, fps, width, height, tmp_dir)
 
    blurpad = await _make_blurpad_fallback(
        local_src, source, scene_duration_frames, fps, width, height, tmp_dir
    )
    if blurpad:
        return blurpad
 
    print("[render] blur-pad composite failed — using color fallback as last resort")
    return await _make_color_fallback(scene_duration_frames, fps, width, height, tmp_dir)
 
 
async def _make_silent_audio(duration_frames: int, fps: int, tmp_dir: str) -> str:
    out_path = os.path.join(tmp_dir, "silence.aac")
    seconds = max(duration_frames / fps, 1 / fps)
    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout={SILENT_AUDIO_CHANNEL_LAYOUT}:sample_rate={SILENT_AUDIO_SAMPLE_RATE}",
        "-t", f"{seconds:.3f}",
        "-c:a", "aac",
        out_path,
    ]
    await _run(cmd)
    return out_path
 
 
async def render_infographic_via_remotion(
    composition_id: str,
    props: dict,
    duration_frames: int,
    fps: int,
    width: int,
    height: int,
    tmp_dir: str,
) -> Optional[str]:
    print(f"[DEBUG] render_infographic_via_remotion called: composition_id={composition_id!r}, REMOTION_PROJECT_DIR={REMOTION_PROJECT_DIR!r}")
    if not REMOTION_PROJECT_DIR:
        print(
            f"[render] REMOTION_PROJECT_DIR not configured — skipping "
            f"infographic '{composition_id}'. Set up a Remotion project and "
            f"point REMOTION_PROJECT_DIR at it to enable overlays."
        )
        return None
 
    out_path = os.path.join(tmp_dir, f"infographic_{uuid.uuid4().hex}.mov")
    props_path = os.path.join(tmp_dir, "props.json")
    with open(props_path, "w") as f:
        json.dump(props, f)
 
    def _build_cmd(frame_range: Optional[str]) -> list[str]:
        cmd = [
            "npx", "remotion", "render",
            composition_id,
            out_path,
            f"--props={props_path}",
        ]
        if frame_range:
            cmd.append(f"--frames={frame_range}")
        cmd += [
            f"--fps={fps}",
            f"--width={width}",
            f"--height={height}",
            "--codec=prores",
            # FIX E (opaque black background instead of transparent):
            # --codec=prores alone defaults to Remotion's "hq" ProRes
            # profile, which is 4:2:2 and has NO alpha channel at all,
            # regardless of the pixel-format requested below. Only the
            # "4444" and "4444-xq" profiles carry an alpha channel.
            "--prores-profile=4444",
            "--pixel-format=yuva444p10le",
        ]
        return cmd
 
    try:
        await _run(_build_cmd(f"0-{duration_frames - 1}"), cwd=REMOTION_PROJECT_DIR)
    except RuntimeError as e:
        msg = str(e)
        m = re.search(
            r"durationInFrames.*?evaluated to be (\d+).*?not inbetween 0-(\d+)",
            msg, re.IGNORECASE | re.DOTALL,
        )
        if not m:
            print(f"[render] Remotion render failed for '{composition_id}': {e}")
            return None
 
        actual_duration = int(m.group(1))
        max_frame = int(m.group(2))
        print(
            f"[render] '{composition_id}' has a fixed durationInFrames of "
            f"{actual_duration} (requested {duration_frames}) — retrying "
            f"within its native bounds (0-{max_frame})"
        )
        try:
            await _run(_build_cmd(f"0-{max_frame}"), cwd=REMOTION_PROJECT_DIR)
        except Exception as e2:
            print(f"[render] retry also failed for '{composition_id}': {e2}")
            return None
 
    return out_path
 
 
def _frames_to_ass_time(frame: int, fps: int) -> str:
    total_seconds = frame / fps
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = total_seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"
 
 
def _build_ass_from_words(
    words: list[dict], scene_start_frame: int, fps: int, width: int, height: int,
    style: Optional[dict] = None,
) -> str:
    style = style or {}
    font_size = style.get("font_size") or 72
    primary_color = _hex_to_ass_color(style.get("text_color") or "#FFFFFF", alpha_hex="00")
    outline_color = _hex_to_ass_color(style.get("outline_color") or "#000000", alpha_hex="00")
    animation_type = style.get("animation_type") or "kinetic_caption"

    # FIX (captions not actually low): this used to hardcode Alignment=2
    # and MarginV=200 directly in the style string below, completely
    # ignoring `vertical_position`/`margin_bottom_percent` — the fields
    # DEFAULT_CAPTION_STYLE sets upstream specifically to push captions
    # very low. Those fields were being computed and stored in the
    # timeline JSON correctly, then silently dropped here at burn-in time.
    #
    # ASS numpad alignment: 2 = bottom-center, 8 = top-center. MarginV is
    # the margin from whichever edge Alignment anchors to (bottom for 1-3,
    # top for 7-9) — computed as a percentage of frame height so "very low"
    # holds regardless of canvas resolution, matching how the JSON side
    # expresses it.
    vertical_position = (style.get("vertical_position") or "bottom").lower()
    try:
        margin_percent = float(style.get("margin_bottom_percent", 3))
    except (TypeError, ValueError):
        margin_percent = 3.0
    alignment = 8 if vertical_position == "top" else 2
    margin_v = max(round(height * margin_percent / 100), 0)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
 
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Caption,Arial Black,{font_size},{primary_color},{outline_color},&H80000000,1,4,0,{alignment},60,60,{margin_v}
 
[Events]
Format: Layer, Start, End, Style, Text
"""
    if animation_type == "static_line" and words:
        start_frame = words[0]["startFrame"] - scene_start_frame
        end_frame = words[-1]["endFrame"] - scene_start_frame
        text = " ".join(w["word"] for w in words)
        if start_frame < 0:
            start_frame = 0
        if end_frame > start_frame:
            start_ts = _frames_to_ass_time(start_frame, fps)
            end_ts = _frames_to_ass_time(end_frame, fps)
            return header + f"Dialogue: 0,{start_ts},{end_ts},Caption,{text}\n"
        return header
 
    lines = []
    chunk: list[dict] = []
    CHUNK_SIZE = 4
    for w in words:
        chunk.append(w)
        if len(chunk) >= CHUNK_SIZE:
            lines.append(chunk)
            chunk = []
    if chunk:
        lines.append(chunk)
 
    events = []
    for line_words in lines:
        start_frame = line_words[0]["startFrame"] - scene_start_frame
        end_frame = line_words[-1]["endFrame"] - scene_start_frame
        if start_frame < 0 or end_frame <= start_frame:
            continue
        text = " ".join(w["word"] for w in line_words)
        start_ts = _frames_to_ass_time(start_frame, fps)
        end_ts = _frames_to_ass_time(end_frame, fps)
        events.append(f"Dialogue: 0,{start_ts},{end_ts},Caption,{text}")
 
    return header + "\n".join(events) + "\n"
 
 
async def _burn_captions(
    input_path: str, words: list[dict], scene_start_frame: int, fps: int,
    width: int, height: int, tmp_dir: str,
    style: Optional[dict] = None,
) -> str:
    if not words:
        return input_path
 
    ass_content = _build_ass_from_words(words, scene_start_frame, fps, width, height, style=style)
    ass_path = os.path.join(tmp_dir, "captions.ass")
    with open(ass_path, "w") as f:
        f.write(ass_content)
 
    out_path = os.path.join(tmp_dir, "with_captions.mp4")
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
        "-vf", "ass=captions.ass",
        *FFMPEG_X264_FLAGS,
        "-c:a", "copy",
        out_path,
    ]
    await _run(cmd, cwd=tmp_dir)
    return out_path
 
 
def _escape_drawtext(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\u2019")
        .replace("%", "\\%")
        .replace("\n", "\\n")
    )


_PLACEMENT_TO_XY = {
    "top_left": ("40", "40"),
    "top_center": ("(w-text_w)/2", "40"),
    "top_right": ("w-text_w-40", "40"),
    "center_left": ("40", "(h-text_h)/2"),
    "center": ("(w-text_w)/2", "(h-text_h)/2"),
    "center_right": ("w-text_w-40", "(h-text_h)/2"),
    "bottom_left": ("40", "h-text_h-160"),
    "bottom_center": ("(w-text_w)/2", "h-text_h-160"),
    "bottom_right": ("w-text_w-40", "h-text_h-160"),
    "full_frame": ("(w-text_w)/2", "(h-text_h)/2"),
}


def _animation_overlay_text(animation: dict, scene: dict, infographic: Optional[dict]) -> Optional[str]:
    if infographic:
        props = infographic.get("props") or {}
        items = props.get("items")
        if items:
            return "\n".join(f"- {i}" for i in items[:4])
        for key in ("title", "quote", "label", "value"):
            if props.get(key):
                text = str(props[key])
                extra = props.get("subtitle") or props.get("caption") or props.get("attribution")
                if extra:
                    text = f"{text}\n{extra}"
                return text

    on_screen_text = (scene.get("on_screen_text") or "").strip()
    return on_screen_text or None


async def _burn_animation_overlay(
    input_path: str,
    animation: dict,
    scene: dict,
    infographic: Optional[dict],
    width: int,
    height: int,
    fps: int,
    tmp_dir: str,
) -> str:
    text = _animation_overlay_text(animation, scene, infographic)
    print(f"[DEBUG] _burn_animation_overlay called for scene {scene.get('scene_id')}: animation_type={(animation or {}).get('animation_type')!r}, resolved_text={text!r}")
    if not text:
        return input_path

    placement = (animation or {}).get("placement") or "bottom_center"
    x_expr, y_expr = _PLACEMENT_TO_XY.get(placement, _PLACEMENT_TO_XY["bottom_center"])
    font_size = max(round(height / 22), 28)
    safe_text = _escape_drawtext(text)

    out_path = os.path.join(tmp_dir, "with_animation.mp4")
    drawtext = (
        f"drawtext=text='{safe_text}':fontcolor=white:fontsize={font_size}:"
        f"box=1:boxcolor=black@0.55:boxborderw=20:"
        f"x={x_expr}:y={y_expr}:line_spacing=8"
    )
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
        "-vf", drawtext,
        *FFMPEG_X264_FLAGS,
        "-c:a", "copy",
        out_path,
    ]
    try:
        await _run(cmd)
        return out_path
    except Exception as e:
        print(
            f"[render] scene {scene.get('scene_id')}: ffmpeg text-overlay "
            f"fallback failed, leaving scene without an overlay: {e}"
        )
        return input_path


async def _lock_clip_to_frame_count(
    input_path: str, duration_frames: int, fps: int, width: int, height: int, tmp_dir: str,
) -> str:
    """
    Beat clips are each rounded to their own duration independently, so a
    concatenated multi-beat scene clip can land a frame or two short of or
    over the scene's own `duration_frames`. Re-encodes to land on EXACTLY
    `duration_frames`: `tpad=stop_mode=clone:stop=100` pads with cloned
    trailing frames if the concat came up short (100 is just a generous
    buffer — it gets hard-cut right after), then `-frames:v` trims to the
    exact count either way. Keeps the caption/audio mux downstream (which
    trusts `duration_frames` as exact) from drifting out of sync.
    """
    out_path = os.path.join(tmp_dir, "locked.mp4")
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
        "-vf", f"fps={fps},tpad=stop_mode=clone:stop=100",
        "-frames:v", str(duration_frames),
        *FFMPEG_X264_FLAGS,
        "-an",
        out_path,
    ]
    await _run(cmd)
    return out_path


async def _render_scene(
    scene: dict, fps: int, width: int, height: int, work_root: str,
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore,
    timeline_tracks_by_scene: Optional[dict] = None,
    caption_tracks_by_scene: Optional[dict] = None,
    infographic_tracks_by_scene: Optional[dict] = None,
    animation_tracks_by_scene: Optional[dict] = None,
) -> str:
    async with semaphore:
        scene_id = scene.get("scene_id", uuid.uuid4().hex)
        tmp_dir = os.path.join(work_root, f"scene_{scene_id}")
        os.makedirs(tmp_dir, exist_ok=True)
 
        trim = scene.get("trim") or {}
        start_sec = scene.get("start")
        if start_sec is None:
            start_sec = trim.get("start", 0.0)
 
        end_sec = scene.get("end")
        if end_sec is None:
            end_sec = trim.get("end", 0.0)
 
        if (start_sec in (None, 0.0)) or (end_sec in (None, 0.0)):
            word_segments = scene.get("word_segments") or []
            timed = [w for w in word_segments if "start" in w and "end" in w]
            if timed:
                if start_sec in (None,):
                    start_sec = timed[0]["start"]
                if end_sec in (None, 0.0):
                    end_sec = timed[-1]["end"]
 
        duration_frames = max(round((end_sec - start_sec) * fps), fps)
        target_seconds = duration_frames / fps
 
        selected = None
        source = None

        # FIX (B-roll never rotates): timeline_tracks_by_scene[scene_id] is
        # now a list of this scene's beat-level broll tracks (sorted by
        # startFrame), not a single track. Render each beat as its own clip
        # at its own duration, then concat them into one scene-length clip —
        # this is what actually makes the footage change mid-scene, instead
        # of stretching a single asset across the whole scene.
        scene_beat_tracks = (timeline_tracks_by_scene or {}).get(scene_id) or []

        if scene_beat_tracks:
            beat_clip_paths = []
            for i, beat_track in enumerate(scene_beat_tracks):
                b_start_sec = beat_track.get("beat_start_sec")
                b_end_sec = beat_track.get("beat_end_sec")
                if b_start_sec is None or b_end_sec is None:
                    # Legacy/no-timing beat (e.g. a silent scene's single
                    # implicit beat) — it covers the whole scene.
                    b_start_sec, b_end_sec = start_sec, end_sec

                beat_duration_frames = max(round((b_end_sec - b_start_sec) * fps), 1)

                beat_selected = beat_track.get("selected_asset")
                beat_broll_track = {"selected_asset": None}
                if beat_selected:
                    beat_broll_track = {"selected_asset": dict(beat_selected)}

                beat_bg_override = beat_track.get("background_color") or scene.get("background_color")

                # NEW: resolved image motion effect for this beat, read
                # straight off the timeline track (build_timeline_from_scenes
                # already resolved override-vs-LLM-choice via
                # _resolve_beat_motion_type). Ignored by _prepare_broll_clip
                # when the beat's selected asset turns out to be a video.
                beat_motion_type = beat_track.get("motion_type")

                beat_tmp_dir = os.path.join(tmp_dir, f"beat_{i}")
                os.makedirs(beat_tmp_dir, exist_ok=True)

                beat_clip = await _prepare_broll_clip(
                    beat_broll_track, beat_duration_frames, fps, width, height, beat_tmp_dir, client,
                    background_color_override=beat_bg_override,
                    motion_type=beat_motion_type,
                )
                beat_clip_paths.append(beat_clip)

            if len(beat_clip_paths) == 1:
                base_clip = beat_clip_paths[0]
            else:
                base_clip = await _concat_scenes(beat_clip_paths, tmp_dir, fps=fps)

            # Beat durations are rounded independently per-beat, so their sum
            # can land a frame or two off `duration_frames` (the scene's own
            # rounding of end_sec - start_sec). Force the concatenated clip
            # to the exact scene frame count here so the caption/audio mux
            # below — which trusts `duration_frames` — stays frame-accurate.
            base_clip = await _lock_clip_to_frame_count(base_clip, duration_frames, fps, width, height, tmp_dir)
        else:
            # Legacy fallback: no beat tracks at all for this scene (older
            # timeline_json predating the beats refactor) — behave exactly
            # as before, one asset for the whole scene, default motion.
            timeline_broll = None
            if not selected:
                media = scene.get("media") or {}
                video_candidates = (media.get("videos") or {}).get("results") or []
                image_candidates = (media.get("images") or {}).get("results") or []
                if video_candidates:
                    selected, source = dict(video_candidates[0]), "video"
                elif image_candidates:
                    selected, source = dict(image_candidates[0]), "image"

            broll_track = {"selected_asset": None}
            if selected:
                broll_track = {
                    "selected_asset": {**selected, "source": source}
                }

            background_color_override = scene.get("background_color")

            base_clip = await _prepare_broll_clip(
                broll_track, duration_frames, fps, width, height, tmp_dir, client,
                background_color_override=background_color_override,
                motion_type=scene.get("motion_type"),
            )
 
        timeline_infographic = (infographic_tracks_by_scene or {}).get(scene_id)
        if timeline_infographic and timeline_infographic.get("composition_id"):
            infographic = {
                "composition_id": timeline_infographic.get("composition_id"),
                "props": timeline_infographic.get("props", {}),
                "duration_frames": timeline_infographic.get("duration_frames") or duration_frames,
            }
        else:
            infographic = scene.get("infographics")

        timeline_animation = (animation_tracks_by_scene or {}).get(scene_id)
        animation = timeline_animation or scene.get("animation") or {}

        current = base_clip
        animation_applied = False

        if infographic and infographic.get("composition_id"):
            overlay_clip = await render_infographic_via_remotion(
                composition_id=infographic["composition_id"],
                props=infographic.get("props", {}),
                duration_frames=infographic.get("duration_frames") or duration_frames,
                fps=fps, width=width, height=height, tmp_dir=tmp_dir,
            )
            if overlay_clip:
                composited = os.path.join(tmp_dir, "composited.mp4")
                cmd = [
                    FFMPEG_BIN, "-y",
                    "-i", current,
                    "-i", overlay_clip,
                    "-filter_complex", "[1:v]format=yuva420p[fg];[0:v][fg]overlay=0:0:eof_action=pass:format=auto",
                    "-r", str(fps),
                    *FFMPEG_X264_FLAGS,
                    composited,
                ]
                await _run(cmd)
                current = composited
                animation_applied = True
            else:
                print(
                    f"[render] scene {scene_id}: Remotion composition "
                    f"'{infographic['composition_id']}' unavailable — using "
                    f"FFmpeg text-overlay fallback so the infographic still renders"
                )
                current = await _burn_animation_overlay(
                    current, animation, scene, infographic, width, height, fps, tmp_dir,
                )
                animation_applied = True

        if not animation_applied and animation.get("animation_type"):
            current = await _burn_animation_overlay(
                current, animation, scene, None, width, height, fps, tmp_dir,
            )
 
        timeline_caption = (caption_tracks_by_scene or {}).get(scene_id)
        caption_style = (timeline_caption or {}).get("style") or scene.get("caption_style")
 
        words = scene.get("word_segments") or []
        words = [
            w for w in words
            if "start" in w and "end" in w
            and w["start"] >= start_sec and w["end"] <= end_sec
        ]
        frame_words = [
            {
                "word": w.get("word", ""),
                "startFrame": max(round((w["start"] - start_sec) * fps), 0),
                "endFrame": max(round((w["end"] - start_sec) * fps), 0),
            }
            for w in words
        ]
        current = await _burn_captions(
            current, frame_words, 0, fps, width, height, tmp_dir, style=caption_style
        )
 
        voiceover = scene.get("voiceover")
        final_scene_path = os.path.join(work_root, f"scene_{scene_id}_final.mp4")
 
        if voiceover and voiceover.get("url"):
            audio_path_raw = os.path.join(tmp_dir, "audio_raw.mp3")
            await _download(voiceover["url"], audio_path_raw, client)
 
            audio_path = os.path.join(tmp_dir, "audio_trimmed.m4a")
            trim_cmd = [
                FFMPEG_BIN, "-y",
                "-i", audio_path_raw,
                "-ss", f"{start_sec:.3f}",
                "-to", f"{end_sec:.3f}",
                "-c:a", "aac",
                audio_path,
            ]
            await _run(trim_cmd)
        else:
            print(f"[render] scene {scene_id} has no voiceover ({scene.get('error')}) — muxing silent audio instead")
            audio_path = await _make_silent_audio(duration_frames, fps, tmp_dir)
 
        locked_audio_path = os.path.join(tmp_dir, "audio_locked.m4a")
        lock_cmd = [
            FFMPEG_BIN, "-y",
            "-i", audio_path,
            "-af", f"apad=whole_dur={target_seconds:.3f}",
            "-t", f"{target_seconds:.3f}",
            "-c:a", "aac",
            locked_audio_path,
        ]
        await _run(lock_cmd)
 
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", current,
            "-i", locked_audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-frames:v", str(duration_frames),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            final_scene_path,
        ]
        try:
            await _run(cmd)
        except Exception as e:
            print(f"[render] stream-copy mux failed for scene {scene_id}, falling back to re-encode: {e}")
            cmd = [
                FFMPEG_BIN, "-y",
                "-i", current,
                "-i", locked_audio_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-r", str(fps),
                "-vsync", "cfr",
                "-frames:v", str(duration_frames),
                *FFMPEG_X264_FLAGS,
                "-c:a", "aac",
                "-shortest",
                final_scene_path,
            ]
            await _run(cmd)
 
        return final_scene_path
 
 
async def _concat_scenes(scene_paths: list[str], work_root: str, fps: int = TIMELINE_FPS) -> str:
    list_path = os.path.join(work_root, "concat_list.txt")
    with open(list_path, "w") as f:
        for p in scene_paths:
            f.write(f"file '{p}'\n")
 
    out_path = os.path.join(work_root, "final_output.mp4")
 
    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        out_path,
    ]
    try:
        await _run(cmd)
    except Exception as e:
        print(f"[render] concat stream-copy failed, falling back to re-encode: {e}")
        cmd = [
            FFMPEG_BIN, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-r", str(fps),
            "-vsync", "cfr",
            *FFMPEG_X264_FLAGS,
            "-c:a", "aac",
            out_path,
        ]
        await _run(cmd)
 
    return out_path
 
 
async def _run_render_job(video_id: str, timeline: dict, scenes: list, orientation: str) -> Optional[str]:
    no_voice_scene_ids = [
        s.get("scene_id") for s in scenes
        if not (s.get("voiceover") and s.get("voiceover", {}).get("url"))
    ]
    if no_voice_scene_ids:
        print(
            f"[render] video {video_id}: scenes with NO voiceover "
            f"(will render silent): {no_voice_scene_ids}"
        )
 
    fps = timeline.get("fps", 30)
 
    if orientation == "portrait":
        resolution = PORTRAIT_RESOLUTION
    else:
        resolution = LANDSCAPE_RESOLUTION
 
    stored_resolution = timeline.get("resolution")
    if stored_resolution and (
        stored_resolution.get("width") != resolution["width"]
        or stored_resolution.get("height") != resolution["height"]
    ):
        print(
            f"[render] video {video_id}: overriding stored timeline resolution "
            f"{stored_resolution} -> {resolution} to match requested "
            f"orientation='{orientation}'"
        )
 
    width, height = resolution["width"], resolution["height"]
 
    # FIX (B-roll never rotates): this used to be a plain
    # {scene_id: track} dict. build_timeline_from_scenes creates ONE
    # "broll" track PER BEAT for a scene, so a multi-beat scene produced
    # several tracks sharing the same scene_id — and each one silently
    # overwrote the last in this dict, leaving only the final beat's asset
    # for the whole scene. That is why footage held static for an entire
    # scene regardless of how many beats were computed upstream.
    #
    # Now scene_id maps to a LIST of that scene's beat tracks, sorted by
    # startFrame, so `_render_scene` can render — and concatenate — one
    # clip per beat instead of one clip for the whole scene. Each of those
    # tracks also carries its own resolved `motion_type` (see
    # build_timeline_from_scenes).
    timeline_tracks_by_scene = {}
    caption_tracks_by_scene = {}
    infographic_tracks_by_scene = {}
    animation_tracks_by_scene = {}
    for track in timeline.get("tracks", []):
        if track.get("type") == "broll" and track.get("scene_id"):
            timeline_tracks_by_scene.setdefault(track["scene_id"], []).append(track)
        elif track.get("type") == "caption_word" and track.get("scene_id"):
            caption_tracks_by_scene[track["scene_id"]] = track
        elif track.get("type") == "infographic" and track.get("scene_id"):
            infographic_tracks_by_scene[track["scene_id"]] = track
        elif track.get("type") == "animation" and track.get("scene_id"):
            animation_tracks_by_scene[track["scene_id"]] = track

    for beat_tracks in timeline_tracks_by_scene.values():
        beat_tracks.sort(key=lambda t: t.get("startFrame", 0))
 
    try:
        supabase.table("videos").update({"render_status": "rendering"}).eq("id", video_id).execute()
    except Exception as e:
        print(f"[render] failed to set render_status=rendering for {video_id}: {e}")
 
    work_root = os.path.join(RENDER_TMP_ROOT, video_id)
    os.makedirs(work_root, exist_ok=True)
    semaphore = asyncio.Semaphore(RENDER_CONCURRENCY)
 
    try:
        async with httpx.AsyncClient() as client:
            # FIX F (rendering takes far longer than it needs to): scenes
            # were previously rendered strictly one at a time in a plain
            # `for` loop that fully awaited each scene before starting the
            # next — even though a concurrency-aware `semaphore` was
            # already being created above and threaded all the way into
            # `_render_scene` (`async with semaphore:` at its top). That
            # semaphore had nothing to actually limit, because nothing ever
            # launched more than one scene at a time in the first place.
            #
            # Each scene independently pays for: a b-roll download, a
            # separate `npx remotion render` invocation (which re-bundles
            # the whole Remotion project from scratch on every call — ~4s
            # of "Bundled code" overhead alone before any real rendering
            # happens), caption burn-in, and a final audio mux. None of
            # that depends on any other scene finishing first. Running
            # them sequentially means total wall-clock time is the SUM of
            # every scene's time — 5 scenes at a few minutes each adds up
            # to the ~30 minutes reported.
            #
            # Launching every scene as a task and awaiting them together
            # with asyncio.gather lets independent scenes' downloads /
            # Remotion renders / ffmpeg passes overlap instead of queuing
            # behind each other. The semaphore inside _render_scene still
            # caps how many run concurrently (RENDER_CONCURRENCY, CPU-based
            # by default), so this doesn't overwhelm the box — it just
            # actually uses the concurrency budget that was already being
            # computed but never spent. On a multi-core machine this is
            # typically a 3-5x wall-clock improvement for a handful of
            # scenes.
            async def _render_one(scene: dict):
                scene_id = scene.get("scene_id")
                try:
                    path = await _render_scene(
                        scene, fps, width, height, work_root, client, semaphore,
                        timeline_tracks_by_scene=timeline_tracks_by_scene,
                        caption_tracks_by_scene=caption_tracks_by_scene,
                        infographic_tracks_by_scene=infographic_tracks_by_scene,
                        animation_tracks_by_scene=animation_tracks_by_scene,
                    )
                    return scene_id, path, None
                except Exception as e:
                    print(f"[render] scene {scene_id} failed: {e}")
                    return scene_id, None, scene_id

            results = await asyncio.gather(*(_render_one(scene) for scene in scenes))

            scene_paths_by_id = {sid: path for sid, path, _ in results if path}
            failed_scenes = [sid for _, path, sid in results if path is None]
            # Concatenation order still matters even though render order no
            # longer does — asyncio.gather returns results in the same
            # order as the input list regardless of completion order, but
            # this makes that ordering guarantee explicit so concat stays
            # correct even if _render_one is refactored later.
            scene_paths = [
                scene_paths_by_id[s.get("scene_id")]
                for s in scenes
                if s.get("scene_id") in scene_paths_by_id
            ]

            if not scene_paths:
                print(f"[render] video {video_id}: all scenes failed to render")
                try:
                    supabase.table("videos").update({
                        "render_status": "failed",
                        "failed_render_scene_ids": failed_scenes,
                    }).eq("id", video_id).execute()
                except Exception as e:
                    print(f"[render] failed to persist all-scenes-failed status for {video_id}: {e}")
                raise HTTPException(status_code=500, detail="All scenes failed to render")
 
            final_path = await _concat_scenes(scene_paths, work_root, fps=fps)
 
        try:
            final_duration = await _probe_duration_seconds(final_path)
            print(f"[render] video {video_id}: final output duration = {final_duration:.3f}s")
        except Exception as e:
            print(f"[render] could not probe final output duration: {e}")
 
        try:
            public_url = _upload_rendered_video_to_supabase(final_path, video_id)
            render_status = "completed" if not failed_scenes else "completed_with_errors"
        except Exception as e:
            print(f"[render] video {video_id}: Supabase upload failed: {e}")
            video_output_dir = os.path.join(RENDER_OUTPUT_DIR, video_id)
            os.makedirs(video_output_dir, exist_ok=True)
            dest_path = os.path.join(video_output_dir, "final.mp4")
            shutil.copy2(final_path, dest_path)
            public_url = None
            render_status = "completed_upload_failed"
 
        try:
            supabase.table("videos").update({
                "final_video_url": public_url,
                "render_status": render_status,
                "failed_render_scene_ids": failed_scenes,
            }).eq("id", video_id).execute()
        except Exception as e:
            print(f"[render] failed to persist final status for {video_id}: {e}")
 
        print(
            f"[render] video {video_id}: done, status={render_status}, "
            f"failed_scenes={failed_scenes}, no_voiceover_scenes={no_voice_scene_ids}"
        )
 
        if public_url is None:
            raise HTTPException(status_code=500, detail="Render completed but upload to storage failed")
 
        return public_url
 
    except HTTPException:
        try:
            supabase.table("videos").update({"render_status": "failed"}).eq("id", video_id).execute()
        except Exception:
            pass
        raise
    except Exception as e:
        print(f"[render] render failed for {video_id}: {e}")
        try:
            supabase.table("videos").update({"render_status": "failed"}).eq("id", video_id).execute()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Render failed: {e}")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
 
 
@app.post("/render/{video_id}")
async def render_video(
    video_id: str,
    request: RenderVideoRequest = RenderVideoRequest(),
):
    print("hit")
    try:
        row = (
            supabase.table("videos")
            .select("timeline_json, raw_scenes, final_video_url")
            .eq("id", video_id)
            .single()
            .execute()
        )
    except Exception as e:
        print(f"[render] failed to fetch video {video_id}: {e}")
        raise HTTPException(status_code=404, detail="Video not found")
 
    if not row.data:
        raise HTTPException(status_code=404, detail="Video not found")
 
    if row.data.get("final_video_url") and not request.force:
        return {"final_video_url": row.data["final_video_url"]}
 
    timeline = row.data.get("timeline_json") or {}
    scenes = row.data.get("raw_scenes") or []
    if not scenes:
        raise HTTPException(status_code=400, detail="No scenes to render for this video")
 
    final_video_url = await _run_render_job(video_id, timeline, scenes, request.orientation)
 
    return {"final_video_url": final_video_url}