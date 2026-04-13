"""
mainResearch.py
Migrated to google-genai (new unified SDK).
pip install google-genai  (replaces google-generativeai)

Bugs fixed vs original:
  1. Old SDK (google.generativeai) → new SDK (google.genai)
  2. genai.configure() was never called in original → now handled by Client(api_key=...)
  3. RAZORPAY_WEBHOOK_SECRET was defined AFTER the webhook endpoint → moved to top
  4. PromptRequest was defined twice → deduplicated
  5. Chrome 91 User-Agent → Chrome 124 + full Sec-Fetch headers (fixes 403s)
  6. EMBEDDING_MODEL migrated to gemini-embedding-001 (text-embedding-004 deprecated Jan 14 2026)

UPDATED (plan section 6.3):
  - add_scraped_data_to_db() now tags every row with source_type='web_scrape'
    and metadata including domain + scraped_at.
"""
import uvicorn
from fastapi import Depends, HTTPException, status, Request, Header, BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from postgrest.exceptions import APIError
from supabase_auth.types import User
from openai import AsyncOpenAI
from auth_dependencies import get_current_user
from tss_v3 import run_tss
from pipeline_response_adapter import adapt_pipeline_payload
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ── NEW unified Google GenAI SDK ─────────────────────────────
from google import genai
from google.genai import types as genai_types
# ─────────────────────────────────────────────────────────────

import os
import asyncio
import time
import re
import json
import hmac
import hashlib
import httpx
import httplib2
import nltk
import datetime
import sqlite3
try:
    import razorpay  # type: ignore
except Exception:
    razorpay = None

from urllib.parse import urlparse
from datetime import datetime as dt
from pathlib import Path
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
from ddgs import DDGS
from readability import Document

# ── urllib3 v2 compatibility patch for pytrends ──────────────
try:
    from urllib3.util.retry import Retry as _Retry
    _orig_retry_init = _Retry.__init__
    def _patched_retry_init(self, *args, **kwargs):
        kwargs.pop('method_whitelist', None)
        _orig_retry_init(self, *args, **kwargs)
    _Retry.__init__ = _patched_retry_init
except Exception:
    pass
from pytrends.request import TrendReq

load_dotenv()

# ── NLTK ─────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.abspath(__file__))
nltk_data_dir = os.path.join(project_root, 'nltk_data')
nltk.data.path.insert(0, nltk_data_dir)
CACHE_DIR = Path(project_root) / "cache"
CACHE_DIR.mkdir(exist_ok=True)
YOUTUBE_SIGNALS_DB = CACHE_DIR / "youtube_topic_signals.sqlite3"

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    print("NLTK data found successfully.")
except LookupError as e:
    print(f"!!! CRITICAL NLTK DATA ERROR: {e} !!!")

# ── Razorpay ─────────────────────────────────────────────────
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")

if razorpay is None:
    print("WARNING: Razorpay package unavailable. Payment endpoints will fail.")
    razorpay_client = None
elif not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    print("WARNING: Razorpay API keys not found. Payment endpoints will fail.")
    razorpay_client = None
else:
    razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    print("Razorpay client initialized.")

# ── YouTube API URLs ──────────────────────────────────────────
YOUTUBE_SEARCH_URL   = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL   = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

# ── Groq (main) ───────────────────────────────────────────────
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found.")
groq_client = AsyncOpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)
GROQ_GENERATION_MODEL = "llama-3.1-8b-instant"

# ── Groq (research agent — separate key/quota) ───────────────
groq_api_key_research = os.getenv("GROQ_API_KEY_Research_Agent")
if not groq_api_key_research:
    raise ValueError("GROQ_API_KEY_Research_Agent not found.")
groq_client_research = AsyncOpenAI(
    api_key=groq_api_key_research,
    base_url="https://api.groq.com/openai/v1",
)

# ── Google GenAI client (NEW SDK) ────────────────────────────
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found.")

# Gemini client — ONLY used for embeddings
gemini_client = genai.Client(api_key=google_api_key)

# text-embedding-004 was deprecated Jan 14 2026 — migrated to gemini-embedding-001.
# output_dimensionality=768 forces 768-dim output to match existing Supabase DB vectors.
EMBEDDING_MODEL = "gemini-embedding-001"

# ── OpenRouter (LLM generation) ──────────────────────────────
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found.")

openrouter_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

openrouter_api_key_2 = os.getenv("OPENROUTER_API_KEY_2")
openrouter_client_2 = AsyncOpenAI(
    api_key=openrouter_api_key_2,
    base_url="https://openrouter.ai/api/v1",
) if openrouter_api_key_2 else None

GENERATION_MODEL        = "google/gemma-3-27b-it:free"
GENERATION_MODEL_BACKUP = "google/gemma-3n-e4b-it:free"
GENERATION_MODEL_EXTRA  = "deepseek/deepseek-r1-0528-qwen3-8b:free"

print("Google GenAI (embeddings), Groq, and OpenRouter clients initialized successfully.")


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
                    break
                elif is_rate_limit and attempt == 0:
                    wait = 1
                    print(f"OpenRouter 429 on {model} — retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"OpenRouter failed on {model}: {e} — trying next slot...")
                    break

    raise Exception(f"All OpenRouter slots exhausted. Last error: {last_error}")


# ── Supabase ─────────────────────────────────────────────────
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("Supabase credentials not found in .env file")
supabase: Client = create_client(supabase_url, supabase_key)
print("Supabase client initialized.")

# ── YouTube service ───────────────────────────────────────────
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    print("WARNING: YOUTUBE_API_KEY not found. YouTube search will fail.")
    youtube_service = None
else:
    http_client_yt = httplib2.Http(timeout=15)
    youtube_service = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY, http=http_client_yt)
    print("YouTube Data API client initialized (15s timeout).")


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = 250, chunk_overlap: int = 50) -> list[str]:
    sentences = sent_tokenize(text)
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
    Background task: chunk → embed (new SDK) → insert to Supabase.

    Metadata shape (web_scrape):
      {
        "category":   str   — from the user's topic request context (empty if unknown)
        "topic":      str   — the user's searched topic
        "tags":       list  — tags inferred from topic context
        "domain":     str   — netloc extracted from article_url
        "scraped_at": str   — ISO timestamp
        "author":     {
          "has_credentials": bool,
          "name":            str or null,   — publication name from domain
          "description":     str or null,
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

        # NEW SDK: batch embed
        embed_response = gemini_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunks,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768),
        )
        embeddings = [e.values for e in embed_response.embeddings]

        domain = urlparse(article_url).netloc.lstrip('www.') if article_url else ""
        scraped_at = dt.now().isoformat()

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
_playwright_semaphore = asyncio.Semaphore(3)


def _extract_text_from_html(html: str) -> tuple[str, str]:
    """Extract title and clean text from raw HTML using readability + BeautifulSoup."""
    doc = Document(html)
    title = doc.title()
    soup = BeautifulSoup(doc.summary(), 'html.parser')
    text = soup.get_text(separator='\n', strip=True)
    return title, text


async def _scrape_with_httpx(url: str) -> tuple[str, str] | None:
    """Tier 1: httpx with full browser headers. Fast, works on most sites."""
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
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return _extract_text_from_html(response.text)


async def _scrape_with_playwright(url: str) -> tuple[str, str] | None:
    """
    Tier 2: Real headless Chromium via Playwright.
    Defeats TLS fingerprinting, JS challenges, and most bot detection.
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
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                await page.wait_for_selector("body", timeout=10000)
                html = await page.content()
                return _extract_text_from_html(html)
            finally:
                await browser.close()

    async with _playwright_semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: asyncio.run(_run()))


# Domains that consistently return junk, non-English content, or block scrapers
_SCRAPE_BLOCKLIST = {
    'zhidao.baidu.com', 'baidu.com', 'en.cppreference.com', 'cppreference.com',
    'stackoverflow.com', 'github.com', 'reddit.com', 'twitter.com', 'x.com',
    'instagram.com', 'facebook.com', 'linkedin.com', 'pinterest.com',
    'researchgate.net', 'academia.edu', 'jstor.org',
}


async def scrape_url(
    client: httpx.AsyncClient,
    url: str,
    scraped_urls: set,
    snippet: str = "",
) -> dict | None:
    """
    3-tier scraping with automatic fallback:
      Tier 1 -> httpx (fast)
      Tier 2 -> Playwright headless Chrome (robust, defeats bot detection)
      Tier 3 -> Use DDGS snippet directly (always works, less text)
    """
    if url in scraped_urls:
        return None
    domain = urlparse(url).netloc.lstrip('www.')
    if any(domain == b or domain.endswith('.' + b) for b in _SCRAPE_BLOCKLIST):
        print(f"  ⊘ Skipped blocklisted domain: {domain}")
        return None

    # Tier 1: httpx
    try:
        title, text = await _scrape_with_httpx(url)
        if text and len(text) > 200:
            scraped_urls.add(url)
            print(f"  v Tier 1 (httpx) succeeded: {url[:60]}")
            return {"url": url, "title": title, "text": text}
    except Exception as e:
        print(f"  x Tier 1 (httpx) failed: {e} -- trying Playwright...")

    # Tier 2: Playwright
    try:
        result = await _scrape_with_playwright(url)
        if result:
            title, text = result
            if text and len(text) > 200:
                scraped_urls.add(url)
                print(f"  v Tier 2 (Playwright) succeeded: {url[:60]}")
                return {"url": url, "title": title, "text": text}
    except Exception as e:
        print(f"  x Tier 2 (Playwright) failed: {e} -- using snippet fallback...")

    # Tier 3: snippet fallback
    if snippet and len(snippet) > 50:
        print(f"  v Tier 3 (snippet fallback) used for: {url[:60]}")
        scraped_urls.add(url)
        return {"url": url, "title": url, "text": snippet}

    print(f"  x All tiers failed for: {url[:60]}")
    return None


async def deep_search_and_scrape(keywords: list[str], scraped_urls: set) -> list[dict]:
    print("--- DEEP WEB SCRAPE: Starting full search... ---")
    urls_to_scrape = []
    with DDGS(timeout=20) as ddgs:
        for keyword in keywords:
            results = list(ddgs.text(keyword, region='wt-wt', max_results=3))
            if results:
                top = results[0]
                urls_to_scrape.append((top['href'], top.get('body', '')))

    seen = set()
    unique = []
    for url, snippet in urls_to_scrape:
        if url not in seen:
            seen.add(url)
            unique.append((url, snippet))

    async with httpx.AsyncClient() as client:
        tasks = [scrape_url(client, url, scraped_urls, snippet) for url, snippet in unique]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r and r.get("text")]


async def get_latest_news_context(topic: str, scraped_urls: set) -> list[dict]:
    print("--- LIGHT WEB SCRAPE: Starting news search... ---")
    try:
        keyword = f"{topic} latest news today"
        url_snippet_pairs = []
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.text(keyword, region='wt-wt', max_results=2))
            for r in results:
                url_snippet_pairs.append((r['href'], r.get('body', '')))
        async with httpx.AsyncClient() as client:
            tasks = [scrape_url(client, url, scraped_urls, snippet)
                     for url, snippet in url_snippet_pairs]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r and r.get("text")]
    except Exception as e:
        print(f"--- WEB TASK: Error: {e} ---")
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

        loop = asyncio.get_running_loop()
        embed_response = await loop.run_in_executor(
            None,
            lambda: gemini_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=hypothetical_document,
                config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768),
            ),
        )
        query_embedding = embed_response.embeddings[0].values

        vector_response = await loop.run_in_executor(
            None,
            lambda: supabase.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.55,   # relaxed: was 0.65
                    'match_count':     8,
                }
            ).execute()
        )
        vector_results = vector_response.data or []
        for row in vector_results:
            combined[row['id']] = row
        print(f"--- DB TASK: Vector search → {len(vector_results)} results ---")

        # ── Stage 2: Keyword fallback if vector was thin ──────
        if len(combined) < 3:
            print(f"--- DB TASK: Thin vector results ({len(combined)}), running keyword search... ---")
            keyword_response = await loop.run_in_executor(
                None,
                lambda: supabase.rpc(
                    'search_documents',
                    {
                        'search_query': topic,
                        'match_count':  8,
                    }
                ).execute()
            )
            keyword_results = keyword_response.data or []
            for row in keyword_results:
                if row['id'] not in combined:
                    combined[row['id']] = row
            print(f"--- DB TASK: Keyword search → {len(keyword_results)} results, total unique: {len(combined)} ---")

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://www.storybit.tech",
        "https://storybit.tech",
    ],
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


class ResearchBriefInput(BaseModel):
    topic_analysis: dict
    strategic_angles: list[dict]
    audience_profile: dict


# ════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════

@app.get("/")
async def read_root():
    return {"status": "Welcome"}



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



@app.post("/pipeline-metrics")
async def pipeline_metrics(
    request: PromptRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Frontend integration endpoint for the new scoring stack.
    Returns the full run_tss payload:
      TSS + CSI + CAGS + verdict
    """
    try:
        result = await run_tss(request.topic)
        return adapt_pipeline_payload(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline metrics failed: {e}")


async def _generate_search_keywords(topic: str) -> list[str]:
    """Generate 3 focused, English-language news search keywords using Groq. ~0.5s."""
    keyword_prompt = f"""
    Generate 3 diverse search engine keyword phrases to find English-language news articles about: '{topic}'.
    Rules:
    1. Return ONLY the 3 phrases — nothing else.
    2. Each phrase must be in English and suitable for Google/news search.
    3. Focus on factual, journalistic angles (not tutorials or code).
    4. One phrase per line, no numbers or bullet points.

    EXAMPLE INPUT: Future Trends: Where Recession in US is Heading
    EXAMPLE OUTPUT:
    US recession outlook 2025 economic forecast
    Federal Reserve interest rates recession risk
    US GDP growth slowdown indicators
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


# ── /process-topic ───────────────────────────────────────────

@app.post("/process-topic")
async def process_topic(
    request: PromptRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    total_start_time = time.time()
    user_id = current_user.id
    print(f"Received topic from user ({user_id}): {request.topic}")

    IDEA_COST = 1
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
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found.")

        credits  = profile.get('credits_remaining', 0)
        user_tier = profile.get('user_tier', 'free')

        if user_tier != 'admin' and credits < IDEA_COST:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient credits. Requires {IDEA_COST}, you have {credits}.",
            )
        print(f"User {user_id} (Tier: {user_tier}) has {credits} credits.")
    except APIError as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e.message}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error checking credits: {e}")
        raise HTTPException(status_code=500, detail="Error checking profile.")

    try:
        # ── Phase 1: DB + keyword gen in parallel ─────────────
        print("--- Phase 1: DB lookup + keyword gen in parallel ---")
        db_results, base_keywords = await asyncio.gather(
            get_db_context(request.topic),
            _generate_search_keywords(request.topic),
        )
        print(f"--- Phase 1 done: {len(db_results)} DB docs, keywords: {base_keywords} ---")

        scraped_urls: set = set()
        source_of_context = ""

        # ── Phase 2: Web scrape strategy based on DB quality ──
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
            db_context = "\n\n".join([item['content'] for item in db_results])
            source_urls.extend(list(set([
                item['source_url'] for item in db_results if item.get('source_url')
            ])))

        if new_articles:
            web_context = "\n\n".join([
                f"Source: {art['title']}\n{art['text']}" for art in new_articles
            ])
            source_urls.extend([art['url'] for art in new_articles])
            for article in new_articles:
                background_tasks.add_task(
                    add_scraped_data_to_db,
                    article['title'], article['text'], article['url'],
                    "",               # category unknown at live-scrape time
                    request.topic,    # topic from user request
                    base_keywords,    # use generated keywords as tags proxy
                )

        if not db_context and not web_context:
            return {"error": "Could not find any information."}

        final_prompt = f"""
        You are an expert YouTube title strategist and scriptwriter.
        Generate 4 distinct, attention-grabbing video titles for: "{request.topic}",
        with a corresponding description for each.

        RULES:
        1. For each idea: provide 'TITLE' and 'DESCRIPTION'.
        2. Each DESCRIPTION MUST be 150-180 words. Write in full sentences, expand on the hook, tease 2-3 specific points the video covers, and end with a curiosity gap or question.
        3. Separate each complete idea with '---'.
        4. NO introductory text, explanations, or anything else.

        EXAMPLE FORMAT:
        TITLE: This Is Why Everyone Is Suddenly Talking About [Topic]
        DESCRIPTION: In this video, we uncover the shocking truth behind [Topic]...
        ---

        RESEARCH FOR TOPIC: "{request.topic}"
        ---
        FOUNDATIONAL KNOWLEDGE:
        {db_context}
        ---
        LATEST NEWS UPDATES:
        {web_context}
        ---
        """

        step3_start = time.time()
        response_text = await openrouter_generate([{"role": "user", "content": final_prompt}])
        print(f"--- Step 3 (Final Idea Gen) took {time.time() - step3_start:.2f}s ---")

        final_ideas, final_descriptions = [], []
        for block in response_text.strip().split('---'):
            title = description = ""
            for line in block.strip().split('\n'):
                if line.startswith('TITLE:'):
                    title = line.replace('TITLE:', '', 1).strip()
                elif line.startswith('DESCRIPTION:'):
                    description = line.replace('DESCRIPTION:', '', 1).strip()
            if title and description:
                final_ideas.append(title)
                final_descriptions.append(description)

        # Decrement credits
        if user_tier != 'admin':
            try:
                new_balance = max(0, credits - IDEA_COST)
                supabase.table('profiles').update(
                    {'credits_remaining': new_balance}
                ).eq('id', user_id).execute()
                print(f"Decremented {IDEA_COST} credit(s) for {user_id}. Balance: {new_balance}")
            except Exception as e:
                print(f"ERROR: Credit decrement failed for {user_id}: {e}")

        print(f"Total request time: {time.time() - total_start_time:.2f}s")

        return {
            "source_of_context": source_of_context,
            "ideas": final_ideas,
            "descriptions": final_descriptions,
            "generated_keywords": base_keywords,
            "source_urls": list(set(source_urls)),
            "scraped_text_context": f"DB CONTEXT:\n{db_context}\n\nWEB CONTEXT:\n{web_context}",
        }

    except Exception as e:
        print(f"Error in /process-topic: {e}")
        return {"error": "An error occurred in the processing pipeline."}


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
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found.")

        credits   = profile.get('credits_remaining', 0)
        user_tier = profile.get('user_tier', 'free')

        if user_tier != 'admin' and credits < IDEA_COST:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient credits. Requires {IDEA_COST}, you have {credits}.",
            )
        print(f"User {user_id} (Tier: {user_tier}) has {credits} credits.")
    except APIError as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e.message}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error checking credits: {e}")
        raise HTTPException(status_code=500, detail="Error checking profile.")

    print(
        f"Duration: {request.duration_minutes}min | Tone: {request.emotional_tone} | "
        f"Type: {request.creator_type} | Audience: {request.audience_description} | Accent: {request.accent}"
    )

    try:
        # ── Phase 1: DB + keyword gen in parallel ─────────────
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

        db_context, web_context = "", ""
        if db_results:
            db_context = "\n\n".join([item['content'] for item in db_results])
        if new_articles:
            web_context = "\n\n".join([
                f"Source: {art['title']}\n{art['text']}" for art in new_articles
            ])
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

        WORDS_PER_MINUTE = 130
        target_duration   = request.duration_minutes or 10
        target_word_count = target_duration * WORDS_PER_MINUTE

        structure_guidance_text = STRUCTURE_GUIDANCE.get(
            request.script_structure or "problem_solution",
            STRUCTURE_GUIDANCE["problem_solution"]
        )

        script_prompt = f"""
        You are a professional YouTube scriptwriter creating natural, engaging, conversational scripts.

        **Creator Profile:**
        * Creator Type: {request.creator_type}
        * Target Audience: {request.audience_description}
        * Desired Emotional Tone: {request.emotional_tone}
        * Accent/Dialect: {request.accent}

        **Task:** Generate a complete YouTube script of ~{target_duration} minutes (~{target_word_count} words).

        **Style:**
        - Spoken dialogue only — no titles, stage directions, or metadata.
        - Direct, friendly, confident, slightly spontaneous.
        - Short/medium sentences, natural pauses (…), rhetorical questions, humor.
        - Personal anecdotes ("I remember…"), vivid imagery ("Imagine this…").
        - Hook viewers emotionally in first 15-30 seconds.
        - Inclusive language: "you guys", "we all", "my friends".
        - Stay close to {target_word_count} words (±50).

        {structure_guidance_text}

        **Topic:** "{request.topic}"

        **Research:**
        FOUNDATIONAL KNOWLEDGE: {db_context}
        LATEST NEWS: {web_context}
        """

        script_text = await openrouter_generate([{"role": "user", "content": script_prompt}])
        print(f"--- Script generation took {time.time() - total_start_time:.2f}s ---")

        # Analyse script with Groq
        ANALYSIS_PROMPT = f"""
        You are a script analyzer. Return ONLY a JSON object — no explanation, no other text.

        {{
          "examples_count": <int>,
          "research_facts_count": <int>,
          "proverbs_count": <int>,
          "emotional_depth": "Low" | "Medium" | "High"
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

        analysis_results = {"examples_count": 0, "research_facts_count": 0,
                            "proverbs_count": 0, "emotional_depth": "Unknown"}
        try:
            clean  = analysis_raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            analysis_results.update({
                "examples_count":       parsed.get("examples_count", 0),
                "research_facts_count": parsed.get("research_facts_count", 0),
                "proverbs_count":       parsed.get("proverbs_count", 0),
                "emotional_depth":      parsed.get("emotional_depth", "Unknown"),
            })
        except Exception as e:
            print(f"Analysis parse error: {e}")

        generated_word_count = len(script_text.split())
        print(f"Word count: {generated_word_count}")
        print(f"Total /generate-script time: {time.time() - total_start_time:.2f}s")

        # Decrement credits
        if user_tier != 'admin':
            try:
                new_balance = max(0, credits - IDEA_COST)
                result = (
                    supabase.table('profiles')
                    .update({'credits_remaining': new_balance})
                    .eq('id', user_id)
                    .execute()
                )
                if result.data:
                    print(f"Decremented {IDEA_COST} credit(s) for {user_id}. Balance: {new_balance}")
            except Exception as e:
                print(f"ERROR: Credit decrement failed for {user_id}: {e}")
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

    user_id  = current_user.id
    amount   = request_data.amount
    currency = request_data.currency

    if amount <= 0:
        raise HTTPException(status_code=400, detail="Invalid amount.")
    if request_data.target_tier not in ['basic', 'pro']:
        raise HTTPException(status_code=400, detail="Invalid target tier.")

    order_data = {
        "amount":   amount,
        "currency": currency,
        "receipt":  request_data.receipt or f"rec_{int(time.time())}",
        "notes": {
            "user_id":     str(user_id),
            "target_tier": request_data.target_tier,
        },
    }
    try:
        order = razorpay_client.order.create(data=order_data)
        print(f"Created Razorpay order {order['id']} for user {user_id}")
        return {
            "order_id": order['id'],
            "key_id":   RAZORPAY_KEY_ID,
            "amount":   amount,
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

    try:
        razorpay_client.utility.verify_webhook_signature(
            body.decode('utf-8'),
            x_razorpay_signature,
            RAZORPAY_WEBHOOK_SECRET,
        )
        print("Webhook signature verified.")
    except Exception as e:
        # Razorpay import/runtime issues should not crash app boot.
        print(f"Webhook verification error: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")

    try:
        event_data = json.loads(body)
        event_type = event_data.get('event')
        print(f"Received webhook event: {event_type}")

        if event_type == 'order.paid':
            order_entity = event_data['payload']['order']['entity']
            order_id   = order_entity.get('id', 'unknown')
            payment_id = event_data['payload']['payment']['entity'].get('id', 'unknown')
            notes        = order_entity.get('notes', {})
            user_id      = notes.get('user_id')
            target_tier  = notes.get('target_tier')

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

                result = (
                    supabase.table('profiles')
                    .update({'user_tier': target_tier, 'credits_remaining': new_credits})
                    .eq('id', user_id)
                    .execute()
                )
                if result.data:
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


# ════════════════════════════════════════════════════════════
# RESEARCH AGENT
# ════════════════════════════════════════════════════════════

async def fetch_social_pulse(topic: str) -> list[dict]:
    print(f"RESEARCH AGENT: Fetching social pulse for '{topic}'...")
    loop = asyncio.get_running_loop()
    try:
        with DDGS(timeout=10) as ddgs:
            forum_results = await loop.run_in_executor(
                None, lambda: ddgs.text(f"{topic} discussion forum", timelimit="m", max_results=10)
            )
            quora_results = await loop.run_in_executor(
                None, lambda: ddgs.text(f"{topic} site:quora.com", timelimit='m', max_results=10)
            )
        all_results = []
        if forum_results: all_results.extend(forum_results)
        if quora_results:  all_results.extend(quora_results)
        return all_results
    except Exception as e:
        print(f"RESEARCH AGENT: Social pulse failed. Error: {e}")
        return []


async def fetch_news_analysis(topic: str) -> list[dict]:
    print(f"RESEARCH AGENT: Fetching news for '{topic}'...")
    loop = asyncio.get_running_loop()
    try:
        with DDGS(timeout=10) as ddgs:
            news_results = await loop.run_in_executor(
                None, lambda: ddgs.news(topic, timelimit='w', max_results=10)
            )
        if not news_results:
            return []
        return [
            {"title": item['title'], "body": item['body'],
             "url": item['url'], "source": item['source']}
            for item in news_results
        ]
    except Exception as e:
        print(f"RESEARCH AGENT: News analysis failed. Error: {e}")
        return []


def _calculate_engagement_pct(stats: dict) -> float:
    try:
        views = int(stats.get('viewCount', '0'))
        if views == 0:
            return 0.0
        likes    = int(stats.get('likeCount', '0'))
        comments = int(stats.get('commentCount', '0'))
        return round(((likes + comments) / views) * 100, 2)
    except Exception:
        return 0.0


def _get_channel_authority(subscriber_count: int) -> str:
    if subscriber_count >= 20_000_000: return "Super High"
    elif subscriber_count >= 5_000_000: return "High"
    elif subscriber_count >= 1_000_000: return "Medium"
    elif subscriber_count >= 100_000:   return "Poor"
    else:                               return "Very Poor"


def _parse_duration_iso8601(duration: str) -> int:
    """Parse ISO 8601 duration (PT4M13S) → total seconds."""
    import re as _re
    m = _re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration or '')
    if not m:
        return 0
    h, mn, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mn * 60 + s


def _iso_utc(dt_obj: datetime.datetime) -> str:
    return dt_obj.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _parse_yt_datetime(value: str | None) -> datetime.datetime | None:
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _normalize_topic_key(topic: str) -> str:
    return re.sub(r"\s+", " ", (topic or "").strip().lower())


_QUERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at", "by",
    "with", "from", "vs", "is", "are", "was", "were", "will", "would", "could",
    "should", "after", "before", "today", "latest", "news", "update", "updates",
    "what", "why", "how", "when", "where",
}


def _is_question_text(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return lowered.endswith("?") or lowered.startswith(
        ("what ", "why ", "how ", "when ", "where ", "who ", "will ", "can ", "should ", "is ", "are ")
    )


def _significant_query_tokens(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in _QUERY_STOPWORDS
    ]


def _extract_topic_entities(text: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Z][a-z]+|[A-Z]{2,}|[A-Z][a-z]+(?:-[A-Z][a-z]+)?", text or "")
    entities = []
    for token in raw_tokens:
        lowered = token.lower()
        if lowered not in _QUERY_STOPWORDS and lowered not in entities:
            entities.append(lowered)
    return entities


def _build_entity_preserving_query_variants(topic: str) -> list[str]:
    original = re.sub(r"\s+", " ", (topic or "").strip())
    tokens = original.split()
    lowered = original.lower()
    variants = [original]

    entities = _extract_topic_entities(original)
    base_entity_phrase = " ".join(word.capitalize() for word in entities) if entities else original
    if base_entity_phrase and base_entity_phrase not in variants:
        variants.append(base_entity_phrase)

    if "war" in lowered:
        variants.extend([
            re.sub(r"\bwar\b", "conflict", original, flags=re.IGNORECASE),
            f"{original} analysis",
        ])
    if "conflict" not in lowered:
        variants.append(f"{base_entity_phrase} conflict")
    if "war" not in lowered:
        variants.append(f"{base_entity_phrase} war")
    if "escalation" not in lowered:
        variants.append(f"{base_entity_phrase} escalation")
    if len(tokens) >= 2:
        reversed_entities = " ".join(reversed(tokens[:2]))
        variants.append(f"{reversed_entities} conflict")

    cleaned = []
    seen = set()
    for item in variants:
        candidate = re.sub(r"\s+", " ", item).strip(" -")
        if candidate and candidate.lower() not in seen:
            seen.add(candidate.lower())
            cleaned.append(candidate)
    return cleaned[:6]


def _query_similarity_score(original: str, candidate: str) -> float:
    original_tokens = set(_significant_query_tokens(original))
    candidate_tokens = set(_significant_query_tokens(candidate))
    if not original_tokens:
        return 1.0 if candidate.strip() else 0.0
    overlap = len(original_tokens & candidate_tokens) / len(original_tokens)
    original_entities = set(_extract_topic_entities(original))
    candidate_entities = set(_extract_topic_entities(candidate))
    if original_entities:
        entity_overlap = len(original_entities & candidate_entities) / len(original_entities)
        return round((overlap * 0.6) + (entity_overlap * 0.4), 4)
    return round(overlap, 4)


def _select_guardrailed_query(topic: str, requested_query: str, alternates: list[str] | None = None) -> tuple[str, str, list[str]]:
    original = re.sub(r"\s+", " ", (topic or "").strip())
    candidate_pool = [requested_query] + (alternates or []) + _build_entity_preserving_query_variants(original)
    original_is_question = _is_question_text(original)
    original_entities = set(_extract_topic_entities(original))

    best_query = original
    best_score = -1.0

    for raw_candidate in candidate_pool:
        candidate = re.sub(r"\s+", " ", (raw_candidate or "").strip())
        if not candidate:
            continue
        if not original_is_question and _is_question_text(candidate):
            continue
        candidate_entities = set(_extract_topic_entities(candidate))
        if original_entities and not (original_entities & candidate_entities):
            continue
        score = _query_similarity_score(original, candidate)
        if score >= 0.55 and score > best_score:
            best_query = candidate
            best_score = score

    final_alternates = []
    seen = set()
    for item in [best_query, original, *(alternates or []), *_build_entity_preserving_query_variants(original)]:
        candidate = re.sub(r"\s+", " ", (item or "").strip())
        if candidate and candidate.lower() not in seen:
            seen.add(candidate.lower())
            final_alternates.append(candidate)

    reason = "guardrailed_refinement" if best_query != original else "guardrailed_original"
    return best_query, reason, final_alternates[:5]


def _init_youtube_signals_db() -> None:
    conn = sqlite3.connect(YOUTUBE_SIGNALS_DB)
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
            CREATE INDEX IF NOT EXISTS idx_youtube_topic_snapshots_lookup
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
                uploads_last_7d INTEGER NOT NULL,
                uploads_prior_30d INTEGER NOT NULL,
                upload_surge_ratio REAL NOT NULL,
                avg_view_velocity REAL NOT NULL,
                channel_diversity REAL NOT NULL,
                avg_snapshot_delta REAL,
                video_count INTEGER NOT NULL,
                raw_payload TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_youtube_topic_scans_lookup
            ON youtube_topic_scans(topic_key, scan_timestamp DESC)
            """
        )
        conn.commit()
    finally:
        conn.close()


def _load_previous_video_snapshots(topic: str, video_ids: list[str]) -> dict[str, dict]:
    if not video_ids:
        return {}
    _init_youtube_signals_db()
    placeholders = ",".join("?" for _ in video_ids)
    conn = sqlite3.connect(YOUTUBE_SIGNALS_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT topic_key, video_id, views_at_scan, scan_timestamp
            FROM youtube_topic_snapshots
            WHERE topic_key = ?
              AND video_id IN ({placeholders})
            ORDER BY scan_timestamp DESC
            """,
            [_normalize_topic_key(topic), *video_ids],
        ).fetchall()
    finally:
        conn.close()

    latest_by_video: dict[str, dict] = {}
    for row in rows:
        if row["video_id"] not in latest_by_video:
            latest_by_video[row["video_id"]] = dict(row)
    return latest_by_video


def _persist_youtube_topic_scan(topic: str, signal_summary: dict, videos: list[dict]) -> None:
    _init_youtube_signals_db()
    topic_key = _normalize_topic_key(topic)
    conn = sqlite3.connect(YOUTUBE_SIGNALS_DB)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO youtube_topic_scans (
                topic_key, topic, scan_timestamp, uploads_last_7d, uploads_prior_30d,
                upload_surge_ratio, avg_view_velocity, channel_diversity,
                avg_snapshot_delta, video_count, raw_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_key,
                topic,
                signal_summary["scan_timestamp"],
                signal_summary["uploads_last_7d"],
                signal_summary["uploads_prior_30d"],
                signal_summary["upload_surge_ratio"],
                signal_summary["avg_view_velocity"],
                signal_summary["channel_diversity"],
                signal_summary.get("avg_snapshot_delta"),
                len(videos),
                json.dumps(signal_summary, ensure_ascii=True),
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
                    video["id"],
                    video.get("channel_id"),
                    video.get("title"),
                    int(video.get("view_count", 0)),
                    video.get("published_at"),
                    signal_summary["scan_timestamp"],
                )
                for video in videos
            ],
        )
        conn.commit()
    finally:
        conn.close()


async def _youtube_search_total_results(
    client: httpx.AsyncClient,
    topic: str,
    *,
    published_after: datetime.datetime,
    published_before: datetime.datetime | None = None,
) -> int:
    params = {
        "key": YOUTUBE_API_KEY,
        "q": topic,
        "part": "snippet",
        "type": "video",
        "order": "date",
        "publishedAfter": _iso_utc(published_after),
        "maxResults": 1,
    }
    if published_before is not None:
        params["publishedBefore"] = _iso_utc(published_before)
    response = await client.get(YOUTUBE_SEARCH_URL, params=params)
    response.raise_for_status()
    payload = response.json()
    page_info = payload.get("pageInfo", {})
    return int(page_info.get("totalResults", 0) or 0)


async def _youtube_search_recent_videos(
    client: httpx.AsyncClient,
    topic: str,
    *,
    published_after: datetime.datetime,
    max_results: int = 20,
) -> list[dict]:
    params = {
        "key": YOUTUBE_API_KEY,
        "q": topic,
        "part": "snippet",
        "type": "video",
        "order": "viewCount",
        "publishedAfter": _iso_utc(published_after),
        "maxResults": max_results,
    }
    response = await client.get(YOUTUBE_SEARCH_URL, params=params)
    response.raise_for_status()
    return response.json().get("items", [])


async def _refine_youtube_market_query(topic: str) -> dict:
    """
    Broad umbrella topics produce inflated YouTube counts.
    This helper narrows the topic to the most commercially or editorially specific
    phrase while preserving the user's original intent.
    """
    fallback = {
        "scan_query": topic,
        "reason": "fallback",
        "alternates": _build_entity_preserving_query_variants(topic),
    }
    try:
        prompt = f"""
You are refining a topic for a YouTube market scan.

TOPIC: "{topic}"

Goal:
- Keep the same intent
- Make the query more specific and less inflated if the topic is broad
- Prefer a 2-6 word query that a creator would actually search on YouTube
- Do not add dates unless the topic is explicitly newsy
- Preserve the main entities from the topic
- Do NOT turn the topic into a speculative question unless the original topic is already a question
- Prefer entity-preserving rewrites like:
  - Israel Iran conflict
  - Israel Iran war analysis
  - Iran Israel escalation

Return ONLY valid JSON:
{{
  "scan_query": "",
  "reason": "",
  "alternates": ["", ""]
}}
"""
        cc = await groq_client_research.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_GENERATION_MODEL,
            response_format={"type": "json_object"},
        )
        data = json.loads(cc.choices[0].message.content)
        alternates = [
            alt.strip() for alt in (data.get("alternates") or []) if isinstance(alt, str) and alt.strip()
        ]
        guarded_query, guard_reason, guarded_alternates = _select_guardrailed_query(
            topic,
            (data.get("scan_query") or topic).strip(),
            alternates,
        )
        return {
            "scan_query": guarded_query,
            "reason": f"{(data.get('reason') or 'llm_refined').strip()} | {guard_reason}",
            "alternates": guarded_alternates,
        }
    except Exception as e:
        print(f"RESEARCH AGENT: Query refinement failed, using raw topic. Error: {e}")
        return fallback


async def fetch_youtube_trends(topic: str) -> dict:
    """
    Quota-aware YouTube market scan.

    Signals:
      - upload_surge_ratio
      - avg_view_velocity (top 5 recent videos by views/day)
      - channel_diversity (unique channels / sampled videos)
      - avg_snapshot_delta (same-video gain/day from last scan)
    """
    if not YOUTUBE_API_KEY:
        print("RESEARCH AGENT: YouTube API not configured. Skipping.")
        return {
            "videos": [],
            "signals": {
                "scan_timestamp": _iso_utc(_utc_now()),
                "uploads_last_7d": 0,
                "uploads_prior_30d": 0,
                "upload_surge_ratio": 0.0,
                "avg_view_velocity": 0.0,
                "channel_diversity": 0.0,
                "avg_snapshot_delta": None,
                "video_sample_size": 0,
                "top_velocity_video_ids": [],
                "quota_estimate_units": 0,
            },
        }

    query_plan = await _refine_youtube_market_query(topic)
    scan_query = query_plan.get("scan_query") or topic
    print(f"RESEARCH AGENT: Fetching YouTube trends for '{topic}' using query '{scan_query}'...")
    now_utc = _utc_now()
    seven_days_ago = now_utc - datetime.timedelta(days=7)
    thirty_seven_days_ago = now_utc - datetime.timedelta(days=37)
    thirty_days_ago = now_utc - datetime.timedelta(days=30)
    scan_timestamp = _iso_utc(now_utc)

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            uploads_last_7d, uploads_prior_30d, search_items = await asyncio.gather(
                _youtube_search_total_results(client, scan_query, published_after=seven_days_ago),
                _youtube_search_total_results(
                    client,
                    scan_query,
                    published_after=thirty_seven_days_ago,
                    published_before=seven_days_ago,
                ),
                _youtube_search_recent_videos(client, scan_query, published_after=thirty_days_ago, max_results=20),
            )

            video_ids, channel_ids, video_snippet_map = [], set(), {}
            for item in search_items:
                vid = item['id'].get('videoId')
                if not vid:
                    continue
                cid = item['snippet']['channelId']
                video_ids.append(vid)
                channel_ids.add(cid)
                video_snippet_map[vid] = item['snippet']

            if not video_ids:
                print(f"RESEARCH AGENT: No YouTube videos found for '{topic}'.")
                signal_summary = {
                    "scan_timestamp": scan_timestamp,
                    "scan_query": scan_query,
                    "scan_query_reason": query_plan.get("reason"),
                    "scan_query_alternates": query_plan.get("alternates", []),
                    "uploads_last_7d": uploads_last_7d,
                    "uploads_prior_30d": uploads_prior_30d,
                    "upload_surge_ratio": 0.0,
                    "avg_view_velocity": 0.0,
                    "channel_diversity": 0.0,
                    "avg_snapshot_delta": None,
                    "video_sample_size": 0,
                    "top_velocity_video_ids": [],
                    "quota_estimate_units": 320,
                }
                return {"videos": [], "signals": signal_summary}

            async def _get_video_details(ids_str: str):
                resp = await client.get(
                    YOUTUBE_VIDEOS_URL,
                    params={
                        'key': YOUTUBE_API_KEY,
                        'part': 'statistics,snippet,contentDetails',
                        'id': ids_str,
                    }
                )
                resp.raise_for_status()
                return resp.json()

            async def _get_channel_details(ids_str: str):
                resp = await client.get(
                    YOUTUBE_CHANNELS_URL,
                    params={
                        'key': YOUTUBE_API_KEY,
                        'part': 'statistics,brandingSettings',
                        'id': ids_str,
                    }
                )
                resp.raise_for_status()
                return resp.json()

            async def _get_comment_themes_async(vid_id: str) -> tuple[str, list]:
                try:
                    params = {
                        'key': YOUTUBE_API_KEY, 'part': 'snippet',
                        'videoId': vid_id, 'order': 'relevance', 'maxResults': 60,
                    }
                    resp = await client.get(YOUTUBE_COMMENTS_URL, params=params)
                    resp.raise_for_status()

                    comment_items = resp.json().get('items', [])
                    comment_items.sort(
                        key=lambda x: int(x['snippet']['topLevelComment']['snippet'].get('likeCount', 0)),
                        reverse=True
                    )
                    raw_comments = [
                        item['snippet']['topLevelComment']['snippet']['textOriginal']
                        for item in comment_items
                    ]
                    if not raw_comments:
                        return vid_id, []

                    analysis_prompt = f"""
                    Identify the Top 3-5 recurring themes/sentiments in these YouTube comments.
                    Comments are sorted by likes (most liked = most representative audience view).
                    Return ONLY a valid JSON object: {{"themes": ["theme1", "theme2", ...]}}

                    COMMENTS: {json.dumps(raw_comments[:30])}
                    """
                    cc = await groq_client_research.chat.completions.create(
                        messages=[{"role": "user", "content": analysis_prompt}],
                        model=GROQ_GENERATION_MODEL,
                        response_format={"type": "json_object"},
                    )
                    data   = json.loads(cc.choices[0].message.content)
                    themes = data.get("themes", [])
                    return vid_id, themes
                except Exception as e:
                    print(f"RESEARCH AGENT: Comment analysis failed for {vid_id}: {e}")
                    return vid_id, []

            video_id_str   = ','.join(video_ids)
            channel_id_str = ','.join(channel_ids)
            comments_tasks = [_get_comment_themes_async(vid) for vid in video_ids[:3]]
            previous_snapshots = _load_previous_video_snapshots(topic, video_ids)

            results = await asyncio.gather(
                _get_video_details(video_id_str),
                _get_channel_details(channel_id_str),
                *comments_tasks,
                return_exceptions=True,
            )

            video_details_response   = results[0] if not isinstance(results[0], Exception) else {}
            channels_response        = results[1] if not isinstance(results[1], Exception) else {}
            comment_results          = [r for r in results[2:] if not isinstance(r, Exception)]

            if isinstance(results[0], Exception):
                print(f"RESEARCH AGENT: CRITICAL — video details error: {results[0]}")
                return {
                    "videos": [],
                    "signals": {
                        "scan_timestamp": scan_timestamp,
                        "uploads_last_7d": uploads_last_7d,
                        "uploads_prior_30d": uploads_prior_30d,
                        "upload_surge_ratio": 0.0,
                        "avg_view_velocity": 0.0,
                        "channel_diversity": 0.0,
                        "avg_snapshot_delta": None,
                        "video_sample_size": 0,
                        "top_velocity_video_ids": [],
                        "quota_estimate_units": 320,
                        "error": "video_details_failed",
                    },
                }
            if isinstance(results[1], Exception):
                print(f"RESEARCH AGENT: CRITICAL — channel details error: {results[1]}")
                return {
                    "videos": [],
                    "signals": {
                        "scan_timestamp": scan_timestamp,
                        "uploads_last_7d": uploads_last_7d,
                        "uploads_prior_30d": uploads_prior_30d,
                        "upload_surge_ratio": 0.0,
                        "avg_view_velocity": 0.0,
                        "channel_diversity": 0.0,
                        "avg_snapshot_delta": None,
                        "video_sample_size": 0,
                        "top_velocity_video_ids": [],
                        "quota_estimate_units": 320,
                        "error": "channel_details_failed",
                    },
                }

            video_details_map = {item['id']: item for item in video_details_response.get('items', [])}
            channel_details_map = {item['id']: item for item in channels_response.get('items', [])}
            comments_map = {vid_id: themes for vid_id, themes in comment_results}

            videos = []
            shorts_filtered = 0
            snapshot_gains = []
            for vid in video_ids:
                snippet        = video_snippet_map.get(vid, {})
                detail_item    = video_details_map.get(vid, {})
                stats          = detail_item.get('statistics', {})
                detail_snippet = detail_item.get('snippet', {})
                content_detail = detail_item.get('contentDetails', {})
                cid            = snippet.get('channelId')
                channel_item   = channel_details_map.get(cid, {})
                subs           = int(channel_item.get('statistics', {}).get('subscriberCount', '0'))

                duration_secs = _parse_duration_iso8601(content_detail.get('duration', ''))
                if duration_secs > 0 and duration_secs < 60:
                    shorts_filtered += 1
                    continue

                published_at = snippet.get('publishedAt')
                published_dt = _parse_yt_datetime(published_at)
                if published_dt is None:
                    age_days = 1.0
                else:
                    age_days = max((now_utc - published_dt).total_seconds() / 86400.0, 1.0)
                view_count = int(stats.get('viewCount', '0'))
                view_velocity = round(view_count / age_days, 2)

                previous = previous_snapshots.get(vid)
                snapshot_delta = None
                if previous:
                    previous_scan = _parse_yt_datetime(previous.get("scan_timestamp"))
                    previous_views = int(previous.get("views_at_scan", 0))
                    if previous_scan is not None:
                        elapsed_days = max((now_utc - previous_scan).total_seconds() / 86400.0, 0.0001)
                        snapshot_delta = round((view_count - previous_views) / elapsed_days, 2)
                        snapshot_gains.append(snapshot_delta)

                tags = detail_snippet.get('tags', [])
                channel_keywords = (
                    channel_item.get('brandingSettings', {})
                    .get('channel', {})
                    .get('keywords', '')
                )
                import shlex
                try:
                    channel_keywords_list = shlex.split(channel_keywords) if channel_keywords else []
                except Exception:
                    channel_keywords_list = channel_keywords.split() if channel_keywords else []

                videos.append({
                    "id": vid,
                    "title": snippet.get('title'),
                    "description": snippet.get('description'),
                    "body": snippet.get('description'),
                    "published_at": published_at,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "channel_id": cid,
                    "channel_link": f"https://www.youtube.com/channel/{cid}",
                    "channel_title": snippet.get('channelTitle'),
                    "duration_seconds": duration_secs,
                    "view_count":     view_count,
                    "like_count":     int(stats.get('likeCount', '0')),
                    "comment_count":  int(stats.get('commentCount', '0')),
                    "subscriber_count": subs,
                    "channel_authority": _get_channel_authority(subs),
                    "engagement_pct":    _calculate_engagement_pct(stats),
                    "age_days":           round(age_days, 2),
                    "view_velocity":      view_velocity,
                    "snapshot_delta":     snapshot_delta,
                    "tags":                tags[:20],
                    "channel_keywords":    channel_keywords_list[:15],
                    "category_id":         detail_snippet.get('categoryId', ''),
                    "default_language":    detail_snippet.get('defaultLanguage', ''),
                    "top_comments": comments_map.get(vid, []),
                })

            if shorts_filtered:
                print(f"RESEARCH AGENT: Filtered out {shorts_filtered} Shorts from results.")

            if not videos:
                signal_summary = {
                    "scan_timestamp": scan_timestamp,
                    "scan_query": scan_query,
                    "scan_query_reason": query_plan.get("reason"),
                    "scan_query_alternates": query_plan.get("alternates", []),
                    "uploads_last_7d": uploads_last_7d,
                    "uploads_prior_30d": uploads_prior_30d,
                    "upload_surge_ratio": 0.0,
                    "avg_view_velocity": 0.0,
                    "channel_diversity": 0.0,
                    "avg_snapshot_delta": None,
                    "video_sample_size": 0,
                    "top_velocity_video_ids": [],
                    "quota_estimate_units": 320,
                }
                return {"videos": [], "signals": signal_summary}

            uploads_last_7d_daily = uploads_last_7d / 7 if uploads_last_7d else 0.0
            prior_30d_daily = uploads_prior_30d / 30 if uploads_prior_30d else 0.0
            upload_surge_ratio = round(
                uploads_last_7d_daily / prior_30d_daily, 2
            ) if prior_30d_daily > 0 else (round(uploads_last_7d_daily, 2) if uploads_last_7d_daily > 0 else 0.0)

            diversity = round(len({v["channel_id"] for v in videos if v.get("channel_id")}) / len(videos), 2)
            top_velocity_videos = sorted(videos, key=lambda v: v.get("view_velocity", 0), reverse=True)[:5]
            avg_view_velocity = round(
                sum(v.get("view_velocity", 0.0) for v in top_velocity_videos) / len(top_velocity_videos), 2
            ) if top_velocity_videos else 0.0
            avg_snapshot_delta = round(sum(snapshot_gains) / len(snapshot_gains), 2) if snapshot_gains else None

            signal_summary = {
                "scan_timestamp": scan_timestamp,
                "scan_query": scan_query,
                "scan_query_reason": query_plan.get("reason"),
                "scan_query_alternates": query_plan.get("alternates", []),
                "uploads_last_7d": uploads_last_7d,
                "uploads_prior_30d": uploads_prior_30d,
                "upload_surge_ratio": upload_surge_ratio,
                "avg_view_velocity": avg_view_velocity,
                "channel_diversity": diversity,
                "avg_snapshot_delta": avg_snapshot_delta,
                "video_sample_size": len(videos),
                "top_velocity_video_ids": [video["id"] for video in top_velocity_videos],
                "quota_estimate_units": 320,
            }
            _persist_youtube_topic_scan(topic, signal_summary, videos)

            print(
                "RESEARCH AGENT: YouTube signals — "
                f"surge={upload_surge_ratio}, velocity={avg_view_velocity}, "
                f"diversity={diversity}, delta={avg_snapshot_delta}"
            )
            print(f"RESEARCH AGENT: Found {len(videos)} YouTube videos (long-form only).")
            return {"videos": videos, "signals": signal_summary}

        except httpx.HTTPStatusError as e:
            print(f"RESEARCH AGENT: YouTube API error: {e.response.status_code} - {e.response.text}")
            return {
                "videos": [],
                "signals": {
                    "scan_timestamp": scan_timestamp,
                    "uploads_last_7d": 0,
                    "uploads_prior_30d": 0,
                    "upload_surge_ratio": 0.0,
                    "avg_view_velocity": 0.0,
                    "channel_diversity": 0.0,
                    "avg_snapshot_delta": None,
                    "video_sample_size": 0,
                    "top_velocity_video_ids": [],
                    "quota_estimate_units": 320,
                    "error": f"{e.response.status_code}",
                },
            }
        except Exception as e:
            print(f"RESEARCH AGENT: YouTube API call failed: {e}")
            return {
                "videos": [],
                "signals": {
                    "scan_timestamp": scan_timestamp,
                    "uploads_last_7d": 0,
                    "uploads_prior_30d": 0,
                    "upload_surge_ratio": 0.0,
                    "avg_view_velocity": 0.0,
                    "channel_diversity": 0.0,
                    "avg_snapshot_delta": None,
                    "video_sample_size": 0,
                    "top_velocity_video_ids": [],
                    "quota_estimate_units": 320,
                    "error": str(e),
                },
            }


# ════════════════════════════════════════════════════════════
# GOOGLE TRENDS
# ════════════════════════════════════════════════════════════

async def fetch_keyword_trends(topic: str) -> dict:
    """
    Fetches 4 Google Trends signals:
      1. interest_over_time  → trend direction + current score
      2. related_queries     → rising search phrases
      3. interest_by_region  → top countries with highest interest
      4. trending_searches   → real-time check if topic is trending now
    """
    print(f"RESEARCH AGENT: Fetching Google Trends for '{topic}'...")
    loop = asyncio.get_running_loop()

    result = {
        "query_used": topic,
        "queries_tried": [topic],
        "trend_direction":   "unknown",
        "current_score":     0,
        "peak_score":        0,
        "rising_queries":    [],
        "top_queries":       [],
        "top_regions":       [],
        "is_trending_now":   False,
    }

    def _score_trends_result(item: dict) -> tuple:
        return (
            1 if item.get("trend_direction") != "unknown" else 0,
            1 if item.get("current_score", 0) > 0 else 0,
            len(item.get("rising_queries", [])),
            len(item.get("top_regions", [])),
            item.get("peak_score", 0),
        )

    def _run_trends():
        candidates = _build_entity_preserving_query_variants(topic)
        tried = []
        best_result = dict(result)
        best_score = _score_trends_result(best_result)

        for candidate in candidates:
            candidate_result = {
                "query_used": candidate,
                "queries_tried": [],
                "trend_direction": "unknown",
                "current_score": 0,
                "peak_score": 0,
                "rising_queries": [],
                "top_queries": [],
                "top_regions": [],
                "is_trending_now": False,
            }
            tried.append(candidate)
            try:
                pytrends = TrendReq(hl='en-US', tz=0, timeout=(10, 25), retries=2, backoff_factor=0.5)
                kw_list = [candidate]

                pytrends.build_payload(kw_list, timeframe='today 12-m', geo='')
                iot_df = pytrends.interest_over_time()

                if not iot_df.empty and candidate in iot_df.columns:
                    series = iot_df[candidate].dropna()
                    if len(series) >= 2:
                        recent_avg = series.tail(4).mean()
                        earlier_avg = series.head(4).mean()
                        candidate_result['current_score'] = int(series.iloc[-1])
                        candidate_result['peak_score'] = int(series.max())
                        if recent_avg > earlier_avg * 1.15:
                            candidate_result['trend_direction'] = 'rising'
                        elif recent_avg < earlier_avg * 0.85:
                            candidate_result['trend_direction'] = 'declining'
                        else:
                            candidate_result['trend_direction'] = 'stable'

                related = pytrends.related_queries()
                topic_data = related.get(candidate, {})
                if topic_data:
                    rising_df = topic_data.get('rising')
                    top_df = topic_data.get('top')
                    if rising_df is not None and not rising_df.empty:
                        candidate_result['rising_queries'] = rising_df['query'].head(10).tolist()
                    if top_df is not None and not top_df.empty:
                        candidate_result['top_queries'] = top_df['query'].head(10).tolist()

                pytrends.build_payload(kw_list, timeframe='today 3-m', geo='')
                region_df = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=False)
                if not region_df.empty and candidate in region_df.columns:
                    top_regions = (
                        region_df[candidate]
                        .sort_values(ascending=False)
                        .head(5)
                    )
                    candidate_result['top_regions'] = [
                        {"country": country, "score": int(score)}
                        for country, score in top_regions.items()
                        if score > 0
                    ]

                try:
                    trending_df = pytrends.trending_searches(pn='united_states')
                    col = trending_df.columns[0] if not trending_df.empty else None
                    if col is not None:
                        trending_list = trending_df[col].str.lower().tolist()
                        candidate_result['is_trending_now'] = candidate.lower() in trending_list
                except Exception:
                    pass
            except Exception as e:
                print(f"RESEARCH AGENT: Google Trends error for '{candidate}': {e}")

            score = _score_trends_result(candidate_result)
            if score > best_score:
                best_result = candidate_result
                best_score = score
            if best_score[0] == 1 and (best_score[1] == 1 or best_score[2] > 0 or best_score[3] > 0):
                break

        best_result["queries_tried"] = tried
        return best_result

    try:
        trends_data = await loop.run_in_executor(None, _run_trends)
        print(
            f"RESEARCH AGENT: Trends fetched using '{trends_data['query_used']}' — direction: {trends_data['trend_direction']}, "
            f"score: {trends_data['current_score']}, rising_queries: {len(trends_data['rising_queries'])}, "
            f"trending_now: {trends_data['is_trending_now']}"
        )
        return trends_data
    except Exception as e:
        print(f"RESEARCH AGENT: fetch_keyword_trends failed: {e}")
        return result


def _fallback_angle_proof(
    angle: dict,
    youtube_results: list[dict],
    youtube_signals: dict,
    trends_data: dict,
) -> list[str]:
    proofs: list[str] = []
    keywords = [str(k).lower() for k in angle.get("suggested_keywords", []) if str(k).strip()]
    angle_text = f"{angle.get('angle', '')} {angle.get('reasoning', '')}".lower()

    ranked_videos: list[tuple[int, int]] = []
    for idx, item in enumerate(youtube_results):
        haystack = " ".join(
            [
                str(item.get("title", "")),
                str(item.get("description", "")),
                " ".join(item.get("tags", [])[:10]),
                " ".join(item.get("channel_keywords", [])[:10]),
            ]
        ).lower()
        score = sum(1 for kw in keywords if kw and kw in haystack)
        if not score:
            score = sum(1 for kw in keywords if kw and kw in angle_text and kw in haystack)
        ranked_videos.append((score, idx))

    ranked_videos.sort(key=lambda pair: (pair[0], youtube_results[pair[1]].get("view_velocity", 0)), reverse=True)
    best_video_idx = ranked_videos[0][1] if ranked_videos else None
    if best_video_idx is not None and youtube_results:
        proofs.append(f"[V{best_video_idx}]")

    if (
        trends_data.get("trend_direction") == "rising"
        or trends_data.get("is_trending_now")
        or (youtube_signals.get("upload_surge_ratio", 0) or 0) >= 1.3
    ):
        proofs.append("[T]")

    if not proofs and youtube_results:
        proofs.append("[V0]")
    return proofs[:3]


def _hydrate_angle_proofs(
    research_brief: dict,
    social_results: list[dict],
    youtube_results: list[dict],
    news_results: list[dict],
    trends_data: dict,
    youtube_signals: dict,
) -> None:
    for angle in research_brief.get("strategic_angles", []):
        raw_proof = angle.get("proof", [])
        if not isinstance(raw_proof, list):
            raw_proof = []

        normalized_ids = []
        for pid in raw_proof:
            if isinstance(pid, str) and re.match(r"^\[(S|V|N)\d+\]$", pid):
                normalized_ids.append(pid)
            elif pid == "[T]":
                normalized_ids.append(pid)

        if not normalized_ids:
            normalized_ids = _fallback_angle_proof(angle, youtube_results, youtube_signals, trends_data)

        source_data = []
        for pid in normalized_ids:
            try:
                if pid.startswith("[S"):
                    idx = int(pid[2:-1])
                    item = social_results[idx]
                    source_data.append({
                        "type": "social",
                        "title": item['title'],
                        "url": item['href'],
                        "snippet": item['body'],
                    })
                elif pid.startswith("[V"):
                    idx = int(pid[2:-1])
                    item = youtube_results[idx]
                    source_data.append({"type": "youtube_video", **item})
                elif pid.startswith("[N"):
                    idx = int(pid[2:-1])
                    item = news_results[idx]
                    source_data.append({
                        "type": "news",
                        "title": item['title'],
                        "url": item['url'],
                        "snippet": item['body'],
                    })
                elif pid == "[T]":
                    source_data.append({
                        "type": "google_trends",
                        "trend_direction": trends_data.get("trend_direction"),
                        "current_score": trends_data.get("current_score"),
                        "peak_score": trends_data.get("peak_score"),
                        "is_trending_now": trends_data.get("is_trending_now"),
                        "rising_queries": trends_data.get("rising_queries", [])[:5],
                        "top_queries": trends_data.get("top_queries", [])[:5],
                        "top_regions": trends_data.get("top_regions", [])[:5],
                    })
            except Exception:
                continue

        if not source_data and youtube_results:
            source_data.append({"type": "youtube_video", **youtube_results[0]})

        angle["proof"] = source_data


RESEARCH_SYNTHESIS_PROMPT = """
You are a world-class YouTube Content Strategist. Analyze real-time market data about: '{topic}'.
Today's date: {current_date}.

Use ALL data sources below:
- YouTube videos → what content is winning, engagement levels, comment sentiment
- YouTube market signals → upload surge, view velocity, diversity, day-over-day gains
- Google Trends   → timing signal, rising search phrases, regional demand
- Community discussions + news → context and angles

1. **Topic Analysis:** Calculate average_views, views_range, average_engagement_pct, channel_niche, comment_themes.
   Also include trend_direction, top competitor SEO tags, and whether YouTube momentum is accelerating.
2. **Strategic Angles:** Identify 3-4 distinct angles. Each needs:
   - reasoning (why this angle works based on data)
   - proof (MUST include 1-3 source IDs, and each angle MUST contain at least one [V...] or [T])
   - strategic_timing (e.g. "Post now — topic is rising fast" or "Wait for next news cycle")
   - suggested_keywords: 3-5 keywords from rising_queries or video tags
3. **Audience Profile:** age_group, interests, sentiment, top_regions (from Trends).

RAW DATA:
---
[YouTube Videos (ID | Title | Channel | Views | Eng% | Subs | Authority | Published | Tags | Comments)]
{youtube_videos}
---
[YouTube Market Signals]
Scan Query: {scan_query}
Upload Surge Ratio: {upload_surge_ratio}
Uploads Last 7d: {uploads_last_7d}
Uploads Prior 30d: {uploads_prior_30d}
Average View Velocity (top 5): {avg_view_velocity}
Channel Diversity: {channel_diversity}
Average Snapshot Delta/day: {avg_snapshot_delta}
---
[Google Trends]
Trend Direction: {trend_direction} | Current Score: {trend_score}/100 | Peak Score: {trend_peak}/100
Trending Now: {is_trending_now}
Rising Queries (breakout phrases): {rising_queries}
Top Queries: {top_queries}
Top Regions: {top_regions}
---
[Community Discussions]
{social_threads}
---
[News Headlines]
{news_headlines}
---

Return ONLY a valid JSON object:
{{
  "topic_analysis": {{
    "average_views": null,
    "views_range": null,
    "average_engagement_pct": null,
    "channel_niche": null,
    "comment_themes": [],
    "youtube_momentum": "",
    "trend_direction": "",
    "top_competitor_tags": []
  }},
  "strategic_angles": [
    {{
      "angle": "",
      "reasoning": "",
      "proof": ["[V0]", "[T]"],
      "strategic_timing": "",
      "suggested_keywords": []
    }}
  ],
  "audience_profile": {{
    "age_group": "",
    "interests": [],
    "sentiment": [],
    "top_regions": []
  }}
}}
"""


@app.post("/research-topic")
async def research_topic(request: PromptRequest):
    total_start_time = time.time()
    print(f"RESEARCH AGENT: Topic: {request.topic}")

    try:
        social_results, youtube_scan, news_results, trends_data = await asyncio.gather(
            fetch_social_pulse(request.topic),
            fetch_youtube_trends(request.topic),
            fetch_news_analysis(request.topic),
            fetch_keyword_trends(request.topic),
        )
        youtube_results = youtube_scan.get("videos", [])
        youtube_signals = youtube_scan.get("signals", {})

        social_titles_for_prompt = "\n".join([
            f"[S{i}]: {item['title']}" for i, item in enumerate(social_results)
        ])
        youtube_videos_for_prompt = "\n".join([
            f"[V{i}]: {item['title']} | {item['channel_title']} | Views: {item['view_count']} | "
            f"Eng%: {item['engagement_pct']} | Subs: {item['subscriber_count']} | "
            f"Authority: {item['channel_authority']} | Published: {item['published_at']} | "
            f"Velocity: {item.get('view_velocity', 0)} views/day | "
            f"Delta/day: {item.get('snapshot_delta')} | "
            f"Tags: {item.get('tags', [])[:8]} | Comments: {item['top_comments']}"
            for i, item in enumerate(youtube_results)
        ])
        news_headlines_for_prompt = "\n".join([
            f"[N{i}]: {item['title']}" for i, item in enumerate(news_results)
        ])

        synthesis_prompt = RESEARCH_SYNTHESIS_PROMPT.format(
            topic=request.topic,
            current_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            youtube_videos=youtube_videos_for_prompt,
            scan_query=youtube_signals.get('scan_query', request.topic),
            upload_surge_ratio=youtube_signals.get('upload_surge_ratio', 0.0),
            uploads_last_7d=youtube_signals.get('uploads_last_7d', 0),
            uploads_prior_30d=youtube_signals.get('uploads_prior_30d', 0),
            avg_view_velocity=youtube_signals.get('avg_view_velocity', 0.0),
            channel_diversity=youtube_signals.get('channel_diversity', 0.0),
            avg_snapshot_delta=youtube_signals.get('avg_snapshot_delta', None),
            social_threads=social_titles_for_prompt,
            news_headlines=news_headlines_for_prompt,
            trend_direction=trends_data.get('trend_direction', 'unknown'),
            trend_score=trends_data.get('current_score', 0),
            trend_peak=trends_data.get('peak_score', 0),
            is_trending_now=trends_data.get('is_trending_now', False),
            rising_queries=trends_data.get('rising_queries', []),
            top_queries=trends_data.get('top_queries', []),
            top_regions=trends_data.get('top_regions', []),
        )

        print("RESEARCH AGENT: Synthesizing with Groq...")
        cc = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model=GROQ_GENERATION_MODEL,
            response_format={"type": "json_object"},
        )
        response_text = cc.choices[0].message.content

        try:
            research_brief = json.loads(response_text)
        except json.JSONDecodeError:
            print(f"RESEARCH AGENT: Failed to parse JSON: {response_text}")
            raise HTTPException(status_code=500, detail="AI failed to generate valid research.")

        _hydrate_angle_proofs(
            research_brief,
            social_results,
            youtube_results,
            news_results,
            trends_data,
            youtube_signals,
        )

        print(f"--- /research-topic took {time.time() - total_start_time:.2f}s ---")
        return {
            **research_brief,
            "youtube_market_signals": youtube_signals,
            "youtube_videos": youtube_results,
            "trends": trends_data,
        }

    except Exception as e:
        print(f"RESEARCH AGENT error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during research.")


# ════════════════════════════════════════════════════════════
# SEO AGENT
# ════════════════════════════════════════════════════════════

SEO_SYNTHESIS_PROMPT = """
You are an expert YouTube SEO Analyst and Title Strategist.

VIDEO ANGLE: "{angle}"
AUDIENCE PROFILE: {audience_profile}

COMPETITIVE DATA:
- Top 5 YouTube Titles: {competing_titles}
- Top 5 PAA Questions: {paa_questions}

Return ONLY a valid JSON object:
{{
  "search_intent_type": "",
  "ctr_potential": "Low" | "Medium" | "High",
  "justification": "",
  "recommended_titles": [],
  "key_questions_to_answer": [],
  "related_keywords": []
}}
"""


async def run_seo_analysis(angle_data: dict, audience_profile: dict) -> dict:
    angle      = angle_data.get("angle", "Unknown Angle")
    proof_data = angle_data.get("proof", [])
    print(f"SEO AGENT: Analyzing angle: '{angle}'")
    loop = asyncio.get_running_loop()

    competing_titles, paa_questions = [], []
    try:
        with DDGS(timeout=10) as ddgs:
            video_results = await loop.run_in_executor(
                None,
                lambda: ddgs.videos(f"{angle}", region='wt-wt', timelimit='m', max_results=5),
            )
            if video_results:
                competing_titles = [v['title'] for v in video_results]

            answer_results = await loop.run_in_executor(
                None,
                lambda: ddgs.answers(f"{angle}", region='wt-wt'),
            )
            if answer_results:
                paa_questions = [a['question'] for a in answer_results[:5]]
    except Exception as e:
        print(f"SEO AGENT: Scraping failed for '{angle}': {e}")

    seo_prompt = SEO_SYNTHESIS_PROMPT.format(
        angle=angle,
        audience_profile=json.dumps(audience_profile),
        competing_titles=str(competing_titles),
        paa_questions=str(paa_questions),
    )

    try:
        cc = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": seo_prompt}],
            model=GROQ_GENERATION_MODEL,
            response_format={"type": "json_object"},
        )
        plan = json.loads(cc.choices[0].message.content)
        plan['angle'] = angle
        plan['proof'] = proof_data
        print(f"SEO AGENT: Battle plan created for '{angle}'")
        return plan
    except Exception as e:
        print(f"SEO AGENT: AI synthesis failed for '{angle}': {e}")
        return {
            "angle": angle, "proof": proof_data,
            "error": f"AI synthesis failed: {e}",
            "ctr_potential": "Unknown", "justification": "AI synthesis failed.",
            "recommended_titles": [], "key_questions_to_answer": [], "related_keywords": [],
        }


@app.post("/seo-agent")
async def seo_agent(request: ResearchBriefInput):
    total_start_time = time.time()
    print("SEO AGENT: Received request.")

    try:
        tasks = [
            run_seo_analysis(angle_data=a, audience_profile=request.audience_profile)
            for a in request.strategic_angles
        ]
        seo_battle_plans = await asyncio.gather(*tasks)

        print(f"--- /seo-agent took {time.time() - total_start_time:.2f}s ---")
        return {
            "topic_analysis":   request.topic_analysis,
            "audience_profile": request.audience_profile,
            "seo_battle_plans": seo_battle_plans,
        }

    except Exception as e:
        print(f"SEO AGENT error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during SEO analysis.")
