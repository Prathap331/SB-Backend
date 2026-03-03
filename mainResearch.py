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
  6. EMBEDDING_MODEL kept as text-embedding-004 (768-dim, matches existing Supabase DB)
"""

from fastapi import Depends, HTTPException, status, Request, Header, BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from postgrest.exceptions import APIError
from supabase_auth.types import User
from openai import AsyncOpenAI
from auth_dependencies import get_current_user
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
import razorpay
import datetime

from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
from ddgs import DDGS
from readability import Document
from pytrends.request import TrendReq

load_dotenv()

# ── NLTK ─────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.abspath(__file__))
nltk_data_dir = os.path.join(project_root, 'nltk_data')
nltk.data.path.insert(0, nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    print("NLTK data found successfully.")
except LookupError as e:
    print(f"!!! CRITICAL NLTK DATA ERROR: {e} !!!")

# ── Razorpay ─────────────────────────────────────────────────
# FIX: All Razorpay vars declared together at the top (RAZORPAY_WEBHOOK_SECRET
# was previously declared AFTER the webhook endpoint — caused NameError at runtime)
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")

if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
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
# LLM generation has moved to OpenRouter
gemini_client = genai.Client(api_key=google_api_key)

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

# Primary model + backup model
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
                    wait = 3
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


def add_scraped_data_to_db(article_title: str, article_text: str, article_url: str):
    """Background task: chunk → embed (new SDK) → insert to Supabase."""
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

        documents_to_insert = [
            {
                "content": chunk,
                "embedding": embeddings[i],
                "source_title": article_title,
                "source_url": article_url,
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
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                await page.wait_for_selector("body", timeout=10000)
                html = await page.content()
                return _extract_text_from_html(html)
            finally:
                await browser.close()

    async with _playwright_semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: asyncio.run(_run()))


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
    print(f"Scraping: {url}")

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
    """HyDE: generate hypothetical doc with Groq, embed with Gemini, search Supabase."""
    print("--- DB TASK: Starting HyDE search (Groq + Gemini embed)... ---")
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
        print(f"--- DB TASK: HyDE doc: {hypothetical_document[:100]}...")

        # NEW SDK: embed in executor (sync call, avoid blocking event loop)
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

        db_results_response = await loop.run_in_executor(
            None,
            lambda: supabase.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.65,
                    'match_count': 5,
                }
            ).execute()
        )
        print(f"--- DB TASK: Found {len(db_results_response.data)} documents. ---")
        return db_results_response.data

    except Exception as e:
        print(f"--- DB TASK: Error: {e} ---")
        return []


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
# FIX: PromptRequest was defined twice in the original — deduplicated here

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
        db_task = asyncio.create_task(get_db_context(request.topic))
        await asyncio.sleep(11)

        db_results = []
        new_articles = []
        scraped_urls = set()
        base_keywords = []
        source_of_context = ""

        if db_task.done():
            db_results = db_task.result()

        if len(db_results) >= 3:
            source_of_context = "DATABASE_WITH_NEWS"
            new_articles = await get_latest_news_context(request.topic, scraped_urls)
        else:
            print("--- DB MISS: Deep web scrape. ---")
            source_of_context = "DEEP_SCRAPE"

            keyword_prompt = f"""
            Generate 3 diverse search engine keyword phrases for: '{request.topic}'.
            Rules: ONLY the 3 phrases, no numbers/markdown/intro, one per line.
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
            base_keywords = keywords_in_quotes if keywords_in_quotes else [
                kw.strip() for kw in raw_text.strip().split('\n') if kw.strip()
            ]
            targeted_keywords = base_keywords  # Reddit removed — blocks all scrapers unconditionally
            new_articles = await deep_search_and_scrape(targeted_keywords, scraped_urls)

        if not db_task.done():
            db_results = await db_task

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
                    article['title'], article['text'], article['url']
                )

        if not db_context and not web_context:
            return {"error": "Could not find any information."}

        final_prompt = f"""
        You are an expert YouTube title strategist and scriptwriter.
        Generate 4 distinct, attention-grabbing video titles for: "{request.topic}",
        with a corresponding description for each.

        RULES:
        1. For each idea: provide 'TITLE' and 'DESCRIPTION'.
        2. Each DESCRIPTION MUST be 90-110 words.
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

        # OpenRouter generation with retry on 429
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
        db_task = asyncio.create_task(get_db_context(request.topic))
        await asyncio.sleep(11)

        db_results, new_articles, scraped_urls, base_keywords = [], [], set(), []

        if db_task.done():
            db_results = db_task.result()

        if len(db_results) >= 3:
            print("--- DB HIT: Light news scrape. ---")
            new_articles = await get_latest_news_context(request.topic, scraped_urls)
        else:
            print("--- DB MISS: Deep web scrape. ---")
            keyword_prompt = f"""
            Generate 3 diverse search engine keyword phrases for: '{request.topic}'.
            Rules: ONLY the 3 phrases, no numbers/markdown/intro, one per line.
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
            base_keywords = keywords_in_quotes if keywords_in_quotes else [
                kw.strip() for kw in raw_text.strip().split('\n') if kw.strip()
            ]
            targeted_keywords = base_keywords  # Reddit removed — blocks all scrapers unconditionally
            new_articles = await deep_search_and_scrape(targeted_keywords, scraped_urls)

        if not db_task.done():
            db_results = await db_task

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
                    article['title'], article['text'], article['url']
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

        # OpenRouter generation with retry on 429
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
    except razorpay.errors.SignatureVerificationError as e:
        print(f"Webhook signature failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")
    except Exception as e:
        print(f"Webhook verification error: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing error.")

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


async def fetch_youtube_trends(topic: str) -> list[dict]:
    if not YOUTUBE_API_KEY:
        print("RESEARCH AGENT: YouTube API not configured. Skipping.")
        return []

    print(f"RESEARCH AGENT: Fetching YouTube trends for '{topic}'...")
    six_months_ago = (
        datetime.datetime.now() - datetime.timedelta(days=180)
    ).isoformat("T") + "Z"

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            search_params = {
                'key': YOUTUBE_API_KEY, 'q': topic, 'part': 'snippet',
                'type': 'video', 'order': 'relevance', 'videoDuration': 'medium',
                'publishedAfter': six_months_ago, 'maxResults': 10,
            }
            search_response = await client.get(YOUTUBE_SEARCH_URL, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()

            video_ids, channel_ids, video_snippet_map = [], set(), {}
            for item in search_data.get('items', []):
                vid = item['id']['videoId']
                cid = item['snippet']['channelId']
                video_ids.append(vid)
                channel_ids.add(cid)
                video_snippet_map[vid] = item['snippet']

            if not video_ids:
                print(f"RESEARCH AGENT: No YouTube videos found for '{topic}'.")
                return []

            async def _get_video_stats(ids_str: str):
                resp = await client.get(YOUTUBE_VIDEOS_URL,
                                        params={'key': YOUTUBE_API_KEY, 'part': 'statistics', 'id': ids_str})
                resp.raise_for_status()
                return resp.json()

            async def _get_channel_stats(ids_str: str):
                resp = await client.get(YOUTUBE_CHANNELS_URL,
                                        params={'key': YOUTUBE_API_KEY, 'part': 'statistics', 'id': ids_str})
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
                    raw_comments = [
                        item['snippet']['topLevelComment']['snippet']['textOriginal']
                        for item in resp.json().get('items', [])
                    ]
                    if not raw_comments:
                        return vid_id, []

                    analysis_prompt = f"""
                    Identify the Top 3-5 recurring themes/sentiments in these YouTube comments.
                    Return ONLY a valid JSON object: {{"themes": ["theme1", "theme2", ...]}}

                    COMMENTS: {json.dumps(raw_comments)}
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

            results = await asyncio.gather(
                _get_video_stats(video_id_str),
                _get_channel_stats(channel_id_str),
                *comments_tasks,
                return_exceptions=True,
            )

            stats_response   = results[0] if not isinstance(results[0], Exception) else {}
            channels_response = results[1] if not isinstance(results[1], Exception) else {}
            comment_results  = [r for r in results[2:] if not isinstance(r, Exception)]

            if isinstance(results[0], Exception):
                print(f"RESEARCH AGENT: CRITICAL — video stats error: {results[0]}")
                return []
            if isinstance(results[1], Exception):
                print(f"RESEARCH AGENT: CRITICAL — channel stats error: {results[1]}")
                return []

            stats_map    = {item['id']: item['statistics'] for item in stats_response.get('items', [])}
            subs_map     = {item['id']: int(item['statistics'].get('subscriberCount', '0'))
                           for item in channels_response.get('items', [])}
            comments_map = {vid_id: themes for vid_id, themes in comment_results}

            videos = []
            for vid in video_ids:
                snippet  = video_snippet_map.get(vid, {})
                cid      = snippet.get('channelId')
                stats    = stats_map.get(vid, {})
                subs     = subs_map.get(cid, 0)
                videos.append({
                    "id": vid, "title": snippet.get('title'),
                    "description": snippet.get('description'),
                    "body": snippet.get('description'),
                    "published_at": snippet.get('publishedAt'),
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "channel_id": cid,
                    "channel_link": f"https://www.youtube.com/channel/{cid}",
                    "channel_title": snippet.get('channelTitle'),
                    "view_count": int(stats.get('viewCount', '0')),
                    "like_count": int(stats.get('likeCount', '0')),
                    "comment_count": int(stats.get('commentCount', '0')),
                    "subscriber_count": subs,
                    "channel_authority": _get_channel_authority(subs),
                    "engagement_pct": _calculate_engagement_pct(stats),
                    "top_comments": comments_map.get(vid, []),
                })

            print(f"RESEARCH AGENT: Found {len(videos)} YouTube videos.")
            return videos

        except httpx.HTTPStatusError as e:
            print(f"RESEARCH AGENT: YouTube API error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            print(f"RESEARCH AGENT: YouTube API call failed: {e}")
            return []


RESEARCH_SYNTHESIS_PROMPT = """
You are a world-class YouTube Content Strategist. Analyze real-time market data about: '{topic}'.
Today's date: {current_date}.

Analyze the YouTube videos to find what's working. Use discussions and news as supporting context.

1. **Topic Analysis:** Calculate average_views, views_range, average_engagement_pct, channel_niche, comment_themes.
2. **Strategic Angles:** Identify 3-4 distinct angles inspired by YouTube data. Each needs reasoning, proof (at least one [V...] ID), and strategic_timing.
3. **Audience Profile:** Describe age_group, interests, sentiment.

RAW DATA:
---
[YouTube Videos (ID | Title | Channel | Views | Eng % | Subs | Authority | Published | Comments)]
{youtube_videos}
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
    "comment_themes": []
  }},
  "strategic_angles": [
    {{
      "angle": "",
      "reasoning": "",
      "proof": [],
      "strategic_timing": ""
    }}
  ],
  "audience_profile": {{
    "age_group": "",
    "interests": [],
    "sentiment": []
  }}
}}
"""


@app.post("/research-topic")
async def research_topic(request: PromptRequest):
    total_start_time = time.time()
    print(f"RESEARCH AGENT: Topic: {request.topic}")

    try:
        social_results, youtube_results, news_results = await asyncio.gather(
            fetch_social_pulse(request.topic),
            fetch_youtube_trends(request.topic),
            fetch_news_analysis(request.topic),
        )

        social_titles_for_prompt = "\n".join([
            f"[S{i}]: {item['title']}" for i, item in enumerate(social_results)
        ])
        youtube_videos_for_prompt = "\n".join([
            f"[V{i}]: {item['title']} | {item['channel_title']} | Views: {item['view_count']} | "
            f"Eng%: {item['engagement_pct']} | Subs: {item['subscriber_count']} | "
            f"Authority: {item['channel_authority']} | Published: {item['published_at']} | "
            f"Comments: {item['top_comments']}"
            for i, item in enumerate(youtube_results)
        ])
        news_headlines_for_prompt = "\n".join([
            f"[N{i}]: {item['title']}" for i, item in enumerate(news_results)
        ])

        synthesis_prompt = RESEARCH_SYNTHESIS_PROMPT.format(
            topic=request.topic,
            current_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            youtube_videos=youtube_videos_for_prompt,
            social_threads=social_titles_for_prompt,
            news_headlines=news_headlines_for_prompt,
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

        # Hydrate proof IDs with full source data
        for angle in research_brief.get("strategic_angles", []):
            source_data = []
            for pid in angle.get("proof", []):
                try:
                    if pid.startswith("[S"):
                        idx  = int(pid[2:-1])
                        item = social_results[idx]
                        source_data.append({"type": "social", "title": item['title'],
                                            "url": item['href'], "snippet": item['body']})
                    elif pid.startswith("[V"):
                        idx  = int(pid[2:-1])
                        item = youtube_results[idx]
                        source_data.append({"type": "youtube_video", **item})
                    elif pid.startswith("[N"):
                        idx  = int(pid[2:-1])
                        item = news_results[idx]
                        source_data.append({"type": "news", "title": item['title'],
                                            "url": item['url'], "snippet": item['body']})
                except Exception:
                    pass
            angle["proof"] = source_data

        print(f"--- /research-topic took {time.time() - total_start_time:.2f}s ---")
        return research_brief

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
