# ============================================================
# main.py — Migrated to google-genai (new unified SDK)
# pip install google-genai  (replaces google-generativeai)
# ============================================================

from fastapi import Depends, HTTPException, status, Request, Header, BackgroundTasks
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from postgrest.exceptions import APIError
from gotrue.types import User
from openai import AsyncOpenAI
from auth_dependencies import get_current_user
from auth_dependencies import login_user

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
import nltk
import razorpay

from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
from ddgs import DDGS
from readability import Document

load_dotenv()

# ── NLTK data ────────────────────────────────────────────────
project_root = os.path.dirname(os.path.abspath(__file__))
nltk_data_dir = os.path.join(project_root, 'nltk_data')
nltk.data.path.insert(0, nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    print("NLTK 'punkt' and 'punkt_tab' data found successfully.")
except LookupError as e:
    print(f"!!! CRITICAL NLTK DATA ERROR: {e} !!!")

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
                    break  # don't retry 404s
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
    """Background task: chunk → embed → upsert to Supabase."""
    print(f"BACKGROUND TASK: Starting upload for '{article_title[:30]}...'")
    try:
        raw_chunks = chunk_text(article_text)
        chunks = [c for c in raw_chunks if c and not c.isspace()]
        if not chunks:
            print("BACKGROUND TASK: No valid chunks.")
            return

        # NEW SDK: sync embed — returns EmbedContentResponse
        embed_response = gemini_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunks,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768),
        )
        # embed_response.embeddings is a list of ContentEmbedding objects
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

    # Semaphore limits concurrent Playwright instances — prevents RAM exhaustion in prod
    async with _playwright_semaphore:
        loop = asyncio.get_running_loop()
        # Run in thread pool so heavy browser process never blocks the event loop
        return await loop.run_in_executor(None, lambda: asyncio.run(_run()))


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
    with DDGS(timeout=20) as ddgs:
        for keyword in keywords:
            results = list(ddgs.text(keyword, region='wt-wt', max_results=3))
            if results:
                # Take top result; keep snippet for Tier 3 fallback
                top = results[0]
                urls_to_scrape.append((top['href'], top.get('body', '')))

    # Deduplicate by URL
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
    print("--- LIGHT WEB SCRAPE: Starting lightweight news search... ---")
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
        print(f"--- WEB TASK: Error during news scraping: {e} ---")
        return []


async def get_db_context(topic: str) -> list[dict]:
    """HyDE: generate a hypothetical doc with Groq, embed with Gemini, search Supabase."""
    print("--- DB TASK: Starting HyDE database search (using Groq + Gemini embed)... ---")
    try:
        hyde_prompt = f"""
        Write a short, factual, encyclopedia-style paragraph that provides a direct answer
        to the following topic. Be concise and include key terms.

        Topic: "{topic}"
        """
        # Generate hypothetical document with Groq
        chat_completion = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": hyde_prompt}],
            model=GROQ_GENERATION_MODEL,
        )
        hypothetical_document = chat_completion.choices[0].message.content
        print(f"--- DB TASK: HyDE doc: {hypothetical_document[:100]}...")

        # Embed with NEW SDK (run in executor to avoid blocking the event loop)
        loop = asyncio.get_running_loop()
        embed_response = await loop.run_in_executor(
            None,
            lambda: gemini_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=hypothetical_document,
                config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768),
            ),
        )
        # Single-string embed → first embedding's values
        query_embedding = embed_response.embeddings[0].values

        # Supabase RPC
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


# ════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════

@app.get("/")
async def read_root():
    return {"status": "Welcome"}
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    return await login_user(form_data)

# ── /process-topic ───────────────────────────────────────────

@app.post("/process-topic")
async def process_topic(request: PromptRequest, background_tasks: BackgroundTasks):
    total_start_time = time.time()
    print(f"Received topic: {request.topic}")

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
            print(f"--- DB task finished early. Found {len(db_results)} docs. ---")

        if len(db_results) >= 3:
            source_of_context = "DATABASE_WITH_NEWS"
            new_articles = await get_latest_news_context(request.topic, scraped_urls)
        else:
            print("--- DB MISS or SLOW: Deep web scrape. ---")
            source_of_context = "DEEP_SCRAPE"

            keyword_prompt = f"""
            Your ONLY task is to generate 3 diverse search engine keyword phrases for: '{request.topic}'.
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
            base_keywords = keywords_in_quotes if keywords_in_quotes else [
                kw.strip() for kw in raw_text.strip().split('\n') if kw.strip()
            ]
            targeted_keywords = base_keywords  # Reddit removed — blocks all scrapers unconditionally
            print(f"Targeted keywords: {targeted_keywords}")
            new_articles = await deep_search_and_scrape(targeted_keywords, scraped_urls)

        if not db_task.done():
            print("--- Waiting for DB task... ---")
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

        Use the provided research:
        - FOUNDATIONAL KNOWLEDGE: deep context, facts, historical background.
        - LATEST NEWS: fresh, timely angles (current date is today).

        RULES:
        1. For each idea: provide 'TITLE' and 'DESCRIPTION'.
        2. Each DESCRIPTION MUST be 90-110 words.
        3. Separate each complete idea with '---'.
        4. NO introductory text, explanations, or anything else.

        EXAMPLE FORMAT:
        TITLE: This Is Why Everyone Is Suddenly Talking About [Topic]
        DESCRIPTION: In this video, we uncover the shocking truth behind [Topic]...
        ---
        TITLE: The Hidden Truth Behind [Related Concept]
        DESCRIPTION: Everyone thinks they understand [Related Concept], but they're wrong...
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

        # Parse titles + descriptions
        final_ideas = []
        final_descriptions = []
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

        print(f"Generated {len(final_ideas)} ideas.")
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
        # ── Step 1: Gather context ────────────────────────────
        db_task = asyncio.create_task(get_db_context(request.topic))
        await asyncio.sleep(11)

        db_results = []
        new_articles = []
        scraped_urls = set()
        base_keywords = []

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

        # ── Step 2: Merge context ─────────────────────────────
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

        # OpenRouter generation with retry on 429
        script_text = await openrouter_generate([{"role": "user", "content": script_prompt}])

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
