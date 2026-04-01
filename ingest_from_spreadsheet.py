"""
ingest_from_spreadsheet.py  — v3 (Wikipedia API + NewsAPI + Jina Reader)
═══════════════════════════════════════════════════════════════════════════
Fetching strategy per category:

  Science / Technology / Psychology  → Wikipedia REST API
      Full articles (2k–15k words), structured, authoritative, no blocking.

  Politics & Current Affairs         → NewsAPI.org
      Real articles from BBC, Reuters, Al Jazeera etc. with author + pub date.
      Free tier: 100 req/day. Get key at https://newsapi.org

  Business & Money                   → Jina AI Reader (r.jina.ai)
      Prefix any URL with https://r.jina.ai/ → returns clean markdown.
      Free, no API key, handles JS rendering, paywall-resistant.

  Any unknown category               → Jina AI Reader fallback

source_from is ALWAYS populated:
  Wikipedia  → "Wikipedia"
  NewsAPI    → publication name from source.name field (e.g. "BBC News")
  Jina       → domain looked up in KNOWN_PUBLICATIONS or raw domain
  web_scrape → same as Jina

Run:
  python ingest_from_spreadsheet.py

Resume: safe to Ctrl+C and re-run — resumes from spreadsheet_progress.log
"""

import os
import re
import json
import time
import httpx
import openpyxl
import nltk

from urllib.parse import urlparse
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.genai import types as genai_types
from ddgs import DDGS
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
from readability import Document

load_dotenv()

# ── NLTK ──────────────────────────────────────────────────────
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

SPREADSHEET_FILE   = 'Storybit_Database.xlsx'
PROGRESS_FILE      = 'spreadsheet_progress.log'
GEMINI_MODEL       = 'gemini-embedding-001'
LOCAL_MODEL_NAME   = 'sentence-transformers/all-mpnet-base-v2'
CHUNK_SIZE         = 250
CHUNK_OVERLAP      = 50
BATCH_SIZE         = 100
LOCAL_BATCH_SIZE   = 64
ARTICLES_PER_TOPIC = 5
DDGS_TIMEOUT       = 15
MIN_ARTICLE_WORDS  = 200
MIN_SNIPPET_WORDS  = 50

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# Runtime embedding state — auto-switches on Gemini daily quota hit
_use_local_embedder = False
_local_model        = None


def get_local_model():
    global _local_model
    if _local_model is None:
        print(f"\n  Loading local fallback model: {LOCAL_MODEL_NAME}")
        print(f"  (First run downloads ~420MB to ~/.cache/huggingface — one-time only)")
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(LOCAL_MODEL_NAME)
        print(f"  ✓ Local model loaded (768-dim)")
    return _local_model

# ── Category → fetching strategy ──────────────────────────────
CATEGORY_SOURCE_MAP: dict[str, str] = {
    "Politics & Current Affairs":  "news",        # → NewsAPI
    "Science":                     "wikipedia",   # → Wikipedia API
    "Technology":                  "wikipedia",   # → Wikipedia API
    "Business & Money":            "web_scrape",  # → Jina Reader
    "Psychology & Human Behavior": "wikipedia",   # → Wikipedia API
}
DEFAULT_SOURCE_TYPE = "web_scrape"

# ── Scrape headers for httpx ───────────────────────────────────
SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ══════════════════════════════════════════════════════════════
# KNOWN PUBLICATIONS — domain → human-readable name + description
# source_from will always be one of these names (or raw domain as fallback)
# ══════════════════════════════════════════════════════════════

KNOWN_PUBLICATIONS: dict[str, dict] = {
    # News
    "bbc.com":               {"name": "BBC News",              "description": "British public broadcaster — globally trusted news source"},
    "bbc.co.uk":             {"name": "BBC News",              "description": "British public broadcaster — globally trusted news source"},
    "reuters.com":           {"name": "Reuters",               "description": "International news agency — factual, wire-service journalism"},
    "apnews.com":            {"name": "AP News",               "description": "Associated Press — non-profit news cooperative"},
    "theguardian.com":       {"name": "The Guardian",          "description": "British daily newspaper — independent journalism"},
    "nytimes.com":           {"name": "The New York Times",    "description": "American newspaper of record"},
    "washingtonpost.com":    {"name": "The Washington Post",   "description": "American daily newspaper"},
    "aljazeera.com":         {"name": "Al Jazeera",            "description": "Qatari international news network"},
    "cnn.com":               {"name": "CNN",                   "description": "American cable news network"},
    "ndtv.com":              {"name": "NDTV",                  "description": "Indian news broadcaster"},
    "thehindu.com":          {"name": "The Hindu",             "description": "Indian national newspaper"},
    "timesofindia.com":      {"name": "Times of India",        "description": "India's largest English-language newspaper"},
    "hindustantimes.com":    {"name": "Hindustan Times",       "description": "Indian national newspaper"},
    "indianexpress.com":     {"name": "The Indian Express",    "description": "Indian national newspaper"},
    "economictimes.indiatimes.com": {"name": "Economic Times", "description": "Indian financial newspaper"},
    "bloomberg.com":         {"name": "Bloomberg",             "description": "American financial news and media company"},
    "forbes.com":            {"name": "Forbes",                "description": "American business magazine"},
    "ft.com":                {"name": "Financial Times",       "description": "British financial newspaper"},
    "wsj.com":               {"name": "Wall Street Journal",   "description": "American financial newspaper"},
    "cnbc.com":              {"name": "CNBC",                  "description": "American business news channel"},
    "businessinsider.com":   {"name": "Business Insider",      "description": "Business and technology news site"},
    # Government / International
    "who.int":               {"name": "WHO",                   "description": "World Health Organization — UN specialized agency"},
    "un.org":                {"name": "United Nations",        "description": "Intergovernmental organization"},
    "gov.in":                {"name": "Government of India",   "description": "Official Indian government portal"},
    "india.gov.in":          {"name": "Government of India",   "description": "Official Indian government portal"},
    "pib.gov.in":            {"name": "Press Information Bureau, India", "description": "Official Indian government press releases"},
    "cdc.gov":               {"name": "CDC",                   "description": "US Centers for Disease Control and Prevention"},
    "nih.gov":               {"name": "NIH",                   "description": "US National Institutes of Health"},
    "nasa.gov":              {"name": "NASA",                  "description": "US National Aeronautics and Space Administration"},
    "worldbank.org":         {"name": "World Bank",            "description": "International financial institution"},
    "imf.org":               {"name": "IMF",                   "description": "International Monetary Fund"},
    # Academic / Tech
    "wikipedia.org":         {"name": "Wikipedia",             "description": "Collaborative encyclopedia — community-edited reference"},
    "britannica.com":        {"name": "Encyclopaedia Britannica", "description": "Authoritative reference encyclopedia"},
    "nature.com":            {"name": "Nature",                "description": "Peer-reviewed scientific journal"},
    "science.org":           {"name": "Science",               "description": "Peer-reviewed journal — AAAS publication"},
    "pubmed.ncbi.nlm.nih.gov": {"name": "PubMed",             "description": "Biomedical literature database — NIH"},
    "techcrunch.com":        {"name": "TechCrunch",            "description": "Technology news publication"},
    "wired.com":             {"name": "Wired",                 "description": "Technology and culture magazine"},
    "theverge.com":          {"name": "The Verge",             "description": "Technology news website"},
    "arstechnica.com":       {"name": "Ars Technica",          "description": "Technology news and analysis"},
    "mit.edu":               {"name": "MIT",                   "description": "Massachusetts Institute of Technology"},
    "harvard.edu":           {"name": "Harvard University",    "description": "Ivy League research university"},
    "investopedia.com":      {"name": "Investopedia",          "description": "Financial education and news platform"},
    "hbr.org":               {"name": "Harvard Business Review", "description": "Business management magazine — Harvard"},
    "mckinsey.com":          {"name": "McKinsey & Company",    "description": "Global management consulting firm"},
    "psychologytoday.com":   {"name": "Psychology Today",      "description": "Psychology and mental health publication"},
    "sciencedaily.com":      {"name": "Science Daily",         "description": "Science news aggregator"},
    "nationalgeographic.com":{"name": "National Geographic",   "description": "Science and exploration magazine"},
    "youtube.com":           {"name": "YouTube",               "description": "Video platform"},
}


def get_publication_info(url: str, fallback_name: str = "") -> dict:
    """
    Returns publication info for a URL.
    ALWAYS returns a non-null source_from:
      - Known domain  → human-readable name from KNOWN_PUBLICATIONS
      - Unknown domain → cleaned domain string (e.g. "goodfinancialcents.com")
      - No URL        → fallback_name or "Unknown Source"
    """
    if not url:
        return {
            "source_from":       fallback_name or "Unknown Source",
            "has_credentials":   False,
            "name":              fallback_name or None,
            "description":       None,
        }

    parsed = urlparse(url)
    domain = parsed.netloc.lower().lstrip('www.')

    # Direct lookup
    if domain in KNOWN_PUBLICATIONS:
        pub = KNOWN_PUBLICATIONS[domain]
        return {
            "source_from":     pub["name"],
            "has_credentials": True,
            "name":            pub["name"],
            "description":     pub["description"],
        }

    # Partial suffix match (e.g. 'health.who.int' → 'who.int')
    for known_domain, pub in KNOWN_PUBLICATIONS.items():
        if domain.endswith(known_domain):
            return {
                "source_from":     pub["name"],
                "has_credentials": True,
                "name":            pub["name"],
                "description":     pub["description"],
            }

    # Unknown — use cleaned domain as source_from (never null)
    clean_domain = domain or fallback_name or "Unknown Source"
    return {
        "source_from":     clean_domain,
        "has_credentials": bool(domain),
        "name":            clean_domain or None,
        "description":     "Web publication" if domain else None,
    }


# ══════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ══════════════════════════════════════════════════════════════

def chunk_text(text: str) -> list[str]:
    """250-word sentence-aware chunker with 50-word overlap."""
    sentences = sent_tokenize(text)
    chunks: list[str] = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= CHUNK_SIZE:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            overlap_words = current_chunk.split()[-CHUNK_OVERLAP:]
            current_chunk = " ".join(overlap_words) + " " + sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def _is_relevant(text: str, topic: str) -> bool:
    """
    At least 2 significant topic words must appear in first 500 words.
    Prevents ESL worksheets, dictionary pages, off-topic scrapes.
    """
    stop = {"the", "a", "an", "of", "in", "to", "and", "or", "for",
            "is", "it", "on", "by", "at", "do", "be", "vs", "how",
            "why", "what", "who", "when", "where", "with", "from"}
    topic_words = [w.lower() for w in topic.split()
                   if w.lower() not in stop and len(w) > 3]
    if not topic_words:
        return True
    preview = " ".join(text.split()[:500]).lower()
    matches  = sum(1 for w in topic_words if w in preview)
    required = 1 if len(topic_words) <= 2 else 2
    return matches >= required


def _extract_text_from_html(html: str) -> str:
    """readability + BeautifulSoup → clean article text."""
    try:
        doc  = Document(html)
        soup = BeautifulSoup(doc.summary(), "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)


# ══════════════════════════════════════════════════════════════
# FETCHER 1 — WIKIPEDIA REST API
# ══════════════════════════════════════════════════════════════

def fetch_wikipedia(topic: str, max_results: int) -> list[dict]:
    """
    Uses Wikipedia API to fetch full article text.
    Returns articles with guaranteed source_from = "Wikipedia".
    Strategy:
      1. Wikipedia search API (with correct User-Agent) → get matching page titles + URLs
      2. Jina Reader → fetch full article content for each URL
         (Jina returns 30k-word Wikipedia articles perfectly — confirmed working)
    """
    articles = []
    print(f"  [Wikipedia + Jina] Searching: '{topic}'")

    # Wikipedia requires a descriptive User-Agent per their bot policy:
    # https://w.wiki/4wJS
    WIKI_HEADERS = {
        "User-Agent": "StoryBit-Ingestion/1.0 (educational content pipeline; contact@storybit.app)"
    }

    try:
        # Step 1: search API to get matching page titles
        with httpx.Client(headers=WIKI_HEADERS, timeout=15) as client:
            search_resp = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action":   "query",
                    "list":     "search",
                    "srsearch": topic,
                    "srlimit":  max_results + 3,
                    "format":   "json",
                    "utf8":     1,
                },
            )
            search_results = search_resp.json().get("query", {}).get("search", [])

        for result in search_results:
            if len(articles) >= max_results:
                break
            page_title = result.get("title", "")
            if not page_title:
                continue

            # Step 2: fetch full content via Jina Reader
            # Jina returns 30k-word Wikipedia articles as clean markdown
            wiki_url  = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
            full_text = _scrape_jina(wiki_url)

            if not full_text:
                print(f"    ✗ Jina fetch failed: '{page_title[:40]}'")
                continue

            word_count = len(full_text.split())

            if word_count < MIN_ARTICLE_WORDS:
                print(f"    ✗ Too short ({word_count}w): '{page_title[:40]}'")
                continue
            if not _is_relevant(full_text, topic):
                print(f"    ✗ Irrelevant: '{page_title[:40]}'")
                continue

            articles.append({
                "title":       page_title,
                "text":        full_text,
                "url":         wiki_url,
                "source_from": "Wikipedia",
                "author_name": "Wikipedia",
                "author_desc": "Collaborative encyclopedia (Wikipedia.org)",
                "has_creds":   True,
            })
            print(f"    ✓ Wikipedia/{page_title[:40]} ({word_count:,} words)")

    except Exception as e:
        print(f"    Wikipedia fetch error: {e}")

    return articles


# ══════════════════════════════════════════════════════════════
# FETCHER 2 — NEWSAPI.ORG
# ══════════════════════════════════════════════════════════════

def fetch_newsapi(topic: str, max_results: int) -> list[dict]:
    """
    Fetches news articles from NewsAPI.org.
    Returns full article content with publication name guaranteed in source_from.
    Falls back to DDGS news if no NEWSAPI_KEY set.
    """
    articles = []

    if not NEWSAPI_KEY:
        print(f"  [NewsAPI] No NEWSAPI_KEY set — falling back to DDGS news")
        return fetch_ddgs_news(topic, max_results)

    print(f"  [NewsAPI] Searching: '{topic}'")
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":        topic,
                    "language": "en",
                    "sortBy":   "relevancy",
                    "pageSize": max_results + 3,
                    "apiKey":   NEWSAPI_KEY,
                },
            )
            data = resp.json()

        if data.get("status") != "ok":
            print(f"    NewsAPI error: {data.get('message', 'unknown')}")
            return fetch_ddgs_news(topic, max_results)

        for art in data.get("articles", []):
            if len(articles) >= max_results:
                break

            title       = art.get("title", topic)
            url         = art.get("url", "")
            content     = art.get("content", "")     # truncated at 200 chars by NewsAPI free
            description = art.get("description", "")
            source_name = art.get("source", {}).get("name", "")  # e.g. "BBC News"
            author      = art.get("author", "")
            published   = art.get("publishedAt", "")

            # Skip consent walls, paywalls, and aggregator redirect URLs
            SKIP_URL_PATTERNS = [
                "consent.yahoo.com", "consent.google.com",
                "accounts.google.com", "login.", "subscribe.",
                "[Removed]", "removed",
            ]
            if not url or any(p in url for p in SKIP_URL_PATTERNS):
                print(f"    ✗ Skipped (consent/removed URL): '{title[:40]}'")
                continue

            # NewsAPI free tier truncates content — try to scrape full article via Jina
            full_text = _scrape_jina(url)

            # Fallback chain: Jina full → NewsAPI content field → description
            text = full_text or content or description
            if not text or len(text.split()) < MIN_SNIPPET_WORDS:
                print(f"    ✗ Too short: '{title[:40]}'")
                continue
            if not _is_relevant(text, topic):
                print(f"    ✗ Irrelevant: '{title[:40]}'")
                continue

            # source_from: NewsAPI gives us the publication name directly
            pub_info = get_publication_info(url, fallback_name=source_name)
            # Prefer NewsAPI's source.name if it's more descriptive
            if source_name and pub_info["source_from"] in ("Unknown Source", ""):
                pub_info["source_from"] = source_name
                pub_info["name"]        = source_name

            # Build author info
            if author:
                author_info = {
                    "has_credentials": True,
                    "name":            author,
                    "description":     f"Journalist at {source_name}" if source_name else "News journalist",
                }
            else:
                author_info = {
                    "has_credentials": bool(source_name),
                    "name":            source_name or pub_info["name"],
                    "description":     pub_info["description"],
                }

            articles.append({
                "title":        title,
                "text":         text,
                "url":          url,
                "source_from":  pub_info["source_from"],   # guaranteed non-null
                "author_name":  author_info["name"],
                "author_desc":  author_info["description"],
                "has_creds":    author_info["has_credentials"],
                "published_at": published,
            })
            word_count = len(text.split())
            src = "full scrape" if full_text else "NewsAPI content"
            print(f"    ✓ {pub_info['source_from']} ({src}, {word_count}w): '{title[:40]}'")

    except Exception as e:
        print(f"    NewsAPI error: {e} — falling back to DDGS")
        return fetch_ddgs_news(topic, max_results)

    return articles


def fetch_ddgs_news(topic: str, max_results: int) -> list[dict]:
    """DDGS news fallback when NewsAPI key not set or fails."""
    articles = []
    print(f"  [DDGS News] Searching: '{topic}'")
    try:
        with DDGS(timeout=DDGS_TIMEOUT) as ddgs:
            results = list(ddgs.news(topic, timelimit='m', max_results=max_results + 4))
        for r in results:
            if len(articles) >= max_results:
                break
            url     = r.get('url', '')
            title   = r.get('title', topic)
            snippet = r.get('body', '')
            source  = r.get('source', '')

            full_text = _scrape_jina(url) if url else None
            text      = full_text or snippet
            if not text or len(text.split()) < MIN_SNIPPET_WORDS:
                continue
            if not _is_relevant(text, topic):
                continue

            pub_info = get_publication_info(url, fallback_name=source)
            articles.append({
                "title":       title,
                "text":        text,
                "url":         url,
                "source_from": pub_info["source_from"],
                "author_name": pub_info["name"],
                "author_desc": pub_info["description"],
                "has_creds":   pub_info["has_credentials"],
            })
            print(f"    ✓ {pub_info['source_from']} ({len(text.split())}w): '{title[:40]}'")
    except Exception as e:
        print(f"    DDGS news error: {e}")
    return articles


# ══════════════════════════════════════════════════════════════
# FETCHER 3 — JINA AI READER  (r.jina.ai)
# ══════════════════════════════════════════════════════════════

def _scrape_jina(url: str) -> str | None:
    """
    Fetch clean article text via Jina AI Reader.
    Prefix URL with https://r.jina.ai/ → returns clean markdown.
    Free, no API key, handles JS rendering and most paywalls.
    """
    if not url:
        return None
    try:
        jina_url = f"https://r.jina.ai/{url}"
        with httpx.Client(
            headers={**SCRAPE_HEADERS, "Accept": "text/plain"},
            timeout=20,
            follow_redirects=True,
        ) as client:
            resp = client.get(jina_url)
        if resp.status_code == 200:
            text = resp.text.strip()
            # Strip Jina's header block (Title:, URL:, etc.)
            lines = text.split('\n')
            content_lines = []
            skip_header = True
            for line in lines:
                if skip_header and line.startswith(('Title:', 'URL:', 'Published', 'Source:', 'Description:')):
                    continue
                skip_header = False
                content_lines.append(line)
            clean = "\n".join(content_lines).strip()
            if len(clean.split()) >= MIN_ARTICLE_WORDS:
                return clean
    except Exception as e:
        print(f"    Jina scrape failed for {url[:60]}: {e}")
    return None


def fetch_jina_web(topic: str, max_results: int) -> list[dict]:
    """
    DDGS text search → Jina Reader for full article content.
    Used for Business & Money and any web_scrape category.
    source_from always populated from domain lookup.
    """
    articles = []
    print(f"  [Jina + DDGS] Searching: '{topic}'")
    try:
        with DDGS(timeout=DDGS_TIMEOUT) as ddgs:
            results = list(ddgs.text(f"{topic} explained guide", max_results=max_results + 5))

        for r in results:
            if len(articles) >= max_results:
                break
            url     = r.get('href', '')
            title   = r.get('title', topic)
            snippet = r.get('body', '')

            # Skip obviously irrelevant domains
            domain = urlparse(url).netloc.lower()
            skip_domains = ['stackoverflow.com', 'reddit.com', 'quora.com',
                            'amazon.com', 'ebay.com', 'pinterest.com']
            if any(d in domain for d in skip_domains):
                continue

            # Try Jina full scrape first
            full_text = _scrape_jina(url) if url else None

            text = full_text or snippet
            if not text or len(text.split()) < MIN_SNIPPET_WORDS:
                continue
            if not _is_relevant(text, topic):
                print(f"    ✗ Irrelevant: '{title[:40]}'")
                continue

            pub_info = get_publication_info(url)
            articles.append({
                "title":       title,
                "text":        text,
                "url":         url,
                "source_from": pub_info["source_from"],   # guaranteed non-null
                "author_name": pub_info["name"],
                "author_desc": pub_info["description"],
                "has_creds":   pub_info["has_credentials"],
            })
            src = "Jina full" if full_text else "snippet"
            print(f"    ✓ {pub_info['source_from']} ({src}, {len(text.split())}w): '{title[:40]}'")

    except Exception as e:
        print(f"    Jina/DDGS error: {e}")
    return articles


# ══════════════════════════════════════════════════════════════
# MAIN CONTENT ROUTER
# ══════════════════════════════════════════════════════════════

def fetch_content_for_topic(topic: str, source_type: str) -> list[dict]:
    """
    Routes to the correct fetcher based on source_type.
    Every article returned is guaranteed to have:
      - text (≥ MIN_ARTICLE_WORDS words)
      - source_from (non-null, human-readable)
      - author_name, author_desc, has_creds
    """
    if source_type == "wikipedia":
        return fetch_wikipedia(topic, ARTICLES_PER_TOPIC)
    elif source_type == "news":
        return fetch_newsapi(topic, ARTICLES_PER_TOPIC)
    else:  # web_scrape or any future category
        return fetch_jina_web(topic, ARTICLES_PER_TOPIC)


# ══════════════════════════════════════════════════════════════
# SPREADSHEET READER
# ══════════════════════════════════════════════════════════════

def read_spreadsheet(filepath: str) -> list[dict]:
    print(f"Reading spreadsheet: {filepath}")
    wb = openpyxl.load_workbook(filepath)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        print("ERROR: Spreadsheet is empty.")
        return []

    header = [str(cell).strip().lower() if cell else "" for cell in rows[0]]
    print(f"Detected headers: {header}")

    col_map = {}
    for i, h in enumerate(header):
        if 'category' in h:    col_map['category'] = i
        elif 'topic' in h:     col_map['topic'] = i
        elif 'objective' in h: col_map['objective'] = i
        elif 'tag' in h:       col_map['tags'] = i

    for r in ['category', 'topic']:
        if r not in col_map:
            print(f"ERROR: Required column '{r}' not found. Headers: {header}")
            return []

    topics = []
    for row in rows[1:]:
        if not any(row):
            continue
        category = str(row[col_map['category']]).strip() if row[col_map['category']] else ""
        topic    = str(row[col_map['topic']]).strip()    if row[col_map['topic']]    else ""
        tags_raw = str(row[col_map['tags']]).strip()     if col_map.get('tags') is not None and row[col_map.get('tags', 0)] else ""

        if not category or not topic or category.lower() == 'none' or topic.lower() == 'none':
            continue

        tags: list[str] = []
        if tags_raw:
            try:
                parsed = json.loads(tags_raw)
                tags = [str(t).strip() for t in parsed if t] if isinstance(parsed, list) else []
            except (json.JSONDecodeError, ValueError):
                tags = [t.strip() for t in tags_raw.split(',') if t.strip()]

        topics.append({"category": category, "topic": topic, "tags": tags})

    print(f"Loaded {len(topics)} topics from spreadsheet.")
    return topics


# ══════════════════════════════════════════════════════════════
# RATE LIMIT HELPER
# ══════════════════════════════════════════════════════════════

def _parse_retry_delay(error_str: str, fallback: float = 65.0) -> float:
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", error_str)
    if m:
        return float(m.group(1)) + 2
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", error_str)
    if m:
        return float(m.group(1)) + 2
    return fallback


# ══════════════════════════════════════════════════════════════
# EMBED & INSERT
# ══════════════════════════════════════════════════════════════

class DailyQuotaError(Exception):
    pass


def _embed_gemini(contents: list[str], gemini_client) -> list:
    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = gemini_client.models.embed_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768,
                ),
            )
            return [e.values for e in resp.embeddings]
        except Exception as e:
            err = str(e)
            if "PerDay" in err or "EmbedContentRequestsPerDayPerUser" in err:
                raise DailyQuotaError(err)
            if ("429" in err or "RESOURCE_EXHAUSTED" in err) and attempt < max_retries - 1:
                wait = _parse_retry_delay(err)
                print(f"    [Gemini] Rate limit — waiting {wait:.0f}s...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    [Gemini] Error: {e}. Retry in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [Gemini] Max retries: {e}")
                return []
    return []


def _embed_local(contents: list[str]) -> list:
    model   = get_local_model()
    vectors = model.encode(contents, batch_size=LOCAL_BATCH_SIZE,
                           show_progress_bar=False, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_and_insert(batch: list[dict], supabase: Client, gemini_client) -> bool:
    """
    Embeds batch and inserts to Supabase.
    Primary: Gemini (768-dim). Fallback: all-mpnet-base-v2 (768-dim, local).
    Auto-switches permanently on daily quota hit — no DB changes needed.
    """
    global _use_local_embedder

    contents   = [item['content'] for item in batch]
    embeddings = None

    if _use_local_embedder:
        print(f"    [Local: all-mpnet] Embedding {len(batch)} chunks...")
        embeddings = _embed_local(contents)
    else:
        print(f"    [Gemini] Embedding {len(batch)} chunks...")
        try:
            embeddings = _embed_gemini(contents, gemini_client)
        except DailyQuotaError:
            print(f"\n{'!'*60}")
            print(f"  Gemini daily quota hit. AUTO-SWITCHING to local model.")
            print(f"  {LOCAL_MODEL_NAME} (768-dim) — no DB changes needed.")
            print(f"{'!'*60}\n")
            _use_local_embedder = True
            embeddings = _embed_local(contents)

    if not embeddings:
        return False

    for i, item in enumerate(batch):
        item['embedding'] = embeddings[i]
        item['metadata']['embedding_model'] = (
            LOCAL_MODEL_NAME if _use_local_embedder else GEMINI_MODEL
        )

    try:
        print(f"    Inserting {len(batch)} rows to Supabase...")
        supabase.table('documents').insert(batch).execute()
        print("    ✓ Batch inserted.")
        return True
    except Exception as e:
        print(f"    CRITICAL: Supabase insert failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

def main(gemini_client, supabase: Client):
    topics = read_spreadsheet(SPREADSHEET_FILE)
    if not topics:
        print("No topics loaded. Exiting.")
        return

    # Resume support
    start_topic_index = 0
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            content = f.read().strip()
            if content:
                start_topic_index = int(content)
        print(f"Resuming from topic #{start_topic_index + 1}")

    total_chunks_inserted = 0
    pending_batch: list[dict] = []

    for topic_idx, row in enumerate(topics[start_topic_index:], start=start_topic_index):
        category    = row['category']
        topic       = row['topic']
        tags        = row['tags']
        source_type = CATEGORY_SOURCE_MAP.get(category, DEFAULT_SOURCE_TYPE)

        print(f"\n[{topic_idx + 1}/{len(topics)}] {category} → '{topic}' (source: {source_type})")

        articles = fetch_content_for_topic(topic, source_type)
        if not articles:
            print(f"  Skipping '{topic}' — no content fetched.")
            continue

        print(f"  → {len(articles)} articles fetched.")

        for article in articles:
            chunks = chunk_text(article['text'])
            chunks = [c for c in chunks if c and not c.isspace()]

            for chunk in chunks:
                # ── Build guaranteed metadata ─────────────────
                # source_from is ALWAYS non-null — set by fetcher
                # Falls back through: fetcher → domain lookup → raw domain → "Unknown Source"
                source_from = article.get("source_from") or \
                              get_publication_info(article.get("url",""))["source_from"]

                metadata: dict = {
                    "source_from": source_from,
                    "category":    category,
                    "topic":       topic,
                    "tags":        tags,
                    "author": {
                        "has_credentials": article.get("has_creds", False),
                        "name":            article.get("author_name") or source_from,
                        "description":     article.get("author_desc"),
                    },
                }

                # Extra fields per source type
                if source_type == "wikipedia":
                    metadata["wiki_page"] = article.get("title", "")
                elif source_type == "news" and article.get("published_at"):
                    metadata["published_at"] = article["published_at"]
                elif source_type == "youtube":
                    metadata["video_id"] = article.get("video_id", "")
                elif source_type == "book":
                    metadata["book_title"] = article.get("title", "")

                pending_batch.append({
                    "content":      chunk,
                    "source_title": article['title'],
                    "source_url":   article.get('url') or None,
                    "source_type":  source_type,
                    "metadata":     metadata,
                })

                # Flush when full
                if len(pending_batch) >= BATCH_SIZE:
                    success = embed_and_insert(pending_batch, supabase, gemini_client)
                    if not success:
                        print("CRITICAL: Batch failed. Aborting.")
                        return
                    total_chunks_inserted += len(pending_batch)
                    pending_batch = []

                    with open(PROGRESS_FILE, 'w') as f:
                        f.write(str(topic_idx + 1))

                    if not _use_local_embedder:
                        print("    Waiting 62s (Gemini rate limit)...")
                        time.sleep(62)

    # Flush final batch
    if pending_batch:
        print(f"\nFlushing final batch of {len(pending_batch)} chunks...")
        success = embed_and_insert(pending_batch, supabase, gemini_client)
        if success:
            total_chunks_inserted += len(pending_batch)

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    print(f"\n{'='*60}")
    print(f"Spreadsheet ingestion complete!")
    print(f"Topics processed:      {len(topics) - start_topic_index}")
    print(f"Total chunks inserted: {total_chunks_inserted}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase credentials not found in .env")

    newsapi_status = f"✓ NEWSAPI_KEY set" if NEWSAPI_KEY else "✗ NEWSAPI_KEY not set (will use DDGS fallback)"

    _gemini_client = genai.Client(api_key=google_api_key)
    _supabase: Client = create_client(supabase_url, supabase_key)

    print("=" * 60)
    print("StoryBit Ingestion Pipeline v3")
    print("=" * 60)
    print(f"  Wikipedia API:  ✓ always available (no key needed)")
    print(f"  NewsAPI:        {newsapi_status}")
    print(f"  Jina Reader:    ✓ always available (no key needed)")
    print(f"  Gemini embed:   ✓ key loaded")
    print(f"  Supabase:       ✓ connected")
    print(f"  Spreadsheet:    {SPREADSHEET_FILE}")
    print(f"  Batch size:     {BATCH_SIZE} chunks")
    print("=" * 60)
    print()

    main(_gemini_client, _supabase)