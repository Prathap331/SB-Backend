#main3

import os
import google.generativeai as genai
import httpx
import uvicorn
import asyncio
import time
import re
from bs4 import BeautifulSoup
from ddgs import DDGS
from readability import Document
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
import nltk
import json
from nltk.tokenize import sent_tokenize
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from shared.schemas.pipeline_context import (
    AgentPipelineContext,
    extract_angle_for_prompt,
    staleness_hours,
)
from openai import AsyncOpenAI
import random
import json
from datetime import datetime
from pydantic import BaseModel

'''

# --- NEW: NLTK Data Check on Startup ---
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' data found.")
# --- THIS IS THE FIX: Catch the correct error type ---
except LookupError: # Changed from nltk.downloader.DownloadError
# ---------------------------------------------------
    print("NLTK 'punkt' data not found. Downloading...")
    nltk.download('punkt')
    print("NLTK 'punkt' data downloaded successfully.")
# ----------------------------------------
'''


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


class SelectedIdea:
    def __init__(self, idea_id, title):
        self.idea_id = idea_id
        self.title = title


class gap_context:
    def __init__(self,problem,insight,angle_string):
        self.problem = problem
        self.insight = insight
        self.angle_string = angle_string


class context(BaseModel):
    def __init__(self,topic,keywords,selected_idea_id,selected_angle_id,pipeline_assembled_at: datetime | None = None):
        self.topic= topic
        self.keywords =keywords
        self.selected_idea_id = selected_idea_id
        self.selected_angle_id = selected_angle_id
        self.pipeline_assembled_at = pipeline_assembled_at
        
class ChannelContextInput(BaseModel):
    channel_id: str | None = None
    channel_niche: str | None = None
    subscriber_count: int | None = None
    top_video_titles: list[str] | None = None
    existing_hashtags: list[str] | None = None
    avg_ctr_pct: float | None = None


# --- Setup and Configuration (Using your specified models) ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print(api_key)
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=api_key)

# Using the models from your working code


pro_model = genai.GenerativeModel('models/gemini-3-flash-preview')

flash_model = genai.GenerativeModel('models/gemini-3-flash-preview')

content_genmodel=genai.GenerativeModel('models/gemini-3-flash-preview')

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found.")
groq_client = AsyncOpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)
GROQ_GENERATION_MODEL = "llama-3.1-8b-instant"

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
embedding_model = 'models/text-embedding-004'

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("Supabase credentials not found in .env file")
supabase: Client = create_client(supabase_url, supabase_key)
print("Clients for Google AI and Supabase initialized successfully.")


class SEOAgentRequest(BaseModel):
    context: AgentPipelineContext
    channel_context: ChannelContextInput | None = None



# --- Helper Functions (Your existing, working code) ---
def chunk_text(text: str, chunk_size: int = 250, chunk_overlap: int = 50) -> list[str]:
    # (This function is unchanged)
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
    # (This function is unchanged)
    print(f"BACKGROUND TASK: Starting to upload '{article_title[:30]}...'")
    try:
        raw_chunks = chunk_text(article_text)
        chunks = [chunk for chunk in raw_chunks if chunk and not chunk.isspace()]
        if not chunks:
            print("BACKGROUND TASK: No valid chunks to process.")
            return
        embedding_result = genai.embed_content(model=embedding_model, content=chunks, task_type="retrieval_document")
        embeddings = embedding_result['embedding']
        documents_to_insert = [{"content": chunk, "embedding": embeddings[i], "source_title": article_title, "source_url": article_url} for i, chunk in enumerate(chunks)]
        supabase.table('documents').insert(documents_to_insert).execute()
        print(f"BACKGROUND TASK: Successfully uploaded {len(documents_to_insert)} chunks.")
    except Exception as e:
        print(f"BACKGROUND TASK: Failed to add data to DB. Error: {e}")

async def scrape_url(client: httpx.AsyncClient, url: str, scraped_urls: set):
    # (This function is unchanged)
    if url in scraped_urls:
        return None
    print(f"Scraping: {url}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = await client.get(url, headers=headers, timeout=10, follow_redirects=True)
        response.raise_for_status()
        scraped_urls.add(url)
        doc = Document(response.text)
        title = doc.title()
        article_html = doc.summary()
        soup = BeautifulSoup(article_html, 'html.parser')
        article_text = soup.get_text(separator='\n', strip=True)
        return {"url": url, "title": title, "text": article_text}
    except Exception as e:
        print(f"An error occurred while processing {url}: {e}")
        return None

async def deep_search_and_scrape(keywords: list[str], scraped_urls: set) -> list[dict]:
    # (This function is unchanged)
    print("--- DEEP WEB SCRAPE: Starting full search... ---")
    urls_to_scrape = set()
    with DDGS(timeout=20) as ddgs:
        for keyword in keywords:
            search_results = list(ddgs.text(keyword, region='wt-wt', max_results=3))
            if search_results:
                urls_to_scrape.add(search_results[0]['href'])
    async with httpx.AsyncClient() as client:
        tasks = [scrape_url(client, url, scraped_urls) for url in urls_to_scrape]
        results = await asyncio.gather(*tasks)
        return [res for res in results if res and res.get("text")]

async def get_latest_news_context(topic: str, scraped_urls: set) -> list[dict]:
    # (This function is unchanged)
    print("--- LIGHT WEB SCRAPE: Starting lightweight news search... ---")
    try:
        keyword = f"{topic} latest news today"
        urls_to_scrape = set()
        with DDGS(timeout=10) as ddgs:
            search_results = list(ddgs.text(keyword, region='wt-wt', max_results=2))
            for result in search_results:
                urls_to_scrape.add(result['href'])
        async with httpx.AsyncClient() as client:
            tasks = [scrape_url(client, url, scraped_urls) for url in urls_to_scrape]
            results = await asyncio.gather(*tasks)
            return [res for res in results if res and res.get("text")]
    except Exception as e:
        print(f"--- WEB TASK: Error during news scraping: {e} ---")
        return []

async def get_db_context(topic: str) -> list[dict]:
    # (This function is unchanged)
    print("--- DB TASK: Starting HyDE database search... ---")
    try:
        hyde_prompt = f"""
        Write a short, factual, encyclopedia-style paragraph that provides a direct answer to the following topic.
        This will be used to find similar documents, so be concise and include key terms.
        
        Topic: "{topic}"
        """
        hyde_response = await flash_model.generate_content_async(hyde_prompt)
        query_embedding = genai.embed_content(model=embedding_model, content=hyde_response.text, task_type="retrieval_query")['embedding']
        db_results = supabase.rpc('match_documents', {'query_embedding': query_embedding, 'match_threshold': 0.65, 'match_count': 5}).execute()
        return db_results.data
    except Exception as e:
        print(f"--- DB TASK: Error during database search: {e} ---")
        return []


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

        response = await flash_model.generate_content_async(prompt)

        category = response.candidates[0].content.parts[0].text.strip()

        return {"category": category}

    except Exception as e:
        return {"category": "UNKNOWN", "error": str(e)}    




def _compute_ctr_signal(
    gap_context: dict[str, any],
    csi_scores: dict[str, any],
    csi_quality: dict[str, any],
    tss_scores: dict[str, any],
) -> tuple[str, float, bool]:
    demand = float(csi_scores.get("demand_score", 50) or 50) / 100.0
    supply = float(csi_scores.get("supply_score", 50) or 50) / 100.0
    openness = 1.0 - supply
    momentum = float(tss_scores.get("m1_score", 50) or 50) / 100.0

    score = (demand * 0.45) + (momentum * 0.30) + (openness * 0.25)
    label = "High" if score >= 0.65 else "Medium" if score >= 0.40 else "Low"
    degraded = bool(
        csi_quality.get("engagement_insufficient")
        or csi_quality.get("redundancy_embedding_failed")
    )
    if degraded:
        label = {"High": "Medium", "Medium": "Low", "Low": "Low"}[label]
    return label, round(score, 3), degraded



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


def _get_title_config(cat_id: str) -> tuple[list[str], bool]:
    return BLOCKED_TITLE_TYPES.get(cat_id, []), CAT_FACE_DEFAULTS.get(cat_id, True)


async def _run_ddgs_scrape(angle: str) -> tuple[list[str], list[str]]:
    loop = asyncio.get_running_loop()
    competing_titles: list[str] = []
    paa_questions: list[str] = []
    try:
        with DDGS(timeout=10) as ddgs:
            videos = await loop.run_in_executor(
                None,
                lambda: list(ddgs.videos(angle, region="wt-wt", timelimit="m", max_results=5)),
            )
            competing_titles = [str(v.get("title") or "").strip() for v in videos if str(v.get("title") or "").strip()]
    except Exception:
        competing_titles = []

    try:
        with DDGS(timeout=10) as ddgs:
            answers = await loop.run_in_executor(
                None,
                lambda: list(ddgs.answers(angle, region="wt-wt")),
            )
            paa_questions = [str(a.get("question") or "").strip() for a in answers[:5] if str(a.get("question") or "").strip()]
    except Exception:
        paa_questions = []
    return competing_titles, paa_questions




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



def _strip_json_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text



def _parse_json_object(raw: str) -> dict[str, any]:
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




def _first_allowed_pipe_token(value: any, allowed: set[str], default: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    tokens = [t.strip() for t in raw.split("|") if t.strip()]
    for token in tokens:
        if token in allowed:
            return token
    return default



def _safe_recommended_titles(raw_titles: any, blocked_types: list[str]) -> list[dict[str, str]]:
    if not isinstance(raw_titles, list):
        return []
    cleaned: list[dict[str, str]] = []
    for item in raw_titles:
        if not isinstance(item, dict):
            continue
        t_type = str(item.get("type") or "").strip()
        if not t_type or t_type in blocked_types:
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        cleaned.append(
            {
                "type": t_type,
                "title": title[:70],
                "rationale": str(item.get("rationale") or "").strip()[:300],
            }
        )
        if len(cleaned) >= 4:
            break
    return cleaned


def _deduplicate_hashtags(generated: list[str], existing_channel_hashtags: list[str]) -> list[dict[str, any]]:
    existing_set = {h.lower().lstrip("#") for h in (existing_channel_hashtags or [])}
    result: list[dict[str, any]] = []
    for h in generated:
        clean = str(h or "").strip()
        if not clean:
            continue
        normalized = clean if clean.startswith("#") else f"#{clean}"
        key = normalized.lower().lstrip("#")
        strategy = "established" if key in existing_set else "expansion"
        result.append({"hashtag": normalized, "strategy": strategy})
    result = result[:5]
    expansions = [x for x in result if x.get("strategy") == "expansion"]
    if len(expansions) < 2:
        result.append({"hashtag": None, "strategy": "expansion", "needs_generation": True})
    return result


def _ensure_hashtag_floor(
    hashtags: list[dict[str, any]],
    topic: str,
    existing_channel_hashtags: list[str],
) -> list[dict[str, any]]:
    out = [h for h in hashtags if h.get("hashtag")]
    if len(out) >= 3:
        return out[:5]
    words = [w for w in re.findall(r"[a-zA-Z0-9]+", topic) if len(w) >= 3]
    candidates = [f"#{''.join(words[:2])}" if words else "#StoryBitTopic"]
    candidates += [f"#{w}" for w in words[:4]]
    existing = {str(h.get("hashtag") or "").lower() for h in out}
    existing_tags = {f"#{str(h).lstrip('#').lower()}" for h in (existing_channel_hashtags or [])}
    for cand in candidates:
        key = cand.lower()
        if key in existing:
            continue
        out.append(
            {
                "hashtag": cand,
                "strategy": "established" if key in existing_tags else "expansion",
            }
        )
        existing.add(key)
        if len(out) >= 5:
            break
    return out[:5]



def _ensure_chapter_structure(chapters: any, fallback_titles: list[str] | None = None) -> list[dict[str, any]]:
    fallback_titles = fallback_titles or ["Hook", "Context", "Analysis", "Takeaway"]
    cleaned: list[dict[str, any]] = []
    if isinstance(chapters, list):
        for idx, item in enumerate(chapters):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()[:40]
            covers = str(item.get("covers") or "").strip()
            pct = item.get("section_pct")
            try:
                pct_f = float(pct)
            except Exception:
                pct_f = 0.0
            if not title:
                continue
            cleaned.append(
                {
                    "index": idx + 1,
                    "title": title,
                    "covers": covers or "Core discussion",
                    "section_pct": pct_f,
                }
            )
    if len(cleaned) < 3:
        cleaned = [
            {"index": 1, "title": fallback_titles[0], "covers": "Open with the central tension", "section_pct": 0.18},
            {"index": 2, "title": fallback_titles[1], "covers": "Set background and stakeholders", "section_pct": 0.24},
            {"index": 3, "title": fallback_titles[2], "covers": "Break down evidence and dynamics", "section_pct": 0.34},
            {"index": 4, "title": fallback_titles[3], "covers": "Close with implications and CTA", "section_pct": 0.24},
        ]
        return cleaned
    total = sum(max(float(c.get("section_pct") or 0.0), 0.0) for c in cleaned)
    if total <= 0.0:
        weight = 1.0 / len(cleaned)
        for c in cleaned:
            c["section_pct"] = round(weight, 4)
    else:
        for c in cleaned:
            c["section_pct"] = round(max(float(c.get("section_pct") or 0.0), 0.0) / total, 4)
    for idx, c in enumerate(cleaned, start=1):
        c["index"] = idx
    return cleaned


def get_context(request):
    return request.context if hasattr(request, "context") else request["context"]

async def seo_agent(request: SEOAgentRequest):
    ctx = get_context(request)
    # warnings = await _validate_seo_entry(ctx)
    angle = extract_angle_for_prompt(ctx.gap_context or {})
    angle_string = angle.get("angle_string") or (ctx.gap_context or {}).get("angle_string") or ""
    if not angle_string:
        raise HTTPException(status_code=400, detail={"error": "missing_angle_string", "message": "gap_context.angle_string is required"})

    tss_scores = ctx.tss_scores or {}
    csi_scores = ctx.csi_scores or {}
    csi_quality = ctx.csi_quality or {}
    ctr_label, ctr_score, ctr_degraded = _compute_ctr_signal(ctx.gap_context or {}, csi_scores, csi_quality, tss_scores)

    cat_id = str(tss_scores.get("cat_id") or tss_scores.get("category_id") or "CAT-08")
    cat_label = str(tss_scores.get("cat_label") or tss_scores.get("category") or "General")
    blocked_types, face_default = _get_title_config(cat_id)

    competing_titles, paa_questions = await _run_ddgs_scrape(angle_string)

    channel_ctx = (request.channel_context.model_dump() if request.channel_context else None) or {}
    existing_hashtags = list(channel_ctx.get("existing_hashtags") or [])
    audience_profile = {
        "channel_niche": channel_ctx.get("channel_niche"),
        "subscriber_count": channel_ctx.get("subscriber_count"),
        "topic": ctx.topic,
    }
    prompt = SEO_SYNTHESIS_PROMPT.format(
        angle_string=angle_string,
        who=angle.get("who", ""),
        what=angle.get("what", ""),
        story_frame=angle.get("story_frame", ""),
        audience_profile=json.dumps(audience_profile),
        cat_id=cat_id,
        cat_label=cat_label,
        competing_titles=json.dumps(competing_titles),
        paa_questions=json.dumps(paa_questions),
        ctr_label=ctr_label,
        ctr_score=ctr_score,
        degraded_note="Signal degraded due to CSI quality flags." if ctr_degraded else "Signal quality healthy.",
        blocked_title_types=json.dumps(blocked_types),
        ctr_signal_degraded=str(ctr_degraded).lower(),
    )

    fallback = {
        "search_intent_type": "educational",
        "recommended_structure": "problem_solution",
        "ctr_potential": ctr_label,
        "ctr_signal_degraded": ctr_degraded,
        "ctr_score": ctr_score,
        "justification": "CTR estimated from demand, momentum, and supply openness signals.",
        "recommended_titles": [],
        "keyword_clusters": {"primary": [], "secondary": [], "longtail": [], "question_based": paa_questions[:5]},
        "description_template": {"hook": "", "body_bullets": [], "outro": ""},
        "thumbnail_brief": [
            {
                "concept_type": "data_driven",
                "text_overlay": "What Changes Next?",
                "visual_theme": "high-contrast context imagery",
                "colour_temperature": "high_contrast",
                "face_recommended": face_default,
                "rationale": "Fallback thumbnail due to seo synthesis failure.",
            }
        ],
        "hashtags": [],
        "chapter_structure": [],
        "key_questions_to_answer": paa_questions[:5],
    }

    try:
        raw = await groq_idea_generate([{"role": "user", "content": prompt}], model=GROQ_GENERATION_MODEL)
        parsed = _parse_json_object(raw)
    except Exception as exc:
        print(f"SEO synthesis failed: {exc}")
        parsed = {}

    merged = {**fallback, **(parsed or {})}
    merged["search_intent_type"] = _first_allowed_pipe_token(
        merged.get("search_intent_type"),
        SEO_INTENT_TYPES,
        "educational",
    )
    merged["recommended_structure"] = _first_allowed_pipe_token(
        merged.get("recommended_structure"),
        SEO_STRUCTURES,
        "problem_solution",
    )
    merged["ctr_potential"] = ctr_label
    merged["ctr_signal_degraded"] = ctr_degraded
    merged["ctr_score"] = ctr_score
    merged["recommended_titles"] = _safe_recommended_titles(merged.get("recommended_titles"), blocked_types)
    deduped = _deduplicate_hashtags(list(merged.get("hashtags") or []), existing_hashtags)
    merged["hashtags"] = _ensure_hashtag_floor(deduped, ctx.topic, existing_hashtags)
    merged["chapter_structure"] = _ensure_chapter_structure(merged.get("chapter_structure"))
    merged["angle"] = angle_string
    merged["angle_id"] = ctx.selected_angle_id
    merged["channel_context_unavailable"] = not bool(request.channel_context and request.channel_context.channel_id)
    # merged["warnings"] = warnings
    merged["key_questions_to_answer"] = list(merged.get("key_questions_to_answer") or paa_questions[:5])[:8]

    return merged




# --- FastAPI App ---
app = FastAPI()

# CORS Configuration
origins = [
    "https://www.storybit.tech",  # ✅ your production frontend
    # "http://localhost:3000",   # (Optional) for local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # List of allowed origins
    allow_credentials=True,           # If you need cookies/auth
    allow_methods=["*"],              # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],              # Allow all headers
)

class PromptRequest(BaseModel):
    topic: str

@app.get("/")
async def read_root(): return {"status": "Welcome"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

@app.post("/process-topic")
async def process_topic(request: PromptRequest, background_tasks: BackgroundTasks):
    total_start_time = time.time()
    print(f"Received topic from user: {request.topic}")
    
    try:
        db_task = asyncio.create_task(get_db_context(request.topic))
        
        await asyncio.sleep(11) # Your working sleep time

        db_results = []
        new_articles = []
        scraped_urls = set()
        # --- NEW: Initialize these here to ensure they exist for the final return ---
        base_keywords = []
        source_of_context = ""
        # --------------------------------------------------------------------------

        if db_task.done():
            db_results = db_task.result()
            print(f"--- DB task finished early. Found {len(db_results)} documents. ---")

        if len(db_results) >= 3:
            source_of_context = "DATABASE_WITH_NEWS"
            new_articles = await get_latest_news_context(request.topic, scraped_urls)
        else:
            source_of_context = "DEEP_SCRAPE"
            print("--- DB MISS or SLOW: Initiating DEEP web scrape. ---")
            keyword_prompt =  f"""
            Your ONLY task is to generate 3 diverse search engine keyword phrases for the topic: '{request.topic}'.
            Follow these rules STRICTLY:
            1. Return ONLY the 3 phrases.
            2. DO NOT add numbers, markdown, explanations, or any introductory text.
            3. Each phrase must be on a new line.
            EXAMPLE INPUT: Is coding dead?
            EXAMPLE OUTPUT:
            future of programming jobs automation
            AI replacing software developers
            demand for software engineers 2025
            """
            response = await flash_model.generate_content_async(keyword_prompt)
            # This parsing logic is from your older, working version
            raw_text = response.text
            keywords_in_quotes = re.findall(r'"(.*?)"', raw_text)
            if keywords_in_quotes:
                base_keywords = keywords_in_quotes
            else:
                base_keywords = [kw.strip() for kw in raw_text.strip().split('\n') if kw.strip()]
            
            targeted_keywords = [kw for kw in base_keywords] + [f"{kw} site:reddit.com" for kw in base_keywords]
            new_articles = await deep_search_and_scrape(targeted_keywords, scraped_urls)

        if not db_task.done():
            print("--- Waiting for DB task to complete... ---")
            db_results = await db_task
            print(f"--- DB task finished. Found {len(db_results)} documents. ---")

        db_context, web_context = "", ""
        source_urls = []
        
        if db_results:
            db_context = "\n\n".join([item['content'] for item in db_results])
            source_urls.extend(list(set([item['source_url'] for item in db_results if item['source_url']])))

        if new_articles:
            web_context = "\n\n".join([f"Source: {art['title']}\n{art['text']}" for art in new_articles])
            source_urls.extend([art['url'] for art in new_articles])
            for article in new_articles:
                background_tasks.add_task(add_scraped_data_to_db, article['title'], article['text'], article['url'])

        if not db_context and not web_context:
            return {"error": "Could not find any information."}

        

        # --- THIS IS THE UPGRADED PROMPT ---
        final_prompt = f"""
        You are an expert YouTube title strategist and scriptwriter.
        Your mission is to generate 4 distinct, attention-grabbing video titles for the topic: "{request.topic}", AND a corresponding description for each.

        Use the provided research material to inform your output:
        - Use the 'FOUNDATIONAL KNOWLEDGE' for deep context, facts, and historical background.
        - Use the 'LATEST NEWS' to find a fresh, timely, or surprising angle, especially considering the current date is October 15, 2025.

        RULES FOR YOUR OUTPUT:
        1.  For each of the 4 ideas, provide a 'TITLE' and a 'DESCRIPTION'.
        2.  Each 'DESCRIPTION' MUST be between 90 and 110 words.
        3.  Separate each complete idea (title + description) with '---'.
        4.  DO NOT add any introductory sentences, explanations, or any text other than the titles and descriptions in the specified format.

        EXAMPLE OUTPUT FORMAT:
        TITLE: This Is Why Everyone Is Suddenly Talking About [Topic]
        DESCRIPTION: In this video, we uncover the shocking truth behind [Topic]. For years, experts have believed one thing, but new data from October 2025 reveals a completely different story. We'll break down the historical context, analyze the latest reports, and explain exactly why this topic is about to become the biggest conversation on the internet. You'll learn about the key players, the secret history, and what this means for the future. Don't miss this deep dive into one of the most misunderstood subjects of our time, it will change everything you thought you knew.
        ---
        TITLE: The Hidden Truth Behind [Related Concept]
        DESCRIPTION: Everyone thinks they understand [Related Concept], but they're wrong. We've dug through the archives and analyzed the latest breaking news to bring you the untold story. This video explores the forgotten origins, the powerful figures who shaped its narrative, and the surprising new developments that are challenging everything we know. We connect the dots from the foundational knowledge to the fresh web updates to give you a complete picture you won't find anywhere else. Get ready to have your mind blown by the real story behind [Related Concept].
        ---
        
        RESEARCH FOR TOPIC: "{request.topic}"
        ---
        FOUNDATIONAL KNOWLEDGE (from our database):
        {db_context}
        ---
        LATEST NEWS UPDATES (from the web):
        {web_context}
        ---
        """
        step3_start_time = time.time()
        final_response = await pro_model.generate_content_async(final_prompt)
        step3_end_time = time.time()
        print(f"--- PROFILING: Step 3 (Final Idea Gen) took {step3_end_time - step3_start_time:.2f} seconds ---")

        # --- THIS IS THE NEW, SMARTER PARSING LOGIC ---
        response_text = final_response.text
        
        final_ideas = []
        final_descriptions = []
        
        # Split the entire response into blocks, one for each idea
        idea_blocks = response_text.strip().split('---')
        
        for block in idea_blocks:
            title = ""
            description = ""
            lines = block.strip().split('\n')
            
            for line in lines:
                if line.startswith('TITLE:'):
                    # Extract text after "TITLE:"
                    title = line.replace('TITLE:', '', 1).strip()
                elif line.startswith('DESCRIPTION:'):
                    # Extract text after "DESCRIPTION:"
                    description = line.replace('DESCRIPTION:', '', 1).strip()
            
            # Only add the pair if both title and description were found
            if title and description:
                final_ideas.append(title)
                final_descriptions.append(description)

        print(f"Final generated ideas: {final_ideas}")
        print(f"Final generated descriptions: {len(final_descriptions)} descriptions found.")
        total_end_time = time.time()
        print(f"--- PROFILING: Total request time was {total_end_time - total_start_time:.2f} seconds ---")
        
        # --- THIS IS THE UPDATED RETURN STATEMENT ---
        return {
            "source_of_context": source_of_context,
            "ideas": final_ideas,
            "descriptions": final_descriptions, # The new descriptions list
            "generated_keywords": base_keywords,
            "source_urls": list(set(source_urls)),
            "scraped_text_context": f"DB CONTEXT:\n{db_context}\n\nWEB CONTEXT:\n{web_context}"
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": "An error occurred in the processing pipeline."}
    
'''
    
# --- Add this new Pydantic model with your other one ---
class ScriptRequest(BaseModel):
    topic: str

# --- Add this new endpoint, now with YOUR smart conditional logic ---
@app.post("/generate-script")
async def generate_script(request: ScriptRequest, background_tasks: BackgroundTasks):
    total_start_time = time.time()
    print(f"SCRIPT GENERATION: Received request for topic: {request.topic}")

    try:
        # --- Step 1: Start the DB search in the background (Your Logic) ---
        db_task = asyncio.create_task(get_db_context(request.topic))
        
        # Give the DB task a head start
        await asyncio.sleep(11) 

        db_results = []
        new_articles = []
        scraped_urls = set()
        base_keywords = []

        # Check if the DB task finished early and was successful
        if db_task.done():
            db_results = db_task.result()
            print(f"--- DB task finished early. Found {len(db_results)} documents. ---")

        # --- Step 2: The Conditional Scrape (Your Logic) ---
        if len(db_results) >= 3:
            # If DB has enough data, just get the latest news
            print("--- DB HIT: Performing LIGHT web scrape for latest news. ---")
            new_articles = await get_latest_news_context(request.topic, scraped_urls)
        else:
            # If DB is slow or has no data, perform a full, deep scrape
            print("--- DB MISS or SLOW: Initiating DEEP web scrape. ---")
            keyword_prompt = f"""
            Your ONLY task is to generate 3 diverse search engine keyword phrases for the topic: '{request.topic}'.
            Follow these rules STRICTLY:
            1. Return ONLY the 3 phrases.
            2. DO NOT add numbers, markdown, explanations, or any introductory text.
            3. Each phrase must be on a new line.
            EXAMPLE INPUT: Is coding dead?
            EXAMPLE OUTPUT:
            future of programming jobs automation
            AI replacing software developers
            demand for software engineers 2025
            """
            response = await flash_model.generate_content_async(keyword_prompt)
            raw_text = response.text
            keywords_in_quotes = re.findall(r'"(.*?)"', raw_text)
            if keywords_in_quotes:
                base_keywords = keywords_in_quotes
            else:
                base_keywords = [kw.strip() for kw in raw_text.strip().split('\n') if kw.strip()]
            
            targeted_keywords = [kw for kw in base_keywords] + [f"{kw} site:reddit.com" for kw in base_keywords]
            new_articles = await deep_search_and_scrape(targeted_keywords, scraped_urls)

        # --- Step 3: Wait for DB task if it's still running ---
        if not db_task.done():
            print("--- Waiting for DB task to complete... ---")
            db_results = await db_task
            print(f"--- DB task finished. Found {len(db_results)} documents. ---")

        # --- Step 4: The Merge ---
        db_context, web_context = "", ""
        if db_results:
            db_context = "\n\n".join([item['content'] for item in db_results])
        if new_articles:
            web_context = "\n\n".join([f"Source: {art['title']}\n{art['text']}" for art in new_articles])
            for article in new_articles:
                background_tasks.add_task(add_scraped_data_to_db, article['title'], article['text'], article['url'])

        if not db_context and not web_context:
            return {"error": "Could not find any research material to write the script."}
            
        # --- Step 5: Use YOUR detailed prompt to generate the script ---
        print("SCRIPT GENERATION: Generating full script with content_gen_model...")
        script_prompt = f"""
        You are an expert YouTube scriptwriter who creates engaging and natural long-form content.

        Your task is to generate a complete YouTube video script of **10 minutes** in length, totaling around **1300 words**, based on the provided **main idea or topic**.

        Follow this exact structure and divide both **time and word count** proportionally across sections:

        1. **Hook & Introduction** (Approx. 1 minute / 130–150 words)
           - Begin with a powerful hook that grabs attention immediately.
           - Briefly introduce the topic and explain why it matters to viewers.
           - End with a line that builds curiosity for what’s coming next.

        2. **Problem Statement** (Approx. 1.5 minutes / 180–200 words)
           - Clearly define the main issue or challenge related to the topic.
           - Explain how it affects people, industries, or society.
           - Keep it relatable and emotionally engaging.

        3. **Evidence & Data** (Approx. 2 minutes / 250–270 words)
           - Present relevant research findings, stats, or scientific explanations from the provided context.
           - Reference credible sources, reports, or studies mentioned in the research.
           - Explain the underlying logic in simple, conversational language.

        4. **Real-world Examples** (Approx. 2.5 minutes / 300–320 words)
           - Use 2–3 case studies, events, or real-world stories from the research to illustrate the topic.
           - Connect these examples to the audience’s understanding.
           - Maintain storytelling tone and flow.

        5. **Potential Solutions** (Approx. 2.5 minutes / 300–320 words)
           - Discuss practical or innovative solutions to the problem found in the research.
           - Include expert opinions or emerging technologies if available.
           - Offer a balanced view of pros and cons.

        6. **Call to Action** (Approx. 0.5 minute / 100–120 words)
           - End with a strong conclusion and call to action.
           - Inspire the audience to think, share, or take meaningful steps.
           - Maintain an optimistic or thought-provoking tone.

        Additional Requirements:
        - Tone: natural, engaging, and conversational (like a smart storyteller).
        - Style: Mix of facts, storytelling, and insights — no fluff or filler.
        - Avoid repetition, and ensure smooth transitions between sections.
        - Keep the total around **1300 words** ±50 words.
        
        ---
        MAIN TOPIC/IDEA: "{request.topic}"

        RESEARCH CONTEXT:
        FOUNDATIONAL KNOWLEDGE (from database): {db_context}
        LATEST NEWS (from web): {web_context}
        ---
        """
        
        script_response = await content_genmodel.generate_content_async(script_prompt)
        
        total_end_time = time.time()
        print(f"--- PROFILING: Script generation took {total_end_time - total_start_time:.2f} seconds ---")
        
        return {"script": script_response.text}

    except Exception as e:
        print(f"SCRIPT GENERATION: An error occurred: {e}")
        return {"error": "An error occurred during the script generation pipeline."}



'''




# --- Define Script Structure Options ---
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
    - Item 1 (Explanation, examples, pros/cons) (~15-20%)
    - Item 2 (...) (~15-20%)
    - Item 3 (...) (~15-20%)
    - Item X (...) (~15-20%) - *Adjust percentages based on number of items*
    - (Optional) Bonus Item / Honorable Mentions (~10%)
    - Conclusion & Call to Action (Summarize, final thought) (~10%)
    """,
        "chronological": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (Introduce topic & relevance) (~10%)
    - Early Beginnings / Origins (~20%)
    - Key Developments / Turning Points (~40%) - *This is the main body*
    - Later Stages / Modern Impact (~20%)
    - Conclusion & Reflection (Call to Action) (~10%)
    """,
        "myth_debunking": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (Introduce common misconception) (~10%)
    - Myth 1 & Fact 1 (State myth, then debunk with evidence) (~25%)
    - Myth 2 & Fact 2 (...) (~25%)
    - Myth 3 & Fact 3 (...) (~25%) - *Adjust percentages based on number of myths*
    - Conclusion & Call to Action (Summarize truths, encourage critical thinking) (~15%)
""",
    "tech_review": """
    **Structure Guidance (for proportion, but do not label in script):**
    - Hook & Introduction (Show product, state review goal) (~10%)
    - Design & Build Quality (Look, feel) (~15%)
    - Key Features & Specs (What it promises, tech details) (~20%)
    - Performance & User Experience (Real-world testing, how it feels to use, battery, camera examples etc.) (~30%)
    - Pros & Cons (Balanced summary of good and bad) (~10%)
    - Verdict & Recommendation (Who is it for? Worth the price? Call to Action) (~15%)
    """
}







## --- FastAPI App ---
#app = FastAPI()

#class PromptRequest(BaseModel):
 #   topic: str

# --- UPDATED: Add duration_minutes ---
class ScriptRequest(BaseModel):
    topic: str
    emotional_tone: str | None = "engaging"
    creator_type: str | None = "educator"
    audience_description: str | None = "a general audience interested in learning"
    accent: str | None = "neutral"
    duration_minutes: int | None = 10 # NEW: Add video duration in minutes, default 10
    script_structure: str | None = "problem_solution" # NEW FIELD
# ------------------------------------

# --- REWRITTEN: The /generate-script endpoint with dynamic duration ---
@app.post("/generate-script")
async def generate_script(request: ScriptRequest, background_tasks: BackgroundTasks):
    total_start_time = time.time()
    print(f"SCRIPT GENERATION: Received request for topic: '{request.topic}'")
    print(f"Personalization - Duration: {request.duration_minutes} min, Tone: {request.emotional_tone}, Type: {request.creator_type}, Audience: {request.audience_description}, Accent: {request.accent}")

    try:

        content_category = await get_structure(request.topic)  
        a = content_category["category"]
        res = supabase.table("documents_structure").select("*").eq("catergory name",a).execute()
        structure = res.data[0]["Structure"]


        # get the seo of the topic
        # selected_idea_id =  random.randint(1,1000)  
        # selected_angle_id = random.randint(1,1000)
            # "selected_idea_id": "{selected_idea_id}",
            # "selected_angle_id": "{selected_angle_id}",
            # "idea_id": "{selected_idea_id}",


        json_generation_prompt = f"""
        You are an expert YouTube SEO strategist and content ideation assistant.

        You will be given a topic. Your task is to generate structured, high-quality SEO and content strategy output for YouTube.

        You must return ONLY valid JSON.

        ---

        OUTPUT FORMAT (STRICT):

        {{
        "context": {{
            "topic": "",
            "keywords": [],
            "selected_idea": {{
            "title": ""
            }},
            "gap_context": {{
            "problem": "",
            "insight": "",
            "angle_string": ""
            }},
            "pipeline_assembled_at": "2026-04-10T16:00:00"
        }}
        }}

        ---

        RULES:
        - Return ONLY valid JSON (no markdown, no explanation)
        - Everything must be inside "context"
        - keywords must be SEO-friendly YouTube search queries (8–15 items)
        - selected_idea_id and selected_angle_id must be used EXACTLY as provided
        - selected_idea.idea_id MUST match selected_idea_id
        - gap_context must be:
        - problem: real misconception people have
        - insight: correct understanding that fixes the misconception
        - angle_string: how to frame the video for YouTube (hook style)

        ---

        INPUT TOPIC:
        Topic : {request.topic}
       Duration:{request.duration_minutes}
        Tone: {request.emotional_tone}
        Type: {request.creator_type}
        Accent : {request.accent}
       Audience : {request.audience_description}
        """


        response = await flash_model.generate_content_async(json_generation_prompt)

        text = response.candidates[0].content.parts[0].text

        data = json.loads(text) 

        request_obj = SEOAgentRequest.model_validate(data)

        res = await seo_agent(request_obj)

        print(res)

        # --- Step 1: Gather Context (Unchanged) ---
        db_task = asyncio.create_task(get_db_context(request.topic))
        await asyncio.sleep(11) # Give DB head start

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
            # (Deep scrape logic remains the same)
            keyword_prompt =  f"""
            Your ONLY task is to generate 3 diverse search engine keyword phrases for the topic: '{request.topic}'.
            Follow these rules STRICTLY:
            1. Return ONLY the 3 phrases.
            2. DO NOT add numbers, markdown, explanations, or any introductory text.
            3. Each phrase must be on a new line.
            EXAMPLE INPUT: Is coding dead?
            EXAMPLE OUTPUT:
            future of programming jobs automation
            AI replacing software developers
            demand for software engineers 2025
            """
            response = await flash_model.generate_content_async(keyword_prompt)
            raw_text = response.text
            keywords_in_quotes = re.findall(r'"(.*?)"', raw_text)
            if keywords_in_quotes: base_keywords = keywords_in_quotes
            else: base_keywords = [kw.strip() for kw in raw_text.strip().split('\n') if kw.strip()]
            targeted_keywords = [kw for kw in base_keywords] + [f"{kw} site:reddit.com" for kw in base_keywords]
            new_articles = await deep_search_and_scrape(targeted_keywords, scraped_urls)

        if not db_task.done():
            print("--- Waiting for DB task to complete... ---")
            db_results = await db_task
            print(f"--- DB task finished. Found {len(db_results)} documents. ---")

        # --- Step 2: Merge Context (Unchanged) ---
        db_context, web_context = "", ""
        if db_results:
            db_context = "\n\n".join([item['content'] for item in db_results])
        if new_articles:
            web_context = "\n\n".join([f"Source: {art['title']}\n{art['text']}" for art in new_articles])
            for article in new_articles:
                background_tasks.add_task(add_scraped_data_to_db, article['title'], article['text'], article['url'])

        if not db_context and not web_context:
            return {"error": "Could not find any research material to write the script."}

        # --- Step 3: Calculate Word Count & Create Personalized Prompt ---
        print("SCRIPT GENERATION: Generating personalized script...")
        
        # --- NEW: Calculate target word count ---
        WORDS_PER_MINUTE = 130
        target_duration = request.duration_minutes if request.duration_minutes else 10 # Use default if not provided
        target_word_count = target_duration * WORDS_PER_MINUTE
        print(f"Targeting {target_duration} minutes / approx. {target_word_count} words.")
        # --------------------------------------
        
# --- NEW: Select the requested structure guidance ---
        requested_structure = request.script_structure if request.script_structure else "problem_solution"
        structure_guidance_text = STRUCTURE_GUIDANCE.get(requested_structure, STRUCTURE_GUIDANCE["problem_solution"]) # Fallback to default
        print(f"Using script structure: {requested_structure}")
        
#         # --------------------------------------
        
        # --- UPDATED PROMPT with dynamic values ---
        script_prompt = f"""
        You are a professional YouTube scriptwriter who creates natural, engaging, and conversational scripts that feel like a real YouTuber speaking directly to the camera.

        **Creator Profile:**
        * **Creator Type:** {request.creator_type}
        * **Target Audience:** {request.audience_description}
        * **Desired Emotional Tone:** {request.emotional_tone}
        * **Accent/Dialect:** {request.accent} (use phrasing natural for this accent)

        **Your Task:**
        Generate a complete YouTube video script of approximately **{target_duration} minutes** (~{target_word_count} words) based on the **main topic** below, using the provided **research context**.

        **Script Style & Flow:**
        - Output only the spoken dialogue — what the YouTuber would actually say aloud.
        - **Do NOT include** section titles, notes, stage directions, or metadata.
        - Speak directly to the viewer — friendly, confident, slightly spontaneous, and off-the-cuff.
        - Use **short and medium-length sentences**, natural pauses (…) or dashes, and occasional repetition for emphasis.
        - Include interjections, rhetorical questions, playful digressions, humor, and brief asides (“Wait, actually…”, “Can you believe that…?”, “By the way…”).
        - Include personal anecdotes or opinions (“I remember…”, “When I tried this…”).
        - Use **visual and emotional imagery** to make scenes vivid (“Imagine this…”, “Picture it like…”).
        - Hook viewers emotionally in the first 15–30 seconds.
        - Alternate between facts, insights, reactions, and short reflections to keep pacing dynamic.
        - Treat the script as a conversation with the audience — inclusive language like “you guys”, “we all”, “my friends”.
        - Build suspense naturally with rhetorical questions, mini cliffhangers, or curiosity hooks.
        - Use relatable analogies or humor when explaining complex topics.
        - Occasionally reference the creator’s regional or cultural context for relatability.
        - Maintain natural pacing as if recording live — mix excitement, storytelling, and factual explanation.
        - Stay close to **{target_word_count} words** (±50).

        
#         {structure_guidance_text} 

#         **Main Topic/Idea:** "{request.topic}"

#         **Research Context:**
#         FOUNDATIONAL KNOWLEDGE (from database): {db_context}
#         LATEST NEWS (from web): {web_context}

#         **Additional Notes:**
#         - Make the opening a curiosity-driven hook that emotionally pulls the viewer in within 15–30 seconds.
#         - Use storytelling techniques: tension, suspense, surprise, and moral dilemmas when relevant.
#         - Make historical or technical details feel immersive and personal, not like a lecture.
#         - Emphasize the narrative arc: build curiosity, climax, and reflection for the audience.
#         - Ensure adaptability: script should feel natural regardless of topic, duration, or target audience.
#         """

#         # ------------------------------------------

        script_response = await content_genmodel.generate_content_async(script_prompt)
        
        total_end_time = time.time()
        print(f"--- PROFILING: Script generation took {total_end_time - total_start_time:.2f} seconds ---")


        # --- NEW: Prompt for Script Analysis ---
        ANALYSIS_PROMPT_TEMPLATE = """
        You are an expert script analyzer. Analyze the provided YouTube script based on the following criteria:

        1.  **Real-world Examples:** Count how many distinct real-world examples, case studies, or specific stories are mentioned.
        2.  **Research Facts/Stats:** Count how many distinct research findings, statistics, or specific data points are cited or explained.
        3.  **Proverbs/Sayings:** Count how many common proverbs, idioms, or well-known sayings are used.
        4.  **Emotional Depth:** Assess the overall emotional depth and engagement level of the script. Rate it as Low, Medium, or High.

        **Your Task:**
        Read the script below and return ONLY a JSON object with the results. Do not add any explanation or other text.

        **EXAMPLE OUTPUT FORMAT:**
        {{
        "examples_count": 3,
        "research_facts_count": 5,
        "proverbs_count": 1,
        "emotional_depth": "Medium"
        }}

        --- SCRIPT TO ANALYZE ---
        {script_text}
        --- END SCRIPT ---
        """


        # --- NEW: Step 6 - Analyze the Generated Script ---
        print("SCRIPT ANALYSIS: Analyzing generated script...")
        analysis_start_time = time.time()
        analysis_prompt_filled = ANALYSIS_PROMPT_TEMPLATE.format(script_text=script_response.text)
        
        analysis_response = await flash_model.generate_content_async(analysis_prompt_filled)
        analysis_end_time = time.time()
        print(f"--- PROFILING: Script analysis took {analysis_end_time - analysis_start_time:.2f} seconds ---")
        
        # --- NEW: Parse the analysis results (with error handling) ---
        analysis_results = {
            "examples_count": 0,
            "research_facts_count": 0,
            "proverbs_count": 0,
            "emotional_depth": "Unknown"
        }
        try:
            # Attempt to parse the JSON response from the analysis model
            analysis_data = json.loads(analysis_response.text)
            analysis_results["examples_count"] = analysis_data.get("examples_count", 0)
            analysis_results["research_facts_count"] = analysis_data.get("research_facts_count", 0)
            analysis_results["proverbs_count"] = analysis_data.get("proverbs_count", 0)
            analysis_results["emotional_depth"] = analysis_data.get("emotional_depth", "Unknown")
            print(f"Script Analysis Results: {analysis_results}")
        except json.JSONDecodeError:
            print("SCRIPT ANALYSIS: Failed to parse analysis JSON response from AI.")
        except Exception as e:
             print(f"SCRIPT ANALYSIS: Error during analysis parsing: {e}")
        # -----------------------------------------------------------

        total_end_time = time.time()
        print(f"--- PROFILING: Total /generate-script analysis request time was {total_end_time - total_start_time:.2f} seconds ---")
        
        
        generated_word_count = len(script_response.text.split())
        print(f"Generated script word count: approx. {generated_word_count}")



        # --- FINAL RETURN STATEMENT with all the data ---
        return {
            "script":script_response.text ,
            "estimated_word_count": generated_word_count,
            "source_urls": list(scraped_urls), # Use the correct list
            "analysis": analysis_results, # Add the analysis results
            "structure" : structure,
            "seo" : res


        }

    except Exception as e:
        print(f"SCRIPT GENERATION: An error occurred: {e}")
        return {"error": "An error occurred during the script generation pipeline."}

        
        # Calculate approximate word count of the generated script
        #generated_word_count = len(script_response.text.split())
        #print(f"Generated script word count: approx. {generated_word_count}")

        #return {"script": script_response.text, "estimated_word_count": generated_word_count}

    # except Exception as e:
    #     print(f"SCRIPT GENERATION: An error occurred: {e}")
    #     return {"error": "An error occurred during the script generation pipeline."}