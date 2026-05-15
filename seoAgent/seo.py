from shared.schemas.pipeline_context import (
    AgentPipelineContext,
    extract_angle_for_prompt,
)
from fastapi import HTTPException
from pydantic import BaseModel
from ddgs import DDGS
import asyncio
import json
from openai import AsyncOpenAI
import re
from datetime import datetime
import os



SEO_SYNTHESIS_PROMPT = """
You are an expert YouTube SEO Analyst and Title Strategist.

VIDEO ANGLE: "{angle_string}"
STAKEHOLDER: {who} | LENS: {what} | FRAME: {story_frame}
AUDIENCE PROFILE: {audience_profile}
CATEGORY: {cat_id} — {cat_label}

COMPETITIVE DATA:
Top 5 YouTube Titles: {competing_titles}
Top 5 PAA Questions: {paa_questions}

CTR SIGNAL:
ctr_potential: {ctr_label} (score: {ctr_score})
{degraded_note}

Return ONLY valid JSON:

{{
  "seo": {{
    "search_intent_type": "",
    "recommended_structure": "",
    "ctr_potential": "",
    "ctr_signal_degraded": false,
    "ctr_score": 0.0,
    "justification": "",
    "angle": "",
    "angle_id": "",
    "key_questions_to_answer": [],
    "chapter_structure": [],
    "channel_context_unavailable": false,
    
    "recommended_titles": [
      {{
        "type": "curiosity_gap",
        "title": "",
        "desc": "",
        "selected": false
      }},
      {{
        "type": "data_led",
        "title": "",
        "desc": "",
        "selected": false
      }},
      {{
        "type": "how_to",
        "title": "",
        "desc": "",
        "selected": false
      }},
      {{
        "type": "narrative",
        "title": "",
        "desc": "",
        "selected": false
      }}
    ],

    "keyword_clusters": {{
      "primary": [],
      "secondary": [],
      "longtail": [],
      "question_based": []
    }},

    "thumbnail_brief": [
      {{
        "type": "curiosity_gap",
        "img_desc": "",
        "prompt": ""
      }},
      {{
        "type": "data_driven",
        "img_desc": "",
        "prompt": ""
      }}
    ],

    "description_template": {{
      "hook": "",
      "body_bullets": [],
      "outro": ""
    }},

    "hashtags": []
  }}
}}
"""




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

groq_api_key = os.getenv("GROQ_API_KEY")



class context(BaseModel):
    def __init__(self,topic,keywords,selected_idea_id=None,selected_angle_id=None,pipeline_assembled_at: datetime | None = None):
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


class SEOAgentRequest(BaseModel):
    context: AgentPipelineContext
    channel_context: ChannelContextInput | None = None



def get_context(request):
    return request.context if hasattr(request, "context") else request["context"]


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

GROQ_GENERATION_MODEL = "llama-3.1-8b-instant"
GROQ_IDEA_CLIENTS = [AsyncOpenAI(api_key=key, base_url="https://api.groq.com/openai/v1") for key in GROQ_IDEA_KEYS]


async def groq_idea_generate(messages: list, model: str = GROQ_GENERATION_MODEL) -> str:
    return await _groq_generate_with_slots(messages, GROQ_IDEA_CLIENTS, model, "idea")



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



async def seo_agent(request: SEOAgentRequest):
    ctx = get_context(request)
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
        raw = await groq_idea_generate(
            [{"role": "user", "content": prompt}],
            model=GROQ_GENERATION_MODEL
        )
        parsed = _parse_json_object(raw)
    except Exception as exc:
        print(f"SEO synthesis failed: {exc}")
        parsed = {}

    # SAFE CLEAN
    def _clean_keys(d: dict):
        if not isinstance(d, dict):
            return {}
        return {str(k).strip(): v for k, v in d.items()}

    parsed = _clean_keys(parsed)
    parsed_seo = _clean_keys(parsed.get("seo", {}))

    seo = fallback.copy()
    seo.update(parsed_seo)

    seo["search_intent_type"] = _first_allowed_pipe_token(
        seo.get("search_intent_type"),
        SEO_INTENT_TYPES,
        "educational",
    )

    seo["recommended_structure"] = _first_allowed_pipe_token(
        seo.get("recommended_structure"),
        SEO_STRUCTURES,
        "problem_solution",
    )

    seo["ctr_potential"] = ctr_label
    seo["ctr_signal_degraded"] = ctr_degraded
    seo["ctr_score"] = ctr_score

    seo["recommended_titles"] = _safe_recommended_titles(
        seo.get("recommended_titles"),
        blocked_types
    )

    deduped = _deduplicate_hashtags(
        list(seo.get("hashtags") or []),
        existing_hashtags
    )

    seo["hashtags"] = _ensure_hashtag_floor(
        deduped,
        ctx.topic,
        existing_hashtags
    )

    seo["chapter_structure"] = _ensure_chapter_structure(
        seo.get("chapter_structure")
    )

    seo["angle"] = angle_string
    seo["angle_id"] = ctx.selected_angle_id
    seo["channel_context_unavailable"] = not bool(
        request.channel_context and request.channel_context.channel_id
    )

    seo["key_questions_to_answer"] = list(
        seo.get("key_questions_to_answer") or paa_questions[:5]
    )[:8]

    return {"seo": seo} 