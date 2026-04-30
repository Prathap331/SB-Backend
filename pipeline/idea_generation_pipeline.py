from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any
import os
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
from typing import Any

DEPTH_TARGET_WORDS = 2600
DEPTH_SUPPRESS_THRESHOLD = 60.0
DEPTH_PASS_THRESHOLD = 65.0
DEPTH_SOFT_MIN_WORDS = 500
DEPTH_CLEAN_MIN_WORDS = 900
DEFAULT_CAGS_THRESHOLD = 40.0
DEFAULT_RELAXED_CAGS_THRESHOLD = 20.0
DEFAULT_MAX_ANGLES = 5
DEFAULT_IDEAS_PER_ANGLE = 3
TOPIC_RELEVANCE_THRESHOLD = 0.42
GEMINI_EMBED_BLOCKED_UNTIL_TS = 0.0

SOURCE_WORD_CAPS = {
    "db_context": 3000,
    "web_context": 3000,
    "social_data": 800,
    "news_data": 1200,
    "angle_richness": 100,
}

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


print("dotenv loaded:", os.path.exists(".env"))
print("key exists:", "DEEPSEEK_API_KEY" in os.environ)
print("value:", os.environ.get("DEEPSEEK_API_KEY"))

def _normalize_topic(topic: str) -> str:
    return " ".join((topic or "").strip().lower().split())


def _payload_has_ideas(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if _payload_uses_fallback_variants(payload):
        return False
    ideas = payload.get("ideas")
    if isinstance(ideas, list) and any(str(item).strip() for item in ideas):
        return True
    clusters = payload.get("idea_clusters")
    if not isinstance(clusters, list) or not clusters:
        return False
    for cluster in clusters:
        variants = (cluster or {}).get("idea_variants")
        if not isinstance(variants, list):
            continue
        for variant in variants:
            title = str((variant or {}).get("title") or "").strip()
            desc = str((variant or {}).get("description") or "").strip()
            if title and desc:
                return True
    return False


def _payload_uses_fallback_variants(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    clusters = payload.get("idea_clusters")
    if not isinstance(clusters, list):
        return False
    for cluster in clusters:
        variants = (cluster or {}).get("idea_variants")
        if not isinstance(variants, list):
            continue
        for variant in variants:
            reason = str((variant or {}).get("gap_reason") or "").lower()
            if "fallback expansion" in reason:
                return True
    return False


# def _cosine_similarity(a: list[float], b: list[float]) -> float:
#     if not a or not b:
#         return 0.0
#     a_arr = np.asarray(a, dtype=np.float32)
#     b_arr = np.asarray(b, dtype=np.float32)
#     denom = float((np.linalg.norm(a_arr) or 0.0) * (np.linalg.norm(b_arr) or 0.0))
#     if denom == 0:
#         return 0.0
#     return float(np.dot(a_arr, b_arr) / denom)


DOMAIN_HINTS: dict[str, set[str]] = {
    "gaming": {
        "gaming", "game", "games", "gamer", "gamers", "esports", "noob", "pro", "hacker",
        "hackers", "stream", "streamer", "streamers", "tournament", "players", "player",
        "ranked", "matchmaking", "fps", "moba", "rpg", "battle", "rank", "clan", "guild",
    },
    "cybersecurity": {
        "cyber", "cybersecurity", "security", "hacker", "hackers", "malware", "phishing",
        "breach", "breaches", "ransomware", "exploit", "exploits", "vulnerability", "vulnerabilities",
    },
    "politics": {
        "government", "policy", "policies", "politics", "election", "elections", "law",
        "regulation", "regulations", "parliament", "congress", "senate", "minister", "agency",
        "agencies", "regulatory", "regulators",
    },
    "war": {
        "war", "conflict", "military", "battle", "ceasefire", "security", "defense",
        "geopolitics", "sanctions", "invasion", "strike", "missile", "drone",
    },
    "finance": {
        "finance", "financial", "market", "markets", "stock", "stocks", "crypto", "bitcoin",
        "economy", "economic", "bank", "banks", "trade", "trading", "investor", "investors",
    },
    "technology": {
        "ai", "artificial", "intelligence", "llm", "software", "developer", "developers",
        "technology", "tech", "automation", "startup", "startups", "computing", "cloud", "data",
    },
    "education": {
        "learn", "learning", "use", "using", "tutorial", "course", "courses", "student",
        "students", "teacher", "teachers", "beginner", "beginners", "skills", "training",
        "curriculum", "classroom", "education",
    },
}



def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def _cap_word_count(text: str, cap: int) -> int:
    return min(_count_words(text), cap)


@dataclass
class CachedIdeaResult:
    topic: str
    topic_key: str
    topic_vector: list[float] | None
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)

    @property
    def age_hours(self) -> float:
        return max((time.time() - self.created_at) / 3600.0, 0.0)


class SemanticIdeaCache:
    def __init__(self):
        self._items: list[CachedIdeaResult] = []

    def lookup(self, topic: str) -> dict[str, Any] | None:
        if not self._items:
            return None

        topic_key = _normalize_topic(topic)

        # exact match only
        exact = next((item for item in self._items if item.topic_key == topic_key), None)
        if exact:
            payload = dict(exact.payload)
            if not _payload_has_ideas(payload):
                return None
            payload["served_from_cache"] = True
            payload["cache_age_hours"] = round(exact.age_hours, 3)
            return payload

        return None

    def store(self, topic: str, payload: dict[str, Any]) -> None:
        topic_key = _normalize_topic(topic)

        # remove duplicates
        self._items = [item for item in self._items if item.topic_key != topic_key]

        self._items.append(
            CachedIdeaResult(
                topic=topic,
                topic_key=topic_key,
                topic_vector=None,
                payload=dict(payload),
            )
        )

        # cap size
        while len(self._items) > 300:
            self._items.pop(0)

TOPIC_CACHE = SemanticIdeaCache()

def _safe_id(x: Any) -> str:
    if isinstance(x, dict):
        return str(x.get("angle_id") or "")
    return str(x)

def select_candidate_angles(
    gap_angles: list[dict[str, Any]],
    used_angle_ids: list[str] | None,
    cags_threshold: float = DEFAULT_CAGS_THRESHOLD,
) -> list[dict[str, Any]]:
    used = set(_safe_id(x) for x in used_angle_ids or [])

    def _filter(threshold: float) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for angle in gap_angles:
            angle_id = angle.get("angle_id")
            if angle_id in used:
                continue

            score = float(angle.get("cags_score", 0.0) or 0.0)
            if score < threshold:
                continue

            kept.append(angle)
        return kept

    selected = _filter(cags_threshold)
    if not selected:
        selected = _filter(DEFAULT_RELAXED_CAGS_THRESHOLD)

    return selected


def apply_diversity_pass(
    candidate_angles: list[dict[str, Any]],
    *,
    max_per_who: int = 2,
    max_angles: int = DEFAULT_MAX_ANGLES,
) -> tuple[list[dict[str, Any]], bool]:
    selected: list[dict[str, Any]] = []
    who_counts: dict[str, int] = {}
    diversity_applied = False
    bypass_limit = max(1, max_angles // 2)

    for angle in candidate_angles:
        who = str(angle.get("who") or "").strip() or "Unknown"
        can_bypass = len(selected) < bypass_limit

        if who_counts.get(who, 0) < max_per_who or can_bypass:
            selected.append(angle)
            who_counts[who] = who_counts.get(who, 0) + 1
            if not can_bypass:
                diversity_applied = True

        if len(selected) >= max_angles:
            break

    return selected, diversity_applied

def _build_video_summary(angle: dict[str, Any]) -> str:
    best_video = angle.get("best_video")
    if not best_video:
        return "No existing video covers this angle."
    sim = float(best_video.get("similarity", 0.0) or 0.0)
    title = best_video.get("title") or "Unknown"
    views = int(best_video.get("views") or 0)
    coverage = "shallow" if sim < 0.6 else "moderate"
    return f'"{title}" - {views:,} views, similarity {sim:.2f} ({coverage} coverage)'


def _build_what_str(what: Any) -> str:
    if isinstance(what, list):
        return ", ".join(str(w) for w in what if str(w).strip())
    return str(what or "")


# def _safe_groq_parse(resp: Any) -> dict[str, Any]:
#     try:
#         content = resp.choices[0].message.content
#         return json.loads(content)
#     except Exception:
#         return {}


def _fallback_variant_title(topic: str, angle: dict[str, Any], index: int) -> str:
    who = str(angle.get("who") or "the audience").strip()
    frame = str(angle.get("story_frame") or "hidden angle").replace("_", " ").strip()
    when = str(angle.get("when") or "now").strip()
    how = str(angle.get("how") or "data-driven").replace("_", " ").strip()
    scale = str(angle.get("scale") or "global").strip()
    topic_clean = " ".join((topic or "").split())
    templates = [
        f"{topic_clean}: {who} {frame.title()} Signals {when.title()}",
        f"Who Gains, Who Pays? {topic_clean} Through {who}",
        f"{topic_clean} {scale.title()} Outlook: {who} on the {how.title()} Trade-offs",
        f"{topic_clean}: The {who} Case Nobody Is Tracking",
        f"{topic_clean} Explained for {who}: What Changes Next",
    ]
    return templates[index % len(templates)]



STYLE_BANK = [
    "an investigative journalist uncovering hidden incentives",
    "an economic analyst breaking down systemic impact",
    "a policy researcher mapping institutional effects",
    "a field reporter documenting real-world consequences",
    "a systems thinker analyzing feedback loops"
]

OPENING_BANK = [
    "{topic} is quietly reshaping incentives across stakeholders.",
    "Behind {topic}, there is a deeper structural shift most people miss.",
    "The real story of {topic} is not what appears on the surface.",
    "{topic} is creating uneven outcomes that are still unfolding.",
    "What looks like {topic} is actually a chain reaction of system-level changes."
]

# def _mutate(base: str, pool: list[str], index: int) -> str:
#     """Pick a controlled variation instead of repeating same field"""
#     if not pool:
#         return base
#     return pool[(hash(base) + index) % len(pool)]


def _fallback_expand_variants(
    angle: dict[str, Any],
    topic: str,
    briefs: list[dict[str, Any]],
    ideas_per_angle: int,
) -> list[dict[str, Any]]:
    variants = []
    for i in range(ideas_per_angle):
        variants.append({
            "variant_index": i + 1,
            "title": f"{topic} Angle {i+1}",
            "description": f"Fallback expansion for {topic}",
            "content_pillars": ["fallback", "fallback", "fallback"],
            "gap_reason": "fallback",
            "target_audience": angle.get("who"),
            "hook_strategy": "fallback hook",
        })
    return variants


def _extract_json_object(raw: str) -> dict[str, Any]:
    if not raw:
        return {}

    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1:
        return {}

    candidate = raw[start:end+1]

    try:
        return json.loads(candidate)
    except Exception:
        return {}


def _normalized_angle_gap_reason(angle: dict[str, Any]) -> str:
    coverage = str(angle.get("coverage_label") or "NOT_COVERED").replace("_", " ").lower()
    who = str(angle.get("who") or "this audience").strip()
    return f"Low direct coverage for {who}; strong opportunity in current content landscape ({coverage})."

def _normalize_variant(
    *,
    variant: dict[str, Any],
    index: int,
    topic: str,
    angle: dict[str, Any],
    briefs: list[dict[str, Any]],
) -> dict[str, Any]:
    seed = get_cags_brief_seed(angle.get("angle_id"), briefs)
    who = str(angle.get("who") or "the audience").strip()
    what = _build_what_str(angle.get("what")) or "the core topic"
    when = str(angle.get("when") or "now").strip()
    scale = str(angle.get("scale") or "global").strip()
    # how = str(angle.get("how") or "contrast").strip()
    # story_frame = str(angle.get("story_frame") or "curiosity").replace("_", " ").strip()

    title = str(variant.get("title") or "").strip()
    if not title:
        if index == 0 and seed.get("suggested_title"):
            title = str(seed.get("suggested_title") or "").strip()
        else:
            title = _fallback_variant_title(topic, angle, index)

    description = str(variant.get("description") or "").strip()
    # if len(description) < 80:
    #     description = fallback:
        # description = (
        #     f"Explore {topic} through {who} while focusing on {what}. "
        #     f"Frame it around a {when} {scale} lens and use a {how} structure to reveal who benefits, "
        #     f"who loses, and why the story still matters. End with a curiosity gap about {story_frame}."
        # )

    pillars = variant.get("content_pillars")
    if not isinstance(pillars, list) or len(pillars) < 3:
        content_pillars = [what, when, scale]
    else:
        clean = [str(p).strip() for p in pillars if str(p).strip()]
        while len(clean) < 3:
            clean.append([what, when, scale][len(clean)])
        content_pillars = clean[:3]

    gap_reason = str(variant.get("gap_reason") or "").strip() or _normalized_angle_gap_reason(angle)
    target_audience = str(variant.get("target_audience") or "").strip() or who
    hook_strategy = str(variant.get("hook_strategy") or "").strip()
    if not hook_strategy:
        hook_strategy = str(seed.get("hook_sentence") or f"What if {topic} looked different through the eyes of {who}?").strip()

    return {
        "variant_index": index + 1,
        "title": title,
        "description": description,
        "content_pillars": content_pillars,
        "gap_reason": gap_reason,
        "target_audience": target_audience,
        "hook_strategy": hook_strategy,
        "who_benefits": str(variant.get("who_benefits") or angle.get("who_benefits") or "unclear"),
        "story_frame": str(variant.get("story_frame") or angle.get("story_frame") or "curiosity"),
    }



def _normalize_variants_payload(
    *,
    parsed: dict[str, Any],
    topic: str,
    angle: dict[str, Any],
    briefs: list[dict[str, Any]],
    ideas_per_angle: int,
) -> list[dict[str, Any]]:

    if not isinstance(parsed, dict):
        return []

    variants_raw = parsed.get("variants")

    if not isinstance(variants_raw, list):
        return []
    if not isinstance(variants_raw, list):
        return []

    normalized: list[dict[str, Any]] = []

    for idx, item in enumerate(variants_raw[:ideas_per_angle]):
        if not isinstance(item, dict):
            continue

        normalized.append(
            _normalize_variant(
                variant=item,
                index=idx,
                topic=topic,
                angle=angle,
                briefs=briefs,
            )
        )

    # fallback fill if model returned fewer variants
    while len(normalized) < ideas_per_angle:
        idx = len(normalized)
        normalized.append(
            _normalize_variant(
                variant={},
                index=idx,
                topic=topic,
                angle=angle,
                briefs=briefs,
            )
        )

    return normalized




async def expand_angle_ideas(
    angle: dict[str, Any],
    topic: str,
    briefs: list[dict[str, Any]],
    ideas_per_angle: int,
    deepseek_client: Any,
) -> list[dict[str, Any]] | None:

    seed = get_cags_brief_seed(angle.get("angle_id"), briefs)

    prompt = MULTI_IDEA_EXPANSION_PROMPT.format(
        n=ideas_per_angle,
        topic=topic,
        angle_string=angle.get("angle_string"),
        who=angle.get("who"),
        what=_build_what_str(angle.get("what")),
        when=angle.get("when"),
        scale=angle.get("scale"),
        how=angle.get("how"),
        who_benefits=angle.get("who_benefits"),
        story_frame=angle.get("story_frame"),
        coverage_label=angle.get("coverage_label"),
        best_video_summary=_build_video_summary(angle),
        demand_score_pct=round(float(angle.get("demand_score", 0.0) or 0.0) * 100),
        cags_score=angle.get("cags_score", 0),
        suggested_title=seed.get("suggested_title"),
        hook_sentence=seed.get("hook_sentence"),
    )

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: deepseek_client.chat.completions.create(  
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}}
            )
    )     

        raw = response.choices[0].message.content
        print("raw", raw)
        parsed = _extract_json_object(raw)
        print("parsed", parsed)

        return _normalize_variants_payload(
            parsed=parsed,
            topic=topic,
            angle=angle,
            briefs=briefs,
            ideas_per_angle=ideas_per_angle,
        )

    except Exception as e:
        print(f"[error] deepseek failed: {e}")
        return _fallback_expand_variants(angle, topic, briefs, ideas_per_angle)
    
    
async def expand_angle_deepseek(angle, topic, briefs, ideas_per_angle, client):
    prompt = MULTI_IDEA_EXPANSION_PROMPT.format(
        n=ideas_per_angle,
        topic=topic,
        angle_string=angle.get("angle_string"),
        who=angle.get("who"),
        what=_build_what_str(angle.get("what")),
        when=angle.get("when"),
        scale=angle.get("scale"),
        how=angle.get("how"),
        who_benefits=angle.get("who_benefits"),
        story_frame=angle.get("story_frame"),
        coverage_label=angle.get("coverage_label"),
        best_video_summary=_build_video_summary(angle),
        demand_score_pct=round(float(angle.get("demand_score", 0.0) or 0.0) * 100),
        cags_score=angle.get("cags_score", 0),
    )

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(  
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}}
        )
    )
    text = response.choices[0].message.content
    parsed = _extract_json_object(text)

    return _normalize_variants_payload(
        parsed=parsed,
        topic=topic,
        angle=angle,
        briefs=briefs,
        ideas_per_angle=ideas_per_angle,
    )


async def expand_all_angles(
    selected_angles,
    topic,
    briefs,
    ideas_per_angle,
    deepseek_client,
):
    tasks = [
        expand_angle_ideas(
            angle,
            topic,
            briefs,
            ideas_per_angle,
            deepseek_client,
        )
        for angle in selected_angles
    ]

    return [
        (angle, result)
        for angle, result in zip(selected_angles, await asyncio.gather(*tasks))
    ]


def build_idea_cluster(source_angle: dict[str, Any], variants: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "angle_id": source_angle["angle_id"],
        "angle_string": source_angle["angle_string"],
        "cags_score": source_angle["cags_score"],
        "who": source_angle["who"],
        "idea_variants": variants,
        "variant_count": len(variants),
    }

def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n\n".join(_coerce_text(v) for v in value)
    if isinstance(value, dict):
        return "\n".join(f"{k}: {_coerce_text(v)}" for k, v in value.items())
    return str(value or "")


def apply_depth_check(
    clusters: list[dict[str, Any]],
    db_context: str,
    web_context: str,
    social_data: list[dict[str, Any]],
    news_data: list[dict[str, Any]],
):
    db_words = _cap_word_count(db_context, SOURCE_WORD_CAPS["db_context"])
    web_words = _cap_word_count(web_context, SOURCE_WORD_CAPS["web_context"])
    social_words = sum(_count_words(str(x)) for x in social_data)
    news_words = sum(_count_words(str(x)) for x in news_data)

    passing = []
    suppressed = []

    for c in clusters:
        total = db_words + web_words + social_words + news_words
        if total >= DEPTH_SOFT_MIN_WORDS:
            passing.append(c)
        else:
            suppressed.append(c)

    summary = {
        "total_words": db_words + web_words + social_words + news_words,
    }

    return passing, suppressed, summary


def assemble_response(
    topic: str,
    gap_angles: list[dict[str, Any]],
    selected_angles: list[dict[str, Any]],
    expanded: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    passing: list[dict[str, Any]],
    suppressed: list[dict[str, Any]],
    depth_summary: dict[str, Any],
    diversity_applied: bool,
    *,
    served_from_cache: bool = False,
    cache_age_hours: float | None = None,
) -> dict[str, Any]:
    passing.sort(key=lambda x: x.get("cags_score", 0), reverse=True)
    angles_expanded = len(expanded)
    failed_expansion = max(0, len(selected_angles) - angles_expanded)
    return {
        "topic": topic,
        "idea_clusters": passing,
        "total_gaps_found": len(gap_angles),
        "angles_selected": len(selected_angles),
        "angles_expanded": angles_expanded,
        "angles_failed_expansion": failed_expansion,
        "angles_processed": angles_expanded,
        "clusters_passed": len(passing),
        "clusters_suppressed": len(suppressed),
        "total_ideas_generated": sum(cluster["variant_count"] for cluster in passing),
        "diversity_applied": diversity_applied,
        "depth_check_summary": depth_summary,
        "served_from_cache": served_from_cache,
        "cache_age_hours": cache_age_hours,
        "mode": "cags_aligned",
    }


def get_cags_brief_seed(angle_id: str | None, briefs: list[dict[str, Any]]) -> dict[str, str]:
    for brief in briefs or []:
        if brief.get("angle_id") == angle_id:
            return {
                "suggested_title": str(brief.get("suggested_title") or ""),
                "hook_sentence": str(brief.get("hook_sentence") or ""),
            }
    return {"suggested_title": "", "hook_sentence": ""}


async def generate_ideas(
    *,
    topic: str,
    gap_angles: list[dict[str, Any]],
    briefs: list[dict[str, Any]],
    perspective_tree: list[dict[str, Any]],
    social_data: list[dict[str, Any]],
    news_data: list[dict[str, Any]],
    db_context: str,
    web_context: str,
    max_angles: int = DEFAULT_MAX_ANGLES,
    ideas_per_angle: int = DEFAULT_IDEAS_PER_ANGLE,
    used_angle_ids: list[str] | None = None,
    deepseek_client: Any | None = None,
    groq_client: Any | None = None,   
):
    
    llm_client = deepseek_client or groq_client

    if llm_client is None:
        raise ValueError("No LLM client provided")
    
    if not topic or not gap_angles:
        raise ValueError("invalid input")

    topic = topic.strip()

    candidates = select_candidate_angles(gap_angles, used_angle_ids)
    selected, diversity = apply_diversity_pass(candidates, max_angles=max_angles)

    expanded = await expand_all_angles(
        selected_angles=selected,
        topic=topic,
        briefs=briefs,
        ideas_per_angle=ideas_per_angle,
        deepseek_client=llm_client,
    )

    clusters = [
        build_idea_cluster(angle, variants)
        for angle, variants in expanded
    ]

    passing, suppressed, depth = apply_depth_check(
        clusters,
        db_context,
        web_context,
        social_data,
        news_data,
    )

    return {
        "topic": topic,
        "idea_clusters": passing,
        "diversity_applied": diversity,
        "depth_check_summary": depth,
        "total_clusters": len(clusters),
    }

async def regenerate_with_expansion(
    client: Any,
    topic: str,
    old_result: dict[str, Any],
    *,
    gap_angles: list[dict[str, Any]],
    ideas_per_angle: int = 3,
) -> dict[str, Any]:
    return await generate_ideas(
        topic=topic,
        gap_angles=gap_angles,
        briefs=[],
        perspective_tree=[],
        social_data=[],
        news_data=[],
        db_context="",
        web_context="",
        ideas_per_angle=ideas_per_angle,
        deepseek_client=client,
    )


MULTI_IDEA_EXPANSION_PROMPT = """
You are generating {n} distinct YouTube idea variants for the same CAGS angle.

Topic: {topic}

Angle:
- angle_string: {angle_string}
- who: {who}
- what: {what}
- when: {when}
- scale: {scale}
- how: {how}
- who_benefits: {who_benefits}
- story_frame: {story_frame}
- coverage_label: {coverage_label}
- best_video_summary: {best_video_summary}
- demand_score_pct: {demand_score_pct}
- cags_score: {cags_score}

Seed brief:
- suggested_title: {suggested_title}
- hook_sentence: {hook_sentence}

Rules:
1. Return exactly {n} variants.
2. Each variant MUST differ in narrative style, hook type, and audience perspective.
3. Avoid repeating structure or phrasing across variants.
4. Each idea must feel like a completely different video concept.
5. Include:
   - at least 1 controversial or contrarian idea
   - at least 1 highly practical or actionable idea

OUTPUT FORMAT (STRICT):
Return ONLY valid JSON.
Do NOT include explanations, markdown, or extra text.

Example format:
{{
  "variants": [
    {{
      "variant_index": 1,
      "title": "Example title",
      "description": "Clear explanation of the video idea",
      "content_pillars": ["hook", "insight", "value"],
      "gap_reason": "Why this angle is underserved",
      "target_audience": "Who this is for",
      "hook_strategy": "Curiosity / controversy / authority etc."
    }}
  ]
}}
"""