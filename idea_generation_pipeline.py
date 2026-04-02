from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

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


def _normalize_topic(topic: str) -> str:
    return " ".join((topic or "").strip().lower().split())


def _payload_has_ideas(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    ideas = payload.get("ideas")
    if isinstance(ideas, list) and len(ideas) > 0:
        return True
    clusters = payload.get("idea_clusters")
    if isinstance(clusters, list) and len(clusters) > 0:
        return True
    return False


def _is_embedding_quota_error(error_text: str) -> bool:
    text = (error_text or "").lower()
    return "resource_exhausted" in text or "quota" in text


def _gemini_embeddings_blocked() -> bool:
    return time.time() < GEMINI_EMBED_BLOCKED_UNTIL_TS


def _block_gemini_embeddings_for(seconds: float = 3600.0) -> None:
    global GEMINI_EMBED_BLOCKED_UNTIL_TS
    GEMINI_EMBED_BLOCKED_UNTIL_TS = max(GEMINI_EMBED_BLOCKED_UNTIL_TS, time.time() + seconds)


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", (text or "").lower())
        if len(token) > 1
    }


def _angle_relevance_text(angle: dict[str, Any]) -> str:
    return " ".join(
        part for part in [
            str(angle.get("angle_string") or ""),
            str(angle.get("who") or ""),
            " ".join(map(str, angle.get("what") or [])),
            str(angle.get("when") or ""),
            str(angle.get("scale") or ""),
            str(angle.get("how") or ""),
            str(angle.get("who_benefits") or ""),
            str(angle.get("story_frame") or ""),
        ]
        if part and str(part).strip()
    ).strip()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    denom = float((np.linalg.norm(a_arr) or 0.0) * (np.linalg.norm(b_arr) or 0.0))
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def _embed_texts_with_gemini(client: Any, texts: list[str]) -> list[list[float]] | None:
    if client is None or not texts or _gemini_embeddings_blocked():
        return None
    try:
        from google.genai import types as gt
    except Exception:
        return None
    models = getattr(client, "models", None)
    if models is None or not hasattr(models, "embed_content"):
        return None
    try:
        resp = models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=gt.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=768,
            ),
        )
        vecs = [list(e.values) for e in getattr(resp, "embeddings", []) or []]
        if len(vecs) != len(texts) or any(len(vec) != 768 for vec in vecs):
            return None
        return vecs
    except Exception as exc:
        if _is_embedding_quota_error(str(exc)):
            _block_gemini_embeddings_for(3600.0)
        return None


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


def _infer_topic_domains(topic: str) -> set[str]:
    tokens = _token_set(topic)
    raw = (topic or "").lower()
    detected: set[str] = set()
    for domain, anchors in DOMAIN_HINTS.items():
        if tokens & anchors or any(anchor in raw for anchor in anchors):
            detected.add(domain)
    return detected


def _topic_relevance_details(
    topic: str,
    angle: dict[str, Any],
    *,
    topic_vector: list[float] | None = None,
    angle_vector: list[float] | None = None,
) -> dict[str, Any]:
    topic_text = _normalize_topic(topic)
    angle_text = _normalize_topic(_angle_relevance_text(angle))
    topic_tokens = _token_set(topic_text)
    angle_tokens = _token_set(angle_text)
    lexical_hits = topic_tokens & angle_tokens
    lexical_score = len(lexical_hits) / max(len(topic_tokens), 1)

    semantic_score = 0.0
    if topic_vector is not None and angle_vector is not None:
        semantic_score = _cosine_similarity(topic_vector, angle_vector)

    topic_domains = _infer_topic_domains(topic_text)
    angle_domains = set()
    for domain, anchors in DOMAIN_HINTS.items():
        if angle_tokens & anchors or any(anchor in angle_text for anchor in anchors):
            angle_domains.add(domain)

    domain_overlap = topic_domains & angle_domains
    score = (semantic_score * 0.55) + (lexical_score * 0.20)
    if topic_domains:
        if domain_overlap:
            score += 0.55
            score += 0.03 * min(len(domain_overlap), 3)
        else:
            score -= 0.35

    who = angle.get("who") or ""
    who_text = str(who).lower()
    topic_requests_policy = any(
        term in topic_text for term in {"policy", "law", "regulation", "regulatory", "government", "geopolitics"}
    )
    learning_intent = any(
        term in topic_text for term in {"learn", "learning", "how to", "tutorial", "beginner", "use ai", "using ai"}
    )

    if topic_domains and "gaming" in topic_domains:
        if any(term in who_text for term in {"government", "policy", "regulatory", "regulation", "law", "agency", "agencies"}):
            score -= 0.25
    if topic_domains and "war" in topic_domains:
        if any(term in who_text for term in {"gaming", "esports", "stream", "streamer"}):
            score -= 0.25
    if learning_intent and ({"technology", "education"} & topic_domains) and not topic_requests_policy:
        if any(
            term in who_text
            for term in {
                "government", "agency", "agencies", "regulator", "regulatory", "policy",
                "international organization", "ethics committee",
            }
        ):
            score -= 0.40
        if any(
            term in who_text
            for term in {
                "student", "teacher", "developer", "creator", "engineer", "professional",
                "freelancer", "entrepreneur", "educator",
            }
        ):
            score += 0.20

    score = max(0.0, min(score, 1.0))
    reasons = []
    if lexical_hits:
        reasons.append(f"lexical:{','.join(sorted(lexical_hits)[:5])}")
    if domain_overlap:
        reasons.append(f"domain:{','.join(sorted(domain_overlap))}")
    if semantic_score:
        reasons.append(f"semantic:{semantic_score:.2f}")
    if not reasons:
        reasons.append("weak_topic_match")
    return {
        "score": round(score, 3),
        "reasons": reasons,
        "topic_domains": sorted(topic_domains),
        "angle_domains": sorted(angle_domains),
    }


def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def _cap_word_count(text: str, cap: int) -> int:
    return min(_count_words(text), cap)


def _angle_payload_text(angle: dict[str, Any]) -> str:
    parts = [
        str(angle.get("angle_string") or ""),
        str(angle.get("who") or ""),
        " ".join(map(str, angle.get("what") or [])),
        str(angle.get("when") or ""),
        str(angle.get("scale") or ""),
        str(angle.get("how") or ""),
        str(angle.get("who_benefits") or ""),
        str(angle.get("story_frame") or ""),
    ]
    return " ".join(p for p in parts if p).strip()


def _extract_social_texts(social_data: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for item in social_data or []:
        if not isinstance(item, dict):
            continue
        pieces = [
            str(item.get("title") or item.get("headline") or ""),
            str(item.get("body") or item.get("snippet") or item.get("text") or ""),
        ]
        text = " ".join(p for p in pieces if p).strip()
        if text:
            texts.append(text)
    return texts


def _extract_news_texts(news_data: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for item in news_data or []:
        if not isinstance(item, dict):
            continue
        pieces = [
            str(item.get("title") or item.get("headline") or ""),
            str(item.get("body") or item.get("snippet") or item.get("description") or item.get("text") or ""),
        ]
        text = " ".join(p for p in pieces if p).strip()
        if text:
            texts.append(text)
    return texts


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
    def __init__(self, similarity_threshold: float = 0.92):
        self._threshold = similarity_threshold
        self._items: list[CachedIdeaResult] = []

    def _vector_from_client(self, topic: str, gemini_client: Any | None) -> list[float] | None:
        if gemini_client is None or _gemini_embeddings_blocked():
            return None
        try:
            from google.genai import types as gt
        except Exception as exc:
            if _is_embedding_quota_error(str(exc)):
                _block_gemini_embeddings_for(3600.0)
            return None
        try:
            resp = gemini_client.models.embed_content(
                model="gemini-embedding-001",
                contents=[_normalize_topic(topic)],
                config=gt.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=768,
                ),
            )
            return list(resp.embeddings[0].values)
        except Exception:
            return None

    def lookup(self, topic: str, gemini_client: Any | None = None) -> dict[str, Any] | None:
        if not self._items:
            return None
        topic_key = _normalize_topic(topic)
        exact = next((item for item in self._items if item.topic_key == topic_key), None)
        if exact:
            payload = dict(exact.payload)
            if not _payload_has_ideas(payload):
                return None
            payload["served_from_cache"] = True
            payload["cache_age_hours"] = round(exact.age_hours, 3)
            return payload

        topic_vec = self._vector_from_client(topic, gemini_client)
        if topic_vec is None:
            return None
        topic_arr = np.asarray(topic_vec, dtype=np.float32)
        topic_norm = float(np.linalg.norm(topic_arr) or 0.0)
        if topic_norm == 0:
            return None

        best_item: CachedIdeaResult | None = None
        best_sim = 0.0
        for item in self._items:
            if not item.topic_vector:
                continue
            cached = np.asarray(item.topic_vector, dtype=np.float32)
            denom = float((np.linalg.norm(cached) or 0.0) * topic_norm)
            if denom == 0:
                continue
            sim = float(np.dot(topic_arr, cached) / denom)
            if sim >= self._threshold and sim > best_sim:
                best_sim = sim
                best_item = item
        if best_item:
            payload = dict(best_item.payload)
            if not _payload_has_ideas(payload):
                return None
            payload["served_from_cache"] = True
            payload["cache_age_hours"] = round(best_item.age_hours, 3)
            payload["cache_similarity"] = round(best_sim, 3)
            return payload
        return None

    def store(self, topic: str, payload: dict[str, Any], gemini_client: Any | None = None) -> None:
        topic_key = _normalize_topic(topic)
        topic_vector = self._vector_from_client(topic, gemini_client)
        self._items = [item for item in self._items if item.topic_key != topic_key]
        self._items.append(
            CachedIdeaResult(
                topic=topic,
                topic_key=topic_key,
                topic_vector=topic_vector,
                payload=dict(payload),
            )
        )
        while len(self._items) > 300:
            self._items.pop(0)


TOPIC_CACHE = SemanticIdeaCache()


def select_candidate_angles(
    topic: str,
    gap_angles: list[dict[str, Any]],
    used_angle_ids: list[str] | None,
    gemini_client: Any | None = None,
    cags_threshold: float = DEFAULT_CAGS_THRESHOLD,
) -> list[dict[str, Any]]:
    used = set(used_angle_ids or [])
    topic_vec = None
    angle_vecs: dict[str, list[float]] = {}
    if gemini_client is not None:
        texts = [_normalize_topic(topic)] + [_angle_relevance_text(angle) for angle in gap_angles]
        vecs = _embed_texts_with_gemini(gemini_client, texts)
        if vecs and len(vecs) == len(texts):
            topic_vec = vecs[0]
            for angle, vec in zip(gap_angles, vecs[1:]):
                angle_id = str(angle.get("angle_id") or "")
                if angle_id:
                    angle_vecs[angle_id] = vec

    def _filter(threshold: float) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for angle in gap_angles:
            angle_id = angle.get("angle_id")
            if angle_id in used:
                continue
            relevance = _topic_relevance_details(
                topic,
                angle,
                topic_vector=topic_vec,
                angle_vector=angle_vecs.get(str(angle_id or "")),
            )
            if relevance["score"] < TOPIC_RELEVANCE_THRESHOLD:
                continue
            if float(angle.get("cags_score", 0.0) or 0.0) < threshold:
                continue
            enriched = dict(angle)
            enriched["topic_relevance_score"] = relevance["score"]
            enriched["topic_relevance_reasons"] = relevance["reasons"]
            kept.append(enriched)
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
    if not candidate_angles:
        return [], False

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


def _safe_groq_parse(resp: Any) -> dict[str, Any]:
    try:
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception:
        return {}


def _fallback_variant_title(topic: str, angle: dict[str, Any], index: int) -> str:
    who = str(angle.get("who") or "the audience").strip()
    frame = str(angle.get("story_frame") or "unexpected").replace("_", " ").strip()
    when = str(angle.get("when") or "now").strip()
    topic_clean = " ".join((topic or "").split())
    return f"{topic_clean}: {who} {frame.title()} {when.title()} Angle {index + 1}"


def _fallback_expand_variants(
    angle: dict[str, Any],
    topic: str,
    briefs: list[dict[str, Any]],
    ideas_per_angle: int,
) -> list[dict[str, Any]]:
    seed = get_cags_brief_seed(angle.get("angle_id"), briefs)
    who = str(angle.get("who") or "the audience").strip()
    what = _build_what_str(angle.get("what")) or "the core topic"
    when = str(angle.get("when") or "now").strip()
    scale = str(angle.get("scale") or "global").strip()
    how = str(angle.get("how") or "contrast").strip()
    who_benefits = str(angle.get("who_benefits") or "unclear").strip()
    story_frame = str(angle.get("story_frame") or "curiosity").replace("_", " ").strip()
    base_hook = seed.get("hook_sentence") or f"What if {topic} looked different through the eyes of {who}?"
    base_title = seed.get("suggested_title") or _fallback_variant_title(topic, angle, 0)

    variants: list[dict[str, Any]] = []
    for index in range(max(1, ideas_per_angle)):
        variant_title = base_title if index == 0 else _fallback_variant_title(topic, angle, index)
        variant_description = (
            f"Explore {topic} through {who} while focusing on {what}. "
            f"Frame it around a {when} {scale} lens and use a {how} structure to reveal who benefits, "
            f"who loses, and why the story still matters. End with a curiosity gap about {story_frame}."
        )
        variants.append(
            {
                "variant_index": index + 1,
                "title": variant_title,
                "description": variant_description,
                "content_pillars": [what, when, scale],
                "gap_reason": f"Fallback expansion for {who} because the model response was unavailable.",
                "target_audience": who,
                "hook_strategy": base_hook,
                "who_benefits": who_benefits,
                "story_frame": story_frame,
            }
        )
    return variants[:ideas_per_angle]


async def expand_angle_ideas(
    angle: dict[str, Any],
    topic: str,
    briefs: list[dict[str, Any]],
    ideas_per_angle: int,
    groq_client: Any,
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
        resp = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1800,
            temperature=0.85,
        )
        parsed = json.loads(resp.choices[0].message.content)
        required = ["title", "description", "content_pillars", "gap_reason", "target_audience", "hook_strategy"]
        valid = [
            variant for variant in parsed.get("variants", [])
            if all(variant.get(field) for field in required)
            and isinstance(variant.get("content_pillars"), list)
            and len(variant["content_pillars"]) == 3
        ]
        return valid if valid else _fallback_expand_variants(angle, topic, briefs, ideas_per_angle)
    except Exception:
        return _fallback_expand_variants(angle, topic, briefs, ideas_per_angle)


async def expand_all_angles(
    selected_angles: list[dict[str, Any]],
    topic: str,
    briefs: list[dict[str, Any]],
    ideas_per_angle: int,
    groq_client: Any,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    tasks = [
        expand_angle_ideas(angle, topic, briefs, ideas_per_angle, groq_client)
        for angle in selected_angles
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    paired: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for angle, result in zip(selected_angles, results):
        if isinstance(result, Exception) or not result:
            paired.append((angle, _fallback_expand_variants(angle, topic, briefs, ideas_per_angle)))
            continue
        paired.append((angle, result))
    return paired


def build_idea_cluster(source_angle: dict[str, Any], variants: list[dict[str, Any]]) -> dict[str, Any]:
    idea_variants = [
        {
            "variant_index": variant["variant_index"],
            "title": variant["title"],
            "description": variant["description"],
            "content_pillars": variant["content_pillars"],
            "gap_reason": variant["gap_reason"],
            "target_audience": variant["target_audience"],
            "hook_strategy": variant["hook_strategy"],
        }
        for variant in variants
    ]
    return {
        "angle_id": source_angle["angle_id"],
        "angle_string": source_angle["angle_string"],
        "cags_score": source_angle["cags_score"],
        "cags_rank": source_angle["rank"],
        "coverage_label": source_angle["coverage_label"],
        "who": source_angle["who"],
        "what": source_angle["what"],
        "when": source_angle["when"],
        "scale": source_angle["scale"],
        "how": source_angle["how"],
        "who_benefits": source_angle["who_benefits"],
        "story_frame": source_angle["story_frame"],
        "best_video": source_angle.get("best_video"),
        "gap_reason": source_angle.get("gap_reason"),
        "variant_count": len(idea_variants),
        "idea_variants": idea_variants,
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    db_words = _cap_word_count(db_context, SOURCE_WORD_CAPS["db_context"])
    web_words = _cap_word_count(web_context, SOURCE_WORD_CAPS["web_context"])
    social_words = min(sum(_count_words(text) for text in _extract_social_texts(social_data)), SOURCE_WORD_CAPS["social_data"])
    news_words = min(sum(_count_words(text) for text in _extract_news_texts(news_data)), SOURCE_WORD_CAPS["news_data"])

    passing: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    assessed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for cluster in clusters:
        angle_words = min(_count_words(_coerce_text(cluster.get("angle_string"))), SOURCE_WORD_CAPS["angle_richness"])
        available_words = db_words + web_words + social_words + news_words + angle_words
        depth_percent = round((available_words / DEPTH_TARGET_WORDS) * 100.0, 1)
        depth_status = "suppressed"
        depth_warning = False
        if available_words >= DEPTH_CLEAN_MIN_WORDS:
            depth_status = "pass"
            depth_warning = depth_percent < 80.0
        elif available_words >= DEPTH_SOFT_MIN_WORDS:
            depth_status = "warning"
            depth_warning = True

        depth_block = {
            "available_words": available_words,
            "required_words": DEPTH_TARGET_WORDS,
            "depth_percent": depth_percent,
            "status": depth_status,
            "depth_warning": depth_warning,
            "depth_assessed_at": assessed_at,
            "source_breakdown": {
                "db_context": db_words,
                "web_context": web_words,
                "social_data": social_words,
                "news_data": news_words,
                "angle_richness": angle_words,
            },
        }
        updated = dict(cluster)
        updated["content_depth"] = depth_block
        updated["depth_assessed_at"] = assessed_at
        updated["depth_warning"] = depth_warning
        updated["depth_status"] = depth_status
        if depth_status != "suppressed":
            passing.append(updated)
        else:
            suppressed.append(updated)

    summary = {
        "target_words": DEPTH_TARGET_WORDS,
        "clean_pass_threshold": DEPTH_PASS_THRESHOLD,
        "suppress_threshold": DEPTH_SUPPRESS_THRESHOLD,
        "soft_min_words": DEPTH_SOFT_MIN_WORDS,
        "clean_min_words": DEPTH_CLEAN_MIN_WORDS,
        "total_support_words": db_words + web_words + social_words + news_words,
        "db_context_words": db_words,
        "web_context_words": web_words,
        "social_data_words": social_words,
        "news_data_words": news_words,
        "depth_assessed_at": assessed_at,
        "clusters_passed": len(passing),
        "clusters_suppressed": len(suppressed),
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
    groq_client: Any | None = None,
    gemini_client: Any | None = None,
    cache_lookup: Callable[[str, Any | None], dict[str, Any] | None] | None = None,
    cache_store: Callable[[str, dict[str, Any], Any | None], None] | None = None,
) -> dict[str, Any]:
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must be a non-empty string")
    if not isinstance(gap_angles, list) or not gap_angles:
        raise ValueError("gap_angles must be a non-empty list")
    if not isinstance(perspective_tree, list) or not perspective_tree:
        raise ValueError("perspective_tree must be a non-empty list")
    if not isinstance(ideas_per_angle, int) or not (1 <= ideas_per_angle <= 5):
        raise ValueError("ideas_per_angle must be between 1 and 5")
    if not isinstance(max_angles, int) or not (1 <= max_angles <= 8):
        raise ValueError("max_angles must be between 1 and 8")
    if not isinstance(db_context, str):
        raise ValueError("db_context must be a string")
    if not isinstance(web_context, str):
        raise ValueError("web_context must be a string")
    if not isinstance(news_data, list):
        raise ValueError("news_data must be a list")
    if not isinstance(social_data, list):
        raise ValueError("social_data must be a list")
    if groq_client is None:
        raise ValueError("groq_client is required")

    if cache_lookup and not used_angle_ids:
        cached = cache_lookup(topic, gemini_client)
        if cached:
            return cached

    candidates = select_candidate_angles(topic, gap_angles, used_angle_ids or [], gemini_client)
    if not candidates:
        raise ValueError("no_viable_angles")
    selected, diversity_applied = apply_diversity_pass(
        candidates,
        max_per_who=2,
        max_angles=max_angles,
    )
    if not selected:
        raise ValueError("no_angles_after_diversity")

    expanded = await expand_all_angles(selected, topic, briefs, ideas_per_angle, groq_client)
    if not expanded:
        raise RuntimeError("idea_expansion_failed")

    clusters = [build_idea_cluster(angle, variants) for angle, variants in expanded]
    passing, suppressed, depth_summary = apply_depth_check(
        clusters,
        db_context=db_context,
        web_context=web_context,
        social_data=social_data,
        news_data=news_data,
    )
    depth_fallback_used = False
    depth_fallback_reason = ""

    response = assemble_response(
        topic=topic,
        gap_angles=gap_angles,
        selected_angles=selected,
        expanded=expanded,
        passing=passing,
        suppressed=suppressed,
        depth_summary=depth_summary,
        diversity_applied=diversity_applied,
        served_from_cache=False,
        cache_age_hours=None,
    )
    if cache_store:
        cache_store(topic, response, gemini_client)
    return response


async def regenerate_with_expansion(
    topic: str,
    old_result: dict[str, Any],
    *,
    gap_angles: list[dict[str, Any]],
    briefs: list[dict[str, Any]],
    perspective_tree: list[dict[str, Any]],
    social_data: list[dict[str, Any]],
    news_data: list[dict[str, Any]],
    db_context: str,
    web_context: str,
    groq_client: Any,
    gemini_client: Any | None = None,
    cache_lookup: Callable[[str, Any | None], dict[str, Any] | None] | None = None,
    cache_store: Callable[[str, dict[str, Any], Any | None], None] | None = None,
) -> dict[str, Any]:
    old_angle_ids = [cluster.get("angle_id") for cluster in old_result.get("idea_clusters", []) if cluster.get("angle_id")]
    return await generate_ideas(
        topic=topic,
        gap_angles=gap_angles,
        briefs=briefs,
        perspective_tree=perspective_tree,
        social_data=social_data,
        news_data=news_data,
        db_context=db_context,
        web_context=web_context,
        max_angles=8,
        ideas_per_angle=5,
        used_angle_ids=old_angle_ids,
        groq_client=groq_client,
        gemini_client=gemini_client,
        cache_lookup=cache_lookup,
        cache_store=cache_store,
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
2. Every variant must be meaningfully different from the others.
3. Keep each description around 150 words and end with a curiosity gap.
4. Do not produce generic ideas.

Return ONLY valid JSON:
{
  "variants": [
    {
      "variant_index": int,
      "title": string,
      "description": string,
      "content_pillars": [string, string, string],
      "gap_reason": string,
      "target_audience": string,
      "hook_strategy": string
    }
  ]
}
"""
