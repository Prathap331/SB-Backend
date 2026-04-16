"""
CAGS (Content Angle Gap Score) — Module 03 pipeline stub.

The module will consume verified outputs from TSS and CSI and
produce perspective-angle coverage insights. All functions are
empty placeholders for later implementation.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import asyncio
import os
from typing import Any, Iterable, Sequence, cast

import numpy as np

__all__ = ["calculate_cags", "collect_corpus_embeddings", "label_youtube_corpus", "tree_interrogation",
           "assess_angle_coverage", "summarise_findings"]

def _collect_google_embed_keys() -> list[str]:
    ordered = [
        (os.getenv("GOOGLE_API_KEY1") or "").strip(),
        (os.getenv("GOOGLE_API_KEY2") or "").strip(),
        (os.getenv("GOOGLE_API_KEY") or "").strip(),
    ]
    keys: list[str] = []
    for key in ordered:
        if key and key not in keys:
            keys.append(key)
    return keys


def _iter_embedding_clients(primary_client: Any | None) -> list[Any]:
    clients: list[Any] = []
    if primary_client is not None:
        clients.append(primary_client)
    keys = _collect_google_embed_keys()
    if not keys:
        return clients
    try:
        from google import genai  # type: ignore
    except Exception:
        return clients
    for key in keys:
        try:
            c = genai.Client(api_key=key)
        except Exception:
            continue
        # Avoid duplicate object instance only.
        if all(c is not existing for existing in clients):
            clients.append(c)
    return clients


def collect_corpus_embeddings(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stage 1: placeholder for corpus embeddings analysis."""
    return []


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of a and b."""
    dot = a @ b.T
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    denom = a_norm * b_norm.T
    denom = np.where(denom == 0, 1.0, denom)
    return dot / denom


def label_youtube_corpus(
    corpus: list[dict[str, Any]],
    corpus_embeddings: np.ndarray,
    angle_vectors: np.ndarray,
    perspective_tree: list[dict[str, Any]],
    corpus_avg_views: float,
) -> list[dict[str, Any]]:
    """Stage 3: label each video with its best-matching angle."""
    corpus_embeddings = np.asarray(corpus_embeddings)
    angle_vectors = np.asarray(angle_vectors)
    if corpus_embeddings.size == 0 or angle_vectors.size == 0 or not perspective_tree:
        return []

    sim_matrix = _cosine_similarity_matrix(corpus_embeddings, angle_vectors)
    results: list[dict[str, Any]] = []
    for vid_idx, video in enumerate(corpus):
        sims = sim_matrix[vid_idx]
        best_idx = int(np.argmax(sims)) if sims.size else 0
        best_sim = float(sims[best_idx]) if sims.size else 0.0
        angle = perspective_tree[best_idx] if best_idx < len(perspective_tree) else {}
        likes = float(video.get("like_count", 0) or 0)
        comments = float(video.get("comment_count", 0) or 0)
        views = float(video.get("view_count", 0) or 1)
        eng_rate = (likes + comments) / max(views, 1.0)
        perf_ratio = views / max(corpus_avg_views, 1.0)
        quality_score = round(
            0.50 * best_sim + 0.30 * min(eng_rate / 0.05, 1.0) + 0.20 * min(perf_ratio, 1.0),
            3,
        )
        if best_sim >= 0.72:
            label = "COVERED_WELL"
        elif best_sim >= 0.45:
            label = "COVERED_LOW_QUALITY"
        else:
            label = "NOT_COVERED"

        augmented = dict(video)
        augmented.update(
            {
                "best_angle_idx": best_idx,
                "best_angle_id": angle.get("angle_id"),
                "best_angle_str": angle.get("angle_string"),
                "similarity": round(best_sim, 3),
                "coverage_label": label,
                "quality_score": quality_score,
                "labelled_who": angle.get("who"),
                "labelled_what": angle.get("what") or [],
                "labelled_when": angle.get("when"),
                "labelled_scale": angle.get("scale"),
                "labelled_how": angle.get("how"),
                "labelled_frame": angle.get("story_frame"),
            }
        )
        results.append(augmented)
    return results


def _safe_groq_parse(resp: Any) -> dict[str, Any]:
    """Guarded JSON parse for Groq responses."""
    try:
        content = resp.choices[0].message.content
        return json.loads(content)
    except (json.JSONDecodeError, KeyError, IndexError, AttributeError, TypeError):
        return {}


def _hash_angle(angle_fields: Sequence[str]) -> str:
    payload = "|".join(angle_fields)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]




STEP_1_PROMPT = """
Topic: "{topic}"

Social signals (sample):
{social_lines}

News signals (sample):
{news_lines}

Task: Identify every stakeholder that has a meaningfully 
distinct perspective on this topic. Include only stakeholders 
where the topic genuinely affects or involves them differently.

Universal stakeholders to consider: government, regulators,
opposition, general_public, media, businesses, workers,
experts, international_actors, future_generations.

Also identify any contextual stakeholders specific to this topic.

Return JSON: {{"stakeholders": [{{"id": str, "label": str,
"type": "universal|contextual", "relevance": str}}]}}
"""

STEP_2_BATCH_PROMPT = """
Topic: "{topic}"

For each stakeholder below identify 1-2 disciplinary lenses
that best explain their relationship to this topic.
Only include lenses that genuinely apply.

Available lenses: history, economics, law, sociology,
technology, psychology, environment, policy, ethics,
geopolitics, anthropology, demography, science.

Stakeholders:
{stakeholder_block}

Return JSON: {{"assignments": [{{"id": str, "lenses": [str]}}]}}
"""

STEPS_3_TO_6_PROMPT = """
Topic: "{topic}"

Complete each angle node below. For each INDEX return a JSON 
object selecting the most applicable option for each step.

STEP 3 - TIME: past | present | future
STEP 3 - SCALE: local | national | global
STEP 4 - SYSTEM DYNAMIC: cause_effect | feedback_loop | 
  trade_off | second_order_effects | risk_scenario
STEP 5 - POWER LAYER: who_gains_loses | inequality | 
  institutional_role | corporate_influence | policy_bias
STEP 6 - NARRATIVE FRAME: conflict | crisis | opportunity | 
  human_story | data_driven | hidden_angle

Then write angle_string using this exact template:
"From the perspective of [WHO], analyzed through [WHAT],
in the [TIME] [SCALE] context, focusing on [HOW],
revealing [WHO_BENEFITS], framed as [FRAME]."

NODES:
{indexed_nodes}

Return JSON: {{"angles": [{{"index": int, "when": str, 
"scale": str, "how": str, "who_benefits": str,
"story_frame": str, "angle_string": str}}]}}
"""


async def tree_interrogation(
    topic: str,
    social_data: list[dict[str, Any]],
    news_data: list[dict[str, Any]],
    groq_client: Any,
) -> list[dict[str, Any]]:
    if groq_client is None:
        raise ValueError('groq_client is None')

    social_lines = "\n".join(
        [
            f'[Social] {r.get("title","")}: {str(r.get("body",""))[:120]}'
            for r in (social_data or [])[:8]
        ]
    ) or "(no social signals available)"
    news_lines = "\n".join(
        [
            f'[News] {r.get("title","")}: {str(r.get("body",""))[:120]}'
            for r in (news_data or [])[:8]
        ]
    ) or "(no news signals available)"

    try:
        r1 = await groq_client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': STEP_1_PROMPT.format(
                        topic=topic,
                        social_lines=social_lines,
                        news_lines=news_lines,
                    ),
                }
            ],
            model='llama-3.1-8b-instant',
            response_format={'type': 'json_object'},
        )
        stakeholders = json.loads(r1.choices[0].message.content).get('stakeholders', [])
    except Exception as exc:
        print(f'CAGS Step 1 error: {exc}', flush=True)
        stakeholders = [
            {
                'id': 'general_public',
                'label': 'General Public',
                'type': 'universal',
                'relevance': 'directly affected',
            }
        ]

    if not stakeholders:
        stakeholders = [
            {
                'id': 'general_public',
                'label': 'General Public',
                'type': 'universal',
                'relevance': 'directly affected',
            }
        ]

    stakeholder_block = "\n".join(
        [f'ID: {s["id"]} | {s["label"]}: {s.get("relevance","")}' for s in stakeholders]
    )
    try:
        r2 = await groq_client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': STEP_2_BATCH_PROMPT.format(
                        topic=topic,
                        stakeholder_block=stakeholder_block,
                    ),
                }
            ],
            model='llama-3.1-8b-instant',
            response_format={'type': 'json_object'},
        )
        assignments = {
            a['id']: a.get('lenses', ['economics'])
            for a in json.loads(r2.choices[0].message.content).get('assignments', [])
        }
    except Exception as exc:
        print(f'CAGS Step 2 error: {exc}', flush=True)
        assignments = {}

    nodes = []
    for s in stakeholders:
        lenses = assignments.get(s['id'], ['economics'])
        nodes.append({
            'who': s['label'],
            'what': lenses,
            'relevance': s.get('relevance', ''),
        })

    indexed = "\n---\n".join(
        [f"INDEX {i}: WHO={n['who']} | WHAT={n['what']}" for i, n in enumerate(nodes)]
    )
    try:
        r3 = await groq_client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': STEPS_3_TO_6_PROMPT.format(
                        topic=topic,
                        indexed_nodes=indexed,
                    ),
                }
            ],
            model='llama-3.1-8b-instant',
            response_format={'type': 'json_object'},
        )
        completions = {
            a['index']: a
            for a in json.loads(r3.choices[0].message.content).get('angles', [])
        }
    except Exception as exc:
        print(f'CAGS Step 3 error: {exc}', flush=True)
        completions = {}

    perspective_tree: list[dict[str, Any]] = []
    for i, node in enumerate(nodes):
        comp = completions.get(i, {})
        who = node['who']
        what = node['what']
        when = comp.get('when', 'present')
        scale = comp.get('scale', 'national')
        how = comp.get('how', 'cause_effect')
        who_b = comp.get('who_benefits', 'who_gains_loses')
        frame = comp.get('story_frame', 'hidden_angle')
        angle = comp.get(
            'angle_string',
            (
                f"From the perspective of {who}, analyzed through {', '.join(what)}, "
                f"in the {when} {scale} context, focusing on {how}, revealing {who_b}, "
                f"framed as {frame}."
            ),
        )
        angle_id = hashlib.sha256(
            f"{who}|{what}|{when}|{scale}|{how}|{who_b}|{frame}".encode()
        ).hexdigest()[:12]
        perspective_tree.append({
            'angle_id': angle_id,
            'angle_string': angle,
            'who': who,
            'what': what,
            'when': when,
            'scale': scale,
            'how': how,
            'who_benefits': who_b,
            'story_frame': frame,
        })
    return perspective_tree

def assess_angle_coverage(angle_id: str, labelled_corpus: list[dict[str, Any]]) -> dict[str, Any]:
    """Stage 4: find coverage stats for a single angle."""
    matches = [video for video in labelled_corpus if video.get("best_angle_id") == angle_id]
    if not matches:
        return {
            "coverage_label": "NOT_COVERED",
            "best_quality": 0.0,
            "matched_count": 0,
            "best_video": None,
        }
    rating = {"NOT_COVERED": 0, "COVERED_LOW_QUALITY": 1, "COVERED_WELL": 2}
    best = max(matches, key=lambda v: (rating.get(v.get("coverage_label"), 0), v.get("quality_score", 0.0)))
    best_video = {
        "title": best.get("title"),
        "views": best.get("view_count"),
        "similarity": best.get("similarity"),
        "url": best.get("url") or best.get("video_id"),
    }
    return {
        "coverage_label": best.get("coverage_label"),
        "best_quality": float(best.get("quality_score", 0.0) or 0.0),
        "matched_count": len(matches),
        "best_video": best_video,
    }


def compute_demand_signal(angle: dict[str, Any], social_data: list[dict[str, Any]]) -> float:
    """Stage 4 demand component based on social text matches."""
    who_terms = (angle.get("who") or "").lower().split()
    what_terms = [str(w).lower() for w in angle.get("what") or []]
    match_score = 0.0
    for post in social_data:
        title = str(post.get("title") or "")
        body = str(post.get("body") or post.get("snippet") or post.get("text") or "")
        text = (title + " " + body).lower()
        who_hit = bool(who_terms) and any(w in text for w in who_terms)
        what_hit = bool(what_terms) and any(w in text for w in what_terms)
        if who_hit and what_hit:
            match_score += 1.0
        elif what_hit:
            match_score += 0.6
        elif who_hit:
            match_score += 0.4
    return min(match_score / 20.0, 1.0)


def score_angle(
    angle: dict[str, Any],
    coverage: dict[str, Any],
    demand_score: float,
    tss_score: float,
) -> dict[str, Any]:
    """Compute calibrated CAGS score for a single angle."""
    label = coverage.get("coverage_label")
    quality = coverage.get("best_quality", 0.0)
    if label == "NOT_COVERED":
        base = 100.0
    elif label == "COVERED_LOW_QUALITY":
        base = 70.0 - (quality * 30.0)
    else:
        base = max(0.0, 20.0 - (quality * 20.0))
    demand_weight = 0.6 + (0.4 * demand_score)
    if tss_score >= 75:
        trend_weight = 1.00
    elif tss_score >= 50:
        trend_weight = 0.85
    elif tss_score >= 20:
        trend_weight = 0.70
    else:
        trend_weight = 0.50
    cags_score = round(base * demand_weight * trend_weight, 1)
    return {
        "angle_id": angle.get("angle_id"),
        "angle_string": angle.get("angle_string"),
        "who": angle.get("who"),
        "what": angle.get("what") or [],
        "when": angle.get("when"),
        "scale": angle.get("scale"),
        "how": angle.get("how"),
        "who_benefits": angle.get("who_benefits"),
        "story_frame": angle.get("story_frame"),
        "coverage": coverage,
        "coverage_label": coverage.get("coverage_label"),
        "best_video": coverage.get("best_video"),
        "matched_count": coverage.get("matched_count", 0),
        "best_quality": float(coverage.get("best_quality", 0.0) or 0.0),
        "demand_score": round(demand_score, 3),
        "trend_weight": trend_weight,
        "cags_score": cags_score,
    }


def score_all_angles(
    perspective_tree: list[dict[str, Any]],
    labelled_corpus: list[dict[str, Any]],
    social_data: list[dict[str, Any]],
    tss_score: float,
) -> list[dict[str, Any]]:
    """Stage 4 orchestrator: assess, demand, and score each angle."""
    scored = []
    for angle in perspective_tree:
        coverage = assess_angle_coverage(angle.get("angle_id"), labelled_corpus)
        demand = compute_demand_signal(angle, social_data)
        scored_angle = score_angle(angle, coverage, demand, tss_score)
        scored_angle["angle"] = angle
        scored.append(scored_angle)
    scored.sort(key=lambda x: x.get("cags_score", 0), reverse=True)
    for i, a in enumerate(scored):
        a["rank"] = i + 1
    return scored


async def generate_briefs(
    scored_angles: list[dict[str, Any]],
    topic: str,
    groq_client: Any | None,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Stage 5: briefs for the top gap angles."""
    # COVERED_WELL angles never receive briefs.
    gap_angles = [
        entry
        for entry in scored_angles
        if entry.get("coverage_label") in {"NOT_COVERED", "COVERED_LOW_QUALITY"}
    ]
    briefs: list[dict[str, Any]] = []
    for entry in gap_angles[:top_n]:
        brief_fields = {"suggested_title": "", "hook_sentence": "", "publish_urgency": "anytime"}
        if groq_client:
            prompt = (
                f"Topic: \"{topic}\"\n"
                f"Content angle: {entry.get('angle_string', '')}\n"
                f"Coverage status: {entry.get('coverage_label', '')}\n"
                f"Closest existing video: {entry.get('best_video')}\n\n"
                "Write a YouTube video brief for this exact angle.\n"
                "You MUST provide non-empty string values for all fields.\n\n"
                "Return JSON with exactly these three string fields:\n"
                "{\n"
                "  \"suggested_title\": \"a specific compelling YouTube title\",\n"
                "  \"hook_sentence\": \"opening sentence that hooks viewers in 15 seconds\",\n"
                "  \"publish_urgency\": \"now\"\n"
                "}\n"
                "publish_urgency must be exactly one of: now | within_1_week | within_1_month | anytime"
            )
            resp = await groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                response_format={"type": "json_object"},
            )
            data = _safe_groq_parse(resp)
            urgency = str(data.get("publish_urgency") or "").strip()
            if urgency not in {"now", "within_1_week", "within_1_month", "anytime"}:
                urgency = "anytime"
            brief_fields = {
                "suggested_title": str(data.get("suggested_title") or "").strip(),
                "hook_sentence": str(data.get("hook_sentence") or "").strip(),
                "publish_urgency": urgency,
            }
        if not brief_fields["suggested_title"]:
            brief_fields["suggested_title"] = f"{entry.get('who', 'Audience')} vs status quo: {topic}"
        if not brief_fields["hook_sentence"]:
            brief_fields["hook_sentence"] = f"What everyone is missing about {topic} from the {entry.get('who', 'stakeholder')} perspective."
        brief = {**entry, **brief_fields}
        briefs.append(brief)
    return briefs


class EmbeddingError(RuntimeError):
    pass


def _build_corpus_embed_input(video: dict[str, Any]) -> str:
    title = str(video.get("title") or video.get("video_id") or "").strip()
    desc = str(video.get("description") or "").strip()[:240]
    return f"{title}. {desc}" if desc else title


def _embed_corpus_for_cags(
    corpus: list[dict[str, Any]],
    gemini_client: Any | None,
) -> np.ndarray | None:
    texts = [_build_corpus_embed_input(v) for v in corpus]
    if not texts:
        return None
    clients = _iter_embedding_clients(gemini_client)
    if not clients:
        return None
    for slot_idx, client in enumerate(clients, start=1):
        try:
            models = getattr(client, "models", None)
            if models is None or not hasattr(models, "embed_content"):
                continue
            config = None
            try:
                from google.genai import types as gt

                config = gt.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=768,
                )
            except Exception:
                config_cls = getattr(models, "EmbedContentConfig", None)
                if config_cls:
                    config = config_cls(
                        task_type="SEMANTIC_SIMILARITY",
                        output_dimensionality=768,
                    )
            resp = models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
                config=config,
            )
            vecs = [list(e.values) for e in getattr(resp, "embeddings", []) or []]
            if len(vecs) != len(texts):
                continue
            if any(len(vec) != 768 for vec in vecs):
                continue
            return np.asarray(vecs, dtype=np.float32)
        except Exception as exc:
            print(f"CAGS corpus embedding client slot {slot_idx} failed: {exc}", flush=True)
            continue
    return None


def embed_landscape(
    angles: list[dict[str, Any]],
    gemini_client: Any | None,
) -> np.ndarray:
    """Stage 2: embed angle strings with semantic task_type."""
    texts = [angle.get("angle_string") or "" for angle in angles]
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    embeddings = None
    clients = _iter_embedding_clients(gemini_client)
    for slot_idx, client in enumerate(clients, start=1):
        try:
            models = getattr(client, "models", None)
            if models is None or not hasattr(models, "embed_content"):
                continue
            config = None
            try:
                from google.genai import types as gt
                config = gt.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=768,
                )
            except Exception:
                config_cls = getattr(models, "EmbedContentConfig", None)
                if config_cls:
                    config = config_cls(
                        task_type="SEMANTIC_SIMILARITY",
                        output_dimensionality=768,
                    )
            resp = models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
                config=config,
            )
            embeddings = np.array(
                [list(e.values) for e in getattr(resp, "embeddings", []) or []],
                dtype=np.float32,
            )
            if embeddings.shape != (len(texts), 768):
                raise EmbeddingError("Gemini returned unexpected shape")
            break
        except Exception as exc:
            print(f"CAGS angle embedding client slot {slot_idx} failed: {exc}", flush=True)
            embeddings = None
    lm_enabled = os.getenv("LM_STUDIO_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    is_render = bool(os.getenv("RENDER"))

    if embeddings is None and lm_enabled and not is_render:
        try:
            import httpx

            resp = httpx.post(
                os.getenv("LMSTUDIO_EMBEDDINGS_URL", "http://127.0.0.1:1234/v1/embeddings"),
                json={
                    "model": os.getenv("LMSTUDIO_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"),
                    "input": texts,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") or []
            vecs = [item.get("embedding") for item in data if item.get("embedding")]
            if not vecs:
                raise EmbeddingError("LM Studio returned no embeddings")
            embeddings = np.array(vecs, dtype=np.float32)
            if embeddings.shape != (len(texts), 768):
                raise EmbeddingError("LM Studio returned unexpected shape")
        except Exception as exc:
            raise EmbeddingError(f"Embedding fallback failed: {exc}") from exc
    elif embeddings is None:
        raise EmbeddingError("Gemini embedding failed and LM Studio fallback disabled in this environment")
    return embeddings


async def calculate_cags(
    topic: str,
    corpus: list[dict[str, Any]],
    corpus_embeddings: "np.ndarray | None",
    social_data: list[dict[str, Any]],
    news_data: list[dict[str, Any]],
    tss_score: float,
    groq_client: Any | None,
    gemini_client: Any | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Entry point for Module 03."""
    cags_warnings: list[str] = []

    if corpus_embeddings is None:
        corpus_embeddings = _embed_corpus_for_cags(corpus, gemini_client)
    if corpus_embeddings is None:
        cags_warnings.append("corpus_embeddings_unavailable")

    corpus_embeddings_arr: np.ndarray | None = None
    if corpus_embeddings is not None:
        corpus_embeddings_arr = np.asarray(corpus_embeddings, dtype=np.float32)
        expected_rows = len(corpus)
        invalid_shape = (
            corpus_embeddings_arr.ndim != 2
            or corpus_embeddings_arr.shape[1] != 768
            or corpus_embeddings_arr.shape[0] != expected_rows
        )
        if invalid_shape:
            # CSI can occasionally hand over vectors with an unexpected shape.
            # Rebuild corpus embeddings here to keep CAGS resilient.
            rebuilt = _embed_corpus_for_cags(corpus, gemini_client)
            if rebuilt is not None:
                corpus_embeddings_arr = np.asarray(rebuilt, dtype=np.float32)
                invalid_shape = (
                    corpus_embeddings_arr.ndim != 2
                    or corpus_embeddings_arr.shape[1] != 768
                    or corpus_embeddings_arr.shape[0] != expected_rows
                )
            if invalid_shape:
                cags_warnings.append("corpus_embeddings_invalid_shape")
                corpus_embeddings_arr = None

    perspective_tree = await tree_interrogation(topic, social_data, news_data, groq_client)
    labelled: list[dict[str, Any]] = []
    if corpus_embeddings_arr is not None:
        try:
            angle_vectors = embed_landscape(perspective_tree, gemini_client)
            angle_vectors = np.asarray(angle_vectors)
            avg_views = float(
                sum(float(v.get("view_count", 0) or 0) for v in corpus) / max(len(corpus), 1)
            )
            labelled = label_youtube_corpus(
                corpus,
                corpus_embeddings_arr,
                angle_vectors,
                perspective_tree,
                avg_views,
            )
        except Exception as exc:
            print(f"CAGS labelling fallback activated: {exc}", flush=True)
            cags_warnings.append("angle_embeddings_unavailable")

    scored_angles = score_all_angles(perspective_tree, labelled, social_data, tss_score)
    gap_angles = [
        entry
        for entry in scored_angles
        if entry.get("coverage", {}).get("coverage_label") in {"NOT_COVERED", "COVERED_LOW_QUALITY"}
    ]
    briefs = await generate_briefs(gap_angles, topic, groq_client)

    payload = {
        "topic": topic,
        "perspective_tree": perspective_tree,
        "labelled_corpus": labelled,
        "scored_angles": scored_angles,
        "gap_angles": gap_angles,
        "briefs": briefs,
    }
    if cags_warnings:
        payload["cags_error"] = ",".join(cags_warnings)
        payload["cags_warnings"] = cags_warnings
    return payload
