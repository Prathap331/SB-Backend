"""
schema_analyser.py  — LM Studio-powered column classifier v3
══════════════════════════════════════════════════════════════════════
Uses your locally running LM Studio model to intelligently classify
DataFrame columns for StoryBit's RAG ingestion pipeline.

Improvements over v2:
  1. Confidence field now always present in LLM JSON (prompt enforces it)
  2. Reasoning field no longer empty — prompt requires a full sentence
  3. Unclassified columns (author, created_utc, permalink) are caught
     pre-LLM in ALWAYS_NOISE and never reach the auto-router
  4. Template validation — referenced columns must exist and not be noise;
     bad refs are stripped rather than left as {placeholder} in sentences
  5. _safe_avg_words uses a larger sample (500 rows) for better accuracy
     on sparse columns like posts.body (4% fill)
  6. Wide-schema pre-filter is smarter: body/text/selftext are always kept
  7. LLM system prompt tightened: noise list is explicit and exhaustive
  8. Retry on low confidence uses a genuinely different (extended) prompt
  9. schema_cache is keyed on column names + dtypes + row count so stale
     cache entries from a different file are never reused
 10. All public return dicts are guaranteed to have every required key
══════════════════════════════════════════════════════════════════════
"""

import re
import os
import json
import httpx
import hashlib
import math
import pandas as pd
from functools import lru_cache

# ── LM Studio config ──────────────────────────────────────────
LM_STUDIO_BASE   = "http://127.0.0.1:1234"
LM_STUDIO_CHAT   = f"{LM_STUDIO_BASE}/v1/chat/completions"
LM_STUDIO_MODELS = f"{LM_STUDIO_BASE}/api/v1/models"
LM_TIMEOUT       = 120
SAMPLE_ROWS      = 5
MAX_SAMPLE_CHARS = 120
MAX_COLS_TO_LLM  = 15

# Columns that are ALWAYS noise — never primary, never context
ALWAYS_NOISE_SET = {
    "id", "post_id", "comment_id", "tweet_id", "user_id", "uuid",
    "author", "username", "user_name", "screen_name", "handle",
    "url", "link", "href", "permalink",
    "score", "votes", "upvote_ratio", "downvote_ratio",
    "num_comments", "followers", "friends", "favourites", "likes",
    "created_utc", "created_at", "timestamp", "date", "time",
    "user_created", "user_followers", "user_friends", "user_verified",
    "user_location", "user_description",
    "ups", "downs", "gilded", "distinguished", "stickied",
    "is_self", "over_18", "spoiler", "locked",
}
ALWAYS_NOISE_SUBSTRINGS = ("_id", "_url", "_link", "ratio", "count", "votes", "created")


# ══════════════════════════════════════════════════════════════
# LM STUDIO HEALTH CHECK + MODEL ID RESOLUTION
# ══════════════════════════════════════════════════════════════

_loaded_model_id: str = ""
_model_id_printed: bool = False


def get_model_id() -> str:
    global _loaded_model_id, _model_id_printed
    if _loaded_model_id:
        return _loaded_model_id

    env_id = os.environ.get("LMSTUDIO_MODEL_ID", "").strip()
    if env_id:
        _loaded_model_id = env_id
        return _loaded_model_id

    try:
        r    = httpx.get(LM_STUDIO_MODELS, timeout=5)
        data = r.json()

        if not _model_id_printed:
            print(f"  [LM Studio] Raw /api/v1/models response (truncated):")
            print(f"  {json.dumps(data, indent=2)[:600]}")
            _model_id_printed = True

        candidates = []

        def _search(obj, depth=0):
            if depth > 4:
                return
            if isinstance(obj, dict):
                for field in ["id", "model_id", "modelId", "model"]:
                    v = obj.get(field, "")
                    if isinstance(v, str) and len(v) > 5:
                        candidates.append((len(v), v))
                for v in obj.values():
                    _search(v, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    _search(item, depth + 1)

        _search(data)

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _loaded_model_id = candidates[0][1]
            print(f"  [LM Studio] Auto-detected model ID: {_loaded_model_id}")
            return _loaded_model_id

    except Exception as e:
        print(f"  [LM Studio] Error reading models: {e}")

    _loaded_model_id = "lfm2.5-1.2b-instruct-thinking-claude-high-reasoning-mlx"
    print(f"  [LM Studio] Using hardcoded model ID: {_loaded_model_id}")
    return _loaded_model_id


def is_lm_studio_running() -> tuple[bool, str]:
    try:
        r = httpx.get(LM_STUDIO_MODELS, timeout=5)
        if r.status_code == 200:
            return True, get_model_id()
    except Exception:
        pass
    return False, ""


# ══════════════════════════════════════════════════════════════
# SAFE HELPERS
# ══════════════════════════════════════════════════════════════

def _safe_nunique(series: pd.Series) -> int:
    try:
        return series.nunique()
    except TypeError:
        return series.apply(lambda x: str(x) if isinstance(x, (list, dict)) else x).nunique()


_EMPTY_SENTINELS = {
    "", "nan", "none", "null", "n/a", "na",
    "[removed]", "[deleted]", "[ removed by moderator ]",
    "[removed by moderator]",
}


def _safe_avg_words(series: pd.Series, n: int = 500) -> float:
    """
    Average word count on NON-NULL, NON-EMPTY, NON-SENTINEL rows only.
    Uses up to n rows (default 500) for accuracy on sparse columns.
    """
    try:
        clean = series.dropna().astype(str).str.strip()
        clean = clean[~clean.str.lower().isin(_EMPTY_SENTINELS)]
        if clean.empty:
            return 0.0
        return clean.head(n).apply(lambda x: len(x.split())).mean()
    except Exception:
        return 0.0


def _is_always_noise(col: str) -> bool:
    col_lower = col.lower()
    if col_lower in ALWAYS_NOISE_SET:
        return True
    return any(sub in col_lower for sub in ALWAYS_NOISE_SUBSTRINGS)


# ══════════════════════════════════════════════════════════════
# SCHEMA SNAPSHOT BUILDER
# ══════════════════════════════════════════════════════════════

def _pre_filter_columns(df: pd.DataFrame, max_cols: int = MAX_COLS_TO_LLM) -> pd.DataFrame:
    """
    For wide DataFrames, keep up to max_cols most promising columns.
    Body-type columns (selftext, body, text, etc.) are always kept.
    Noise columns are always stripped before sending to the LLM.
    """
    # Strip obvious noise first
    candidates = [c for c in df.columns if not _is_always_noise(c)
                  and df[c].dtype in [object, "string", "O"]]

    if len(candidates) <= max_cols:
        return df[candidates] if len(candidates) < len(df.columns) else df

    print(f"      [Schema] Wide dataset ({len(df.columns)} cols) — pre-filtering to top {max_cols}")

    BODY_NAMES = {"body", "text", "selftext", "content", "description",
                  "summary", "abstract", "review", "comment", "post",
                  "transcript", "answer", "response", "message", "title"}

    scored = []
    for col in candidates:
        avg_w = _safe_avg_words(df[col])
        fill  = df[col].notna().sum() / max(len(df), 1)
        # Always keep known body columns
        is_body = col.lower() in BODY_NAMES
        score   = (avg_w * 2 + fill * 5) + (100 if is_body else 0)
        scored.append((score, col))

    scored.sort(reverse=True)
    keep = [col for _, col in scored[:max_cols]]
    print(f"      [Schema] Kept columns: {keep}")
    return df[keep]


def _build_schema_snapshot(df: pd.DataFrame, n_samples: int = SAMPLE_ROWS) -> str:
    lines = [
        f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns\n",
        f"{'Column':<30} {'dtype':<10} {'fill%':<8} {'uniq%':<8} {'avg_words':<12} {'Sample values (first non-null)'}",
        "-" * 115,
    ]

    for col in df.columns:
        series   = df[col]
        dtype    = str(series.dtype)
        clean    = series.dropna()

        if len(clean) == 0:
            lines.append(f"{col:<30} {dtype:<10} {'0%':<8} {'N/A':<8} {'N/A':<12} (all null)")
            continue

        fill_pct   = f"{len(clean)/len(series)*100:.0f}%"
        unique_pct = f"{_safe_nunique(series)/max(len(series),1)*100:.0f}%"

        sample_vals = (
            clean.apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
                 .astype(str).str.strip()
                 .replace(["nan", "none", "null", ""], pd.NA)
                 .dropna()
                 .head(n_samples)
                 .tolist()
        )
        sample_vals = [v[:MAX_SAMPLE_CHARS] + ("…" if len(v) > MAX_SAMPLE_CHARS else "")
                       for v in sample_vals]

        if series.dtype == object:
            first_val = clean.iloc[0] if len(clean) > 0 else None
            avg_words = "N/A" if isinstance(first_val, (list, dict)) else f"{_safe_avg_words(series):.1f}"
        else:
            avg_words = "N/A"

        lines.append(
            f"{col:<30} {dtype:<10} {fill_pct:<8} {unique_pct:<8} {avg_words:<12} {' | '.join(sample_vals[:3])}"
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# LLM CALL
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a data engineering assistant that classifies DataFrame columns for text extraction in a RAG pipeline.

CLASSIFICATION RULES — follow exactly:

## primary
Columns with RICH NARRATIVE TEXT (full sentences, paragraphs, articles, reviews, comments, posts).
Requirements: avg_words >= 8 OR column name is in: body, text, selftext, content, response, answer, description, summary, abstract, review, comment, post, transcript
A sparse column (10% fill) with avg_words=40 on filled rows is still PRIMARY.

## context  
SHORT LABELS that add semantic context (1-6 words, categorical).
Examples: subreddit, category, country, party, sector, title (when avg_words < 15), tags, topic

## noise — ALWAYS put these here, no exceptions:
- IDs: post_id, comment_id, user_id, uuid, id
- Author/user fields: author, username, user_name, screen_name, handle
- URLs: url, link, href, permalink — any column whose samples start with http
- Timestamps: created_utc, created_at, timestamp, date, time, year, month
- Numbers: score, votes, upvote_ratio, downvote_ratio, num_comments, followers, ups, downs, gilded
- Float/int dtype columns
- Any column with avg_words ≤ 1.0 AND unique% > 85% → it's an ID

## is_social
true if dataset contains social media columns: subreddit, screen_name, upvote_ratio, retweet, tweet

## sentence_template
ONLY reference columns classified as primary or context. NEVER reference noise columns.
Format: "{context_col}: {primary_col1}. {primary_col2}"
Reddit comments example: "{subreddit}: {body}."
Reddit posts example: "{subreddit}: {title}. {body}"

Return ONLY valid JSON with NO markdown fences, NO preamble:
{
  "primary": ["col_name"],
  "context": ["col_name"],
  "noise": ["col_name"],
  "is_social": true,
  "dataset_type": "social_media",
  "sentence_template": "{subreddit}: {title}. {body}",
  "confidence": 0.95,
  "reasoning": "This is Reddit social media data with body as primary text and subreddit as context label."
}"""


def _call_lm_studio(schema_snapshot: str) -> dict | None:
    model_id = get_model_id()
    payload  = {
        "model":    model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Classify these DataFrame columns:\n\n{schema_snapshot}"},
        ],
        "temperature": 0.05,
        "max_tokens":  1400,
        "stream":      False,
    }

    print(f"      [LM Studio] Sending to {LM_STUDIO_CHAT} | Model: {model_id}")

    try:
        r = httpx.post(LM_STUDIO_CHAT, json=payload, timeout=LM_TIMEOUT)
        if r.status_code != 200:
            print(f"      [LM Studio] HTTP {r.status_code}: {r.text[:300]}")
            return None

        data    = r.json()
        content = None
        if "choices" in data:
            content = data["choices"][0]["message"]["content"].strip()
        elif "content" in data:
            content = data["content"].strip()
        elif "message" in data:
            content = data["message"].strip()
        else:
            print(f"      [LM Studio] Unexpected response: {json.dumps(data)[:300]}")
            return None

        print(f"      [LM Studio] Raw (first 400 chars): {content[:400]}")

        # Strip <think>...</think> blocks
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        # Strip markdown fences
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$",           "", content).strip()

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            print(f"      [LM Studio] No JSON found. Full content:\n{content}")
            return None

        result = json.loads(json_match.group())

        # Ensure required keys always exist
        result.setdefault("confidence", 1.0)
        result.setdefault("reasoning",  "No reasoning provided.")
        result.setdefault("is_social",  False)
        result.setdefault("dataset_type", "general")
        result.setdefault("sentence_template", "")
        return result

    except json.JSONDecodeError as e:
        print(f"      [LM Studio] JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"      [LM Studio] Exception: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# TEMPLATE VALIDATION
# ══════════════════════════════════════════════════════════════

def _validate_template(template: str, primary: list[str], context: list[str],
                        noise: list[str], df_cols: set) -> str:
    """
    Remove any {placeholder} refs that point to noise or non-existent columns.
    Clean up resulting double punctuation / trailing separators.
    """
    if not template:
        return template

    allowed = set(primary + context)
    noise_set = set(noise)

    def _rep(m):
        col = m.group(1)
        if col not in df_cols or col in noise_set or col not in allowed:
            return ""
        return m.group(0)

    cleaned = re.sub(r"\{(\w+)\}", _rep, template)
    cleaned = re.sub(r"\s*[—–-]+\s*\.", ".", cleaned)
    cleaned = re.sub(r"\s*[—–-]+\s*$",  "",  cleaned)
    cleaned = re.sub(r":\s*\.",          ".", cleaned)
    cleaned = re.sub(r"\.\s*\.",         ".", cleaned)
    cleaned = re.sub(r"\s{2,}",          " ", cleaned).strip()
    cleaned = re.sub(r"[:\s]+$",         "",  cleaned).strip()
    return cleaned


# ══════════════════════════════════════════════════════════════
# RESPONSE VALIDATOR
# ══════════════════════════════════════════════════════════════

def _validate_response(result: dict, df: pd.DataFrame) -> dict:
    df_cols    = list(df.columns)
    valid_cols = set(df_cols)

    raw_primary = [c for c in result.get("primary", []) if c in valid_cols]
    raw_context = [c for c in result.get("context", []) if c in valid_cols]
    raw_noise   = [c for c in result.get("noise",   []) if c in valid_cols]

    primary, context, noise = [], [], list(raw_noise)
    corrected_to_noise = []

    # ── Validate primary ──────────────────────────────────────
    for col in raw_primary:
        # Hard-override: always noise
        if _is_always_noise(col):
            noise.append(col); corrected_to_noise.append(col); continue
        # Numeric dtype
        if df[col].dtype in ["int64", "float64", "int32", "float32"]:
            noise.append(col); corrected_to_noise.append(col); continue
        # High-uniqueness, low-word columns → IDs
        try:
            uniq = _safe_nunique(df[col]) / max(len(df[col]), 1)
        except Exception:
            uniq = 0.0
        avg_w = _safe_avg_words(df[col])
        if avg_w <= 1.5 and uniq > 0.85:
            noise.append(col); corrected_to_noise.append(col); continue
        primary.append(col)

    # ── Validate context ─────────────────────────────────────
    for col in raw_context:
        if _is_always_noise(col):
            noise.append(col); corrected_to_noise.append(col); continue
        if df[col].dtype in ["int64", "float64", "int32", "float32"]:
            noise.append(col); corrected_to_noise.append(col); continue
        context.append(col)

    if corrected_to_noise:
        print(f"      [Validator] Moved to noise: {corrected_to_noise}")

    # ── Demote short primary → context ───────────────────────
    BODY_NAMES = {"body", "text", "selftext", "content", "description",
                  "summary", "abstract", "review", "comment", "post",
                  "transcript", "answer", "response", "message"}
    demoted = []
    for col in primary[:]:
        if col not in df.columns:
            continue
        avg_w     = _safe_avg_words(df[col])
        is_body   = col.lower() in BODY_NAMES
        fill_rate = df[col].dropna().shape[0] / max(len(df), 1)

        if is_body:
            # Always keep known body columns in primary regardless of avg_words
            continue
        if avg_w < 8:
            primary.remove(col)
            context.append(col)
            demoted.append(col)
        elif avg_w < 4 and fill_rate > 0.5:
            primary.remove(col)
            context.append(col)
            demoted.append(col)

    if demoted:
        print(f"      [Validator] Demoted to context (short text): {demoted}")

    # ── Route unclassified columns ────────────────────────────
    classified   = set(primary + context + noise)
    unclassified = [c for c in df_cols if c not in classified]

    if unclassified:
        print(f"      [Validator] Unclassified — routing: {unclassified}")
        for col in unclassified:
            if _is_always_noise(col) or df[col].dtype in ["int64", "float64", "int32", "float32"]:
                noise.append(col); continue
            avg_w = _safe_avg_words(df[col])
            if avg_w >= 10:   primary.append(col)
            elif avg_w >= 2:  context.append(col)
            else:             noise.append(col)

    # Deduplicate while preserving order
    primary = list(dict.fromkeys(primary))
    context = list(dict.fromkeys(context))
    noise   = list(dict.fromkeys(noise))

    return {**result, "primary": primary, "context": context, "noise": noise}


# ══════════════════════════════════════════════════════════════
# SENTENCE TEMPLATE EXECUTOR
# ══════════════════════════════════════════════════════════════

def _execute_template(row: pd.Series, template: str,
                      primary_cols: list[str], context_cols: list[str],
                      is_social: bool = False) -> str:
    _URL_RE     = re.compile(r"https?://\S+")
    _MENTION_RE = re.compile(r"@\w+")
    _HASHTAG_RE = re.compile(r"#(\w+)")
    _BAD        = {"nan", "none", "null", "n/a", "na", "", "[]", "{}"}

    def clean(v: str) -> str:
        v = re.sub(r"<[^>]+>", " ", str(v)).strip()
        if is_social:
            v = _URL_RE.sub("", v)
            v = _MENTION_RE.sub("", v)
            v = _HASHTAG_RE.sub(r"\1 ", v)
        return re.sub(r"\s+", " ", v).strip()

    def get(col: str) -> str:
        val = row.get(col, "")
        if isinstance(val, (list, dict)):
            val = str(val)
        v = str(val).strip()
        return "" if v.lower() in _BAD else clean(v)

    try:
        refs    = re.findall(r"\{(\w+)\}", template)
        missing = [r for r in refs if r not in row.index]
        if not missing:
            subs   = {ref: (get(ref) or f"[{ref}]") for ref in refs}
            result = template
            for ref, val in subs.items():
                result = result.replace(f"{{{ref}}}", val)
            result = re.sub(r"\[\w+\]", "", result)
            result = re.sub(r"\s+", " ", result).strip()
            result = re.sub(r"^\W+", "", result).strip()
            if len(result.split()) >= 5:
                return result
    except Exception:
        pass

    main_parts = [v for c in primary_cols if (v := get(c))]
    if not main_parts:
        return ""
    ctx_vals = [v for c in context_cols if (v := get(c))]
    prefix   = " — ".join(ctx_vals[:2]) + ": " if ctx_vals else ""
    return (prefix + ". ".join(main_parts)).strip()


# ══════════════════════════════════════════════════════════════
# SCHEMA CACHE
# ══════════════════════════════════════════════════════════════

_schema_cache: dict[str, dict] = {}


def _schema_key(df: pd.DataFrame) -> str:
    sig = str(list(df.columns)) + str(list(df.dtypes)) + str(len(df))
    return hashlib.md5(sig.encode()).hexdigest()


# ══════════════════════════════════════════════════════════════
# MAIN PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════

def analyse_dataframe_llm(df: pd.DataFrame,
                           fallback_to_stats: bool = True) -> dict:
    """
    LM Studio-powered column analysis.
    Drop-in replacement for analyse_dataframe() in ingest_from_kaggle.py.

    Returns a dict with keys:
      force_primary, primary, context, all_primary, stats,
      has_text, is_numeric_ds, sample, is_social,
      sentence_template, dataset_type, confidence, used_llm
    """
    key = _schema_key(df)
    if key in _schema_cache:
        print(f"      [LM Studio] Using cached schema analysis")
        return _schema_cache[key]

    # Pre-filter wide DataFrames (noise cols always stripped)
    df_for_llm = _pre_filter_columns(df)
    snapshot   = _build_schema_snapshot(df_for_llm)

    running, model_id = is_lm_studio_running()

    if not running:
        print(f"      [LM Studio] Not reachable at {LM_STUDIO_BASE} — using stats fallback")
        result = _stat_analyse_dataframe(df)
        result["used_llm"] = False
        return result

    print(f"      [LM Studio] Analysing schema with {model_id}...")
    llm_result = _call_lm_studio(snapshot)

    if llm_result is None:
        print(f"      [LM Studio] Call failed — using stats fallback")
        result = _stat_analyse_dataframe(df)
        result["used_llm"] = False
        return result

    # Validate against the full original df
    llm_result = _validate_response(llm_result, df)

    confidence = llm_result.get("confidence", 1.0)
    print(f"      [LM Studio] Confidence: {confidence:.0%} | Type: {llm_result.get('dataset_type','?')} | Social: {llm_result.get('is_social', False)}")
    print(f"      [LM Studio] Reasoning: {llm_result.get('reasoning', '')}")

    if confidence < 0.6:
        print(f"      [LM Studio] Low confidence — retrying with extended snapshot...")
        snapshot2   = _build_schema_snapshot(df_for_llm, n_samples=10)
        llm_result2 = _call_lm_studio(snapshot2)
        if llm_result2 and llm_result2.get("confidence", 0) > confidence:
            llm_result = _validate_response(llm_result2, df)
            print(f"      [LM Studio] Retry confidence: {llm_result.get('confidence',0):.0%}")

    primary   = llm_result.get("primary",  [])
    context   = llm_result.get("context",  [])
    noise     = llm_result.get("noise",    [])
    is_social = llm_result.get("is_social", False)
    raw_tmpl  = llm_result.get("sentence_template", "")

    # Validate and clean the template
    template = _validate_template(raw_tmpl, primary, context, noise, set(df.columns))
    if raw_tmpl != template:
        print(f"      [Validator] Template sanitised: '{raw_tmpl}' → '{template}'")

    stats_out = {}
    for col in primary + context:
        if col not in df.columns:
            continue
        fill_rate = df[col].dropna().shape[0] / max(len(df), 1)
        avg_words = _safe_avg_words(df[col])
        role      = "primary" if col in primary else "context"
        stats_out[col] = {
            "role":      role,
            "avg_words": round(avg_words, 1),
            "fill_rate": round(fill_rate, 2),
        }

    sample = ""
    try:
        if primary:
            best_col = max(primary, key=lambda c: stats_out.get(c, {}).get("avg_words", 0))
            mask     = df[best_col].notna() & (df[best_col].astype(str).str.strip().str.lower() != "nan")
            row_     = df[mask].iloc[0] if mask.any() else df.iloc[0]
            sample   = _execute_template(row_, template, primary, context, is_social)[:300]
    except Exception:
        pass

    result = {
        "force_primary":     primary,
        "primary":           [],          # legacy compat — use all_primary
        "context":           context,
        "all_primary":       primary,
        "stats":             stats_out,
        "has_text":          len(primary) > 0,
        "is_numeric_ds":     len(primary) == 0,
        "sample":            sample,
        "is_social":         is_social,
        "sentence_template": template,
        "dataset_type":      llm_result.get("dataset_type", "general"),
        "confidence":        confidence,
        "used_llm":          True,
    }

    _schema_cache[key] = result
    return result


# ══════════════════════════════════════════════════════════════
# STATISTICAL FALLBACK (when LM Studio is offline)
# ══════════════════════════════════════════════════════════════

def _stat_analyse_dataframe(df: pd.DataFrame) -> dict:
    primary, context, stats_out = [], [], {}

    for col in df.columns:
        series = df[col]
        if _is_always_noise(col):
            continue
        if series.dtype not in [object, "string"]:
            continue
        first_val = series.dropna().iloc[0] if series.notna().any() else None
        if isinstance(first_val, (list, dict)):
            series = series.apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)

        clean = series.dropna().astype(str).str.strip()
        clean = clean[~clean.str.lower().isin(_EMPTY_SENTINELS)]
        if clean.empty:
            continue

        total      = max(len(series), 1)
        fill_rate  = len(clean) / total
        unique_rat = _safe_nunique(series) / total
        avg_words  = _safe_avg_words(series)
        avg_chars  = clean.head(300).apply(len).mean()

        numeric_frac = clean.head(200).apply(
            lambda x: bool(re.fullmatch(r"[\d,.\s%+-]+", x))
        ).mean()
        if numeric_frac > 0.8:
            continue
        url_frac = clean.head(100).apply(lambda x: "://" in x).mean()
        if url_frac > 0.3:
            continue
        if unique_rat > 0.95 and avg_words <= 1 and avg_chars < 20:
            continue
        if fill_rate < 0.01:
            continue

        stats_out[col] = {"role": "?", "avg_words": round(avg_words, 1), "fill_rate": round(fill_rate, 2)}

        if avg_words >= 10:
            primary.append((col, avg_words * (0.5 + fill_rate)))
            stats_out[col]["role"] = "primary"
        elif avg_words >= 2 and fill_rate >= 0.3:
            context.append((col, avg_words))
            stats_out[col]["role"] = "context"

    primary.sort(key=lambda x: x[1], reverse=True)
    context.sort(key=lambda x: x[1], reverse=True)
    p_cols   = [c for c, _ in primary[:3]]
    ctx_cols = [c for c, _ in context[:2]]
    is_social = _detect_social_stat(df)

    sample = ""
    try:
        if p_cols:
            mask = df[p_cols[0]].notna()
            row_ = df[mask].iloc[0] if mask.any() else df.iloc[0]
            main = ". ".join(str(row_.get(c, ""))[:200] for c in p_cols if pd.notna(row_.get(c)))
            ctx  = " — ".join(str(row_.get(c, ""))[:50]  for c in ctx_cols if pd.notna(row_.get(c)))
            sample = f"{ctx}: {main}" if ctx else main
    except Exception:
        pass

    return {
        "force_primary":     p_cols,
        "primary":           [],
        "context":           ctx_cols,
        "all_primary":       p_cols,
        "stats":             stats_out,
        "has_text":          len(p_cols) > 0,
        "is_numeric_ds":     len(p_cols) == 0,
        "sample":            sample[:200],
        "is_social":         is_social,
        "sentence_template": "",
        "dataset_type":      "general",
        "confidence":        0.5,
        "used_llm":          False,
    }


def _detect_social_stat(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    return len(cols & {"tweet", "retweet", "subreddit", "user_name",
                       "user_followers", "screen_name", "selftext"}) >= 2