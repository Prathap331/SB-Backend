"""
ingest_from_kaggle.py  — StoryBit Kaggle Ingestion Pipeline v5
══════════════════════════════════════════════════════════════════════
Design:
  • Each CSV row → exactly ONE Supabase document row
  • content  = natural-language sentence built from that row by LLM template
  • metadata = {category, topic, tags, file, kaggle_slug, author,
                published_at (from date columns in original data), source_from}
  • No Wikipedia fallback — if dataset has no text it is skipped
  • LLM reads 50 actual CSV rows (not just stats) to auto-configure itself
  • Gemini primary embedder (768-dim), nomic via LM Studio fallback

Table columns populated per row:
  content       TEXT          ← sentence built from this CSV row
  embedding     vector(768)   ← Gemini or nomic
  source_title  TEXT          ← dataset name
  source_url    TEXT          ← kaggle.com/datasets/{slug}
  source_type   TEXT          ← reddit | twitter | news | book | youtube |
                                 social_media | web_scrape
  metadata      JSONB         ← {category, topic, tags, file, kaggle_slug,
                                  author, published_at, source_from}
  category      TEXT          ← top-level column (also inside metadata)
  topic         TEXT          ← top-level column (also inside metadata)
  source_from   TEXT          ← top-level column (also inside metadata)

Run full pipeline (reads Storybit_Database_updated.xlsx):
  python ingest_from_kaggle.py

Test a single dataset (built-in, no separate file needed):
  python ingest_from_kaggle.py --test --slug owner/dataset-name
  python ingest_from_kaggle.py --test --slug owner/dataset-name --dry-run
  python ingest_from_kaggle.py --test --slug owner/dataset-name --limit 10

LM Studio must have loaded:
  1. lfm2.5-1.2b-instruct-thinking-claude-high-reasoning-mlx  (schema LLM)
  2. text-embedding-nomic-embed-text-v1.5                       (fallback embedder)
══════════════════════════════════════════════════════════════════════
"""

import os, re, sys, json, time, glob, hashlib, argparse
import httpx
import openpyxl
import pandas as pd
import nltk

from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.genai import types as genai_types
from nltk.tokenize import sent_tokenize

load_dotenv()

for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

SPREADSHEET_FILE = "Storybit_Database.xlsx"
PROGRESS_FILE    = "kaggle_progress.log"

GEMINI_MODEL         = "gemini-embedding-001"
LM_STUDIO_BASE       = "http://127.0.0.1:1234"
NOMIC_MODEL_ID       = "text-embedding-nomic-embed-text-v1.5"
NOMIC_EMBED_ENDPOINT = f"{LM_STUDIO_BASE}/v1/embeddings"
NOMIC_PREFIX         = "search_document: "

BATCH_SIZE       = 100     # rows per embed+insert call
NOMIC_BATCH_SIZE = 50
MAX_ROWS_PER_DS  = 2000    # max rows extracted per dataset file
MIN_ROW_WORDS    = 8       # skip sentences shorter than this

_LM_CHAT_URL = f"{LM_STUDIO_BASE}/v1/chat/completions"
_LM_MODELS   = f"{LM_STUDIO_BASE}/api/v1/models"

_use_nomic   = False       # runtime state: switches permanently on Gemini quota hit


# ══════════════════════════════════════════════════════════════
# SPREADSHEET READER
# ══════════════════════════════════════════════════════════════

def read_spreadsheet(filepath: str) -> list[dict]:
    wb   = openpyxl.load_workbook(filepath)
    ws   = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []

    header = [str(c).strip().lower() if c else "" for c in rows[0]]
    col    = {}
    for i, h in enumerate(header):
        if "category" in h:                                       col["category"] = i
        elif "topic"    in h:                                     col["topic"]    = i
        elif "tag"      in h:                                     col["tags"]     = i
        elif "vector"   in h or "kaggle" in h or "database" in h: col["db_link"] = i

    topics = []
    for row in rows[1:]:
        if not any(row):
            continue
        category = str(row[col.get("category", 0)] or "").strip()
        topic    = str(row[col.get("topic",    1)] or "").strip()
        tags_raw = str(row[col.get("tags",     2)] or "").strip()
        db_link  = str(row[col.get("db_link",  4)] or "").strip()

        if not category or not topic:
            continue

        tags: list[str] = []
        if tags_raw and tags_raw.lower() not in ("none", ""):
            try:
                parsed = json.loads(tags_raw)
                tags   = [str(t).strip() for t in parsed if t] if isinstance(parsed, list) else []
            except (json.JSONDecodeError, ValueError):
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

        url = ""
        for line in db_link.splitlines():
            if "kaggle.com" in line:
                url = line.strip()
                break

        topics.append({"category": category, "topic": topic, "tags": tags, "db_link": url})

    print(f"Loaded {len(topics)} topics from {filepath}")
    return topics


# ══════════════════════════════════════════════════════════════
# KAGGLE DOWNLOADER
# ══════════════════════════════════════════════════════════════

def extract_slug(url: str) -> str | None:
    m = re.search(r"kaggle\.com/datasets/([^/\s]+/[^/\s]+)", url.strip())
    return m.group(1) if m else None


def download_dataset(slug: str) -> str | None:
    try:
        import kagglehub
        print(f"  Downloading: {slug}")
        path = kagglehub.dataset_download(slug)
        print(f"  ✓ {path}")
        return path
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# FILE LOADER
# ══════════════════════════════════════════════════════════════

def find_data_files(dataset_path: str) -> list[str]:
    path  = Path(dataset_path)
    files = []
    for ext in ["*.csv", "*.json", "*.jsonl", "*.tsv"]:
        files.extend(glob.glob(str(path / "**" / ext), recursive=True))
    files = [f for f in files if os.path.getsize(f) < 500 * 1024 * 1024]
    files.sort(key=lambda f: os.path.getsize(f), reverse=True)
    return files


def load_dataframe(filepath: str) -> pd.DataFrame | None:
    ext = Path(filepath).suffix.lower()
    try:
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(filepath, sep=sep, encoding=enc,
                                     on_bad_lines="skip", low_memory=False)
                    if not df.empty:
                        return df
                except UnicodeDecodeError:
                    continue

        elif ext == ".json":
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                return pd.DataFrame(raw)
            if isinstance(raw, dict):
                for key in ["data", "items", "records", "results", "articles", "rows"]:
                    if key in raw and isinstance(raw[key], list):
                        return pd.DataFrame(raw[key])
                return pd.DataFrame([raw])

        elif ext == ".jsonl":
            records = []
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return pd.DataFrame(records) if records else None

    except Exception as e:
        print(f"      Load error: {e}")
    return None


# ══════════════════════════════════════════════════════════════
# LM STUDIO SCHEMA ANALYSER
# Sends schema stats + 50 actual CSV rows so LLM fully understands
# the data and writes a correct sentence_template automatically.
# No hardcoded column names, no manual config per dataset.
# ══════════════════════════════════════════════════════════════

_schema_cache: dict[str, dict] = {}

# Columns that are always noise regardless of what LLM says
_ALWAYS_NOISE = {
    "id", "post_id", "comment_id", "tweet_id", "user_id", "uuid",
    "author", "username", "user_name", "screen_name", "handle",
    "url", "link", "href", "permalink",
    "score", "votes", "upvote_ratio", "downvote_ratio", "num_comments",
    "followers", "friends", "favourites", "likes", "ups", "downs",
    "gilded", "distinguished", "stickied", "is_self", "over_18",
    "spoiler", "locked", "user_verified", "user_location",
}
_NOISE_SUBS = ("_id", "_url", "_link", "ratio", "_count", "_votes")

# Known date column names — used to extract published_at
_DATE_NAMES = {
    "date", "created_at", "created_utc", "published_at", "publish_date",
    "timestamp", "post_date", "article_date", "news_date", "time",
    "created", "updated_at", "pub_date", "publication_date", "datetime",
}

# Body-type column names — always primary even if sparse
_BODY_NAMES = {
    "body", "text", "selftext", "content", "description", "summary",
    "abstract", "review", "comment", "post", "transcript", "answer",
    "response", "message", "article", "paragraph", "details",
}


def _is_noise(col: str) -> bool:
    cl = col.lower()
    if cl in _ALWAYS_NOISE:
        return True
    return any(sub in cl for sub in _NOISE_SUBS)


def _safe_avg_words(series: pd.Series, n: int = 300) -> float:
    try:
        clean = series.dropna().astype(str).str.strip()
        clean = clean[~clean.str.lower().isin(
            {"", "nan", "none", "null", "n/a", "[removed]", "[deleted]"})]
        return clean.head(n).apply(lambda x: len(x.split())).mean() if not clean.empty else 0.0
    except Exception:
        return 0.0


def _find_date_col(df: pd.DataFrame) -> str:
    """Return the best date column name, or empty string."""
    for col in df.columns:
        if col.lower() in _DATE_NAMES:
            return col
    for col in df.columns:
        cl = col.lower()
        if "date" in cl or "publish" in cl or ("time" in cl and "sentiment" not in cl):
            return col
    return ""


def _lm_running() -> tuple[bool, str]:
    """Returns (is_running, model_id)."""
    try:
        r    = httpx.get(_LM_MODELS, timeout=5)
        data = r.json()
        mid  = ""
        def _find(obj, d=0):
            nonlocal mid
            if d > 4 or mid:
                return
            if isinstance(obj, dict):
                for field in ["id", "model_id", "modelId"]:
                    v = obj.get(field, "")
                    if isinstance(v, str) and len(v) > 5:
                        mid = v; return
                for val in obj.values():
                    _find(val, d + 1)
            elif isinstance(obj, list):
                for item in obj:
                    _find(item, d + 1)
        _find(data)
        if not mid:
            mid = "lfm2.5-1.2b-instruct-thinking-claude-high-reasoning-mlx"
        return True, mid
    except Exception:
        return False, ""


_SCHEMA_SYSTEM = """You are a data engineer for a RAG pipeline. You receive:
1. SCHEMA TABLE — column statistics (name, dtype, fill%, avg words, sample values)
2. ACTUAL DATA ROWS — 50 real rows from the CSV file

Read BOTH carefully. The actual rows are ground truth — they tell you what the data really contains.

OUTPUT: JSON that configures how each CSV row becomes one document in a vector database.

CLASSIFICATION:
  primary  = columns with RICH NARRATIVE TEXT to embed as the document content
             Rule: avg_words >= 8, OR column named body/text/selftext/content/
             description/summary/abstract/review/comment/post/transcript/answer
             A sparse column (10% fill) with rich sentences is still PRIMARY.
  context  = SHORT LABELS used as sentence prefix (1-8 words, categorical)
             Examples: subreddit, category, country, title, platform, source
  noise    = SKIP entirely: IDs, URLs, usernames, timestamps, numbers, booleans
             Any column with avg_words ≤ 1.5 AND unique% > 85% is an ID → noise
  date_col = The column holding the ORIGINAL publish/creation date of the content.
             Look for: date, created_at, created_utc, published_at, timestamp, etc.
             This goes into metadata.published_at so we know WHEN the content was published.
             Set to "" if no date column exists.

SENTENCE TEMPLATE — write a Python f-string using {col_name} placeholders.
Read the actual rows to understand what a natural sentence looks like.
  Reddit:  "{subreddit}: {title}. {body}"
  News:    "{source} — {category}: {title}. {summary}"
  Books:   "{title} by {authors}: {description}"
  Twitter: "{screen_name}: {text}"
  Generic: "{category}: {content}"
ONLY reference primary and context columns. NEVER reference noise columns.

Return ONLY valid JSON (no markdown, no extra text):
{
  "primary":           ["col"],
  "context":           ["col"],
  "noise":             ["col"],
  "date_col":          "created_utc",
  "is_social":         true,
  "dataset_type":      "reddit_comments",
  "sentence_template": "{subreddit}: {title}. {body}",
  "confidence":        0.95,
  "reasoning":         "single sentence explaining your decision"
}"""


def _call_schema_llm(df: pd.DataFrame, model_id: str) -> dict | None:
    """Sends schema stats + 50 actual rows to LM Studio. Returns parsed JSON."""

    # ── 1. Schema stats table ─────────────────────────────────
    stat_lines = [
        f"{'Column':<30} {'dtype':<10} {'fill%':<7} {'avg_words':<11} {'Samples (first 3)'}",
        "-" * 100,
    ]
    for col in df.columns:
        s      = df[col]
        dtype  = str(s.dtype)
        fill   = f"{s.notna().sum() / max(len(s), 1) * 100:.0f}%"
        first  = s.dropna().iloc[0] if s.notna().any() else None
        if isinstance(first, (list, dict)) or s.dtype not in [object, "string"]:
            avg_w = "N/A"
        else:
            avg_w = f"{_safe_avg_words(s):.1f}"
        samples = (
            s.dropna()
             .apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
             .astype(str).str.strip()
             .replace(["nan", "none", "null", ""], pd.NA)
             .dropna().head(3).tolist()
        )
        samples = [v[:90] + ("…" if len(v) > 90 else "") for v in samples]
        stat_lines.append(
            f"{col:<30} {dtype:<10} {fill:<7} {avg_w:<11} {' | '.join(samples)}"
        )

    schema_block = (
        f"SCHEMA — {df.shape[0]:,} rows × {df.shape[1]} columns\n"
        + "\n".join(stat_lines)
    )

    # ── 2. Actual data rows (up to 50, spread across file) ────
    n_sample = min(50, len(df))
    step     = max(1, len(df) // n_sample)
    indices  = list(range(0, len(df), step))[:n_sample]

    # For wide dataframes (>18 cols) show only richest columns
    cols_show = list(df.columns)
    if len(cols_show) > 18:
        scored = []
        for col in cols_show:
            first = df[col].dropna().iloc[0] if df[col].notna().any() else None
            if isinstance(first, (list, dict)) or df[col].dtype not in [object, "string"]:
                scored.append((col, 0.0))
            else:
                scored.append((col, _safe_avg_words(df[col])))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [c for c, _ in scored[:18]]
        cols_show = [c for c in df.columns if c in top]  # preserve original order

    row_lines = [
        "",
        f"ACTUAL DATA ROWS — {len(indices)} sample rows from {len(df):,} total",
        "(Read these to understand what the data really contains)",
        "",
    ]
    for idx in indices:
        row = df.iloc[idx]
        row_lines.append(f"  ── row {idx} ──")
        for col in cols_show:
            val = row[col]
            try:
                if pd.isna(val):
                    continue
            except (TypeError, ValueError):
                pass
            val_str = str(val).strip()
            if not val_str or val_str.lower() in (
                "nan", "none", "null", "[removed]", "[deleted]", "[ removed by moderator ]"
            ):
                continue
            if len(val_str) > 250:
                val_str = val_str[:247] + "…"
            row_lines.append(f"    {col}: {val_str}")
        row_lines.append("")

    full_prompt = (
        schema_block + "\n" + "\n".join(row_lines) +
        "\nBased on the schema AND the actual rows above, "
        "classify all columns and write a sentence_template that produces "
        "a natural, readable sentence from each row."
    )

    payload = {
        "model":    model_id,
        "messages": [
            {"role": "system", "content": _SCHEMA_SYSTEM},
            {"role": "user",   "content": full_prompt},
        ],
        "temperature": 0.05,
        "max_tokens":  900,
        "stream":      False,
    }

    try:
        r = httpx.post(_LM_CHAT_URL, json=payload, timeout=150)
        if r.status_code != 200:
            print(f"      [LM Studio] HTTP {r.status_code}: {r.text[:200]}")
            return None

        content = r.json()["choices"][0]["message"]["content"].strip()
        # Strip <think> blocks (thinking models)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content).strip()

        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            print(f"      [LM Studio] No JSON found in response")
            return None

        result = json.loads(m.group())
        result.setdefault("confidence",        1.0)
        result.setdefault("reasoning",         "")
        result.setdefault("is_social",         False)
        result.setdefault("dataset_type",      "general")
        result.setdefault("sentence_template", "")
        result.setdefault("date_col",          "")
        return result

    except json.JSONDecodeError as e:
        print(f"      [LM Studio] JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"      [LM Studio] Exception: {e}")
        return None


def _validate_schema(result: dict, df: pd.DataFrame) -> dict:
    """Hard-correct LLM output: enforce noise rules, demote short cols, fix template."""
    valid   = set(df.columns)
    primary, context, noise = [], [], list(result.get("noise", []))

    for col in result.get("primary", []):
        if col not in valid:
            continue
        if _is_noise(col) or df[col].dtype in ["int64", "float64", "int32", "float32"]:
            noise.append(col); continue
        avg_w = _safe_avg_words(df[col])
        if avg_w < 8 and col.lower() not in _BODY_NAMES:
            context.append(col)       # demote to context
        else:
            primary.append(col)

    for col in result.get("context", []):
        if col not in valid:
            continue
        if _is_noise(col) or df[col].dtype in ["int64", "float64", "int32", "float32"]:
            noise.append(col); continue
        context.append(col)

    # Route any column the LLM didn't classify
    classified = set(primary + context + noise)
    for col in df.columns:
        if col in classified:
            continue
        if _is_noise(col) or df[col].dtype in ["int64", "float64", "int32", "float32"]:
            noise.append(col)
        else:
            avg_w = _safe_avg_words(df[col])
            if avg_w >= 8 or col.lower() in _BODY_NAMES:
                primary.append(col)
            elif avg_w >= 2:
                context.append(col)
            else:
                noise.append(col)

    # Sanitise template — remove refs to noise or missing cols
    tmpl    = result.get("sentence_template", "")
    allowed = set(primary + context)
    for ref in re.findall(r"\{(\w+)\}", tmpl):
        if ref not in valid or ref not in allowed:
            tmpl = tmpl.replace(f"{{{ref}}}", "")
    tmpl = re.sub(r"\.\s*\.", ".", tmpl)
    tmpl = re.sub(r":\s*\.",  ".", tmpl)
    tmpl = re.sub(r"—\s*\.",  ".", tmpl)
    tmpl = re.sub(r"\s{2,}",  " ", tmpl).strip()
    tmpl = re.sub(r"[—:\s]+$","",  tmpl).strip()

    # Validate / auto-detect date column
    date_col = result.get("date_col", "")
    if date_col not in valid:
        date_col = _find_date_col(df)

    return {
        **result,
        "primary":           list(dict.fromkeys(primary)),
        "context":           list(dict.fromkeys(context)),
        "noise":             list(dict.fromkeys(noise)),
        "sentence_template": tmpl,
        "date_col":          date_col,
    }


def _stat_fallback(df: pd.DataFrame) -> dict:
    """Statistical fallback when LM Studio is offline."""
    primary, context = [], []
    for col in df.columns:
        if _is_noise(col) or df[col].dtype not in [object, "string"]:
            continue
        first = df[col].dropna().iloc[0] if df[col].notna().any() else None
        if isinstance(first, (list, dict)):
            continue
        avg_w = _safe_avg_words(df[col])
        if avg_w >= 8 or col.lower() in _BODY_NAMES:
            primary.append(col)
        elif avg_w >= 2:
            context.append(col)
    date_col = _find_date_col(df)
    tmpl     = ""
    if primary:
        pfx  = f"{{{context[0]}}}: " if context else ""
        tmpl = pfx + ". ".join(f"{{{c}}}" for c in primary[:2])
    return {
        "primary":           primary[:3],
        "context":           context[:2],
        "noise":             [],
        "date_col":          date_col,
        "is_social":         False,
        "dataset_type":      "general",
        "sentence_template": tmpl,
        "confidence":        0.5,
        "reasoning":         "stat fallback — LM Studio offline",
        "used_llm":          False,
    }


def analyse_schema(df: pd.DataFrame) -> dict:
    """
    Analyse a DataFrame → return schema dict.
    LM Studio reads 50 actual rows to auto-configure. Falls back to stats.
    Result cached per column signature.
    """
    key = hashlib.md5((str(list(df.columns)) + str(list(df.dtypes))).encode()).hexdigest()
    if key in _schema_cache:
        print(f"      [LM Studio] Using cached schema")
        return _schema_cache[key]

    running, model_id = _lm_running()
    if not running:
        print(f"      [LM Studio] Offline — using stats fallback")
        result = _stat_fallback(df)
        _schema_cache[key] = result
        return result

    print(f"      [LM Studio] Sending {min(50, len(df))} rows to {model_id}...")
    result = _call_schema_llm(df, model_id)

    if result is None:
        print(f"      [LM Studio] LLM failed — using stats fallback")
        result = _stat_fallback(df)
        result["used_llm"] = False
    else:
        result = _validate_schema(result, df)
        result["used_llm"] = True
        print(f"      type={result['dataset_type']}  "
              f"confidence={result['confidence']:.0%}  "
              f"date_col={result['date_col'] or 'none'}")
        print(f"      template: {result['sentence_template']}")
        print(f"      reasoning: {result['reasoning']}")
        print(f"      primary: {result['primary']}")
        print(f"      context: {result['context']}")

        # Low-confidence retry
        if result["confidence"] < 0.65:
            print(f"      Low confidence — retrying...")
            r2 = _call_schema_llm(df, model_id)
            if r2 and r2.get("confidence", 0) > result["confidence"]:
                result = _validate_schema(r2, df)
                result["used_llm"] = True
                print(f"      Retry confidence: {result['confidence']:.0%}")

    _schema_cache[key] = result
    return result


# ══════════════════════════════════════════════════════════════
# ROW → SENTENCE
# Each CSV row becomes one natural-language sentence
# ══════════════════════════════════════════════════════════════

_URL_RE  = re.compile(r"https?://\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE   = re.compile(r"\s+")
_BAD     = {"nan","none","null","n/a","na","","[]","{}","unknown",
            "[removed]","[deleted]","[ removed by moderator ]"}


def _clean(v: str, social: bool = False) -> str:
    v = _HTML_RE.sub(" ", str(v).strip())
    if social:
        v = _URL_RE.sub("", v)
    return _WS_RE.sub(" ", v).strip()


def _get(row: pd.Series, col: str, social: bool = False) -> str:
    val = row.get(col, "")
    if isinstance(val, (list, dict)):
        val = json.dumps(val)
    v = str(val).strip()
    return "" if v.lower() in _BAD else _clean(v, social)


def build_row_sentence(row: pd.Series, schema: dict) -> str:
    """
    Converts one CSV row into a sentence using the LLM template.
    Returns "" if the row has no usable content.
    """
    primary  = schema.get("primary",   [])
    context  = schema.get("context",   [])
    template = schema.get("sentence_template", "")
    social   = schema.get("is_social", False)

    if not primary:
        return ""

    # Gate: require the first primary column to have content
    if not _get(row, primary[0], social):
        return ""

    result = ""

    if template:
        filled = template
        for ref in re.findall(r"\{(\w+)\}", template):
            filled = filled.replace(f"{{{ref}}}", _get(row, ref, social))
        # Clean up empty slots
        filled = re.sub(r"\.\s*\.", ".", filled)
        filled = re.sub(r":\s*\.",  ".", filled)
        filled = re.sub(r"—\s*\.",  ".", filled)
        filled = re.sub(r"\s{2,}",  " ", filled).strip()
        filled = re.sub(r"[—:\s]+$","",  filled).strip()
        result = filled

    if not result or len(result.split()) < MIN_ROW_WORDS:
        # Fallback: context prefix + join primary values
        parts  = [_get(row, c, social) for c in primary]
        parts  = [p for p in parts if p]
        if not parts:
            return ""
        ctx    = [_get(row, c, social) for c in context[:2] if _get(row, c, social)]
        result = (" — ".join(ctx) + ": " if ctx else "") + ". ".join(parts)

    return result.strip() if len(result.split()) >= MIN_ROW_WORDS else ""


# ══════════════════════════════════════════════════════════════
# SOURCE TYPE DETECTOR
# ══════════════════════════════════════════════════════════════

_SOURCE_RULES = [
    (["reddit", "subreddit", "r-"],                              "reddit"),
    (["twitter", "tweet", "x-com"],                              "twitter"),
    (["news", "article", "headline", "bbc", "cnn", "reuters",
      "guardian", "nyt", "times", "press", "journal"],           "news"),
    (["book", "novel", "literature", "fiction", "goodreads"],    "book"),
    (["youtube", "video", "channel", "transcript"],              "youtube"),
    (["instagram", "tiktok", "facebook", "linkedin", "social"],  "social_media"),
    (["election", "vote", "ballot", "parliament", "congress"],   "news"),
    (["wikipedia", "wiki"],                                       "wikipedia"),
]
_src_cache: dict[str, str] = {}


def detect_source_type(slug: str) -> str:
    if slug in _src_cache:
        return _src_cache[slug]
    name = slug.split("/")[-1].lower()
    for keywords, st in _SOURCE_RULES:
        if any(kw in name for kw in keywords):
            _src_cache[slug] = st
            return st
    # Quick LLM call
    try:
        ok, mid = _lm_running()
        if ok:
            r = httpx.post(_LM_CHAT_URL, json={
                "model": mid,
                "messages": [
                    {"role": "system",
                     "content": "Classify this Kaggle dataset name as ONE of: "
                                "reddit twitter news book youtube social_media web_scrape. "
                                "Reply with the single word only."},
                    {"role": "user", "content": name},
                ],
                "max_tokens": 5, "temperature": 0,
            }, timeout=10)
            if r.status_code == 200:
                word = r.json()["choices"][0]["message"]["content"].strip().lower().split()[0].strip(".,;:")
                if word in {"reddit","twitter","news","book","youtube","social_media","web_scrape"}:
                    _src_cache[slug] = word
                    return word
    except Exception:
        pass
    _src_cache[slug] = "web_scrape"
    return "web_scrape"


# ══════════════════════════════════════════════════════════════
# DOCUMENT BUILDER  — one per CSV row
# ══════════════════════════════════════════════════════════════

def _make_doc(sentence: str, row: pd.Series, schema: dict,
              slug: str, fname: str,
              category: str, topic: str, tags: list[str]) -> dict:
    """
    Builds the exact dict inserted into Supabase documents table.

    content      = sentence built from this CSV row
    source_title = dataset name
    source_url   = kaggle URL
    source_type  = auto-detected
    category     = top-level column + in metadata
    topic        = top-level column + in metadata
    source_from  = top-level column + in metadata
    metadata     = full JSONB with published_at extracted from date column
    """
    dataset_name = slug.split("/")[-1]
    owner        = slug.split("/")[0]
    source_from  = f"Kaggle: {dataset_name}"

    # Extract original publish date from the row if a date column was found
    published_at = None
    date_col     = schema.get("date_col", "")
    if date_col and date_col in row.index:
        raw = row[date_col]
        try:
            is_na = pd.isna(raw)
        except (TypeError, ValueError):
            is_na = False
        if not is_na:
            v = str(raw).strip()
            if v and v.lower() not in ("", "nan", "none", "null"):
                published_at = v

    metadata: dict = {
        "category":    category,
        "topic":       topic,
        "tags":        tags,
        "source_from": source_from,
        "kaggle_slug": slug,
        "file":        fname,
        "author": {
            "has_credentials": True,
            "name":            owner,
            "description":     "Kaggle dataset contributor",
        },
    }
    if published_at:
        metadata["published_at"] = published_at

    return {
        "content":      sentence,
        "source_title": dataset_name,
        "source_url":   f"https://www.kaggle.com/datasets/{slug}",
        "source_type":  detect_source_type(slug),
        # Top-level denormalised columns (also in metadata)
        "category":    category,
        "topic":       topic,
        "source_from": source_from,
        "metadata":    metadata,
    }


# ══════════════════════════════════════════════════════════════
# DATASET PROCESSOR
# Downloads dataset → iterates CSV files → one doc per row
# ══════════════════════════════════════════════════════════════

def process_dataset(dataset_path: str, topic: str, category: str,
                    tags: list[str], slug: str) -> list[dict]:
    """
    All files in dataset → list of doc dicts ready for embedding.
    Each CSV row becomes exactly one document.
    Returns [] if no text columns found.
    """
    files = find_data_files(dataset_path)
    if not files:
        print(f"    No data files found.")
        return []

    print(f"    Files: {[Path(f).name for f in files]}")

    all_docs    = []
    seen_hashes = set()
    total_rows  = 0

    for filepath in files:
        if total_rows >= MAX_ROWS_PER_DS:
            break

        fname   = Path(filepath).name
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"\n    ── {fname}  ({size_mb:.1f} MB)")

        df = load_dataframe(filepath)
        if df is None or df.empty:
            print(f"      Empty or unreadable — skipping")
            continue

        rows_left = MAX_ROWS_PER_DS - total_rows
        df = df.head(rows_left)
        print(f"      Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

        # LM Studio reads actual rows to auto-configure
        schema  = analyse_schema(df)
        primary = schema.get("primary", [])

        if not primary:
            print(f"      ✗ No text columns found — skipping file")
            continue

        added = skipped_empty = skipped_dup = 0

        for _, row in df.iterrows():
            sentence = build_row_sentence(row, schema)
            if not sentence:
                skipped_empty += 1
                continue

            h = hashlib.md5(sentence.encode()).hexdigest()
            if h in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(h)

            all_docs.append(_make_doc(sentence, row, schema,
                                       slug, fname, category, topic, tags))
            added += 1

        total_rows += len(df)
        print(f"      → {added:,} rows  "
              f"(skipped: {skipped_empty} empty, {skipped_dup} dup)")

    print(f"\n    Total docs: {len(all_docs):,}")
    return all_docs


# ══════════════════════════════════════════════════════════════
# EMBEDDING  — Gemini primary, nomic fallback
# ══════════════════════════════════════════════════════════════

class DailyQuotaError(Exception):
    pass


def _parse_retry_delay(err: str) -> float:
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"]([\d.]+)s", err)
    if m: return float(m.group(1)) + 2
    m = re.search(r"retry in ([\d.]+)s", err, re.IGNORECASE)
    if m: return float(m.group(1)) + 2
    return 65.0


def _embed_gemini(contents: list[str], gemini_client) -> list:
    for attempt in range(5):
        try:
            resp = gemini_client.models.embed_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768,
                ),
            )
            return [list(e.values) for e in resp.embeddings]
        except Exception as e:
            err = str(e)
            if "PerDay" in err or "EmbedContentRequestsPerDayPerUser" in err:
                raise DailyQuotaError(err)
            if ("429" in err or "RESOURCE_EXHAUSTED" in err) and attempt < 4:
                time.sleep(_parse_retry_delay(err))
            elif attempt < 4:
                time.sleep(5 * (attempt + 1))
            else:
                return []
    return []


def _embed_nomic(contents: list[str]) -> list:
    prefixed = [NOMIC_PREFIX + c for c in contents]
    all_emb  = []
    for i in range(0, len(prefixed), NOMIC_BATCH_SIZE):
        batch = prefixed[i:i + NOMIC_BATCH_SIZE]
        try:
            r = httpx.post(NOMIC_EMBED_ENDPOINT,
                           json={"model": NOMIC_MODEL_ID, "input": batch},
                           timeout=120)
            if r.status_code != 200:
                print(f"    [nomic] HTTP {r.status_code}")
                return []
            items = sorted(r.json()["data"], key=lambda x: x["index"])
            all_emb.extend(item["embedding"] for item in items)
        except Exception as e:
            print(f"    [nomic] Failed: {e}")
            return []
    return all_emb


def _verify_nomic() -> bool:
    try:
        r = httpx.post(NOMIC_EMBED_ENDPOINT,
                       json={"model": NOMIC_MODEL_ID, "input": ["test"]},
                       timeout=15)
        return r.status_code == 200
    except Exception:
        return False


def embed_and_insert(batch: list[dict], supabase: Client, gemini_client) -> bool:
    """Embeds a batch of docs and inserts to Supabase. Returns True on success."""
    global _use_nomic
    contents = [item["content"] for item in batch]

    if _use_nomic:
        print(f"    [nomic] Embedding {len(batch)} rows...")
        embeddings = _embed_nomic(contents)
    else:
        print(f"    [Gemini] Embedding {len(batch)} rows...")
        try:
            embeddings = _embed_gemini(contents, gemini_client)
        except DailyQuotaError:
            print(f"\n  {'!'*50}")
            print(f"  Gemini daily quota hit — checking nomic fallback...")
            if not _verify_nomic():
                print(f"  ✗ nomic not loaded in LM Studio — aborting")
                return False
            print(f"  ✓ nomic available. AUTO-SWITCHING (permanent for this run)")
            print(f"  {'!'*50}\n")
            _use_nomic = True
            embeddings = _embed_nomic(contents)

    if not embeddings:
        print(f"    ✗ Embedding returned empty — skipping batch")
        return False

    if len(embeddings[0]) != 768:
        print(f"    ✗ Wrong dimension: {len(embeddings[0])} (expected 768)")
        return False

    rows = [{**item, "embedding": emb} for item, emb in zip(batch, embeddings)]

    try:
        print(f"    Inserting {len(rows)} rows...")
        supabase.table("documents").insert(rows).execute()
        print(f"    ✓ Inserted.")
        return True
    except Exception as e:
        print(f"    ✗ Insert failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# PROGRESS LOG
# ══════════════════════════════════════════════════════════════

def load_progress() -> set[str]:
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE) as f:
        return {line.strip() for line in f if line.strip()}


def mark_done(slug: str):
    with open(PROGRESS_FILE, "a") as f:
        f.write(slug + "\n")


# ══════════════════════════════════════════════════════════════
# FULL PIPELINE  — reads spreadsheet, processes all datasets
# ══════════════════════════════════════════════════════════════

def run_full_pipeline(gemini_client, supabase: Client):
    topics  = read_spreadsheet(SPREADSHEET_FILE)
    done    = load_progress()
    batch   = []
    total   = 0
    skipped = 0

    print(f"Topics: {len(topics)} | Done: {len(done)} | Remaining: {len(topics)-len(done)}\n")

    for idx, row in enumerate(topics, 1):
        category = row["category"]
        topic    = row["topic"]
        tags     = row["tags"]
        db_link  = row["db_link"]

        print(f"\n{'='*60}")
        print(f"[{idx:02d}/{len(topics)}] {category} → '{topic}'")

        if not db_link:
            print(f"  ⚠  No Kaggle link — skipping")
            skipped += 1; continue

        slug = extract_slug(db_link)
        if not slug:
            print(f"  ✗ Cannot parse slug from: {db_link}")
            skipped += 1; continue

        if slug in done:
            print(f"  ✓ Already ingested"); continue

        path = download_dataset(slug)
        if not path:
            skipped += 1; continue

        docs = process_dataset(path, topic, category, tags, slug)
        if not docs:
            print(f"  ✗ No content extracted")
            mark_done(slug); skipped += 1; continue

        ds_rows = 0
        for doc in docs:
            batch.append(doc)
            ds_rows += 1
            if len(batch) >= BATCH_SIZE:
                ok = embed_and_insert(batch, supabase, gemini_client)
                if not ok:
                    print("  CRITICAL: batch failed — aborting")
                    return
                total += len(batch)
                batch  = []
                if not _use_nomic:
                    print(f"    Waiting 62s (Gemini per-minute rate limit)...")
                    time.sleep(62)

        print(f"  → {ds_rows:,} rows queued")
        mark_done(slug)

    if batch:
        print(f"\nFlushing final {len(batch)} rows...")
        ok = embed_and_insert(batch, supabase, gemini_client)
        if ok:
            total += len(batch)

    print(f"\n{'='*60}")
    print(f"Done!  Rows inserted: {total:,}  |  Datasets skipped: {skipped}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════
# TEST MODE  — single dataset, no separate file needed
#
# Usage:
#   python ingest_from_kaggle.py --test
#   python ingest_from_kaggle.py --test --slug owner/name
#   python ingest_from_kaggle.py --test --slug owner/name --dry-run
#   python ingest_from_kaggle.py --test --slug owner/name --limit 10
# ══════════════════════════════════════════════════════════════

def run_test_mode(args, gemini_client, supabase):
    slug     = args.slug
    topic    = args.topic
    category = args.category
    tags     = [t.strip() for t in args.tags.split(",") if t.strip()]
    dry_run  = args.dry_run
    limit    = args.limit

    W = 66
    print(f"\n{'═'*W}")
    print(f"  StoryBit — Single Dataset Test")
    print(f"{'═'*W}")
    print(f"  Slug:     {slug}")
    print(f"  Topic:    {topic}")
    print(f"  Category: {category}")
    print(f"  Tags:     {tags}")
    print(f"  Dry run:  {dry_run}")
    print(f"  Limit:    {limit if limit > 0 else 'all rows'}")
    print(f"{'═'*W}\n")

    # Download
    print("── Step 1: Download ──────────────────────────────────")
    path = download_dataset(slug)
    if not path:
        sys.exit("✗ Download failed")

    # Schema analysis + extraction
    print("\n── Step 2: LM Studio reads CSV → extracts rows ───────")
    docs = process_dataset(path, topic, category, tags, slug)
    if not docs:
        sys.exit("✗ No rows extracted — check schema analysis output above")

    if limit > 0:
        docs = docs[:limit]
        print(f"\n  (limited to {limit} rows for this test)")

    # Preview
    print(f"\n── Step 3: Preview — exactly what Supabase will store ─")
    for i, doc in enumerate(docs[:3]):
        print(f"\n  ┌─ Row {i+1} {'─'*54}")
        def pr(label, val):
            print(f"  │  {label:<14} {str(val)[:80]}")
        pr("content",      doc["content"])
        pr("source_title", doc["source_title"])
        pr("source_url",   doc["source_url"])
        pr("source_type",  doc["source_type"])
        pr("category",     doc["category"])
        pr("topic",        doc["topic"])
        pr("source_from",  doc["source_from"])
        m = doc["metadata"]
        print(f"  │  metadata      {{")
        print(f"  │    category:     {m.get('category','')}")
        print(f"  │    topic:        {m.get('topic','')}")
        print(f"  │    tags:         {m.get('tags','')}")
        print(f"  │    source_from:  {m.get('source_from','')}")
        print(f"  │    file:         {m.get('file','')}")
        print(f"  │    kaggle_slug:  {m.get('kaggle_slug','')}")
        pa = m.get('published_at', None)
        print(f"  │    published_at: {pa if pa else '(no date column in data)'}")
        a = m.get('author', {})
        print(f"  │    author.name:  {a.get('name','')}")
        print(f"  │  }}")
        print(f"  └{'─'*58}")

    print(f"\n  Total rows to insert: {len(docs)}")

    if dry_run:
        print(f"\n── DRY RUN — nothing written to Supabase ─────────────")
        print(f"\n  When you run for real, verify with:")
        _print_sql(topic)
        return

    # Embed + insert
    print(f"\n── Step 4: Embed + Insert → Supabase ─────────────────")
    ok = embed_and_insert(docs, supabase, gemini_client)
    if not ok:
        sys.exit("✗ Insert failed")

    print(f"\n── Done! {len(docs)} rows inserted.\n")
    _print_sql(topic)


def _print_sql(topic: str):
    print("  Supabase SQL to verify:")
    print(f"""
  -- View your rows
  SELECT id, content, source_type, category, topic,
         source_from, metadata->>'published_at' AS published_at,
         created_at
  FROM documents
  WHERE topic = '{topic}'
  ORDER BY created_at DESC
  LIMIT 20;

  -- Count rows by source_type
  SELECT source_type, COUNT(*) FROM documents GROUP BY source_type;

  -- Check metadata published_at is populated
  SELECT id, metadata->>'published_at' AS published_at
  FROM documents
  WHERE topic = '{topic}' AND metadata->>'published_at' IS NOT NULL
  LIMIT 5;

  -- Delete test rows
  DELETE FROM documents WHERE topic = '{topic}';

  -- Full wipe (fresh start)
  TRUNCATE TABLE documents RESTART IDENTITY;
""")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StoryBit Kaggle ingestion — full pipeline + test mode")
    parser.add_argument("--test",     action="store_true",
                        help="Test mode: run on one dataset (does not touch progress log)")
    parser.add_argument("--slug",     default="umerhaddii/world-news-reddit-data-global-event-discussions",
                        help="[test] Kaggle dataset slug  e.g.  owner/dataset-name")
    parser.add_argument("--topic",    default="Geopolitical conflicts",
                        help="[test] Topic label")
    parser.add_argument("--category", default="Politics & Current Affairs",
                        help="[test] Category label")
    parser.add_argument("--tags",     default="geopolitics",
                        help="[test] Comma-separated tags")
    parser.add_argument("--dry-run",  action="store_true",
                        help="[test] Preview rows, do NOT insert to Supabase")
    parser.add_argument("--limit",    type=int, default=0,
                        help="[test] Max rows to insert (0 = no limit)")
    args = parser.parse_args()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    supabase_url   = os.getenv("SUPABASE_URL")
    supabase_key   = os.getenv("SUPABASE_KEY")

    if not google_api_key: sys.exit("✗  GOOGLE_API_KEY missing in .env")
    if not supabase_url:   sys.exit("✗  SUPABASE_URL missing in .env")
    if not supabase_key:   sys.exit("✗  SUPABASE_KEY missing in .env")

    ku = os.getenv("KAGGLE_USERNAME", "")
    kk = os.getenv("KAGGLE_KEY", "")
    if ku and kk:
        os.environ["KAGGLE_USERNAME"] = ku
        os.environ["KAGGLE_KEY"]      = kk

    _gemini = genai.Client(api_key=google_api_key)
    _supa   = create_client(supabase_url, supabase_key)

    if args.test or args.dry_run:
        run_test_mode(args, _gemini, _supa)
    else:
        print("=" * 60)
        print("StoryBit Kaggle Ingestion v5")
        print(f"Spreadsheet: {SPREADSHEET_FILE}")
        print("=" * 60)
        run_full_pipeline(_gemini, _supa)