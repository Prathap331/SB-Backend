import os
import csv
import json
import glob
from datasets import load_dataset, get_dataset_config_names
from sentence_transformers import SentenceTransformer
from supabase import create_client
from urllib.parse import urlparse

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

SIZE_LIMIT = 250 * 250


def safe_load_dataset(dsname):
    try:
        return load_dataset(dsname, split="train", streaming=True)
    except:
        try:
            configs = get_dataset_config_names(dsname)
            return load_dataset(dsname, configs[0], split="train", streaming=True)
        except:
            return None


def extract_field(row, keys):
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if key in row and row[key]:
            return row[key]
    return None



def iter_file(filepath):
    ext = os.path.splitext(filepath)[-1].lower()

    if ext == ".csv":
        for encoding in ("utf-8", "latin-1", "cp1252", "iso-8859-1"):
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    rows = list(csv.DictReader(f))
                print(f"  ✅ Read CSV with encoding: {encoding} ({len(rows)} rows)")
                yield from rows
                return
            except (UnicodeDecodeError, Exception):
                continue
        raise ValueError(f"Could not decode CSV: {filepath}")

    elif ext == ".json":
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    yield from data
                elif isinstance(data, dict):
                    for key in ("data", "rows", "records", "items"):
                        if key in data and isinstance(data[key], list):
                            yield from data[key]
                            return
                    yield data
                return
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        raise ValueError(f"Could not decode JSON: {filepath}")

    elif ext == ".jsonl":
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    rows = [json.loads(line) for line in f if line.strip()]
                print(f"  ✅ Read JSONL with encoding: {encoding} ({len(rows)} rows)")
                yield from rows
                return
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        raise ValueError(f"Could not decode JSONL: {filepath}")

    elif ext in (".parquet", ".pq"):
        try:
            import pandas as pd
            df = pd.read_parquet(filepath)
            print(f"  ✅ Read Parquet: {len(df)} rows, columns: {list(df.columns)}")
            for _, row in df.iterrows():
                yield row.where(row.notna(), other=None).to_dict()
        except ImportError:
            raise ImportError("pip install pandas pyarrow")

    elif ext in (".xlsx", ".xls"):
        try:
            import pandas as pd
            df = pd.read_excel(filepath)
            print(f"  ✅ Read Excel: {len(df)} rows, columns: {list(df.columns)}")
            for _, row in df.iterrows():
                yield row.where(row.notna(), other=None).to_dict()
        except ImportError:
            raise ImportError("pip install pandas openpyxl")

    elif ext == ".tsv":
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    rows = list(csv.DictReader(f, delimiter="\t"))
                print(f"  ✅ Read TSV with encoding: {encoding} ({len(rows)} rows)")
                yield from rows
                return
            except (UnicodeDecodeError, Exception):
                continue
        raise ValueError(f"Could not decode TSV: {filepath}")

    else:
        raise ValueError(f"Unsupported file type: {ext} — supported: csv, tsv, json, jsonl, parquet, xlsx, xls")



def iter_kaggle(dataset_slug, col_text, username=None, key_env="KAGGLE_KEY", extract_dir="kaggle_tmp"):
    import kaggle

    kaggle_user = username or os.getenv("KAGGLE_USERNAME")
    kaggle_key  = os.getenv(key_env)

    if not kaggle_user or not kaggle_key:
        raise ValueError("Kaggle credentials missing. Set KAGGLE_USERNAME and KAGGLE_KEY env vars.")

    os.environ["KAGGLE_USERNAME"] = kaggle_user
    os.environ["KAGGLE_KEY"]      = kaggle_key
    os.makedirs(extract_dir, exist_ok=True)

    print(f"📥 Downloading Kaggle dataset: {dataset_slug}")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_slug, path=extract_dir, unzip=True, quiet=False)

    extensions = ("*.csv", "*.json", "*.jsonl", "*.parquet", "*.pq", "*.xlsx", "*.xls", "*.tsv")
    found_files = []
    for ext in extensions:
        found_files += glob.glob(f"{extract_dir}/**/{ext}", recursive=True)

    if not found_files:
        raise FileNotFoundError(f"No supported files found in {extract_dir} after download.")

    print(f"📂 Found {len(found_files)} file(s): {[os.path.basename(f) for f in found_files]}")

    for filepath in found_files:
        print(f"  ↳ Reading: {os.path.basename(filepath)}")
        try:
            yield from iter_file(filepath)
        except Exception as e:
            print(f"  ⚠️ Skipping {filepath}: {e}")



def iter_source(dsname_or_path, source_type="dataset", kaggle_slug=None, col_text="text", extract_dir="kaggle_tmp"):
    if source_type == "file":
        yield from iter_file(dsname_or_path)
    elif source_type == "kaggle":
        yield from iter_kaggle(kaggle_slug or dsname_or_path, col_text=col_text, extract_dir=extract_dir)
    else:
        ds = safe_load_dataset(dsname_or_path)
        if ds is None:
            return
        yield from ds



def ingest_dataset(
    dsname,
    source,
    source_type="dataset",   
    col_text="text",
    col_language=None,
    col_url=None,
    extra_cols=None,
    kaggle_slug=None,
    extract_dir="kaggle_tmp"
):
    print(f"\n🚀 Processing: {dsname} (type={source_type})")

    text_keys     = [col_text] if isinstance(col_text, str) else col_text
    language_keys = ([col_language] if isinstance(col_language, str) else col_language) if col_language else ["language", "lang", "locale"]
    url_keys      = ([col_url] if isinstance(col_url, str) else col_url) if col_url else ["url", "link", "source_url", "href"]
    extra_cols    = extra_cols or {}

    batch_texts  = []
    batch_meta   = []
    current_size = 0

    try:
        for row in iter_source(dsname, source_type, kaggle_slug=kaggle_slug, col_text=col_text, extract_dir=extract_dir):
            text = extract_field(row, text_keys)
            if not text:
                continue

            language = extract_field(row, language_keys)
            row_url  = extract_field(row, url_keys)
            extras   = {db_col: extract_field(row, col_keys) for db_col, col_keys in extra_cols.items()}

            batch_texts.append(text)
            batch_meta.append((language, row_url, extras))
            current_size += len(text.encode())

            if current_size >= SIZE_LIMIT:
                print(f"  ⚡ Size limit reached at {len(batch_texts)} rows")
                break

        if not batch_texts:
            print("❌ No text data found")
            return

        print(f"🧠 Encoding {len(batch_texts)} texts...")
        embeddings = model.encode(batch_texts, show_progress_bar=True)

        print("💾 Inserting into Supabase...")
        for text, emb, (language, row_url, extras) in zip(batch_texts, embeddings, batch_meta):
            host = urlparse(row_url).netloc.replace("www.", "") if row_url else dsname

            record = {
                "content":     text,
                "embeddings":  emb.tolist(),
                "source":      host or source,
                "source_link": row_url or dsname,
                "author":      source,
                "language":    language,
                **extras,
            }

            supabase.table("Web & Social Intelligence").insert(record).execute()

        print(f"✅ Done — {len(batch_texts)} records inserted")

    except Exception as e:
        print("❌ Error:", e)



# HuggingFace dataset
ingest_dataset(
    "HuggingFaceFW/fineweb",
    source="FineWeb",
    source_type="dataset",
    col_text="text",
    col_language="language",
    col_url="url",
)

# HuggingFace dataset (edu)
ingest_dataset(
    "HuggingFaceFW/fineweb-edu",
    source="FineWeb-Edu",
    source_type="dataset",
    col_text="text",
    col_language="language",
    col_url="url",
)

# Local file (CSV / JSON / JSONL / Parquet / Excel / TSV)
ingest_dataset(
    "test.csv",
    source="Local CSV Test",
    source_type="file",
    col_text="text",
    col_language="language",
    col_url="url",
)

# Kaggle dataset
ingest_dataset(
    "uciml/sms-spam-collection-dataset",
    source="Kaggle SMS Spam",
    source_type="kaggle",
    kaggle_slug="uciml/sms-spam-collection-dataset",
    col_text="v2",
    col_language=None,
    col_url=None,
)