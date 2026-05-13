# import os
# from datasets import load_dataset, get_dataset_config_names
# from sentence_transformers import SentenceTransformer
# from supabase import create_client
# from urllib.parse import urlparse
# import csv
# import json
# import os

# url = os.getenv("SUPABASE_URL")
# key = os.getenv("SUPABASE_KEY")

# supabase = create_client(url, key)
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# def safe_load_dataset(dsname):
#     try:
#         return load_dataset(dsname, split="train", streaming=True)
#     except:
#         try:
#             configs = get_dataset_config_names(dsname)
#             return load_dataset(dsname, configs[0], split="train", streaming=True)
#         except:
#             return None



# def extract_field(row, keys):
#     """Extract a field from a row by trying multiple possible key names."""
#     if isinstance(keys, str):
#         keys = [keys]
#     for key in keys:
#         if key in row and row[key]:
#             return row[key]
#     return None


# def iter_file(filepath):
#     """Yield rows from a CSV or JSON file."""
#     ext = os.path.splitext(filepath)[-1].lower()

#     if ext == ".csv":
#         with open(filepath, "r", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 yield row

#     elif ext == ".json":
#         with open(filepath, "r", encoding="utf-8") as f:
#             data = json.load(f)
#             if isinstance(data, list):
#                 for row in data:
#                     yield row
#             elif isinstance(data, dict):
#                 yield data

#     else:
#         raise ValueError(f"Unsupported file type: {ext}")


# def iter_source(dsname_or_path, source_type="dataset"):
#     """Unified row iterator for HuggingFace datasets, CSV, and JSON files."""
#     if source_type == "file":
#         yield from iter_file(dsname_or_path)
#     else:
#         ds = safe_load_dataset(dsname_or_path)
#         if ds is None:
#             return
#         yield from ds


# def ingest_dataset(
#     dsname,
#     source,
#     source_type="dataset",  
#     col_text="text",         
#     col_language=None,        
#     col_url=None,             
#     extra_cols=None,          
# ):
#     print(f"\n🚀 Processing: {dsname}")

#     text_keys     = [col_text] if isinstance(col_text, str) else col_text
#     language_keys = ([col_language] if isinstance(col_language, str) else col_language) if col_language else ["language", "lang", "locale"]
#     url_keys      = ([col_url] if isinstance(col_url, str) else col_url) if col_url else ["url", "link", "source_url", "href"]
#     extra_cols    = extra_cols or {}

#     batch_texts = []
#     batch_meta  = []
#     size_limit  = 50 * 250 * 250
#     current_size = 0

#     try:
#         for row in iter_source(dsname, source_type):
#             text = extract_field(row, text_keys)
#             if not text:
#                 continue

#             language = extract_field(row, language_keys)
#             row_url  = extract_field(row, url_keys)
#             extras   = {db_col: extract_field(row, col_keys) for db_col, col_keys in extra_cols.items()}

#             batch_texts.append(text)
#             batch_meta.append((language, row_url, extras))
#             current_size += len(text.encode())

#             if current_size >= size_limit:
#                 break

#         if not batch_texts:
#             print("❌ No text data found")
#             return

#         print(f"🧠 Encoding {len(batch_texts)} texts...")
#         embeddings = model.encode(batch_texts, show_progress_bar=True)

#         print("💾 Inserting into Supabase...")
#         for text, emb, (language, row_url, extras) in zip(batch_texts, embeddings, batch_meta):
#             host = urlparse(row_url).netloc.replace("www.", "") if row_url else dsname

#             record = {
#                 "content":     text,
#                 "embeddings":  emb.tolist(),
#                 "source":      host or source,
#                 "source_link": row_url or dsname,
#                 "author":      source,
#                 "language":    language,
#             }

#             supabase.table("Web & Social Intelligence").insert(record).execute()

#         print(f"✅ Done — {len(batch_texts)} records inserted")

#     except Exception as e:
#         print("❌ Error:", e)





# HuggingFace dataset with custom column names
# ingest_dataset(
#     "HuggingFaceFW/fineweb",
#     source="FineWeb",
#     col_text="text",
#     col_language="language",
#     col_url="url",
# )

# ingest_dataset(
#     "HuggingFaceFW/fineweb-edu",
#     source="FineWeb-Edu",
#     col_text="text",
#     col_language="language",
#     col_url="url",
# )


# ingest_dataset(
#     "test.csv",
#     source="Local CSV Test",
#     source_type="file",
#     col_text="text",
#     col_language="language",
#     col_url="url",
# )

