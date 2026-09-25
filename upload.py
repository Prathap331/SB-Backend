# import os
# import json
# from typing import List, Dict, Any

# from supabase import create_client, Client
# from sentence_transformers import SentenceTransformer


# # ============================================================
# # CONFIG
# # ============================================================

# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# JSON_FILE = "script_structures.json"

# TABLE_NAME = "script_structures"

# # BAAI/bge-m3 -> 1024 dimensions
# MODEL_NAME = "BAAI/bge-m3"

# # Number of rows sent to Supabase at once
# BATCH_SIZE = 10


# # ============================================================
# # VALIDATE ENV
# # ============================================================

# if not SUPABASE_URL:
#     raise RuntimeError("SUPABASE_URL environment variable is missing")

# if not SUPABASE_KEY:
#     raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY environment variable is missing")


# # ============================================================
# # INITIALIZE SUPABASE
# # ============================================================

# supabase: Client = create_client(
#     SUPABASE_URL,
#     SUPABASE_KEY
# )


# # ============================================================
# # LOAD EMBEDDING MODEL
# # ============================================================

# print(f"Loading embedding model: {MODEL_NAME}")

# model = SentenceTransformer(MODEL_NAME)

# print("Embedding model loaded.")


# # ============================================================
# # LOAD JSON
# # ============================================================

# print(f"Loading JSON file: {JSON_FILE}")

# with open(JSON_FILE, "r", encoding="utf-8") as f:
#     script_structures: List[Dict[str, Any]] = json.load(f)

# if not isinstance(script_structures, list):
#     raise ValueError("JSON root must be a list of script structures")

# print(f"Found {len(script_structures)} script structures.")


# # ============================================================
# # HELPER
# # ============================================================

# def build_embedding_text(item: Dict[str, Any]) -> str:
#     """
#     Create the text that will be embedded.

#     Embedding text:
#         title + about
#     """

#     title = item.get("title", "").strip()
#     about = item.get("about", "").strip()

#     return f"{title}\n\n{about}"


# def validate_item(item: Dict[str, Any], index: int):
#     """
#     Validate required fields before uploading.
#     """

#     required_fields = [
#         "key",
#         "title",
#         "cluster",
#         "about",
#         "segments",
#     ]

#     for field in required_fields:
#         if field not in item:
#             raise ValueError(
#                 f"Item {index} is missing required field: {field}"
#             )

#     if not isinstance(item["segments"], list):
#         raise ValueError(
#             f"Item {index} ({item.get('key')}) "
#             f"'segments' must be a list"
#         )


# # ============================================================
# # PREPARE ALL EMBEDDING TEXTS
# # ============================================================

# embedding_texts = []

# for index, item in enumerate(script_structures):

#     validate_item(item, index)

#     text = build_embedding_text(item)

#     embedding_texts.append(text)


# # ============================================================
# # GENERATE EMBEDDINGS
# # ============================================================

# print("Generating embeddings...")

# embeddings = model.encode(
#     embedding_texts,
#     batch_size=8,
#     show_progress_bar=True,
#     normalize_embeddings=True,
# )

# print(f"Generated {len(embeddings)} embeddings.")


# # ============================================================
# # PREPARE SUPABASE ROWS
# # ============================================================

# rows = []

# for index, item in enumerate(script_structures):

#     embedding = embeddings[index].tolist()

#     row = {
#         "key": item["key"],
#         "title": item["title"],
#         "cluster": item["cluster"],
#         "about": item["about"],

#         # Preserve the array exactly as it exists in JSON
#         "best_fit_categories": item.get(
#             "best_fit_categories",
#             []
#         ),

#         "human_texture_tier": item.get(
#             "human_texture_tier"
#         ),

#         # Preserve segments as JSON
#         "segments": item["segments"],

#         # Combined title + about
#         "template_text": embedding_texts[index],

#         # 1024-dimensional BGE-M3 embedding
#         "embedding": embedding,

#         # Don't insert id.
#         # PostgreSQL bigserial will generate it.

#         # Don't insert created_at / updated_at.
#         # PostgreSQL defaults will handle them.
#     }

#     rows.append(row)


# # ============================================================
# # UPLOAD TO SUPABASE
# # ============================================================

# print("Uploading to Supabase...")

# total = len(rows)

# for start in range(0, total, BATCH_SIZE):

#     end = min(start + BATCH_SIZE, total)

#     batch = rows[start:end]

#     print(
#         f"Uploading rows {start + 1}-{end} "
#         f"of {total}..."
#     )

#     try:

#         response = (
#             supabase
#             .table(TABLE_NAME)
#             .upsert(
#                 batch,
#                 on_conflict="key"
#             )
#             .execute()
#         )

#         print(
#             f"Successfully uploaded rows "
#             f"{start + 1}-{end}"
#         )

#     except Exception as e:

#         print(
#             f"ERROR uploading rows "
#             f"{start + 1}-{end}"
#         )

#         print(e)

#         raise


# # ============================================================
# # DONE
# # ============================================================

# print()
# print("=" * 60)
# print("UPLOAD COMPLETE")
# print("=" * 60)
# print(f"Total structures processed: {total}")
# print(f"Table: {TABLE_NAME}")
# print(f"Embedding model: {MODEL_NAME}")
# print("Embedding dimensions: 1024")
# print("=" * 60)


import os
import json

from supabase import create_client, Client
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

JSON_FILE = "templates.json"
TABLE_NAME = "animations_template"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 10


# ============================================================
# VALIDATE ENV
# ============================================================

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL environment variable is missing")

if not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_SERVICE_ROLE_KEY environment variable is missing"
    )


# ============================================================
# SUPABASE
# ============================================================

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# ============================================================
# LOAD JSON
# ============================================================

print(f"Loading JSON file: {JSON_FILE}")

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, dict):
    raise ValueError(
        "metadata.json must contain a JSON object at the top level."
    )

print(f"Found {len(data)} templates.")


# ============================================================
# PREPARE TEMPLATES
# ============================================================

templates = []

for name, template_data in data.items():

    name = str(name).strip()

    if not name:
        print("Skipping template with empty name.")
        continue

    if not isinstance(template_data, dict):
        print(f"Skipping invalid template: {name}")
        continue

    # Get props_schema from JSON
    props = template_data.get("props_schema", {})

    # Get pickWhen from JSON
    pick_when = template_data.get("pickWhen", "")

    templates.append({
        "name": name,
        "props": props,
        "pick_when": pick_when
    })


print(f"Prepared {len(templates)} templates.")


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================
# IMPORTANT:
# Embedding is generated ONLY from the template name.
#
# Example:
#
# "Title Card"
#
# NOT:
# "Title Card + pickWhen + props_schema"
# ============================================================

names = [
    template["name"]
    for template in templates
]

print("Generating embeddings for template names...")

embeddings = model.encode(
    names,
    batch_size=BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Embeddings generated.")


# ============================================================
# PREPARE SUPABASE ROWS
# ============================================================

rows = []

for template, embedding in zip(templates, embeddings):

    rows.append({
        "name": template["name"],
        "props": template["props"],
        "pick_when": template["pick_when"],
        "embedding": embedding.tolist()
    })


# ============================================================
# INSERT INTO SUPABASE
# ============================================================

print(
    f"Inserting {len(rows)} templates "
    f"into {TABLE_NAME}..."
)

for i in range(0, len(rows), BATCH_SIZE):

    batch = rows[i:i + BATCH_SIZE]

    response = (
        supabase
        .table(TABLE_NAME)
        .insert(batch)
        .execute()
    )

    print(
        f"Inserted {len(batch)} templates "
        f"({i + len(batch)}/{len(rows)})"
    )


# ============================================================
# DONE
# ============================================================

print()
print("============================================")
print("Upload completed successfully.")
print("============================================")