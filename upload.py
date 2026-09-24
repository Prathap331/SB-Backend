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



# import os
# import json

# from supabase import create_client, Client
# from sentence_transformers import SentenceTransformer


# # ============================================================
# # CONFIG
# # ============================================================

# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# JSON_FILE = "metadata.json"
# TABLE_NAME = "animations_template"

# MODEL_NAME = "BAAI/bge-m3"
# BATCH_SIZE = 10


# # ============================================================
# # VALIDATE ENV
# # ============================================================

# if not SUPABASE_URL:
#     raise RuntimeError("SUPABASE_URL environment variable is missing")

# if not SUPABASE_KEY:
#     raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY environment variable is missing")


# # ============================================================
# # SUPABASE
# # ============================================================

# supabase: Client = create_client(
#     SUPABASE_URL,
#     SUPABASE_KEY
# )


# # ============================================================
# # EMBEDDING MODEL
# # ============================================================

# print(f"Loading embedding model: {MODEL_NAME}")

# model = SentenceTransformer(MODEL_NAME)

# print("Embedding model loaded.")


# # ============================================================
# # LOAD JSON
# # ============================================================

# print(f"Loading JSON file: {JSON_FILE}")

# with open(JSON_FILE, "r", encoding="utf-8") as f:
#     metadata = json.load(f)

# templates = metadata.get("templates", [])

# if not templates:
#     raise RuntimeError("No templates found in JSON")


# print(f"Found {len(templates)} templates")


# # ============================================================
# # MATCH OBJECT -> TEXT
# # ============================================================

# def match_to_text(match: dict) -> str:
#     """
#     Convert ONLY the match object into text.

#     Nothing outside match is included.
#     """

#     parts = []

#     if match.get("name"):
#         parts.append(f"Name: {match['name']}")

#     if match.get("category"):
#         parts.append(f"Category: {match['category']}")

#     if match.get("motion"):
#         parts.append(f"Motion: {match['motion']}")

#     if match.get("description"):
#         parts.append(f"Description: {match['description']}")

#     if match.get("use_when"):
#         parts.append(f"Use when: {match['use_when']}")

#     if match.get("avoid_when"):
#         parts.append(f"Avoid when: {match['avoid_when']}")

#     tags = match.get("tags", [])

#     if tags:
#         parts.append(
#             "Tags: " + ", ".join(tags)
#         )

#     return "\n".join(parts)


# # ============================================================
# # PREPARE TEMPLATES
# # ============================================================

# records = []
# embedding_texts = []

# for template in templates:

#     # --------------------------------------------------------
#     # ONLY MATCH IS USED FOR EMBEDDING
#     # --------------------------------------------------------

#     match = template.get("match", {})

#     embedding_text = match_to_text(match)

#     embedding_texts.append(embedding_text)

#     # --------------------------------------------------------
#     # EVERYTHING ELSE IS STORED AS-IS
#     # --------------------------------------------------------

#     props_schema = template.get("props_schema", {})

#     record = {
#         "category": match.get("category"),
#         "motion": match.get("motion"),

#         "use_when": match.get("use_when"),
#         "avoid_when": match.get("avoid_when"),

#         "tags": match.get("tags", []),

#         "editing": template.get("editing"),
#         "timing": template.get("timing"),
#         "filters": template.get("filters"),
#         "render": template.get("render"),
#         "props_schema": props_schema,

#         "subtitle": props_schema.get("subtitle"),
#         "data": props_schema.get("data"),
#         "color": props_schema.get("color"),

#         "sample_props": template.get("sample_props"),
#     }

#     records.append(record)


# # ============================================================
# # GENERATE EMBEDDINGS
# # ============================================================

# print("Generating embeddings from MATCH objects only...")

# embeddings = model.encode(
#     embedding_texts,
#     batch_size=BATCH_SIZE,
#     normalize_embeddings=True,
#     show_progress_bar=True
# )


# # ============================================================
# # ADD EMBEDDINGS
# # ============================================================

# for record, embedding in zip(records, embeddings):

#     record["embedding"] = embedding.tolist()


# # ============================================================
# # INSERT INTO SUPABASE
# # ============================================================

# print("Inserting into Supabase...")

# for start in range(0, len(records), BATCH_SIZE):

#     batch = records[start:start + BATCH_SIZE]

#     print(
#         f"Inserting "
#         f"{start + 1}-{start + len(batch)} "
#         f"of {len(records)}"
#     )

#     supabase \
#         .table(TABLE_NAME) \
#         .insert(batch) \
#         .execute()

# print("==========================================")
# print("DONE")
# print("==========================================")
# print(f"Inserted {len(records)} templates.")


