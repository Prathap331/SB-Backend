"""
upload_to_supabase.py
Migrated to google-genai (new unified SDK).
pip install google-genai  (replaces google-generativeai)

UPDATED (plan section 6.2):
  - process_batch() now tags every row with source_type='news' and metadata.
"""

import os
import json
import time

# ── NEW unified Google GenAI SDK ─────────────────────────────
from google import genai
from google.genai import types as genai_types
# ─────────────────────────────────────────────────────────────

from dotenv import load_dotenv
from supabase import create_client, Client
import nltk
from nltk.tokenize import sent_tokenize

# --- CONFIGURATION ---
INPUT_FILE    = 'news_history_data.json'
PROGRESS_FILE = 'progress.log'
EMBEDDING_MODEL = 'gemini-embedding-001'  # text-embedding-004 deprecated Jan 14 2026
CHUNK_SIZE    = 250
CHUNK_OVERLAP = 50
BATCH_SIZE    = 100
MAX_ARTICLES  = 5000
# ---------------------


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        words = sentence.split()
        if len(current_chunk.split()) + len(words) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            overlap_words = current_chunk.split()[-chunk_overlap:]
            current_chunk = " ".join(overlap_words) + " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def main():
    print("--- Starting Supabase Ingestion Script ---")

    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY not found.")
        return

    # NEW SDK: create a single client object
    gemini_client = genai.Client(api_key=google_api_key)
    print("Google GenAI client (new SDK) initialized.")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        print("ERROR: Supabase credentials not found.")
        return
    supabase: Client = create_client(supabase_url, supabase_key)
    print("Supabase client initialized.")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        articles_to_process = articles[:MAX_ARTICLES]
    except FileNotFoundError:
        print(f"ERROR: {INPUT_FILE} not found.")
        return

    # Resume feature
    start_index = 0
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            start_index = int(f.read())
        print(f"--- Resuming from article #{start_index + 1} ---")

    batch_to_upload = []
    total_chunks = 0

    for i, article in enumerate(articles_to_process[start_index:], start=start_index):
        print(f"Processing article {i + 1}/{len(articles_to_process)}: {article['title'][:50]}...")

        article_chunks = chunk_text(article['text'], CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_content in article_chunks:
            # CNN/DailyMail news dataset — no URL or byline available.
            # source_from = "CNN / Daily Mail" as dataset origin.
            batch_to_upload.append({
                "content":      chunk_content,
                "source_title": article['title'],
                "source_url":   None,
                "source_type":  "news",
                "metadata": {
                    "source_from": "CNN / Daily Mail",
                    "category":    "Politics & Current Affairs",
                    "topic":       article['title'],
                    "tags":        [],
                    "author": {
                        "has_credentials": True,
                        "name":            "CNN / Daily Mail",
                        "description":     "News dataset — CNN and Daily Mail articles",
                    },
                },
            })

            if len(batch_to_upload) >= BATCH_SIZE:
                success = process_batch(batch_to_upload, supabase, gemini_client)
                if not success:
                    print("!!! Critical error in batch processing. Aborting. !!!")
                    return

                total_chunks += len(batch_to_upload)
                batch_to_upload = []

                with open(PROGRESS_FILE, 'w') as f:
                    f.write(str(i + 1))

                print("    Waiting 62s between batches (free tier rate limit)...")
                time.sleep(62)

    if batch_to_upload:
        process_batch(batch_to_upload, supabase, gemini_client)
        total_chunks += len(batch_to_upload)

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    print(f"\n--- Ingestion complete! Total chunks uploaded: {total_chunks} ---")


def process_batch(batch: list[dict], supabase: Client, gemini_client) -> bool:
    """Embeds and uploads a single batch with retry logic."""
    print(f"  Processing batch of {len(batch)} chunks...")

    contents_to_embed = [item['content'] for item in batch]

    embeddings = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"    Generating embeddings (Attempt {attempt + 1}/{max_retries})...")

            # NEW SDK: embed_content via client, returns EmbedContentResponse
            embed_response = gemini_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents_to_embed,
                config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768),
            )
            # NEW SDK: response.embeddings is a list of ContentEmbedding objects
            embeddings = [e.values for e in embed_response.embeddings]
            print("    Embeddings generated successfully.")
            break

        except Exception as e:
            print(f"    ERROR generating embeddings: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"    Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("    Max retries reached. Skipping batch.")
                return False

    if not embeddings:
        return False

    for i, item in enumerate(batch):
        item['embedding'] = embeddings[i]

    try:
        print(f"    Uploading {len(batch)} records to Supabase...")
        supabase.table('documents').insert(batch).execute()
        print("    Batch uploaded successfully.")
        return True
    except Exception as e:
        print(f"    CRITICAL ERROR uploading to Supabase: {e}")
        return False


if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    main()