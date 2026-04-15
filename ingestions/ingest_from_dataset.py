"""
ingest_from_dataset.py
────────────────────────────────────────────────────────────────────────────
Generic dataset ingester built on top of ingest_from_kaggle.py.

Supported sources:
  • Kaggle slug or Kaggle dataset URL
  • Hugging Face dataset id or dataset URL
  • Direct dataset file/archive URL

Examples:
  python ingest_from_dataset.py --source kanujulamamatha/indian-general-election-results-2009
  python ingest_from_dataset.py --provider huggingface --source imdb
  python ingest_from_dataset.py --source https://huggingface.co/datasets/imdb
  python ingest_from_dataset.py --provider url --source https://example.com/data.zip --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.parse import urlparse

from google import genai
from supabase import create_client

import ingest_from_kaggle as engine


def infer_provider(source: str, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    s = source.strip()
    if "kaggle.com/datasets/" in s.lower():
        return "kaggle"
    if "huggingface.co/datasets/" in s.lower():
        return "huggingface"
    parsed = urlparse(s)
    if parsed.scheme in {"http", "https"}:
        return "url"
    return "kaggle"


def resolve_source_id(source: str, provider: str) -> tuple[str, str]:
    if provider == "kaggle":
        slug = engine.extract_slug(source) or source.strip()
        return slug, f"https://www.kaggle.com/datasets/{slug}" if "http" not in source else source.strip()
    if provider == "huggingface":
        dataset_id = engine.parse_huggingface_dataset_id(source)
        if not dataset_id:
            raise ValueError(f"Could not parse Hugging Face dataset id from: {source}")
        return dataset_id, f"https://huggingface.co/datasets/{dataset_id}" if "http" not in source else source.strip()
    if provider == "url":
        parsed = urlparse(source.strip())
        dataset_id = parsed.path.rstrip("/").split("/")[-1] or parsed.netloc or "external-dataset"
        return dataset_id, source.strip()
    raise ValueError(f"Unsupported provider: {provider}")


def download_source(source_id: str, source_url: str, provider: str) -> str | None:
    if provider == "kaggle":
        return engine.download_dataset(source_id)
    if provider == "huggingface":
        return engine.download_huggingface_dataset(source_id)
    if provider == "url":
        return engine.download_dataset_from_url(source_url, dataset_id=source_id)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a dataset from Kaggle, Hugging Face, or a direct URL")
    parser.add_argument("--source", required=True, help="Kaggle slug/URL, Hugging Face dataset id/URL, or direct dataset URL")
    parser.add_argument("--provider", default="auto", choices=["auto", "kaggle", "huggingface", "url"],
                        help="Dataset provider. 'auto' infers from the source string.")
    parser.add_argument("--dry-run", action="store_true", help="Preview rows, do not insert into Supabase")
    parser.add_argument("--limit", type=int, default=0, help="Preview/insert only first N built docs (0 = all)")
    parser.add_argument("--preview-rows", type=int, default=10,
                        help="How many built docs to print in terminal during preview")
    parser.add_argument("--preview-jsonl", default="",
                        help="Optional path to write exact would-be-inserted docs as JSONL without uploading")
    parser.add_argument("--category", default="auto", help="Optional category hint")
    parser.add_argument("--topic", default="auto", help="Optional topic hint")
    parser.add_argument("--objective", default="auto", help="Optional objective hint")
    parser.add_argument("--tags", default="auto", help="Optional comma-separated tags hint")
    parser.add_argument("--max-rows-per-dataset", type=int, default=engine.MAX_ROWS_PER_DS,
                        help="Max rows to read per dataset (0 = full dataset)")
    args = parser.parse_args()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not google_api_key:
        sys.exit("✗ GOOGLE_API_KEY missing in .env")
    if not supabase_url:
        sys.exit("✗ SUPABASE_URL missing in .env")
    if not supabase_key:
        sys.exit("✗ SUPABASE_KEY missing in .env")

    provider = infer_provider(args.source, args.provider)
    source_id, source_url = resolve_source_id(args.source, provider)
    source_context = engine.build_source_context(source_id, provider=provider, source_url=source_url)

    hint_topic = args.topic if args.topic != "auto" else ""
    hint_category = args.category if args.category != "auto" else ""
    hint_objective = args.objective if args.objective != "auto" else ""
    hint_tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags != "auto" else []

    engine._runtime_max_rows_per_ds = max(0, args.max_rows_per_dataset)

    gemini_client = genai.Client(api_key=google_api_key)
    supabase = create_client(supabase_url, supabase_key)

    print("=" * 66)
    print("StoryBit Generic Dataset Ingestion")
    print("=" * 66)
    print(f"Provider:          {provider}")
    print(f"Source id:         {source_id}")
    print(f"Source url:        {source_url}")
    print(f"Dry run:           {args.dry_run}")
    print(f"Limit:             {args.limit if args.limit > 0 else 'all rows'}")
    print(f"Dataset rows cap:  {engine._runtime_max_rows_per_ds if engine._runtime_max_rows_per_ds > 0 else 'no cap (full dataset)'}")
    print("=" * 66)

    dataset_path = download_source(source_id, source_url, provider)
    if not dataset_path:
        sys.exit("✗ Download failed")

    docs = engine.process_dataset(
        dataset_path,
        source_id,
        hint_category=hint_category,
        hint_topic=hint_topic,
        hint_objective=hint_objective,
        hint_tags=hint_tags,
        source_context=source_context,
    )
    if not docs:
        sys.exit("✗ No rows extracted")

    if args.limit > 0:
        docs = docs[:args.limit]
        print(f"\n(limited to {args.limit} rows)")

    print(f"\nBuilt docs: {len(docs)}")
    for i, doc in enumerate(docs[:max(0, args.preview_rows)], start=1):
        print(f"\nRow {i}")
        print(f"  source_title: {doc['source_title']}")
        print(f"  source_type:  {doc['source_type']}")
        print(f"  source_url:   {doc['source_url']}")
        print(f"  content:      {doc['content'][:160]}")
        print(f"  metadata:     {str(doc['metadata'])[:240]}")

    if args.preview_jsonl:
        with open(args.preview_jsonl, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"\nPreview JSONL written to: {args.preview_jsonl}")

    if args.dry_run:
        return 0

    ok = engine.embed_and_insert(docs, supabase, gemini_client)
    if not ok:
        sys.exit("✗ Insert failed")

    print(f"\nDone. Inserted {len(docs)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
