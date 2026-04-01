-- StoryBit data health checks for documents table
-- Run in Supabase SQL editor after ingestion.

-- 1) Row counts
SELECT COUNT(*) AS total_rows FROM documents;

-- 2) Critical null/empty checks
SELECT
  COUNT(*) FILTER (WHERE content IS NULL OR btrim(content) = '') AS empty_content,
  COUNT(*) FILTER (WHERE metadata IS NULL) AS null_metadata,
  COUNT(*) FILTER (WHERE source_title IS NULL OR btrim(source_title) = '') AS empty_source_title,
  COUNT(*) FILTER (WHERE source_type IS NULL OR btrim(source_type) = '') AS empty_source_type,
  COUNT(*) FILTER (WHERE embedding IS NULL) AS null_embedding
FROM documents;

-- 3) Invalid source_type against target schema
SELECT source_type, COUNT(*) AS rows
FROM documents
GROUP BY source_type
HAVING source_type NOT IN ('web_scrape', 'news', 'wikipedia', 'book', 'youtube');

-- 4) Embedding dimension sanity (pgvector)
SELECT
  COUNT(*) FILTER (WHERE embedding IS NOT NULL AND vector_dims(embedding) <> 768) AS wrong_dim_rows
FROM documents;

-- 5) Metadata key coverage
SELECT
  COUNT(*) FILTER (WHERE metadata->>'kaggle_slug' IS NULL) AS missing_kaggle_slug,
  COUNT(*) FILTER (WHERE metadata->>'file' IS NULL) AS missing_file,
  COUNT(*) FILTER (WHERE metadata->>'category' IS NULL) AS missing_category,
  COUNT(*) FILTER (WHERE metadata->>'topic' IS NULL) AS missing_topic,
  COUNT(*) FILTER (WHERE metadata->>'objective' IS NULL) AS missing_objective
FROM documents;

-- 6) Suspicious content length outliers
SELECT
  COUNT(*) FILTER (WHERE length(content) < 30) AS very_short_content,
  COUNT(*) FILTER (WHERE length(content) > 8000) AS very_long_content
FROM documents;

-- 7) Top duplicates by content hash
SELECT md5(content) AS content_hash, COUNT(*) AS dup_count
FROM documents
GROUP BY md5(content)
HAVING COUNT(*) > 1
ORDER BY dup_count DESC
LIMIT 50;

-- 8) Per-dataset quality summary
SELECT
  metadata->>'kaggle_slug' AS dataset,
  COUNT(*) AS rows,
  COUNT(*) FILTER (WHERE content IS NULL OR btrim(content) = '') AS empty_content,
  COUNT(*) FILTER (WHERE embedding IS NULL) AS null_embedding,
  COUNT(*) FILTER (WHERE metadata->>'published_at' IS NOT NULL) AS rows_with_published_at
FROM documents
GROUP BY metadata->>'kaggle_slug'
ORDER BY rows DESC;
