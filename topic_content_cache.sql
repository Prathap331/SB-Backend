-- Shared semantic cache for idea-generation topics.
-- Apply this in Supabase/Postgres if you want cache sharing across instances.

create extension if not exists vector;

create table if not exists topic_content_cache (
    id bigserial primary key,
    topic_key text not null unique,
    topic_canonical text not null,
    topic_vec vector(768),
    payload jsonb not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    expires_at timestamptz not null
);

create index if not exists idx_topic_content_cache_expires_at
    on topic_content_cache (expires_at);

create index if not exists idx_topic_content_cache_topic_key
    on topic_content_cache (topic_key);

-- Optional vector index for nearest-neighbour lookup.
-- Tune the index type/ops based on your pgvector version and data volume.
create index if not exists idx_topic_content_cache_topic_vec
    on topic_content_cache using ivfflat (topic_vec vector_cosine_ops);

create or replace function match_topic_content_cache(
    query_embedding vector(768),
    match_threshold double precision default 0.92,
    match_count integer default 1
)
returns table (
    topic_key text,
    topic_canonical text,
    topic_vec vector(768),
    payload jsonb,
    created_at timestamptz,
    updated_at timestamptz,
    expires_at timestamptz,
    similarity double precision
)
language sql
stable
as $$
    select
        t.topic_key,
        t.topic_canonical,
        t.topic_vec,
        t.payload,
        t.created_at,
        t.updated_at,
        t.expires_at,
        1 - (t.topic_vec <=> query_embedding) as similarity
    from topic_content_cache t
    where t.expires_at > now()
      and t.topic_vec is not null
      and 1 - (t.topic_vec <=> query_embedding) >= match_threshold
    order by t.topic_vec <=> query_embedding
    limit match_count;
$$;
