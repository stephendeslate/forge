-- 001_initial.sql — Base schema for Forge
-- Safe for existing databases: all CREATE statements use IF NOT EXISTS.

CREATE EXTENSION IF NOT EXISTS vector;

-- Chunks table: stores code chunks with their embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    project         TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    chunk_type      TEXT NOT NULL,
    name            TEXT,
    content         TEXT NOT NULL,
    start_line      INTEGER NOT NULL,
    end_line        INTEGER NOT NULL,
    token_count     INTEGER NOT NULL,
    embedding       vector(768),
    file_hash       TEXT NOT NULL,
    indexed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_project_file
    ON chunks (project, file_path);

CREATE INDEX IF NOT EXISTS idx_chunks_file_hash
    ON chunks (project, file_path, file_hash);

-- Conversation history
CREATE TABLE IF NOT EXISTS conversations (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    model           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_session
    ON conversations (session_id, created_at);

-- Memories table: cross-session persistent context
CREATE TABLE IF NOT EXISTS memories (
    id          BIGSERIAL PRIMARY KEY,
    project     TEXT NOT NULL,
    category    TEXT NOT NULL CHECK (category IN ('feedback','project','user','reference')),
    subject     TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(768),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memories_project ON memories (project);
CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
    ON memories USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Session metadata
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    project     TEXT,
    mode        TEXT NOT NULL DEFAULT 'chat',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Conversation checkpoints
CREATE TABLE IF NOT EXISTS checkpoints (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    name            TEXT NOT NULL,
    agent_history   TEXT NOT NULL,
    task_store      TEXT,
    message_count   INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(session_id, name)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_session
    ON checkpoints (session_id, created_at);

-- rollback
DROP TABLE IF EXISTS checkpoints CASCADE;
DROP TABLE IF EXISTS memories CASCADE;
DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;
DROP EXTENSION IF EXISTS vector;
