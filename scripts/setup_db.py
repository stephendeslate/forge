#!/usr/bin/env python3
"""Set up the forge PostgreSQL database with pgvector extension.

Usage:
    python scripts/setup_db.py              # Uses default connection
    python scripts/setup_db.py --drop       # Drop and recreate tables

Requires: PostgreSQL 16+ with pgvector extension installed.
Pre-requisite system commands:
    sudo apt install -y postgresql-16 postgresql-16-pgvector
    sudo -u postgres psql -c "CREATE USER stephen WITH SUPERUSER;"
    sudo -u postgres psql -c "CREATE DATABASE forge OWNER stephen;"
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import asyncpg


DATABASE_URL = "postgresql://stephen@/forge?host=/var/run/postgresql&port=5433"

SETUP_SQL = """
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Chunks table: stores code chunks with their embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    project         TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    chunk_type      TEXT NOT NULL,          -- 'function', 'class', 'block', etc.
    name            TEXT,                   -- function/class name if applicable
    content         TEXT NOT NULL,
    start_line      INTEGER NOT NULL,
    end_line        INTEGER NOT NULL,
    token_count     INTEGER NOT NULL,
    embedding       vector(768),           -- nomic-embed-text-v2-moe dimensions
    file_hash       TEXT NOT NULL,          -- SHA256 of source file for incremental updates
    indexed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Index for incremental updates (find chunks by file)
CREATE INDEX IF NOT EXISTS idx_chunks_project_file
    ON chunks (project, file_path);

-- Index for file hash lookups (skip unchanged files)
CREATE INDEX IF NOT EXISTS idx_chunks_file_hash
    ON chunks (project, file_path, file_hash);

-- Conversation history (Phase 4, created now for schema stability)
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

-- Session metadata (Phase 4)
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    project     TEXT,
    mode        TEXT NOT NULL DEFAULT 'chat',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Conversation checkpoints (Phase 8)
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
"""

DROP_SQL = """
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;
"""


async def main(drop: bool = False) -> None:
    try:
        conn = await asyncpg.connect(DATABASE_URL)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        print("\nMake sure PostgreSQL is running and the database exists:")
        print("  sudo -u postgres psql -c \"CREATE USER stephen WITH SUPERUSER;\"")
        print("  sudo -u postgres psql -c \"CREATE DATABASE forge OWNER stephen;\"")
        sys.exit(1)

    try:
        if drop:
            print("Dropping existing tables...")
            await conn.execute(DROP_SQL)

        print("Setting up database schema...")
        await conn.execute(SETUP_SQL)
        print("Database setup complete.")

        # Verify
        row = await conn.fetchrow("SELECT count(*) FROM chunks")
        print(f"  chunks table: {row['count']} rows")
        row = await conn.fetchrow("SELECT count(*) FROM conversations")
        print(f"  conversations table: {row['count']} rows")
        row = await conn.fetchrow("SELECT count(*) FROM sessions")
        print(f"  sessions table: {row['count']} rows")

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up forge database")
    parser.add_argument("--drop", action="store_true", help="Drop and recreate tables")
    args = parser.parse_args()
    asyncio.run(main(drop=args.drop))
