-- Exemplar learning: store cloud model successes for local model few-shot injection
CREATE TABLE IF NOT EXISTS exemplars (
    id                BIGSERIAL PRIMARY KEY,
    project           TEXT NOT NULL,
    task_type         TEXT NOT NULL CHECK (task_type IN ('recovery','planning','critique')),
    task_description  TEXT NOT NULL,
    solution_approach TEXT NOT NULL,
    outcome_score     FLOAT NOT NULL DEFAULT 0.5,
    model_source      TEXT NOT NULL,
    embedding         vector(768) NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    used_count        INTEGER NOT NULL DEFAULT 0,
    last_used_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_exemplars_project ON exemplars (project);
CREATE INDEX IF NOT EXISTS idx_exemplars_embedding_hnsw
    ON exemplars USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
