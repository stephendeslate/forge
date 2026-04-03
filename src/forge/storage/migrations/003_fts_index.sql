-- Add tsvector column and GIN index for full-text search on chunks
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector;

-- Populate from existing content
UPDATE chunks SET tsv = to_tsvector('english', coalesce(name, '') || ' ' || content)
WHERE tsv IS NULL;

-- Auto-update on insert/update
CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', coalesce(NEW.name, '') || ' ' || NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chunks_tsv_update ON chunks;
CREATE TRIGGER chunks_tsv_update BEFORE INSERT OR UPDATE OF content, name
ON chunks FOR EACH ROW EXECUTE FUNCTION chunks_tsv_trigger();

CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin(tsv);

-- rollback
DROP TRIGGER IF EXISTS chunks_tsv_update ON chunks;
DROP FUNCTION IF EXISTS chunks_tsv_trigger();
DROP INDEX IF EXISTS idx_chunks_tsv;
ALTER TABLE chunks DROP COLUMN IF EXISTS tsv;
