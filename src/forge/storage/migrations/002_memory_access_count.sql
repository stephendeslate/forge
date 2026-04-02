-- Add access_count to memories for smart pruning frequency tracking
ALTER TABLE memories ADD COLUMN IF NOT EXISTS access_count INTEGER NOT NULL DEFAULT 0;

-- Rollback
ALTER TABLE memories DROP COLUMN IF EXISTS access_count;
