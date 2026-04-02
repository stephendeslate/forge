"""Tests for the schema migration system."""

import pytest
from pathlib import Path

from forge.storage.migrations import (
    list_available_migrations,
    parse_migration_sql,
    _VERSION_RE,
)


class TestMigrationDiscovery:
    def test_finds_migrations(self):
        migrations = list_available_migrations()
        assert len(migrations) >= 1
        # First migration should be 001_initial
        version, name, path = migrations[0]
        assert version == 1
        assert name == "initial"
        assert path.exists()

    def test_sorted_by_version(self):
        migrations = list_available_migrations()
        versions = [v for v, _, _ in migrations]
        assert versions == sorted(versions)

    def test_version_regex(self):
        assert _VERSION_RE.match("001_initial.sql")
        assert _VERSION_RE.match("042_add_users.sql")
        assert not _VERSION_RE.match("initial.sql")
        assert not _VERSION_RE.match("1_short.sql")
        assert not _VERSION_RE.match("001_initial.txt")


class TestParseMigrationSQL:
    def test_parse_with_rollback(self):
        migrations = list_available_migrations()
        assert len(migrations) >= 1
        _, _, path = migrations[0]
        forward, rollback = parse_migration_sql(path)
        assert "CREATE TABLE" in forward
        assert "DROP TABLE" in rollback

    def test_parse_forward_only(self, tmp_path):
        f = tmp_path / "002_test.sql"
        f.write_text("CREATE TABLE test (id INT);")
        forward, rollback = parse_migration_sql(f)
        assert "CREATE TABLE" in forward
        assert rollback == ""

    def test_parse_preserves_content(self, tmp_path):
        content = "-- forward\nCREATE TABLE foo (id INT);\n\n-- rollback\nDROP TABLE foo;"
        f = tmp_path / "003_test.sql"
        f.write_text(content)
        forward, rollback = parse_migration_sql(f)
        assert "CREATE TABLE foo" in forward
        assert "DROP TABLE foo" in rollback


class TestInitialMigration:
    """Verify the 001_initial migration contains all required tables."""

    def test_has_all_tables(self):
        migrations = list_available_migrations()
        _, _, path = migrations[0]
        forward, _ = parse_migration_sql(path)
        for table in ("chunks", "conversations", "memories", "sessions", "checkpoints"):
            assert table in forward, f"Missing table: {table}"

    def test_has_vector_extension(self):
        migrations = list_available_migrations()
        _, _, path = migrations[0]
        forward, _ = parse_migration_sql(path)
        assert "CREATE EXTENSION IF NOT EXISTS vector" in forward

    def test_rollback_drops_all_tables(self):
        migrations = list_available_migrations()
        _, _, path = migrations[0]
        _, rollback = parse_migration_sql(path)
        for table in ("chunks", "conversations", "memories", "sessions", "checkpoints"):
            assert table in rollback, f"Rollback missing table: {table}"
