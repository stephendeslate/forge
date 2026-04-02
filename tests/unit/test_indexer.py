"""Tests for the walk + chunk + embed + store indexing pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.rag.indexer import (
    _file_hash,
    _should_index,
    _walk_files,
    index_directory,
    _flush_chunks,
    _SKIP_DIRS,
    _SKIP_FILES,
    _MAX_FILE_SIZE,
)
from forge.rag.chunker import Chunk


# ---------------------------------------------------------------------------
# _file_hash
# ---------------------------------------------------------------------------


class TestFileHash:
    def test_deterministic(self):
        assert _file_hash("hello") == _file_hash("hello")

    def test_different_content_different_hash(self):
        assert _file_hash("hello") != _file_hash("world")

    def test_returns_hex_string(self):
        h = _file_hash("test")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest
        int(h, 16)  # should be valid hex

    def test_empty_string(self):
        h = _file_hash("")
        assert len(h) == 64


# ---------------------------------------------------------------------------
# _should_index
# ---------------------------------------------------------------------------


class TestShouldIndex:
    def test_python_file(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("print('hello')")
        assert _should_index(f) is True

    def test_javascript_file(self, tmp_path):
        f = tmp_path / "app.js"
        f.write_text("console.log('hi')")
        assert _should_index(f) is True

    def test_skip_lock_files(self, tmp_path):
        for name in ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock", "poetry.lock", "uv.lock"):
            f = tmp_path / name
            f.write_text("{}")
            assert _should_index(f) is False, f"Should skip {name}"

    def test_skip_ds_store(self, tmp_path):
        f = tmp_path / ".DS_Store"
        f.write_text("")
        assert _should_index(f) is False

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "photo.jpg"
        f.write_text("binary data")
        assert _should_index(f) is False

    def test_large_file_skipped(self, tmp_path):
        f = tmp_path / "huge.py"
        f.write_text("x" * (_MAX_FILE_SIZE + 1))
        assert _should_index(f) is False

    def test_file_at_size_limit(self, tmp_path):
        f = tmp_path / "edge.py"
        f.write_text("x" * _MAX_FILE_SIZE)
        assert _should_index(f) is True

    def test_txt_file(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("some notes")
        assert _should_index(f) is True

    def test_sql_file(self, tmp_path):
        f = tmp_path / "schema.sql"
        f.write_text("CREATE TABLE t (id int);")
        assert _should_index(f) is True


# ---------------------------------------------------------------------------
# _walk_files
# ---------------------------------------------------------------------------


class TestWalkFiles:
    def test_finds_python_files(self, tmp_path):
        (tmp_path / "main.py").write_text("pass")
        (tmp_path / "utils.py").write_text("pass")
        result = _walk_files(tmp_path)
        names = {f.name for f in result}
        assert "main.py" in names
        assert "utils.py" in names

    def test_skips_git_directory(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config.py").write_text("pass")
        (tmp_path / "main.py").write_text("pass")
        result = _walk_files(tmp_path)
        assert all(".git" not in str(f) for f in result)

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "lib.js").write_text("module.exports = {}")
        (tmp_path / "app.js").write_text("console.log('hi')")
        result = _walk_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "app.js"

    def test_skips_all_skip_dirs(self, tmp_path):
        for d in _SKIP_DIRS:
            (tmp_path / d).mkdir(exist_ok=True)
            (tmp_path / d / "hidden.py").write_text("pass")
        (tmp_path / "visible.py").write_text("pass")
        result = _walk_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "visible.py"

    def test_recursive_walk(self, tmp_path):
        sub = tmp_path / "src" / "lib"
        sub.mkdir(parents=True)
        (sub / "deep.py").write_text("pass")
        (tmp_path / "top.py").write_text("pass")
        result = _walk_files(tmp_path)
        names = {f.name for f in result}
        assert "deep.py" in names
        assert "top.py" in names

    def test_sorted_output(self, tmp_path):
        for name in ("z.py", "a.py", "m.py"):
            (tmp_path / name).write_text("pass")
        result = _walk_files(tmp_path)
        names = [f.name for f in result]
        assert names == sorted(names)

    def test_empty_directory(self, tmp_path):
        result = _walk_files(tmp_path)
        assert result == []

    def test_skips_non_indexable_files(self, tmp_path):
        (tmp_path / "photo.jpg").write_text("binary")
        (tmp_path / "code.py").write_text("pass")
        result = _walk_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "code.py"


# ---------------------------------------------------------------------------
# _flush_chunks
# ---------------------------------------------------------------------------


class TestFlushChunks:
    @pytest.fixture
    def mock_db(self):
        db = AsyncMock()
        db.insert_chunks = AsyncMock(return_value=2)
        return db

    def _make_pending(self, n: int = 2) -> list[tuple[Chunk, str, str]]:
        return [
            (
                Chunk(
                    file_path=f"file{i}.py",
                    chunk_type="function_definition",
                    name=f"func{i}",
                    content=f"def func{i}(): pass",
                    start_line=1,
                    end_line=1,
                    token_count=10,
                ),
                f"hash{i}",
                f"file{i}.py",
            )
            for i in range(n)
        ]

    async def test_embeds_and_stores(self, mock_db):
        pending = self._make_pending(2)
        fake_embeddings = [[0.1] * 768, [0.2] * 768]

        with patch("forge.rag.indexer.embed_texts", new_callable=AsyncMock, return_value=fake_embeddings):
            with patch("forge.rag.indexer.format_embedding_for_pg", side_effect=lambda e: str(e)):
                count = await _flush_chunks(pending, "test-project", mock_db)

        assert count == 2
        mock_db.insert_chunks.assert_called_once()
        records = mock_db.insert_chunks.call_args[0][0]
        assert len(records) == 2
        assert records[0]["project"] == "test-project"
        assert records[0]["file_path"] == "file0.py"
        assert records[0]["file_hash"] == "hash0"

    async def test_passes_content_to_embed(self, mock_db):
        pending = self._make_pending(1)
        fake_embeddings = [[0.1] * 768]

        with patch("forge.rag.indexer.embed_texts", new_callable=AsyncMock, return_value=fake_embeddings) as mock_embed:
            with patch("forge.rag.indexer.format_embedding_for_pg", return_value="[0.1]"):
                await _flush_chunks(pending, "proj", mock_db)

        mock_embed.assert_called_once()
        texts = mock_embed.call_args[0][0]
        assert texts == ["def func0(): pass"]


# ---------------------------------------------------------------------------
# index_directory
# ---------------------------------------------------------------------------


class TestIndexDirectory:
    @pytest.fixture
    def mock_db(self):
        db = AsyncMock()
        db.get_file_hash = AsyncMock(return_value=None)
        db.delete_file_chunks = AsyncMock(return_value=0)
        db.insert_chunks = AsyncMock(return_value=1)
        return db

    @pytest.fixture
    def project_dir(self, tmp_path):
        """Create a small project directory with indexable files."""
        (tmp_path / "main.py").write_text(
            'def main():\n    print("hello world from main function")\n'
        )
        (tmp_path / "utils.py").write_text(
            'def helper():\n    return "helper function result value"\n'
        )
        (tmp_path / "readme.jpg").write_text("not indexable")
        return tmp_path

    async def test_returns_stats(self, project_dir, mock_db):
        with patch("forge.rag.indexer.embed_texts", new_callable=AsyncMock, return_value=[[0.1] * 768] * 10):
            with patch("forge.rag.indexer.format_embedding_for_pg", return_value="[0.1]"):
                stats = await index_directory(project_dir, mock_db)

        assert "files_scanned" in stats
        assert "files_indexed" in stats
        assert "files_skipped" in stats
        assert "chunks_stored" in stats
        # Two .py files, jpg is not scanned
        assert stats["files_scanned"] == 2
        assert stats["files_indexed"] >= 1

    async def test_project_name_defaults_to_dir_name(self, project_dir, mock_db):
        with patch("forge.rag.indexer.embed_texts", new_callable=AsyncMock, return_value=[[0.1] * 768] * 10):
            with patch("forge.rag.indexer.format_embedding_for_pg", return_value="[0.1]"):
                await index_directory(project_dir, mock_db)

        # check that db calls used dir name as project
        if mock_db.get_file_hash.call_args_list:
            first_call = mock_db.get_file_hash.call_args_list[0]
            assert first_call[0][0] == project_dir.name

    async def test_custom_project_name(self, project_dir, mock_db):
        with patch("forge.rag.indexer.embed_texts", new_callable=AsyncMock, return_value=[[0.1] * 768] * 10):
            with patch("forge.rag.indexer.format_embedding_for_pg", return_value="[0.1]"):
                await index_directory(project_dir, mock_db, project="custom-name")

        if mock_db.get_file_hash.call_args_list:
            first_call = mock_db.get_file_hash.call_args_list[0]
            assert first_call[0][0] == "custom-name"

    async def test_incremental_skips_unchanged(self, project_dir, mock_db):
        # Return matching hash for each file so all are skipped
        from forge.rag.indexer import _file_hash
        file_hashes = {}
        for name in ("main.py", "utils.py"):
            content = (project_dir / name).read_text()
            file_hashes[name] = _file_hash(content)

        async def fake_get_hash(project, file_path):
            return file_hashes.get(file_path)

        mock_db.get_file_hash = AsyncMock(side_effect=fake_get_hash)

        with patch("forge.rag.indexer.embed_texts", new_callable=AsyncMock, return_value=[[0.1] * 768] * 10):
            with patch("forge.rag.indexer.format_embedding_for_pg", return_value="[0.1]"):
                stats = await index_directory(project_dir, mock_db)

        assert stats["files_skipped"] == stats["files_scanned"]
        assert stats["files_indexed"] == 0

    async def test_force_reindexes_all(self, project_dir, mock_db):
        # Even with matching hash, force=True should reindex
        content = (project_dir / "main.py").read_text()
        from forge.rag.indexer import _file_hash
        existing_hash = _file_hash(content)
        mock_db.get_file_hash = AsyncMock(return_value=existing_hash)

        with patch("forge.rag.indexer.embed_texts", new_callable=AsyncMock, return_value=[[0.1] * 768] * 10):
            with patch("forge.rag.indexer.format_embedding_for_pg", return_value="[0.1]"):
                stats = await index_directory(project_dir, mock_db, force=True)

        assert stats["files_indexed"] >= 1
        # get_file_hash should not be called at all when force=True
        mock_db.get_file_hash.assert_not_called()

    async def test_deletes_old_chunks_before_reindex(self, project_dir, mock_db):
        with patch("forge.rag.indexer.embed_texts", new_callable=AsyncMock, return_value=[[0.1] * 768] * 10):
            with patch("forge.rag.indexer.format_embedding_for_pg", return_value="[0.1]"):
                await index_directory(project_dir, mock_db)

        assert mock_db.delete_file_chunks.call_count >= 1

    async def test_empty_directory(self, tmp_path, mock_db):
        stats = await index_directory(tmp_path, mock_db)
        assert stats["files_scanned"] == 0
        assert stats["files_indexed"] == 0
        assert stats["chunks_stored"] == 0

    async def test_uses_relative_paths(self, project_dir, mock_db):
        with patch("forge.rag.indexer.embed_texts", new_callable=AsyncMock, return_value=[[0.1] * 768] * 10):
            with patch("forge.rag.indexer.format_embedding_for_pg", return_value="[0.1]"):
                await index_directory(project_dir, mock_db)

        # File paths passed to DB should be relative, not absolute
        for call in mock_db.get_file_hash.call_args_list:
            file_path = call[0][1]
            assert not file_path.startswith("/"), f"Path should be relative: {file_path}"

    async def test_unreadable_file_skipped(self, project_dir, mock_db):
        # Create file then make it unreadable
        bad_file = project_dir / "bad.py"
        bad_file.write_text("x = 1")
        bad_file.chmod(0o000)

        try:
            with patch("forge.rag.indexer.embed_texts", new_callable=AsyncMock, return_value=[[0.1] * 768] * 10):
                with patch("forge.rag.indexer.format_embedding_for_pg", return_value="[0.1]"):
                    stats = await index_directory(project_dir, mock_db)
            assert stats["files_skipped"] >= 1
        finally:
            bad_file.chmod(0o644)
