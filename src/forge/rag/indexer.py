"""Walk + chunk + embed + store pipeline for indexing codebases."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from forge.models.embeddings import embed_texts, format_embedding_for_pg
from forge.rag.chunker import Chunk, chunk_file, supported_extensions
from forge.storage.database import Database

console = Console()

# File patterns to skip
_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    ".env", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".next", ".nuxt", "target", "out",
    ".claude", ".cursor",
}

_SKIP_FILES = {
    ".DS_Store", "Thumbs.db", "package-lock.json", "yarn.lock",
    "pnpm-lock.yaml", "Cargo.lock", "poetry.lock", "uv.lock",
}

# Extensions we can meaningfully chunk (code + config)
_INDEXABLE_EXTENSIONS = supported_extensions() | {
    ".txt", ".cfg", ".ini", ".env.example", ".sql",
    ".dockerfile", ".makefile",
}

_MAX_FILE_SIZE = 512 * 1024  # 512 KiB — skip huge files


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _should_index(path: Path) -> bool:
    """Check if a file should be indexed."""
    if path.name in _SKIP_FILES:
        return False
    if path.suffix.lower() not in _INDEXABLE_EXTENSIONS:
        return False
    if path.stat().st_size > _MAX_FILE_SIZE:
        return False
    return True


def _walk_files(root: Path) -> list[Path]:
    """Recursively walk a directory, respecting skip rules."""
    files: list[Path] = []
    for item in sorted(root.iterdir()):
        if item.is_dir():
            if item.name in _SKIP_DIRS:
                continue
            files.extend(_walk_files(item))
        elif item.is_file():
            if _should_index(item):
                files.append(item)
    return files


async def index_directory(
    root: str | Path,
    db: Database,
    *,
    project: str | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Index a directory: walk → chunk → embed → store.

    Returns stats dict with files_scanned, files_indexed, chunks_stored.
    """
    root = Path(root).resolve()
    if project is None:
        project = root.name

    files = _walk_files(root)
    stats = {"files_scanned": len(files), "files_indexed": 0, "files_skipped": 0, "chunks_stored": 0}

    if not files:
        console.print("[dim]No indexable files found.[/dim]")
        return stats

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing files", total=len(files))

        # Process in batches for efficient embedding
        pending_chunks: list[tuple[Chunk, str, str]] = []  # (chunk, file_hash, rel_path)

        for file_path in files:
            progress.update(task, advance=1)
            rel_path = str(file_path.relative_to(root))

            try:
                content = file_path.read_text(errors="replace")
            except (OSError, UnicodeDecodeError):
                stats["files_skipped"] += 1
                continue

            fhash = _file_hash(content)

            # Skip unchanged files (incremental indexing)
            if not force:
                stored_hash = await db.get_file_hash(project, rel_path)
                if stored_hash == fhash:
                    stats["files_skipped"] += 1
                    continue

            # Delete old chunks for this file
            await db.delete_file_chunks(project, rel_path)

            # Chunk the file
            chunks = chunk_file(str(file_path), content)
            if not chunks:
                stats["files_skipped"] += 1
                continue

            for chunk in chunks:
                pending_chunks.append((chunk, fhash, rel_path))

            stats["files_indexed"] += 1

            # Flush batch when it gets large enough
            if len(pending_chunks) >= 50:
                stored = await _flush_chunks(pending_chunks, project, db)
                stats["chunks_stored"] += stored
                pending_chunks = []

        # Flush remaining
        if pending_chunks:
            stored = await _flush_chunks(pending_chunks, project, db)
            stats["chunks_stored"] += stored

    return stats


async def _flush_chunks(
    pending: list[tuple[Chunk, str, str]],
    project: str,
    db: Database,
) -> int:
    """Embed and store a batch of chunks."""
    texts = [c.content for c, _, _ in pending]
    embeddings = await embed_texts(texts)

    records = []
    for (chunk, fhash, rel_path), embedding in zip(pending, embeddings):
        records.append({
            "project": project,
            "file_path": rel_path,
            "chunk_type": chunk.chunk_type,
            "name": chunk.name,
            "content": chunk.content,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "token_count": chunk.token_count,
            "embedding": format_embedding_for_pg(embedding),
            "file_hash": fhash,
        })

    return await db.insert_chunks(records)
