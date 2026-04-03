"""Project detection utilities — shared by REPL, agent, and code modes."""

from __future__ import annotations

import hashlib
from pathlib import Path

# Instruction files to look for in project directories (checked in order)
INSTRUCTION_FILES = ["CLAUDE.md", "FORGE.md", ".forge/instructions.md"]

# Budget limits for instruction file loading
MAX_INSTRUCTION_FILE_CHARS = 4_000
MAX_TOTAL_INSTRUCTION_CHARS = 12_000

# Project manifest files and their types
MANIFEST_FILES = {
    "pyproject.toml": "Python",
    "package.json": "Node.js",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java (Maven)",
    "build.gradle": "Java (Gradle)",
    "Gemfile": "Ruby",
    "composer.json": "PHP",
}


def detect_project_type(cwd: Path) -> str:
    """Detect project type from manifest files."""
    detected = []
    for filename, project_type in MANIFEST_FILES.items():
        if (cwd / filename).is_file():
            detected.append(project_type)
    return ", ".join(detected) if detected else ""


def load_project_instructions(cwd: Path) -> str:
    """Load instruction files from cwd and all ancestor directories.

    Walks from cwd to filesystem root, checking INSTRUCTION_FILES at each level.
    Deduplicates by content hash. Applies per-file (4K) and total (12K) budgets.
    """
    seen_hashes: set[str] = set()
    found: list[tuple[Path, str]] = []
    total_chars = 0

    current = cwd.resolve()
    root = Path(current.anchor)

    while True:
        for filename in INSTRUCTION_FILES:
            path = current / filename
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace").strip()
                if not content:
                    continue

                content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                if len(content) > MAX_INSTRUCTION_FILE_CHARS:
                    content = content[:MAX_INSTRUCTION_FILE_CHARS] + "\n... (truncated)"

                if total_chars + len(content) > MAX_TOTAL_INSTRUCTION_CHARS:
                    break

                found.append((path, content))
                total_chars += len(content)
            except Exception:
                pass

        if total_chars >= MAX_TOTAL_INSTRUCTION_CHARS:
            break
        parent = current.parent
        if parent == current or current == root:
            break
        current = parent

    if not found:
        return ""

    parts: list[str] = []
    for path, content in found:
        try:
            rel = path.relative_to(cwd)
        except ValueError:
            rel = path
        parts.append(f"\n\n## Project Instructions (from {rel})\n\n{content}")

    return "".join(parts)


def build_project_context(cwd: Path) -> str:
    """Build a project-awareness string to append to any system prompt."""
    parts: list[str] = []

    project_type = detect_project_type(cwd)
    if project_type:
        parts.append(f"Detected project type: {project_type}")

    parts.append(f"Working directory: {cwd}")

    instructions = load_project_instructions(cwd)

    return "\n".join(parts) + instructions
