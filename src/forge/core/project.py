"""Project detection utilities — shared by REPL, agent, and code modes."""

from __future__ import annotations

from pathlib import Path

# Instruction files to look for in project directories (checked in order)
INSTRUCTION_FILES = ["CLAUDE.md", "FORGE.md", ".forge/instructions.md"]

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
    """Load project instruction files if present."""
    for filename in INSTRUCTION_FILES:
        path = cwd / filename
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                if content.strip():
                    return f"\n\n## Project Instructions (from {filename})\n\n{content}"
            except Exception:
                pass
    return ""


def build_project_context(cwd: Path) -> str:
    """Build a project-awareness string to append to any system prompt."""
    parts: list[str] = []

    project_type = detect_project_type(cwd)
    if project_type:
        parts.append(f"Detected project type: {project_type}")

    parts.append(f"Working directory: {cwd}")

    instructions = load_project_instructions(cwd)

    return "\n".join(parts) + instructions
