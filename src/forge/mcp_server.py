"""Forge MCP server — expose Forge tools to external clients."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

from fastmcp import FastMCP

from forge.config import settings
from forge.log import get_logger

logger = get_logger(__name__)

mcp = FastMCP("Forge")

# Working directory — set at startup, tools operate relative to this
_cwd: Path = Path.cwd()


def _resolve(path: str) -> Path:
    """Resolve a path relative to the working directory."""
    p = Path(path)
    if not p.is_absolute():
        p = _cwd / p
    return p.resolve()


@mcp.tool()
def read_file(
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read a file and return its contents with line numbers.

    Args:
        file_path: Path to the file (relative to working directory or absolute).
        offset: Line offset to start reading from (0-based).
        limit: Maximum number of lines to return.
    """
    from forge.agent.sandbox import check_path_boundary
    boundary_err = check_path_boundary(file_path, _cwd)
    if boundary_err:
        return f"Error: {boundary_err}"
    resolved = _resolve(file_path)
    if not resolved.exists():
        return f"Error: File not found: {file_path}"
    if resolved.is_dir():
        return f"Error: {file_path} is a directory, not a file."

    # Detect image files — return metadata instead of garbled binary
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
    if resolved.suffix.lower() in _IMAGE_EXTS:
        size = resolved.stat().st_size
        return f"[Image file: {resolved} ({resolved.suffix.lower()}, {size:,} bytes)]"

    try:
        lines = resolved.read_text().splitlines()
    except Exception as e:
        return f"Error reading {file_path}: {e}"

    total = len(lines)
    selected = lines[offset : offset + limit]
    numbered = [f"{i + offset + 1:>6}\t{line}" for i, line in enumerate(selected)]
    header = f"File: {file_path} ({total} lines)"
    if offset > 0 or len(selected) < total:
        header += f" — showing lines {offset + 1}-{offset + len(selected)}"
    return header + "\n" + "\n".join(numbered)


@mcp.tool()
def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed.

    Args:
        file_path: Path to the file.
        content: The content to write.
    """
    from forge.agent.sandbox import check_path_boundary
    boundary_err = check_path_boundary(file_path, _cwd)
    if boundary_err:
        return f"Error: {boundary_err}"
    resolved = _resolve(file_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content)
    lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"Wrote {lines} lines to {file_path}"


@mcp.tool()
def edit_file(file_path: str, old_text: str, new_text: str) -> str:
    """Replace text in a file. Tries exact match first, then whitespace-normalized
    and fuzzy matching as fallbacks.

    Args:
        file_path: Path to the file.
        old_text: The text to find and replace.
        new_text: The replacement text.
    """
    from forge.agent.edit_utils import EditMatchError, find_and_replace
    from forge.agent.sandbox import check_path_boundary
    boundary_err = check_path_boundary(file_path, _cwd)
    if boundary_err:
        return f"Error: {boundary_err}"

    resolved = _resolve(file_path)
    if not resolved.exists():
        return f"Error: File not found: {file_path}"

    content = resolved.read_text()
    try:
        new_content, method, warning = find_and_replace(content, old_text, new_text)
    except EditMatchError as e:
        return f"Error: {e}"

    resolved.write_text(new_content)
    msg = f"Edited {file_path}"
    if method != "exact":
        msg += f" (matched via {method})"
    if warning:
        msg += f"\nNote: {warning}"

    # Append unified diff for visibility
    import difflib
    old_lines = content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=file_path, tofile=file_path, lineterm=""))
    if diff:
        msg += "\n\n" + "\n".join(diff[:50])
        if len(diff) > 50:
            msg += f"\n... ({len(diff)} diff lines total)"
    return msg


@mcp.tool()
def search_code(
    pattern: str,
    path: str = ".",
    glob_filter: str | None = None,
) -> str:
    """Search for a regex pattern in files using ripgrep.

    Args:
        pattern: Regex pattern to search for.
        path: Directory to search in (relative to working directory).
        glob_filter: Optional glob to filter files (e.g. "*.py").
    """
    resolved = _resolve(path)
    from forge.agent.sandbox import check_path_boundary
    boundary_err = check_path_boundary(str(resolved), _cwd)
    if boundary_err:
        return f"Error: {boundary_err}"
    from forge.config import settings as _settings
    max_matches = _settings.agent.search_max_matches
    cmd = ["rg", "--no-heading", "-n", "--color=never", f"--max-count={max_matches}", pattern, str(resolved)]
    if glob_filter:
        cmd.extend(["--glob", glob_filter])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, cwd=str(_cwd)
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            lines = output.split("\n")
            if len(lines) > max_matches:
                output = "\n".join(lines[:max_matches]) + f"\n... ({len(lines)} total matches)"
            return output or "No matches found."
        if result.returncode == 1:
            return "No matches found."
        return f"Error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out."
    except FileNotFoundError:
        return "Error: ripgrep (rg) not installed."


@mcp.tool()
def list_files(path: str = ".", pattern: str = "**/*") -> str:
    """List files matching a glob pattern.

    Args:
        path: Directory to list from.
        pattern: Glob pattern (default: all files).
    """
    resolved = _resolve(path)
    from forge.agent.sandbox import check_path_boundary
    boundary_err = check_path_boundary(str(resolved), _cwd)
    if boundary_err:
        return f"Error: {boundary_err}"
    if not resolved.is_dir():
        return f"Error: Not a directory: {path}"

    skip = {".git", "__pycache__", "node_modules", ".venv", ".forge", ".mypy_cache", ".ruff_cache"}
    files: list[str] = []
    for p in sorted(resolved.glob(pattern)):
        if any(part in skip for part in p.parts):
            continue
        if p.is_file():
            try:
                rel = p.relative_to(resolved)
            except ValueError:
                rel = p
            files.append(str(rel))
            if len(files) >= 500:
                break

    if not files:
        return f"No files matching '{pattern}' in {path}"
    result = f"{len(files)} files"
    if len(files) >= 500:
        result += " (truncated at 500)"
    return result + ":\n" + "\n".join(files)


@mcp.tool()
def run_command(command: str, timeout: float = 30.0) -> str:
    """Run a shell command in the working directory.

    Args:
        command: The shell command to execute.
        timeout: Maximum execution time in seconds.
    """
    from forge.agent.sandbox import check_command_blocklist
    block_err = check_command_blocklist(command)
    if block_err:
        return f"Error: {block_err}"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_cwd),
            start_new_session=True,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        output = output.strip()
        max_output = 50000
        if len(output) > max_output:
            output = output[:max_output] + "\n... (truncated)"
        return f"{output}\n\nExit code: {result.returncode}"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


@mcp.resource("forge://status")
def forge_status() -> str:
    """Current Forge configuration and working directory."""
    return (
        f"Working directory: {_cwd}\n"
        f"Heavy model: {settings.ollama.heavy_model}\n"
        f"Fast model: {settings.ollama.fast_model}\n"
        f"Embed model: {settings.ollama.embed_model}\n"
        f"Ollama URL: {settings.ollama.base_url}\n"
    )


def run(cwd: str | None = None) -> None:
    """Entry point for the MCP server."""
    global _cwd
    if cwd:
        _cwd = Path(cwd).resolve()
    mcp.run()
