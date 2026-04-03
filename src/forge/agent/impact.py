"""Impact analysis — find what depends on a file before modifying it.

Uses tree-sitter to extract symbols and ripgrep to find reverse dependencies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from forge.log import get_logger

logger = get_logger(__name__)

# Import patterns per language for ripgrep
_IMPORT_PATTERNS: dict[str, list[str]] = {
    ".py": [
        r"from\s+{module}\s+import",
        r"import\s+{module}",
    ],
    ".js": [r"(?:import|require)\s*.*from\s*['\"].*{module}"],
    ".jsx": [r"(?:import|require)\s*.*from\s*['\"].*{module}"],
    ".ts": [r"(?:import|require)\s*.*from\s*['\"].*{module}"],
    ".tsx": [r"(?:import|require)\s*.*from\s*['\"].*{module}"],
    ".rs": [r"use\s+.*{module}"],
    ".go": [r"import\s+.*{module}"],
}


@dataclass
class ImpactReport:
    file: str
    symbols_defined: list[str] = field(default_factory=list)
    imported_by: dict[str, list[str]] = field(default_factory=dict)
    total_dependents: int = 0

    def format(self) -> str:
        parts = [f"Impact analysis for: {self.file}\n"]

        if self.symbols_defined:
            parts.append(f"Symbols defined ({len(self.symbols_defined)}):")
            for sym in self.symbols_defined[:30]:
                parts.append(f"  - {sym}")
            if len(self.symbols_defined) > 30:
                parts.append(f"  ... and {len(self.symbols_defined) - 30} more")

        if self.imported_by:
            parts.append(f"\nDependent files ({self.total_dependents}):")
            for symbol, files in sorted(self.imported_by.items()):
                if files:
                    parts.append(f"  {symbol}:")
                    for f in files[:10]:
                        parts.append(f"    - {f}")
                    if len(files) > 10:
                        parts.append(f"    ... and {len(files) - 10} more")
        elif self.symbols_defined:
            parts.append("\nNo other files import symbols from this file.")

        if not self.symbols_defined:
            parts.append("No top-level symbols found (file may not be parseable with tree-sitter).")

        return "\n".join(parts)


def _extract_symbols(file_path: Path) -> list[str]:
    """Extract top-level symbol names from a file using tree-sitter."""
    try:
        from forge.rag.chunker import _get_parser, _node_name
    except ImportError:
        logger.debug("tree-sitter chunker not available")
        return []

    ext = file_path.suffix
    result = _get_parser(ext)
    if result is None:
        return []

    parser, top_types = result

    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except Exception:
        logger.debug("Failed to parse %s", file_path, exc_info=True)
        return []

    symbols = []
    for node in tree.root_node.children:
        if node.type in top_types:
            name = _node_name(node)
            if name:
                symbols.append(name)
    return symbols


async def _ripgrep_search(pattern: str, cwd: Path, glob_filter: str = "") -> list[str]:
    """Run ripgrep and return matching file paths."""
    cmd = ["rg", "--files-with-matches", "--no-heading", "--color=never"]
    if glob_filter:
        cmd.extend(["--glob", glob_filter])
    cmd.extend(["--", pattern, str(cwd)])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
    except (FileNotFoundError, asyncio.TimeoutError):
        return []

    if proc.returncode != 0:
        return []

    output = stdout.decode("utf-8", errors="replace").strip()
    if not output:
        return []

    # Return relative paths
    files = []
    for line in output.splitlines():
        try:
            p = Path(line)
            if p.is_absolute():
                p = p.relative_to(cwd)
            files.append(str(p))
        except (ValueError, OSError):
            files.append(line)
    return files


def _module_name_from_path(file_path: Path, cwd: Path) -> str:
    """Convert a file path to a Python module-style name for import searching."""
    try:
        rel = file_path.relative_to(cwd)
    except ValueError:
        rel = file_path

    # Remove extension and convert path separators to dots
    parts = list(rel.with_suffix("").parts)
    return ".".join(parts)


async def build_impact_report(file_path: str, cwd: Path) -> ImpactReport:
    """Build an impact report for a file — what symbols it defines and what depends on it."""
    path = Path(file_path)
    if not path.is_absolute():
        path = (cwd / path).resolve()

    report = ImpactReport(file=str(path.relative_to(cwd) if path.is_relative_to(cwd) else path))

    if not path.exists():
        return report

    # Extract symbols
    report.symbols_defined = _extract_symbols(path)

    ext = path.suffix

    # Build search patterns
    search_tasks: list[tuple[str, str]] = []

    if ext == ".py":
        module_name = _module_name_from_path(path, cwd)
        # Search for import patterns
        for symbol in report.symbols_defined:
            search_tasks.append((symbol, rf"(?:from\s+\S+\s+import\s+.*\b{symbol}\b|{symbol})"))
        # Also search for module-level imports
        if module_name:
            # Convert "src.forge.config" style to search pattern
            search_tasks.append((f"module:{module_name}", rf"(?:from\s+{module_name.replace('.', r'\.')}|import\s+{module_name.replace('.', r'\.')})"))
    else:
        # Generic: search for each symbol name
        for symbol in report.symbols_defined:
            search_tasks.append((symbol, rf"\b{symbol}\b"))

    # Run searches concurrently
    glob_filter = f"*{ext}" if ext else ""
    all_dependents: set[str] = set()

    async def _search_symbol(name: str, pattern: str) -> tuple[str, list[str]]:
        files = await _ripgrep_search(pattern, cwd, glob_filter=glob_filter)
        # Exclude the file itself
        rel_self = str(path.relative_to(cwd) if path.is_relative_to(cwd) else path)
        return name, [f for f in files if f != rel_self]

    results = await asyncio.gather(
        *[_search_symbol(name, pattern) for name, pattern in search_tasks],
    )

    for name, files in results:
        if files:
            report.imported_by[name] = files
            all_dependents.update(files)

    report.total_dependents = len(all_dependents)
    return report
