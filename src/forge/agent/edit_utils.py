"""Edit matching engine — multi-layer find-and-replace with fuzzy fallback."""

from __future__ import annotations

import difflib
import re


# Tuning constants
FUZZY_LINE_THRESHOLD = 0.7
FUZZY_AMBIGUITY_THRESHOLD = 0.55
MIN_FUZZY_LINES = 2


class EditMatchError(Exception):
    """Raised when no match can be found after all layers."""

    pass


def find_and_replace(text: str, old_text: str, new_text: str) -> tuple[str, str, str]:
    """Find old_text in text and replace it, trying multiple matching strategies.

    Returns:
        (new_content, match_method, warning) on success.

    Raises:
        EditMatchError with diagnostic information on failure.
    """
    # Layer 1: Exact match
    count = text.count(old_text)
    if count == 1:
        return text.replace(old_text, new_text, 1), "exact", ""
    if count > 1:
        raise EditMatchError(
            f"old_text appears {count} times — provide more surrounding "
            "context to make the match unique"
        )

    # Layer 2: Whitespace-normalized match
    result = _try_whitespace_normalized(text, old_text, new_text)
    if result is not None:
        return result

    # Layer 3: Fuzzy line-based match (only for multi-line old_text)
    old_lines = old_text.splitlines()
    if len(old_lines) >= MIN_FUZZY_LINES:
        result = _try_fuzzy_line_match(text, old_text, new_text, old_lines)
        if result is not None:
            return result

    # All layers failed — generate diagnostic error
    raise EditMatchError(_build_diagnostic(text, old_text))


def _normalize_ws(s: str) -> str:
    """Collapse whitespace runs to single space, strip trailing per line, normalize newlines."""
    lines = s.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    return "\n".join(re.sub(r"[ \t]+", " ", line).rstrip() for line in lines)


def _normalize_ws_aggressive(s: str) -> str:
    """Remove ALL non-essential whitespace for loose comparison."""
    lines = s.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    return "\n".join(re.sub(r"\s+", "", line) for line in lines)


def _try_whitespace_normalized(
    text: str, old_text: str, new_text: str
) -> tuple[str, str, str] | None:
    """Layer 2: Match after normalizing whitespace in both text and old_text.

    Uses aggressive (all-whitespace-removed) comparison for matching,
    then uses standard normalization for line-level alignment.
    """
    # First try standard normalization
    norm_text = _normalize_ws(text)
    norm_old = _normalize_ws(old_text)

    if not norm_old.strip():
        return None

    # Try standard normalization first, then aggressive
    count = norm_text.count(norm_old)
    if count != 1:
        # Fall back to aggressive normalization for matching
        agg_text = _normalize_ws_aggressive(text)
        agg_old = _normalize_ws_aggressive(old_text)
        if not agg_old.strip():
            return None
        agg_count = agg_text.count(agg_old)
        if agg_count != 1:
            return None
        # Use aggressive line matching
        agg_text_lines = agg_text.splitlines()
        agg_old_lines = agg_old.splitlines()
        start_idx = _find_line_block(agg_text_lines, agg_old_lines)
        if start_idx is None:
            return None
    else:
        # Find which original lines correspond to the normalized match
        norm_text_lines = norm_text.splitlines()
        old_norm_lines = norm_old.splitlines()
        start_idx = _find_line_block(norm_text_lines, old_norm_lines)
        if start_idx is None:
            return None

    text_lines = text.splitlines(keepends=True)
    old_line_count = len(old_text.splitlines())

    end_idx = start_idx + old_line_count

    # Reconstruct: lines before + new_text + lines after
    before = "".join(text_lines[:start_idx])
    after = "".join(text_lines[end_idx:])

    # Preserve trailing newline style of the replaced block
    if text_lines and start_idx < len(text_lines):
        # Check if new_text needs a trailing newline
        if after and not new_text.endswith("\n"):
            new_text_with_nl = new_text + "\n"
        else:
            new_text_with_nl = new_text
    else:
        new_text_with_nl = new_text

    new_content = before + new_text_with_nl + after
    return new_content, "whitespace_normalized", "Matched after normalizing whitespace"


def _find_line_block(haystack: list[str], needle: list[str]) -> int | None:
    """Find the starting index of needle lines within haystack lines."""
    if not needle:
        return None
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def _try_fuzzy_line_match(
    text: str, old_text: str, new_text: str, old_lines: list[str]
) -> tuple[str, str, str] | None:
    """Layer 3: Fuzzy line-based matching using SequenceMatcher."""
    text_lines = text.splitlines()
    n = len(old_lines)

    if n > len(text_lines):
        return None

    best_ratio = 0.0
    best_start = -1
    second_ratio = 0.0

    # Slide a window of size n across text_lines
    for i in range(len(text_lines) - n + 1):
        window = text_lines[i : i + n]
        ratio = difflib.SequenceMatcher(None, old_lines, window).ratio()
        if ratio > best_ratio:
            second_ratio = best_ratio
            best_ratio = ratio
            best_start = i
        elif ratio > second_ratio:
            second_ratio = ratio

    if best_ratio < FUZZY_LINE_THRESHOLD:
        return None

    # Ambiguity guard: second-best match must be below threshold
    if second_ratio >= FUZZY_AMBIGUITY_THRESHOLD:
        return None

    # Replace the matched lines
    text_lines_with_ends = text.splitlines(keepends=True)
    before = "".join(text_lines_with_ends[:best_start])
    after = "".join(text_lines_with_ends[best_start + n :])

    if after and not new_text.endswith("\n"):
        replacement = new_text + "\n"
    else:
        replacement = new_text

    new_content = before + replacement + after
    warning = (
        f"Fuzzy matched at lines {best_start + 1}-{best_start + n} "
        f"(similarity: {best_ratio:.2f})"
    )
    return new_content, "fuzzy_line", warning


def _build_diagnostic(text: str, old_text: str) -> str:
    """Build a diagnostic error message showing the closest match."""
    text_lines = text.splitlines()
    old_lines = old_text.splitlines()

    if not old_lines:
        return "old_text is empty"

    if len(old_lines) == 1:
        # Single-line: find closest line
        best_ratio = 0.0
        best_line = -1
        for i, line in enumerate(text_lines):
            ratio = difflib.SequenceMatcher(None, old_text.strip(), line.strip()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_line = i

        if best_ratio > 0.4:
            ctx_start = max(0, best_line - 2)
            ctx_end = min(len(text_lines), best_line + 3)
            context = "\n".join(
                f"  {i + 1:>4}: {text_lines[i]}" for i in range(ctx_start, ctx_end)
            )
            diff = list(
                difflib.unified_diff(
                    [old_text.strip()],
                    [text_lines[best_line].strip()],
                    fromfile="your old_text",
                    tofile="actual text",
                    lineterm="",
                )
            )
            diff_str = "\n".join(diff) if diff else ""
            return (
                f"old_text not found in file. Closest match at line {best_line + 1} "
                f"(similarity: {best_ratio:.0%}):\n{context}\n\n"
                f"{diff_str}\n\n"
                f"Read the file around lines {ctx_start + 1}-{ctx_end} and use the exact text."
            )
        return "old_text not found in file — read the file first to get the exact text"

    # Multi-line: sliding window approach
    n = len(old_lines)
    best_ratio = 0.0
    best_start = -1

    for i in range(max(1, len(text_lines) - n + 1)):
        end = min(i + n, len(text_lines))
        window = text_lines[i:end]
        ratio = difflib.SequenceMatcher(None, old_lines, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    if best_ratio > 0.3 and best_start >= 0:
        end = min(best_start + n, len(text_lines))
        actual = text_lines[best_start:end]
        diff = list(
            difflib.unified_diff(
                old_lines,
                actual,
                fromfile="your old_text",
                tofile="actual text",
                lineterm="",
            )
        )
        diff_str = "\n".join(diff)
        return (
            f"old_text not found in file. Closest match at lines {best_start + 1}-{end} "
            f"(similarity: {best_ratio:.0%}):\n{diff_str}\n\n"
            f"Read the file around lines {best_start + 1}-{end} and use the exact text."
        )

    return "old_text not found in file — read the file first to get the exact text"


def apply_edits(
    content: str, edits: list[tuple[str, str]]
) -> tuple[str, list[str], list[str]]:
    """Apply multiple find-and-replace edits sequentially on the same buffer.

    Each edit operates on the content produced by the previous edit.

    Args:
        content: Original file content.
        edits: List of (old_text, new_text) pairs.

    Returns:
        (final_content, match_methods, warnings) — match_methods[i] is
        the method used for edit i. On failure, raises EditMatchError
        indicating which edit index failed.
    """
    methods: list[str] = []
    warnings: list[str] = []
    for i, (old_text, new_text) in enumerate(edits):
        try:
            content, method, warning = find_and_replace(content, old_text, new_text)
        except EditMatchError as e:
            raise EditMatchError(f"Edit {i + 1}/{len(edits)} failed: {e}") from e
        methods.append(method)
        warnings.append(warning)
    return content, methods, warnings
