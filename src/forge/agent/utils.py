"""Shared utilities for agent tools."""

from __future__ import annotations


def head_tail_truncate(text: str, max_chars: int, *, ratio: float = 0.5) -> str:
    """Keep first `ratio` and last `1-ratio` of text, drop the middle."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * ratio)
    tail = max_chars - head
    dropped = len(text) - max_chars
    return (
        text[:head]
        + f"\n\n... ({dropped:,} chars truncated) ...\n\n"
        + text[-tail:]
    )
