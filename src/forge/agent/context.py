"""Context management — history processing to keep conversations within token budget."""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRequestPart,
    ToolReturnPart,
)

# Rough estimate: ~4 chars per token
CHARS_PER_TOKEN = 4
DEFAULT_TOKEN_BUDGET = 16_000
TOOL_RESULT_TRUNCATE = 500
MIN_RECENT_MESSAGES = 4  # Always keep last N messages


def estimate_tokens(text: str) -> int:
    """Rough token count estimate."""
    return len(text) // CHARS_PER_TOKEN


def _message_text(msg: ModelMessage) -> str:
    """Extract all text content from a message for token estimation."""
    parts_text: list[str] = []
    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                content = str(part.content) if part.content else ""
                parts_text.append(content)
            elif hasattr(part, "content"):
                parts_text.append(str(part.content))
        return "\n".join(parts_text)
    elif isinstance(msg, ModelResponse):
        for part in msg.parts:
            if hasattr(part, "content"):
                parts_text.append(str(part.content))
            if hasattr(part, "args"):
                parts_text.append(str(part.args))
        return "\n".join(parts_text)
    return str(msg)


def _truncate_tool_results(msg: ModelMessage, max_len: int = TOOL_RESULT_TRUNCATE) -> ModelMessage:
    """Truncate tool return content in old messages to save context space."""
    if not isinstance(msg, ModelRequest):
        return msg

    new_parts: list[ModelRequestPart] = []
    changed = False
    for part in msg.parts:
        if isinstance(part, ToolReturnPart):
            content_str = str(part.content) if part.content else ""
            if len(content_str) > max_len:
                # Create a truncated copy
                truncated = content_str[:max_len] + "... (truncated)"
                new_part = ToolReturnPart(
                    tool_name=part.tool_name,
                    content=truncated,
                    tool_call_id=part.tool_call_id,
                    timestamp=part.timestamp,
                )
                new_parts.append(new_part)
                changed = True
                continue
        new_parts.append(part)

    if changed:
        return ModelRequest(parts=new_parts)
    return msg


def compact_history(
    messages: list[ModelMessage],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[ModelMessage]:
    """Compact message history to fit within token budget.

    Strategy:
    1. Always keep the last MIN_RECENT_MESSAGES messages intact.
    2. Truncate tool results in older messages.
    3. Drop oldest messages if still over budget.
    """
    if not messages:
        return messages

    # Step 1: Split into recent (kept intact) and older
    recent_count = min(MIN_RECENT_MESSAGES, len(messages))
    older = list(messages[:-recent_count]) if recent_count < len(messages) else []
    recent = list(messages[-recent_count:])

    # Step 2: Truncate tool results in older messages
    older = [_truncate_tool_results(m) for m in older]

    # Step 3: Estimate total tokens and drop oldest if over budget
    recent_tokens = sum(estimate_tokens(_message_text(m)) for m in recent)
    budget_for_older = token_budget - recent_tokens

    if budget_for_older <= 0:
        # Recent messages alone exceed budget — keep them anyway
        return recent

    # Keep as many older messages as fit, starting from the most recent
    kept_older: list[ModelMessage] = []
    used_tokens = 0
    for msg in reversed(older):
        msg_tokens = estimate_tokens(_message_text(msg))
        if used_tokens + msg_tokens > budget_for_older:
            break
        kept_older.append(msg)
        used_tokens += msg_tokens

    kept_older.reverse()
    return kept_older + recent


def count_messages_tokens(messages: list[ModelMessage]) -> tuple[int, int]:
    """Return (message_count, estimated_tokens) for a message list."""
    total_tokens = sum(estimate_tokens(_message_text(m)) for m in messages)
    return len(messages), total_tokens
