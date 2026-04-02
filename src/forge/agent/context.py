"""Context management — history processing to keep conversations within token budget."""

from __future__ import annotations

import os

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRequestPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
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


def _message_to_readable(msg: ModelMessage) -> str:
    """Convert a ModelMessage to a human-readable text representation for summarization."""
    lines: list[str] = []
    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                lines.append(f"User: {part.content}")
            elif isinstance(part, ToolReturnPart):
                content = str(part.content) if part.content else ""
                # Truncate long tool results for summarization
                if len(content) > 300:
                    content = content[:300] + "..."
                lines.append(f"Tool result ({part.tool_name}): {content}")
            elif hasattr(part, "content"):
                lines.append(f"Request: {part.content}")
    elif isinstance(msg, ModelResponse):
        for part in msg.parts:
            if isinstance(part, TextPart):
                lines.append(f"Assistant: {part.content}")
            elif isinstance(part, ToolCallPart):
                args_str = str(part.args)
                if len(args_str) > 200:
                    args_str = args_str[:200] + "..."
                lines.append(f"Tool call: {part.tool_name}({args_str})")
    return "\n".join(lines) if lines else str(msg)[:200]


async def summarize_for_compaction(messages: list[ModelMessage]) -> str | None:
    """Use the fast LLM to summarize older messages into a compact summary.

    Returns None on any failure (LLM error, timeout, etc.).
    """
    from pydantic_ai import Agent
    from pydantic_ai.settings import ModelSettings

    from forge.config import settings

    # Build readable text from messages
    text_parts = []
    for msg in messages:
        readable = _message_to_readable(msg)
        if readable.strip():
            text_parts.append(readable)

    if not text_parts:
        return None

    conversation_text = "\n---\n".join(text_parts)
    # Cap input to avoid overwhelming the fast model
    if len(conversation_text) > 8000:
        conversation_text = conversation_text[:8000] + "\n... (truncated)"

    prompt = (
        "Summarize this conversation history concisely. Focus on:\n"
        "- What the user asked for\n"
        "- What files were read/modified and key changes made\n"
        "- Important decisions and outcomes\n"
        "- Any errors encountered and how they were resolved\n\n"
        "Be concise (under 500 words). This summary replaces the original messages.\n\n"
        f"Conversation:\n{conversation_text}"
    )

    # Ensure OLLAMA_BASE_URL is set
    if "OLLAMA_BASE_URL" not in os.environ:
        base = settings.ollama.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        os.environ["OLLAMA_BASE_URL"] = base

    try:
        summarizer: Agent[None, str] = Agent(
            model=f"ollama:{settings.ollama.fast_model}",
            instructions="You are a concise conversation summarizer.",
            tools=[],
            model_settings=ModelSettings(timeout=60),
        )
        result = await summarizer.run(prompt)
        summary = result.output
        if isinstance(summary, str) and len(summary.strip()) > 20:
            return summary.strip()
        return None
    except Exception:
        return None


async def smart_compact_history(
    messages: list[ModelMessage],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[ModelMessage]:
    """Compact history using LLM summarization for older messages.

    Falls back to mechanical compact_history() on LLM failure.
    """
    if not messages or len(messages) <= MIN_RECENT_MESSAGES:
        return messages

    # Split into older + recent
    recent_count = min(MIN_RECENT_MESSAGES, len(messages))
    older = list(messages[:-recent_count])
    recent = list(messages[-recent_count:])

    # Try LLM summarization of older messages
    summary = await summarize_for_compaction(older)

    if summary:
        # Create a synthetic user message with the summary
        summary_msg = ModelRequest(
            parts=[UserPromptPart(
                content=f"[Conversation summary of {len(older)} earlier messages]\n\n{summary}"
            )]
        )
        return [summary_msg] + recent

    # Fallback to mechanical compaction
    return compact_history(messages, token_budget)
