"""Context management — history processing to keep conversations within token budget."""

from __future__ import annotations

import os

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from forge.log import get_logger

logger = get_logger(__name__)

# Token estimation constants
DEFAULT_TOKEN_BUDGET = 16_000
TOOL_RESULT_TRUNCATE = 500
MIN_RECENT_MESSAGES = 4  # Always keep last N messages
MESSAGE_FRAMING_TOKENS = 4  # Per-message overhead
TOOL_METADATA_TOKENS = 10  # Per tool call/return overhead
CHARS_PER_TOKEN_CODE = 3.2
CHARS_PER_TOKEN_PROSE = 3.8

# Domain-aware compaction prompt
COMPACTION_PROMPT = """\
Summarize the following agent conversation for context continuity.
You are compressing {count} messages into a concise summary.

## PRESERVE (these are critical for the agent to continue working):
- Files touched and WHY (path + what was read/modified/created)
- Tool outcomes: success/failure, exit codes, key error messages
- Decisions made and their rationale
- Current task state: what's done, what's in progress, what's next
- Unresolved blockers or errors
- User preferences or corrections expressed during the conversation

## DISCARD (the agent can re-read these if needed):
- Raw file contents (just note filename + approximate line count)
- Verbose command output (keep exit code + key errors only)
- Intermediate search results (note what was searched and whether it helped)
- Redundant acknowledgments and filler

## Output format:
### Files Touched
- `path/to/file` — read/modified/created: brief reason

### Key Decisions
- Decision: rationale

### Current State
- What's completed, what's pending, any blockers

### Errors & Resolutions
- Error encountered → how it was resolved (or still open)

Keep under 400 words. Be specific about file paths and error messages.

---
Conversation to summarize:
{conversation}"""


def _is_code_heavy(text: str) -> bool:
    """Detect if text is code-heavy based on newline and brace density."""
    if not text:
        return False
    newline_ratio = text.count("\n") / max(len(text), 1)
    brace_count = text.count("{") + text.count("}") + text.count("(") + text.count(")")
    brace_ratio = brace_count / max(len(text), 1)
    return newline_ratio > 0.03 or brace_ratio > 0.02


def _estimate_part_tokens(part) -> int:
    """Estimate tokens for a single message part with type-specific overhead."""
    tokens = 0

    if isinstance(part, ToolReturnPart):
        content = str(part.content) if part.content else ""
        cpt = CHARS_PER_TOKEN_CODE if _is_code_heavy(content) else CHARS_PER_TOKEN_PROSE
        tokens = int(len(content) / cpt) + TOOL_METADATA_TOKENS
    elif isinstance(part, ToolCallPart):
        args_str = str(part.args) if part.args else ""
        tokens = int(len(args_str) / CHARS_PER_TOKEN_CODE) + TOOL_METADATA_TOKENS
    elif isinstance(part, (UserPromptPart, TextPart)):
        content = str(part.content) if part.content else ""
        cpt = CHARS_PER_TOKEN_CODE if _is_code_heavy(content) else CHARS_PER_TOKEN_PROSE
        tokens = int(len(content) / cpt)
    elif hasattr(part, "content"):
        content = str(part.content) if part.content else ""
        tokens = int(len(content) / CHARS_PER_TOKEN_PROSE)

    return tokens


def estimate_message_tokens(msg: ModelMessage) -> int:
    """Estimate tokens for a single message with part-level granularity."""
    tokens = MESSAGE_FRAMING_TOKENS

    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            tokens += _estimate_part_tokens(part)
    elif isinstance(msg, ModelResponse):
        for part in msg.parts:
            tokens += _estimate_part_tokens(part)
    else:
        tokens += int(len(str(msg)) / CHARS_PER_TOKEN_PROSE)

    return tokens


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (legacy compat)."""
    cpt = CHARS_PER_TOKEN_CODE if _is_code_heavy(text) else CHARS_PER_TOKEN_PROSE
    return int(len(text) / cpt)


def count_messages_tokens(messages: list[ModelMessage]) -> tuple[int, int]:
    """Return (message_count, estimated_tokens) for a message list."""
    total_tokens = sum(estimate_message_tokens(m) for m in messages)
    return len(messages), total_tokens


def get_token_count(
    messages: list[ModelMessage],
    deps: object | None = None,
) -> int:
    """Return real token count if available from deps, else estimate.

    Args:
        messages: Conversation messages (used for estimation fallback).
        deps: AgentDeps instance with tokens_in field (optional).
    """
    if deps and getattr(deps, "tokens_in", 0) > 0:
        return deps.tokens_in  # Last request's input tokens ≈ conversation size
    _, estimated = count_messages_tokens(messages)
    return estimated


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

    recent_count = min(MIN_RECENT_MESSAGES, len(messages))
    older = list(messages[:-recent_count]) if recent_count < len(messages) else []
    recent = list(messages[-recent_count:])

    older = [_truncate_tool_results(m) for m in older]

    recent_tokens = sum(estimate_message_tokens(m) for m in recent)
    budget_for_older = token_budget - recent_tokens

    if budget_for_older <= 0:
        return recent

    kept_older: list[ModelMessage] = []
    used_tokens = 0
    for msg in reversed(older):
        msg_tokens = estimate_message_tokens(msg)
        if used_tokens + msg_tokens > budget_for_older:
            break
        kept_older.append(msg)
        used_tokens += msg_tokens

    kept_older.reverse()
    return kept_older + recent


def _message_to_readable(msg: ModelMessage) -> str:
    """Convert a ModelMessage to a human-readable text representation for summarization."""
    lines: list[str] = []
    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                lines.append(f"User: {part.content}")
            elif isinstance(part, ToolReturnPart):
                content = str(part.content) if part.content else ""
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


def _group_task_sequences(
    messages: list[ModelMessage],
) -> list[list[ModelMessage]]:
    """Group messages into task sequences (user prompt → tool calls → response → next user prompt).

    Each group starts with a UserPromptPart and continues until the next UserPromptPart.
    """
    groups: list[list[ModelMessage]] = []
    current: list[ModelMessage] = []

    for msg in messages:
        # Check if this message starts a new user prompt
        is_user_prompt = False
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    is_user_prompt = True
                    break

        if is_user_prompt and current:
            groups.append(current)
            current = []
        current.append(msg)

    if current:
        groups.append(current)

    return groups


def _extract_preservable_refs(messages: list[ModelMessage]) -> str:
    """Stub: extract memory/task refs that must survive compaction."""
    return ""


async def _summarize_with_prompt(
    messages: list[ModelMessage],
    prompt_template: str | None = None,
) -> str | None:
    """Use the fast LLM to summarize messages.

    Returns None on any failure.
    """
    from pydantic_ai import Agent

    from forge.config import settings

    text_parts = []
    for msg in messages:
        readable = _message_to_readable(msg)
        if readable.strip():
            text_parts.append(readable)

    if not text_parts:
        return None

    conversation_text = "\n---\n".join(text_parts)
    input_limit = settings.agent.compaction_input_limit
    if len(conversation_text) > input_limit:
        conversation_text = conversation_text[:input_limit] + "\n... (truncated)"

    if prompt_template:
        prompt = prompt_template.format(
            count=len(messages),
            conversation=conversation_text,
        )
    else:
        prompt = (
            "Summarize this conversation history concisely. Focus on:\n"
            "- What the user asked for\n"
            "- What files were read/modified and key changes made\n"
            "- Important decisions and outcomes\n"
            "- Any errors encountered and how they were resolved\n\n"
            "Be concise (under 500 words). This summary replaces the original messages.\n\n"
            f"Conversation:\n{conversation_text}"
        )

    if "OLLAMA_BASE_URL" not in os.environ:
        base = settings.ollama.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        os.environ["OLLAMA_BASE_URL"] = base

    try:
        from forge.models.ollama import _model_settings
        model_name = settings.agent.compaction_model or settings.ollama.fast_model
        summarizer: Agent[None, str] = Agent(
            model=f"ollama:{model_name}",
            instructions="You are a concise conversation summarizer for an AI coding assistant.",
            tools=[],
            model_settings=_model_settings(timeout=60, num_ctx=8192),
        )
        result = await summarizer.run(prompt)
        summary = result.output
        if isinstance(summary, str) and len(summary.strip()) > 20:
            return summary.strip()
        return None
    except Exception:
        logger.debug("LLM summarization failed", exc_info=True)
        return None


async def summarize_for_compaction(messages: list[ModelMessage]) -> str | None:
    """Use the fast LLM to summarize older messages into a compact summary."""
    return await _summarize_with_prompt(messages)


async def smart_compact_history(
    messages: list[ModelMessage],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[ModelMessage]:
    """Tiered compaction: truncate → targeted summarize → full summarize.

    Tier 1: Truncate tool results in older messages (no LLM).
    Tier 2: Summarize completed task sequences individually (targeted LLM).
    Tier 3: Full summarization of all older messages (improved prompt).
    Falls back to mechanical compact_history() on LLM failure.
    """
    if not messages or len(messages) <= MIN_RECENT_MESSAGES:
        return messages

    recent_count = min(MIN_RECENT_MESSAGES, len(messages))
    older = list(messages[:-recent_count])
    recent = list(messages[-recent_count:])

    # Tier 1: Truncate tool results (no LLM call)
    truncated_older = [_truncate_tool_results(m) for m in older]
    trial = truncated_older + recent
    _, tokens = count_messages_tokens(trial)
    if tokens <= token_budget:
        logger.debug("Compact tier 1 (truncate): %d tokens, under budget", tokens)
        return trial

    # Tier 2: Summarize completed task sequences individually
    groups = _group_task_sequences(truncated_older)
    if len(groups) > 1:
        compacted_groups: list[ModelMessage] = []
        for group in groups:
            if len(group) <= 2:
                # Small group — keep as-is
                compacted_groups.extend(group)
                continue

            # Extract first user prompt for labeling
            preview = ""
            for msg in group:
                if isinstance(msg, ModelRequest):
                    for part in msg.parts:
                        if isinstance(part, UserPromptPart):
                            preview = str(part.content)[:60]
                            break
                    if preview:
                        break

            summary = await _summarize_with_prompt(group, COMPACTION_PROMPT)
            if summary:
                summary_msg = ModelRequest(parts=[UserPromptPart(
                    content=f"[Compacted task: {preview}]\n\n{summary}"
                )])
                compacted_groups.append(summary_msg)
            else:
                # Keep original if summarization failed
                compacted_groups.extend(group)

        trial = compacted_groups + recent
        _, tokens = count_messages_tokens(trial)
        if tokens <= token_budget:
            logger.debug("Compact tier 2 (targeted): %d tokens, under budget", tokens)
            return trial

        # If tier 2 helped but not enough, use its output as the older messages
        truncated_older = compacted_groups

    # Tier 3: Full summarization with domain-aware prompt
    summary = await _summarize_with_prompt(truncated_older, COMPACTION_PROMPT)

    if summary:
        preservable = _extract_preservable_refs(older)
        content = f"[Context compacted — {len(older)} messages summarized, tier 3]\n\n{summary}"
        if preservable:
            content += f"\n\n{preservable}"
        summary_msg = ModelRequest(parts=[UserPromptPart(content=content)])
        logger.debug("Compact tier 3 (full): summarized %d messages", len(older))
        return [summary_msg] + recent

    # Fallback to mechanical compaction
    logger.debug("Compact fallback: mechanical compaction")
    return compact_history(messages, token_budget)
