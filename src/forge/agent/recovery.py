"""Cloud recovery and error handling for circuit breaker trips."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.tools import READ_ONLY_TOOLS
from forge.config import settings
from forge.log import get_logger

if TYPE_CHECKING:
    from forge.agent.circuit_breaker import ToolCallTracker

logger = get_logger(__name__)


def _extract_text_from_messages(messages: list[ModelMessage]) -> str:
    """Extract assistant text from a message list for exemplar capture."""
    from pydantic_ai.messages import ModelResponse, TextPart

    parts = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    parts.append(part.content)
    return "\n".join(parts)


def _fork_history(
    message_history: list[ModelMessage],
    circuit_breaker: ToolCallTracker | None,
    identical_threshold: int = 3,
) -> list[ModelMessage]:
    """Truncate toxic loop tail from history before appending recovery.

    Removes the failed turns that led to the circuit breaker trip, preserving
    all successful earlier turns. Appends a synthetic bridge message so the
    local model understands the discontinuity.
    """
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    if not message_history:
        return []

    # Determine truncation point
    loop_start = None
    if circuit_breaker and circuit_breaker.loop_start_index is not None:
        loop_start = circuit_breaker.loop_start_index

    if loop_start is not None and 0 < loop_start < len(message_history):
        truncated = message_history[:loop_start]
    else:
        # Fallback: remove last 2 * identical_threshold messages (conservative)
        remove_count = 2 * identical_threshold
        if remove_count < len(message_history):
            truncated = message_history[:-remove_count]
        else:
            # History too short to truncate meaningfully — keep first message
            truncated = message_history[:1] if message_history else []

    # Append synthetic bridge message
    bridge = ModelRequest(parts=[
        UserPromptPart(
            content=(
                "[System note: The previous approach got stuck in a loop and was interrupted. "
                "A cloud model was consulted to resolve the issue. "
                "The following messages contain the cloud model's solution.]"
            ),
        ),
    ])
    truncated.append(bridge)

    logger.info(
        "Forked history: %d → %d messages (removed %d toxic tail messages)",
        len(message_history), len(truncated),
        len(message_history) - len(truncated) + 1,  # +1 for bridge
    )
    return truncated


async def _cloud_recovery(
    user_input: str,
    deps: AgentDeps,
    console: Console,
    turn_number: int,
) -> list[ModelMessage] | None:
    """Attempt cloud recovery after circuit breaker trip.

    Uses a fresh agent (no conversation history, no dynamic prompts) to avoid
    leaking session context to the cloud API. Tries primary model first,
    falls back to fallback model on failure.

    Returns recovery messages to append to history, or None on failure.
    """
    from forge.agent.gemini import get_gemini_model_settings, get_gemini_model_string
    from forge.agent.turn import _run_with_status
    from forge.agent.loop import AGENT_SYSTEM

    # Reset circuit breaker for the retry
    if deps.circuit_breaker:
        deps.circuit_breaker.reset_state()

    recovery_prompt = (
        f"The previous attempt with a local model got stuck in a loop. "
        f"Here's what was requested: {user_input}\n\n"
        f"Take a fresh approach. Think carefully and solve this step by step."
    )

    # Build a fresh agent — no history, no dynamic prompt injections
    from forge.core.project import build_project_context

    recovery_system = AGENT_SYSTEM
    if deps.cwd:
        recovery_system += "\n\n" + build_project_context(deps.cwd)

    # models_to_try: list of (model_id: str|Model, settings, label)
    models_to_try: list[tuple] = []

    # Gemini primary/fallback
    gemini_primary = get_gemini_model_string(fallback=False)
    gemini_fallback = get_gemini_model_string(fallback=True)
    if gemini_primary:
        models_to_try.append((gemini_primary, get_gemini_model_settings(), "gemini-primary"))
    if gemini_fallback and gemini_fallback != gemini_primary:
        models_to_try.append((gemini_fallback, get_gemini_model_settings(), "gemini-fallback"))

    # Deduplicate by label
    seen: set[str] = set()
    unique_models: list[tuple] = []
    for m, ms, label in models_to_try:
        if label not in seen:
            seen.add(label)
            unique_models.append((m, ms, label))

    for model_id, model_settings, label in unique_models:
        try:
            # Cloud recovery uses read-only tools to limit filesystem access
            recovery_agent: Agent[AgentDeps, str] = Agent(
                model=model_id,
                instructions=recovery_system,
                tools=READ_ONLY_TOOLS,
                deps_type=AgentDeps,
                model_settings=model_settings,
                retries=3,
            )
            result = await _run_with_status(
                recovery_agent, recovery_prompt, deps, None,
                turn_number=turn_number,
            )
            # Capture successful recovery as exemplar for local model learning
            if deps.memory_db:
                try:
                    from forge.agent.exemplars import capture_exemplar
                    solution = _extract_text_from_messages(result)
                    if solution:
                        await capture_exemplar(
                            deps.memory_db,
                            deps.memory_project or "default",
                            "recovery",
                            user_input,
                            solution,
                            label,
                            outcome_score=0.7,
                        )
                except Exception:
                    logger.debug("Exemplar capture failed", exc_info=True)
            return result
        except Exception as exc:
            exc_str = str(exc)
            # Summarize verbose API errors
            if "429" in exc_str or "quota" in exc_str.lower():
                short_err = "rate limited by cloud API"
            elif len(exc_str) > 200:
                short_err = exc_str[:200] + "..."
            else:
                short_err = exc_str
            if len(unique_models) > 1:
                console.print(
                    f"[yellow]Cloud recovery ({label}) failed: {short_err}[/yellow]\n"
                    f"[yellow]Trying next model...[/yellow]"
                )
            else:
                console.print(f"[red]Cloud recovery failed:[/red] {short_err}")

    return None


def _handle_agent_error(console: Console, e: Exception, deps: AgentDeps | None = None) -> None:
    """Print a user-friendly agent error message."""
    from pydantic_ai.exceptions import UsageLimitExceeded

    from forge.agent.circuit_breaker import CircuitBreakerTripped

    if isinstance(e, CircuitBreakerTripped):
        if deps and settings.agent.mode != "local":
            from forge.agent.gemini import is_gemini_available
            if is_gemini_available(deps):
                console.print(
                    f"[yellow]Circuit breaker tripped:[/yellow] {e}\n"
                    "[yellow]Escalating to cloud reasoning (Gemini) for recovery...[/yellow]"
                )
                deps._cloud_recovery_pending = True
                return
        console.print(
            f"[yellow]Circuit breaker tripped:[/yellow] {e}\n"
            "The model was stuck in a loop. Try rephrasing your request "
            "or breaking it into smaller steps."
        )
        return

    if isinstance(e, UsageLimitExceeded):
        console.print(
            f"[yellow]Agent hit the request limit ({settings.agent.request_limit} iterations).[/yellow] "
            "This usually means the model was stuck in a loop. "
            "The partial result has been preserved in history."
        )
        return

    err_str = str(e).lower()
    if "429" in err_str or "rate" in err_str and "limit" in err_str or "quota" in err_str:
        # Extract retry delay if present
        retry_hint = ""
        if "retry" in err_str:
            import re
            match = re.search(r"retry.*?(\d+)s", err_str)
            if match:
                retry_hint = f" Try again in ~{match.group(1)}s."
        console.print(
            f"[yellow]Rate limited by cloud API.[/yellow]{retry_hint}"
        )
    elif "connection" in err_str or "connect" in err_str:
        console.print(
            "[red]Cannot connect to Ollama.[/red] Is it running? "
            "Check with: [bold]systemctl status ollama[/bold]"
        )
    elif "timeout" in err_str or "timed out" in err_str:
        console.print(
            "[red]Request timed out.[/red] "
            "The model may still be loading — try again in a moment."
        )
    elif "404" in err_str:
        console.print(
            "[red]Model not found.[/red] "
            f"Pull it with: [bold]ollama pull {settings.ollama.heavy_model}[/bold]"
        )
    else:
        console.print(f"[red]Agent error:[/red] {e}")
