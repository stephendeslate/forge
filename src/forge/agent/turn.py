"""Turn execution — run a single agent turn with status tracking."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits
from rich.markdown import Markdown

from forge.agent.deps import AgentDeps
from forge.agent.hooks import TurnEnd, TurnStart
from forge.agent.render import render_events
from forge.agent.status import StatusTracker
from forge.agent.turn_buffer import TurnBuffer
from forge.config import settings
from forge.log import get_logger

logger = get_logger(__name__)


async def _run_with_status(
    agent: Agent[AgentDeps, str],
    prompt: str | Sequence,
    deps: AgentDeps,
    message_history: list[ModelMessage] | None,
    turn_number: int = 0,
    model_override: str | Model | None = None,
) -> list[ModelMessage]:
    """Run agent with status tracker lifecycle management.

    Returns the updated message history.
    """
    # Determine active model name for status bar display
    active_model = ""
    if model_override:
        active_model = model_override.model_name if isinstance(model_override, Model) else model_override
    elif deps.model_override:
        active_model = deps.model_override.model_name if isinstance(deps.model_override, Model) else deps.model_override
    else:
        active_model = settings.ollama.heavy_model

    tracker = StatusTracker(
        console=deps.console,
        visible=deps.status_visible,
        model_name=active_model,
        token_budget=settings.agent.token_budget,
        mode=settings.agent.mode,
    )
    deps.status_tracker = tracker

    # Create turn buffer for this turn
    turn_buffer = TurnBuffer(console=deps.console)
    deps.turn_buffer = turn_buffer

    # Show active task in status detail
    if deps.task_store:
        active_task = deps.task_store.get_active()
        if active_task and active_task.active_form:
            tracker.set_phase(tracker._phase, active_task.active_form)

    def _on_toggle(visible: bool) -> None:
        deps.status_visible = visible

    def _on_tools_toggle() -> None:
        deps.tools_visible = not deps.tools_visible

    tracker.start(on_toggle=_on_toggle, on_tools_toggle=_on_tools_toggle)

    # Emit TurnStart
    registry = deps.hook_registry
    if registry:
        await registry.emit_sequential(TurnStart(turn_number=turn_number, prompt=prompt))

    import time as _time
    turn_start = _time.monotonic()

    try:
        run_kwargs: dict = dict(
            deps=deps,
            message_history=message_history,
            event_stream_handler=render_events,
            usage_limits=UsageLimits(request_limit=settings.agent.request_limit),
        )
        if model_override:
            run_kwargs["model"] = model_override  # Model object or qualified string
        elif deps.model_override:
            if isinstance(deps.model_override, Model):
                run_kwargs["model"] = deps.model_override
            elif deps.model_override.startswith(("google-gla:", "openai:")):
                run_kwargs["model"] = deps.model_override
            else:
                run_kwargs["model"] = f"ollama:{deps.model_override}"

        from forge.models.retry import with_retry

        result = await with_retry(
            lambda: agent.run(prompt, **run_kwargs),
            max_retries=settings.ollama.max_retries,
            backoff_base=settings.ollama.retry_backoff_base,
        )
        # Capture real token counts from Ollama API response
        try:
            usage = result.usage()
            prompt_eval = usage.input_tokens or 0
            deps.tokens_in += prompt_eval
            deps.tokens_out += usage.output_tokens or 0
            # Prompt cache monitoring — detect KV cache invalidation
            if (
                deps._last_prompt_eval_count > 0
                and prompt_eval > deps._last_prompt_eval_count * 1.5
            ):
                logger.info(
                    "Prompt cache likely invalidated: prompt_eval %d → %d (%.0f%% increase)",
                    deps._last_prompt_eval_count, prompt_eval,
                    (prompt_eval / deps._last_prompt_eval_count - 1) * 100,
                )
            deps._last_prompt_eval_count = prompt_eval
        except Exception:
            pass  # usage() may not be available for all backends
        # Pause tracker before printing to prevent status line garbling output
        tracker.pause()

        # Add final answer to turn buffer unless already streamed by render_events
        streamed = any(not it.is_tool and it.was_printed for it in turn_buffer._items)
        if not streamed and isinstance(result.output, str) and result.output.strip():
            turn_buffer.add(Markdown(result.output), is_tool=False)
        tracker.tokens_in = deps.tokens_in
        tracker.tokens_out = deps.tokens_out
        tracker.stop()
        summary = tracker.summary()
        if summary:
            turn_buffer.add(summary, is_tool=False)
        turn_buffer.print_final(deps.tools_visible)

        # Emit TurnEnd
        if registry:
            elapsed = _time.monotonic() - turn_start
            await registry.emit_sequential(TurnEnd(
                turn_number=turn_number,
                tool_call_count=tracker.tool_calls,
                elapsed=elapsed,
                tokens_in=deps.tokens_in,
                tokens_out=deps.tokens_out,
            ))

        deps.status_tracker = None
        return result.all_messages()
    except BaseException:
        tracker.tokens_in = deps.tokens_in
        tracker.tokens_out = deps.tokens_out
        tracker.stop()
        deps.console.print(tracker.summary())
        deps.status_tracker = None
        deps.turn_buffer = None
        raise


def _maybe_prepend_think(prompt: str | Sequence, deps: AgentDeps) -> str | Sequence:
    """Prepend /think or /no_think tag based on thinking_enabled setting."""
    if not deps.thinking_enabled:
        return prompt
    if isinstance(prompt, str):
        return f"/think\n{prompt}"
    # Multimodal: prepend think tag to first text element
    result = list(prompt)
    for i, part in enumerate(result):
        if isinstance(part, str):
            result[i] = f"/think\n{part}"
            return result
    # No text part found — prepend one
    return ["/think", *result]


async def _execute_turn(
    agent: Agent,
    user_input: str,
    deps: AgentDeps,
    message_history: list[ModelMessage] | None,
    turn_counter: int,
    db,
    session_id: str,
    *,
    is_initial_turn: bool = False,
) -> list[ModelMessage] | None:
    """Execute a single agent turn: parse input, route vision, retrieve exemplars, run agent.

    Shared between initial-prompt and REPL-loop paths.
    Returns updated message_history.
    """
    from forge.agent.multimodal import parse_multimodal_input
    from forge.agent.persistence import _save_agent_session

    # Reset turn-scoped state from previous turn
    deps.reset_turn()

    parsed = parse_multimodal_input(user_input, deps.cwd)

    # Route to vision model if images detected
    vision_override = None
    if isinstance(parsed, list) and settings.ollama.vision_model:
        vision_override = deps.model_override
        deps.model_override = settings.ollama.vision_model

    # Retrieve relevant exemplars
    deps._active_exemplar_ids = []
    deps._exemplar_context = None
    if (
        settings.agent.exemplar_enabled
        and deps.memory_db
        and deps.memory_project
        and isinstance(parsed, str)
    ):
        try:
            from forge.agent.exemplars import retrieve_exemplars
            context, exemplar_ids = await retrieve_exemplars(
                deps.memory_db, deps.memory_project, parsed,
            )
            deps._active_exemplar_ids = exemplar_ids
            deps._exemplar_context = context or None
        except Exception:
            logger.debug("Exemplar retrieval failed", exc_info=True)

    prompt = _maybe_prepend_think(parsed, deps)
    message_history = await _run_with_status(
        agent, prompt, deps, message_history, turn_number=turn_counter,
    )

    # Positive exemplar signal if model did real work
    if deps._active_exemplar_ids and deps.memory_db and deps._files_modified_this_turn:
        try:
            from forge.agent.exemplars import update_active_exemplars
            await update_active_exemplars(
                deps.memory_db, deps._active_exemplar_ids, success=True,
            )
        except Exception:
            logger.debug("Exemplar outcome update failed", exc_info=True)
    deps._active_exemplar_ids = []

    # Restore model after vision turn
    if vision_override is not None or (isinstance(parsed, list) and settings.ollama.vision_model):
        deps.model_override = vision_override

    # Persist session
    if db and message_history:
        asyncio.create_task(_save_agent_session(db, session_id, message_history))

    # Set title on initial turn
    if db and is_initial_turn:
        title = user_input[:60].strip()
        if len(user_input) > 60:
            title = title.rsplit(" ", 1)[0] + "…"
        try:
            await db.update_session_title(session_id, title)
        except Exception:
            logger.debug("Title update failed", exc_info=True)

    return message_history


async def _handle_exemplar_failure(deps: AgentDeps, exc: Exception) -> None:
    """Send negative exemplar signal on circuit breaker trips."""
    from forge.agent.circuit_breaker import CircuitBreakerTripped as _CBT

    if deps._active_exemplar_ids and deps.memory_db and isinstance(exc, _CBT):
        try:
            from forge.agent.exemplars import update_active_exemplars
            await update_active_exemplars(
                deps.memory_db, deps._active_exemplar_ids, success=False,
            )
        except Exception:
            logger.debug("Exemplar outcome update failed", exc_info=True)
    deps._active_exemplar_ids = []
