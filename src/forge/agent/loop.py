"""Agent REPL loop — multi-turn agentic coding with tool use."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import UsageLimits
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from forge.agent.context import (
    count_messages_tokens,
    smart_compact_history,
)
from forge.agent.deps import AgentDeps
from forge.agent.hooks import (
    HookRegistry,
    TurnEnd,
    TurnStart,
    make_permission_handler,
)
from forge.agent.permissions import PermissionPolicy
from forge.agent.render import render_events
from forge.agent.status import StatusTracker
from forge.agent.task_store import TaskStore
from forge.agent.tools import ALL_TOOLS, READ_ONLY_TOOLS
from forge.agent.turn_buffer import TurnBuffer
from forge.config import settings
from forge.log import get_logger
from forge.models.ollama import _ensure_ollama_env, _model_settings

if TYPE_CHECKING:
    from forge.storage.database import Database

logger = get_logger(__name__)

AGENT_SYSTEM = """\
You are Forge, a versatile local AI assistant. You help users with coding, writing, analysis, research, and general questions — as well as reading, understanding, editing, and creating code.

You have access to tools for reading files, writing files, editing files, running shell commands, searching code, listing files, web search, and web fetch. Use them to accomplish the user's request.

## Web research rules
1. Search snippets are often enough — if they answer the question, STOP and respond immediately.
2. Fetch a page ONLY when snippets lack the specific detail needed.
3. Budget per turn: at most 2 web_search + at most 2 web_fetch calls total. If you still lack info after that, answer with what you have and note gaps.
4. Never re-fetch a URL that returned an error.
5. Never fetch more than 3 URLs total in one turn.
6. For well-known facts, history, math, and science fundamentals, answer from your knowledge without searching.

## Internal reasoning
You may use <analysis>...</analysis> tags for chain-of-thought reasoning on complex problems.
These tags are automatically stripped during context compaction, so use them freely without worrying about context waste.

## General guidelines
- Read files before editing them to understand existing code.
- Use edit_file for targeted changes (exact string replacement). Use write_file only for new files or complete rewrites.
- When searching, use search_code with ripgrep patterns. Use list_files to understand project structure.
- Run commands to test changes, run builds, or gather information.
- Be concise in explanations. Show what you did, not what you plan to do.
- When making changes, verify them by reading the modified file or running tests.
- All file paths are relative to the working directory unless absolute.
- IMPORTANT: When you use tools like web_search or web_fetch, the results are raw data for YOU to process. Always synthesize tool results into a direct, natural answer to the user's question. Never dump raw search results or tool output as your response — interpret, summarize, and answer the question.
- If the user asks a question (e.g. "what's the weather?"), use the appropriate tool to gather information, then respond with a clear answer based on what you found.

## Task management
When your work involves 3+ distinct steps, create tasks to track progress.
Mark tasks in_progress when starting, completed when done.
If you discover additional work, create new tasks.
Do not create tasks for single simple operations.

## Memory
When the user asks you to remember something, use save_memory with an appropriate category.
When you need to recall past context, use recall_memories.
Categories: feedback (user corrections), project (decisions/context), user (role/preferences), reference (external pointers).

## Delegation
Use the delegate tool for contained, single-file implementation tasks with a clear spec.
The sub-agent runs a fast local model in an isolated git worktree.
Good for: "write tests for X", "add docstrings to Y", "implement function Z per this spec".
Bad for: multi-file refactors, architectural decisions, tasks requiring your conversation context.
After delegation, verify the sub-agent's work before reporting success.

Use delegate_parallel for parallelizable tasks: "write tests for modules A, B, C", "add docstrings to these 4 files".
Each subtask must be self-contained and independent — no cross-subtask dependencies. Max 4 concurrent.
"""

# Marker for prompt cache optimization — everything above is static and cacheable,
# dynamic content (tasks, memories, lint) is injected below via @agent.system_prompt
SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "---DYNAMIC-BOUNDARY---"

PLAN_OVERLAY = """\
You are in PLANNING mode. You have read-only tools to explore the codebase BEFORE planning.

## Process
1. First, use your tools to explore: read relevant files, search for patterns, list directory structures.
2. Understand the existing code, conventions, and dependencies.
3. THEN produce a structured plan based on what you actually found.

## Plan format
1. **Goal**: What needs to be accomplished
2. **Context**: Key files and patterns you discovered (reference specific files, functions, line numbers)
3. **Steps**: Numbered steps to achieve the goal, each referencing specific files/functions to modify
4. **Dependencies**: What existing code depends on the files you'll change
5. **Risks**: Potential issues or edge cases based on the actual codebase

Do NOT plan from imagination. Every claim about the codebase must come from tool results.
Do NOT use write_file, edit_file, or run_command — you are in read-only mode.
"""


def create_agent(
    system: str = AGENT_SYSTEM,
    cwd: Path | None = None,
    model: str | None = None,
    tools: list[Tool] | None = None,
    toolsets: list | None = None,
) -> Agent[AgentDeps, str]:
    """Create a pydantic-ai Agent with coding tools."""
    from forge.core.project import build_project_context

    _ensure_ollama_env()

    full_system = system
    if cwd:
        full_system += "\n\n" + build_project_context(cwd)

    model_name = model or settings.ollama.heavy_model

    return Agent(
        model=f"ollama:{model_name}",
        instructions=full_system,
        tools=tools if tools is not None else ALL_TOOLS,
        toolsets=toolsets or [],
        deps_type=AgentDeps,
        model_settings=_model_settings(),
        retries=3,
    )


async def _run_with_status(
    agent: Agent[AgentDeps, str],
    prompt: str | Sequence,
    deps: AgentDeps,
    message_history: list[ModelMessage] | None,
    turn_number: int = 0,
) -> list[ModelMessage]:
    """Run agent with status tracker lifecycle management.

    Returns the updated message history.
    """
    tracker = StatusTracker(console=deps.console, visible=deps.status_visible)
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
        await registry.emit(TurnStart(turn_number=turn_number, prompt=prompt))

    import time as _time
    turn_start = _time.monotonic()

    try:
        run_kwargs: dict = dict(
            deps=deps,
            message_history=message_history,
            event_stream_handler=render_events,
            usage_limits=UsageLimits(request_limit=settings.agent.request_limit),
        )
        if deps.model_override:
            run_kwargs["model"] = f"ollama:{deps.model_override}"

        result = await agent.run(prompt, **run_kwargs)
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
        turn_buffer.add(tracker.summary(), is_tool=False)
        turn_buffer.print_final(deps.tools_visible)

        # Emit TurnEnd
        if registry:
            elapsed = _time.monotonic() - turn_start
            await registry.emit(TurnEnd(
                turn_number=turn_number,
                tool_call_count=tracker.tool_calls,
                elapsed=elapsed,
                tokens_in=deps.tokens_in,
                tokens_out=deps.tokens_out,
            ))

        deps.status_tracker = None
        return result.all_messages()
    except BaseException:
        tracker.stop()
        deps.console.print(tracker.summary())
        deps.status_tracker = None
        deps.turn_buffer = None
        raise


async def _plan_and_execute(
    agent: Agent[AgentDeps, str],
    prompt: str,
    deps: AgentDeps,
    message_history: list[ModelMessage] | None,
) -> list[ModelMessage] | None:
    """Two-phase plan-then-execute workflow.

    Phase 1: Generate a plan (no tools).
    Phase 2: On approval, execute with full tools.
    Returns updated message_history, or the original if cancelled.
    """
    console = deps.console

    # Phase 1: Planning — agent with read-only tools to explore before planning
    _ensure_ollama_env()
    from forge.core.project import build_project_context

    plan_system = PLAN_OVERLAY
    if deps.cwd:
        plan_system += "\n\n" + build_project_context(deps.cwd)

    plan_agent: Agent[AgentDeps, str] = Agent(
        model=f"ollama:{settings.ollama.heavy_model}",
        instructions=plan_system,
        tools=READ_ONLY_TOOLS,
        deps_type=AgentDeps,
        model_settings=_model_settings(num_ctx=65536),
    )

    console.print("[dim]Exploring codebase and planning...[/dim]")
    try:
        plan_result = await plan_agent.run(
            prompt,
            deps=deps,
            message_history=None,  # Fresh context for planning
            usage_limits=UsageLimits(request_limit=10),  # Cap exploration
        )
        plan_text = plan_result.output
    except Exception as e:
        _handle_agent_error(console, e)
        return message_history

    # Display the plan
    console.print(
        Panel(
            plan_text,
            title="[bold]Plan[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # Phase 2: Ask for approval
    try:
        from prompt_toolkit import prompt as pt_prompt

        answer = await asyncio.get_running_loop().run_in_executor(
            None, lambda: pt_prompt("Execute this plan? [Y/n] ")
        )
        answer = answer.strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]Plan cancelled.[/dim]")
        return message_history

    if answer in ("n", "no"):
        console.print("[dim]Plan cancelled.[/dim]")
        return message_history

    # Phase 3: Execute with full agent
    deps.active_plan = plan_text
    execution_prompt = (
        f"Execute this plan step by step:\n\n{plan_text}\n\n"
        f"Original request: {prompt}\n\n"
        "As you complete each step, note it as done. If a step fails, "
        "explain why and adjust your approach before continuing."
    )
    try:
        result = await _run_with_status(agent, execution_prompt, deps, message_history)
    finally:
        deps.active_plan = None
    return result


from pydantic_ai.messages import ModelMessagesTypeAdapter

_message_list_adapter = ModelMessagesTypeAdapter


async def _save_agent_session(
    db: Database,
    session_id: str,
    messages: list[ModelMessage],
) -> None:
    """Persist agent message history as JSON to the conversations table."""
    try:
        history_json = _message_list_adapter.dump_json(messages).decode()
        await db.delete_agent_history(session_id)
        await db.save_message(session_id, "agent_history", history_json, model="")
    except Exception:
        logger.debug("Failed to save session", exc_info=True)


async def _load_agent_history(
    db: Database,
    session_id: str,
) -> list[ModelMessage] | None:
    """Load agent message history from the database. Returns None if not found."""
    try:
        raw = await db.load_agent_history(session_id)
        if raw is None:
            return None
        return _message_list_adapter.validate_json(raw)
    except Exception:
        logger.debug("Failed to load history", exc_info=True)
        return None


def _rebuild_agent(
    deps: AgentDeps,
    system: str,
    extra_tools: list[Tool] | None = None,
    toolsets: list | None = None,
) -> Agent[AgentDeps, str]:
    """Rebuild agent with fresh project context for current deps.cwd."""
    tools = ALL_TOOLS + (extra_tools or [])
    return create_agent(system=system, cwd=deps.cwd, tools=tools, toolsets=toolsets)


async def agent_repl(
    initial_prompt: str | None = None,
    permission: PermissionPolicy | None = None,
    resume_session_id: str | None = None,
    system: str = AGENT_SYSTEM,
    worktree_name: str | None = None,
) -> None:
    """Run the agentic REPL with tool use."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings

    from forge.agent.commands import _UNCHANGED, CommandContext, dispatch
    from forge.agent.session import (
        build_agent_with_tools,
        cleanup,
        print_welcome,
        setup_mcp,
        setup_memory,
        setup_model_monitor,
        setup_persistence,
        setup_rag,
        setup_worktree,
        wire_dynamic_prompts,
        wire_lint_hooks,
        wire_rag_hooks,
        wire_syntax_hooks,
    )

    console = Console()
    cwd = Path.cwd()

    # --- Setup phase ---
    try:
        cwd, worktree_info = setup_worktree(cwd, worktree_name, console)
    except SystemExit:
        return

    deps = AgentDeps(
        cwd=cwd,
        console=console,
        permission=permission or PermissionPolicy.AUTO,
        worktree=worktree_info,
    )

    hook_registry = HookRegistry()
    from forge.agent.hooks import PreToolUse

    # Sandbox hooks run first (priority 50) — before permission hook (default 0)
    # Lower priority number = runs first, so sandbox at -50 runs before permission at 0
    from forge.agent.sandbox import (
        make_command_blocklist_handler,
        make_path_boundary_handler,
        make_write_command_detector,
    )
    hook_registry.on(PreToolUse, make_command_blocklist_handler(), priority=-50)
    hook_registry.on(PreToolUse, make_path_boundary_handler(cwd), priority=-50)
    hook_registry.on(PreToolUse, make_write_command_detector(deps), priority=-25)

    hook_registry.on(PreToolUse, make_permission_handler(deps))
    deps.hook_registry = hook_registry
    deps.task_store = TaskStore()

    # Circuit breaker
    if settings.agent.cb_enabled:
        from forge.agent.circuit_breaker import ToolCallTracker, wire_circuit_breaker

        tracker = ToolCallTracker(
            identical_threshold=settings.agent.cb_identical,
            failure_threshold=settings.agent.cb_failures,
            oscillation_window=settings.agent.cb_oscillation_window,
            post_warning_grace=settings.agent.cb_post_warning_grace,
            history_size=settings.agent.cb_history_size,
        )
        deps.circuit_breaker = tracker
        wire_circuit_breaker(tracker, deps)

    await setup_model_monitor(deps, console)

    db, session_id, message_history = await setup_persistence(
        resume_session_id, deps, console, system,
    )
    memory_count = await setup_memory(db, deps, cwd)
    rag_available, rag_project_name = await setup_rag(db, deps, cwd, console, worktree_info)
    mcp_servers, mcp_stack = await setup_mcp(cwd)

    agent, extra_tools, system = build_agent_with_tools(
        system, cwd, deps, rag_available, mcp_servers,
    )
    wire_dynamic_prompts(agent, deps)
    wire_syntax_hooks(hook_registry, deps)
    wire_lint_hooks(hook_registry, deps)

    if rag_available and db:
        wire_rag_hooks(hook_registry, deps, db, rag_project_name)

    # Emit SessionStart
    from forge.agent.hooks import SessionStart
    await hook_registry.emit(SessionStart(
        session_id=session_id, cwd=str(cwd), permission=deps.permission.value,
    ))

    print_welcome(console, deps, session_id, db, system, rag_available, memory_count, mcp_servers)

    # --- Key bindings ---
    kb = KeyBindings()

    @kb.add("c-o")
    def _toggle_status(event):
        deps.status_visible = not deps.status_visible
        state = "visible" if deps.status_visible else "hidden"
        if deps.status_tracker:
            deps.status_tracker.visible = deps.status_visible
            if not deps.status_visible:
                deps.status_tracker._clear_line()
        sys.stderr.write(f"\r\033[2K\033[2mStatus line: {state} (Ctrl-O to toggle)\033[0m\n")
        sys.stderr.flush()

    @kb.add("c-r")
    def _toggle_tools(event):
        deps.tools_visible = not deps.tools_visible
        if deps.turn_buffer and deps.turn_buffer._items:
            event.app.renderer.erase()
            deps.turn_buffer.rerender(deps.tools_visible)
            event.app.renderer.reset()
            event.app.invalidate()
        else:
            state = "visible" if deps.tools_visible else "hidden"
            sys.stderr.write(f"\r\033[2K\033[2mTool results: {state} (Ctrl-R to toggle)\033[0m\n")
            sys.stderr.flush()

    pt_session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(), key_bindings=kb,
    )

    # Model escalation
    turn_counter_ref = [0]
    if settings.agent.auto_escalation:
        from forge.agent.escalation import ModelEscalator, wire_escalation

        escalator = ModelEscalator(
            deps, console, threshold=settings.agent.escalation_threshold,
        )
        deps.escalator = escalator
        wire_escalation(escalator, deps, turn_counter_ref)

    # --- Initial prompt ---
    turn_counter = 0
    if initial_prompt:
        turn_counter += 1
        turn_counter_ref[0] = turn_counter
        console.print(f"\n[bold]> {initial_prompt}[/bold]")
        try:
            from forge.agent.multimodal import parse_multimodal_input
            parsed_initial = parse_multimodal_input(initial_prompt, deps.cwd)

            # Route to vision model if images detected
            vision_override = None
            if isinstance(parsed_initial, list) and settings.ollama.vision_model:
                vision_override = deps.model_override
                deps.model_override = settings.ollama.vision_model

            prompt = _maybe_prepend_think(parsed_initial, deps)
            message_history = await _run_with_status(
                agent, prompt, deps, message_history, turn_number=turn_counter,
            )

            # Restore model after vision turn
            if vision_override is not None or (isinstance(parsed_initial, list) and settings.ollama.vision_model):
                deps.model_override = vision_override
            if db and message_history:
                asyncio.create_task(_save_agent_session(db, session_id, message_history))
            if db:
                title = initial_prompt[:60].strip()
                if len(initial_prompt) > 60:
                    title = title.rsplit(" ", 1)[0] + "…"
                try:
                    await db.update_session_title(session_id, title)
                except Exception:
                    logger.debug("Title update failed", exc_info=True)
        except Exception as e:
            _handle_agent_error(console, e)

    # --- REPL loop ---
    title_set = initial_prompt is not None
    try:
        while True:
            try:
                user_input = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: pt_session.prompt("\n❯ ")
                )
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Dispatch slash commands via registry
            if user_input.startswith("/"):
                cmd_ctx = CommandContext(
                    console=console,
                    deps=deps,
                    agent=agent,
                    message_history=message_history,
                    session_id=session_id,
                    db=db,
                    mcp_servers=mcp_servers,
                    rag_available=rag_available,
                    rag_project_name=rag_project_name,
                    extra_tools=extra_tools,
                    system=system,
                    turn_counter=turn_counter,
                )
                result = await dispatch(cmd_ctx, user_input)
                if result is not None:
                    if result.should_break:
                        break
                    if result.message_history is not _UNCHANGED:
                        message_history = result.message_history
                    if result.agent is not _UNCHANGED:
                        agent = result.agent
                    if result.rag_available is not None:
                        rag_available = result.rag_available
                    if result.turn_counter is not None:
                        turn_counter = result.turn_counter
                continue

            # Auto-compact if history is getting large
            budget = settings.agent.token_budget
            if message_history and len(message_history) > 40:
                _, tokens = count_messages_tokens(message_history)
                if tokens > budget * settings.agent.compaction_threshold:
                    before = len(message_history)
                    message_history = await smart_compact_history(message_history, budget)
                    after = len(message_history)
                    console.print(
                        f"[dim]Auto-compacted: {before} → {after} messages[/dim]"
                    )

            # Auto-title from first user message
            if db and not title_set:
                title = user_input[:60].strip()
                if len(user_input) > 60:
                    title = title.rsplit(" ", 1)[0] + "…"
                try:
                    await db.update_session_title(session_id, title)
                except Exception:
                    logger.debug("Title update failed", exc_info=True)
                title_set = True

            # Run agent turn
            try:
                turn_counter += 1
                turn_counter_ref[0] = turn_counter

                # Parse multimodal input (@image.png references)
                from forge.agent.multimodal import parse_multimodal_input
                parsed = parse_multimodal_input(user_input, deps.cwd)

                # Route to vision model if images detected
                vision_override = None
                if isinstance(parsed, list) and settings.ollama.vision_model:
                    vision_override = deps.model_override
                    deps.model_override = settings.ollama.vision_model

                prompt = _maybe_prepend_think(parsed, deps)
                message_history = await _run_with_status(
                    agent, prompt, deps, message_history, turn_number=turn_counter,
                )

                # Restore model after vision turn
                if vision_override is not None or (isinstance(parsed, list) and settings.ollama.vision_model):
                    deps.model_override = vision_override
                if db and message_history:
                    asyncio.create_task(_save_agent_session(db, session_id, message_history))
                if db and deps.task_store:
                    try:
                        await db.save_task_store(session_id, deps.task_store.to_json())
                    except Exception:
                        logger.debug("Task store save failed", exc_info=True)
            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted.[/dim]")
            except Exception as e:
                _handle_agent_error(console, e)
    finally:
        await cleanup(
            deps, hook_registry, mcp_stack, db, session_id, message_history, console,
        )


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


async def _connect_db():
    """Connect to DB for persistence. Returns None on failure."""
    if not settings.persist_history:
        return None
    try:
        from forge.storage.database import Database

        db = Database()
        await db.connect()
        return db
    except Exception:
        logger.info("Database unavailable — persistence disabled", exc_info=True)
        return None


def _handle_agent_error(console: Console, e: Exception) -> None:
    """Print a user-friendly agent error message."""
    from pydantic_ai.exceptions import UsageLimitExceeded

    from forge.agent.circuit_breaker import CircuitBreakerTripped

    if isinstance(e, CircuitBreakerTripped):
        console.print(
            f"[yellow]Circuit breaker tripped:[/yellow] {e}\n"
            "The model was stuck in a loop. Try rephrasing your request "
            "or breaking it into smaller steps."
        )
        return

    if isinstance(e, UsageLimitExceeded):
        console.print(
            "[yellow]Agent hit the request limit (15 iterations).[/yellow] "
            "This usually means the model was stuck in a loop. "
            "The partial result has been preserved in history."
        )
        return

    err_str = str(e).lower()
    if "connection" in err_str or "connect" in err_str:
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
