"""Agent REPL loop — multi-turn agentic coding with tool use."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import UsageLimits
from rich.console import Console
from rich.panel import Panel

from forge.agent.context import (
    count_messages_tokens,
    smart_compact_history,
)
from forge.agent.deps import AgentDeps
from forge.agent.hooks import (
    HookRegistry,
    make_permission_handler,
)
from forge.agent.permissions import PermissionPolicy
from forge.agent.persistence import (  # noqa: F401 — re-exported for external callers
    _connect_db,
    _load_agent_history,
    _message_list_adapter,
    _save_agent_session,
)
from forge.agent.recovery import (  # noqa: F401 — re-exported for external callers
    _cloud_recovery,
    _extract_text_from_messages,
    _fork_history,
    _handle_agent_error,
)
from forge.agent.task_store import TaskStore
from forge.agent.tools import ALL_TOOLS, READ_ONLY_TOOLS
from forge.agent.turn import (  # noqa: F401 — re-exported for external callers
    _execute_turn,
    _handle_exemplar_failure,
    _maybe_prepend_think,
    _run_with_status,
)
from forge.config import settings
from forge.log import get_logger
from forge.models.ollama import _ensure_ollama_env, _model_settings

if TYPE_CHECKING:
    from forge.agent.circuit_breaker import ToolCallTracker
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
- When you need to read 2+ files, prefer batch_read over multiple read_file calls — it's faster.
- Use edit_file for single targeted changes. Use multi_edit when making 2+ replacements in the same file — it applies all edits in one call.
- Use write_file only for new files or complete rewrites.
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

    # Model selection: explicit override → local heavy (cloud used only for recovery/critique/planning)
    # Known cloud provider prefixes that pydantic-ai resolves directly
    _CLOUD_PREFIXES = ("google-gla:", "openai:", "anthropic:")
    if model:
        # Explicit model passed — use as-is if it's a known cloud provider, else assume Ollama
        if model.startswith(_CLOUD_PREFIXES):
            model_id = model
            ms = _model_settings()
        else:
            model_name = model.removeprefix("ollama:")
            model_id = f"ollama:{model_name}"
            ms = _model_settings()
    else:
        # Local primary for both local and balanced modes
        model_id = f"ollama:{settings.ollama.heavy_model}"
        ms = _model_settings()

    return Agent(
        model=model_id,
        instructions=full_system,
        tools=tools if tools is not None else ALL_TOOLS,
        toolsets=toolsets or [],
        deps_type=AgentDeps,
        model_settings=ms,
        retries=3,
    )


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

    from forge.agent.gemini import get_gemini_model_settings, get_gemini_model_string, is_gemini_available

    plan_model = None
    plan_model_settings = None

    if is_gemini_available(deps):
        gemini_model = get_gemini_model_string()
        if gemini_model:
            plan_model = gemini_model
            plan_model_settings = get_gemini_model_settings()
            console.print("[yellow dim]Using cloud reasoning for planning (Gemini)...[/yellow dim]")

    if not plan_model:
        plan_model = f"ollama:{settings.ollama.heavy_model}"
        plan_model_settings = _model_settings(num_ctx=65536)
        console.print("[dim]Cloud unavailable, falling back to local model...[/dim]")

    plan_agent: Agent[AgentDeps, str] = Agent(
        model=plan_model,
        instructions=plan_system,
        tools=READ_ONLY_TOOLS,
        deps_type=AgentDeps,
        model_settings=plan_model_settings,
    )

    console.print("[dim]Exploring codebase and planning...[/dim]")
    used_cloud = plan_model != f"ollama:{settings.ollama.heavy_model}"
    try:
        plan_result = await plan_agent.run(
            prompt,
            deps=deps,
            message_history=None,  # Fresh context for planning
            usage_limits=UsageLimits(request_limit=10),  # Cap exploration
        )
        plan_text = plan_result.output
    except Exception as e:
        if used_cloud:
            # Cloud model failed — fall back to local model
            _handle_agent_error(console, e, deps=deps)
            console.print("[yellow]Falling back to local model for planning...[/yellow]")
            local_model = f"ollama:{settings.ollama.heavy_model}"
            plan_agent = Agent(
                model=local_model,
                instructions=plan_system,
                tools=READ_ONLY_TOOLS,
                deps_type=AgentDeps,
                model_settings=_model_settings(num_ctx=65536),
            )
            try:
                plan_result = await plan_agent.run(
                    prompt,
                    deps=deps,
                    message_history=None,
                    usage_limits=UsageLimits(request_limit=10),
                )
                plan_text = plan_result.output
            except Exception as e2:
                _handle_agent_error(console, e2, deps=deps)
                return message_history
        else:
            _handle_agent_error(console, e, deps=deps)
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

    # Capture accepted plan as exemplar (cloud planning success)
    if deps.memory_db and plan_model != f"ollama:{settings.ollama.heavy_model}":
        try:
            from forge.agent.exemplars import capture_exemplar
            await capture_exemplar(
                deps.memory_db,
                deps.memory_project or "default",
                "planning",
                prompt,
                plan_text,
                "gemini",
                outcome_score=0.8,
            )
        except Exception:
            logger.debug("Plan exemplar capture failed", exc_info=True)

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
    headless: bool = False,
    headless_max_turns: int = 50,
    headless_request_limit: int = 200,
) -> None:
    """Run the agentic REPL with tool use.

    When headless=True, auto-feeds "Continue building." prompts instead of
    reading from stdin. Exits after headless_max_turns or when the agent
    reports completion. Uses a higher request_limit to avoid hitting the
    per-turn cap during large builds.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
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
        wire_critique_hooks,
        wire_dynamic_prompts,
        wire_lint_hooks,
        wire_rag_hooks,
        wire_syntax_hooks,
        wire_test_hooks,
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
        cloud_reasoning_enabled=settings.gemini.enabled,
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
    wire_test_hooks(hook_registry, deps)
    wire_critique_hooks(hook_registry, deps)

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

    history_path = Path.home() / ".config" / "forge" / "history.txt"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)), key_bindings=kb,
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

    # --- Headless: raise per-turn request limit and enable auto-background ---
    _original_request_limit = settings.agent.request_limit
    _original_bg_threshold = settings.agent.run_command_background_threshold
    if headless:
        settings.agent.request_limit = headless_request_limit  # type: ignore[assignment]
        # Auto-background commands after 15s to prevent server-start hangs
        if settings.agent.run_command_background_threshold <= 0:
            settings.agent.run_command_background_threshold = 15.0  # type: ignore[assignment]

    # --- Initial prompt ---
    turn_counter = 0
    if initial_prompt:
        turn_counter += 1
        turn_counter_ref[0] = turn_counter
        console.print(f"\n[bold]> {initial_prompt}[/bold]")
        try:
            message_history = await _execute_turn(
                agent, initial_prompt, deps, message_history, turn_counter,
                db, session_id, is_initial_turn=True,
            )
        except Exception as e:
            _handle_agent_error(console, e, deps=deps)
            await _handle_exemplar_failure(deps, e)

    # --- REPL loop ---
    title_set = initial_prompt is not None
    headless_turns_used = 0
    headless_verify_count = 0
    headless_build_turns = 0  # Tracks how many build-phase turns we've sent
    try:
        while True:
            if headless:
                # Headless mode: auto-feed prompts that drive toward completion
                headless_turns_used += 1
                if headless_turns_used > headless_max_turns:
                    console.print("\n[dim]Headless: max turns reached.[/dim]")
                    break

                # Decide: build prompt or verification prompt?
                # Check if the model's last response mentions all phases complete
                # or if we've sent enough build turns.
                should_verify = False

                if headless_build_turns >= 2:
                    # After 2+ build turns, check if model thinks it's done
                    if message_history:
                        from pydantic_ai.messages import ModelResponse, TextPart
                        for msg in reversed(message_history[-3:]):
                            if isinstance(msg, ModelResponse):
                                text = " ".join(
                                    p.content for p in msg.parts
                                    if isinstance(p, TextPart)
                                )
                                # Model signals it's done building
                                if any(phrase in text.lower() for phrase in [
                                    "all phases", "phase 5", "phase 4",
                                    "complete", "finished", "done",
                                    "verification_complete",
                                ]):
                                    should_verify = True
                                break

                if should_verify or headless_turns_used > 5:
                    # Verification loop: the key insight from telehealth-booking
                    headless_verify_count += 1
                    console.print(f"\n[cyan]Headless: verification pass {headless_verify_count}[/cyan]")
                    user_input = (
                        "Verify the completion of the project against BUILD_PLAN.md. "
                        "Read BUILD_PLAN.md, then check what actually exists on disk:\n"
                        "1. Run `pnpm build` — does it pass with 0 errors?\n"
                        "2. Run `pnpm test` — do tests exist AND pass? If no test files exist, write them.\n"
                        "3. Check each phase in BUILD_PLAN.md — is every item actually implemented "
                        "with real code (not stubs)?\n"
                        "4. Check for common issues: port conflicts between API and web, "
                        "missing CORS configuration, missing error handling for duplicate entries, "
                        "empty packages with no real code, missing database migrations, "
                        ".bak files or compiled .js/.d.ts artifacts in source directories.\n"
                        "5. Verify the API server starts: `timeout 5 node apps/api/dist/main.js 2>&1 || true`\n\n"
                        "If ANYTHING is incomplete or broken, fix it now — write the missing code, "
                        "add the missing tests, fix the config issues. Do NOT skip items.\n"
                        "Only when the build passes, tests pass, and every BUILD_PLAN item is "
                        "implemented, say the exact phrase VERIFICATION_COMPLETE in your response text."
                    )

                    # Check if model said VERIFICATION_COMPLETE in its actual response
                    # (not in command output — only in ModelResponse TextParts)
                    if message_history and headless_verify_count >= 3:
                        from pydantic_ai.messages import ModelResponse, TextPart
                        verified = False
                        for msg in reversed(message_history[-3:]):
                            if isinstance(msg, ModelResponse):
                                for part in msg.parts:
                                    if isinstance(part, TextPart) and "VERIFICATION_COMPLETE" in part.content:
                                        verified = True
                                        break
                            if verified:
                                break
                        if verified:
                            console.print("\n[dim]Headless: verification complete.[/dim]")
                            break
                else:
                    # Build phase: keep building
                    headless_build_turns += 1
                    user_input = (
                        "Continue building. Work through the BUILD_PLAN phases. "
                        "Write all files for the current phase, run build + tests, "
                        "fix any errors, commit, then move to the next phase."
                    )
            else:
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

            # Progressive tool result aging — every 5 turns, truncate old results
            if message_history and turn_counter % 5 == 0:
                from forge.agent.context import age_tool_results
                message_history = age_tool_results(message_history)

            # Auto-compact if history is getting large
            budget = settings.agent.token_budget
            if message_history:
                # Prefer real token count from last Ollama response over estimation
                if deps._last_prompt_eval_count > 0:
                    tokens = deps._last_prompt_eval_count
                else:
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

                # Update circuit breaker with current message count for fork tracking
                if deps.circuit_breaker and message_history:
                    deps.circuit_breaker.set_message_count(len(message_history))

                message_history = await _execute_turn(
                    agent, user_input, deps, message_history, turn_counter,
                    db, session_id,
                )

                # Update conversation summary for sub-agent context
                if message_history:
                    from forge.agent.context import summarize_for_delegation
                    deps._conversation_summary = summarize_for_delegation(message_history)

                # Persist task store
                if db and deps.task_store:
                    try:
                        await db.save_task_store(session_id, deps.task_store.to_json())
                    except Exception:
                        logger.debug("Task store save failed", exc_info=True)
            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted.[/dim]")
            except Exception as e:
                _handle_agent_error(console, e, deps=deps)
                await _handle_exemplar_failure(deps, e)

                # Gemini recovery: retry with fresh cloud agent (no history leak)
                if deps._cloud_recovery_pending:
                    deps._cloud_recovery_pending = False
                    recovery_result = await _cloud_recovery(
                        user_input, deps, console, turn_counter,
                    )
                    if recovery_result is not None:
                        if message_history is None:
                            message_history = recovery_result
                        else:
                            # Fork: remove toxic loop tail, then append recovery
                            message_history = _fork_history(
                                message_history,
                                deps.circuit_breaker,
                                settings.agent.cb_identical,
                            )
                            message_history.extend(recovery_result)

    finally:
        if headless:
            settings.agent.request_limit = _original_request_limit  # type: ignore[assignment]
            settings.agent.run_command_background_threshold = _original_bg_threshold  # type: ignore[assignment]
        await cleanup(
            deps, hook_registry, mcp_stack, db, session_id, message_history, console,
        )


