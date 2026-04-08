"""Forge CLI — REPL + commands via typer."""

from __future__ import annotations

import asyncio

import typer

from forge import __version__
from forge.prompts.system import CHAT_SYSTEM

from ._helpers import console, resolve_route

app = typer.Typer(
    name="forge",
    help="Local AI orchestration — smart routing, RAG, and self-correction.",
    no_args_is_help=False,
    invoke_without_command=True,
)


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
    worktree: bool = typer.Option(False, "--worktree", help="Create isolated git worktree"),
    worktree_name: str = typer.Option(None, "--worktree-name", help="Name for the worktree"),
    local: bool = typer.Option(False, "--local", help="Local-only mode (no cloud calls)"),
    max_mode: bool = typer.Option(False, "--max", help="Max mode (cloud-orchestrated with Opus)"),
    prompt: str = typer.Argument(None, help="Optional initial prompt (shortcut for 'forge agent \"prompt\"')"),
) -> None:
    """Forge — local AI orchestration."""
    if version:
        console.print(f"forge {__version__}")
        raise typer.Exit()

    if local and max_mode:
        console.print("[red]Cannot use --local and --max together.[/red]")
        raise typer.Exit(1)

    if local or max_mode:
        from forge.config import apply_mode
        apply_mode("local" if local else "max")

    if ctx.invoked_subcommand is None:
        from forge.agent.loop import agent_repl

        wt_name = worktree_name if worktree or worktree_name else None
        asyncio.run(agent_repl(initial_prompt=prompt, system=CHAT_SYSTEM, worktree_name=wt_name))


@app.command()
def agent(
    prompt: str = typer.Argument(None, help="Optional initial prompt"),
    yolo: bool = typer.Option(False, "--yolo", help="Allow all tool calls without prompting"),
    ask: bool = typer.Option(False, "--ask", help="Prompt for every tool call"),
    worktree: bool = typer.Option(False, "--worktree", help="Create isolated git worktree"),
    worktree_name: str = typer.Option(None, "--worktree-name", help="Name for the worktree"),
    local: bool = typer.Option(False, "--local", help="Local-only mode (no cloud calls)"),
    max_mode: bool = typer.Option(False, "--max", help="Max mode (cloud-orchestrated with Opus)"),
) -> None:
    """Agentic coding mode — read, edit, search, and run commands."""
    from forge.agent.loop import agent_repl
    from forge.agent.permissions import PermissionPolicy

    if local and max_mode:
        console.print("[red]Cannot use --local and --max together.[/red]")
        raise typer.Exit(1)

    if local or max_mode:
        from forge.config import apply_mode
        apply_mode("local" if local else "max")

    if yolo:
        policy = PermissionPolicy.YOLO
    elif ask:
        policy = PermissionPolicy.ASK
    else:
        policy = PermissionPolicy.AUTO

    wt_name = worktree_name if worktree or worktree_name else None
    asyncio.run(agent_repl(prompt, permission=policy, worktree_name=wt_name))


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Question or prompt"),
    gpu: bool = typer.Option(False, "--gpu", help="Force heavy GPU model"),
    fast: bool = typer.Option(False, "--fast", help="Force fast GPU model"),
    npu: bool = typer.Option(False, "--npu", help="Force NPU model"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming"),
) -> None:
    """One-shot query — ask a question and get a response."""
    from ._streaming import one_shot

    force = resolve_route(gpu, fast, npu)
    asyncio.run(one_shot(prompt, force_route=force, stream=not no_stream))


@app.command()
def code(
    prompt: str = typer.Argument(None, help="Optional initial prompt"),
    gpu: bool = typer.Option(False, "--gpu", help="Force heavy GPU model"),
    fast: bool = typer.Option(False, "--fast", help="Force fast GPU model"),
    npu: bool = typer.Option(False, "--npu", help="Force NPU model"),
    project: str = typer.Option(None, "--project", "-p", help="Project name (defaults to cwd name)"),
) -> None:
    """REPL with codebase RAG context from indexed project."""
    from ._code import code_command

    asyncio.run(code_command(prompt, gpu, fast, npu, project))


@app.command()
def status() -> None:
    """Show model status and configuration."""
    from ._status import status_command

    status_command()


@app.command()
def history(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of sessions to show"),
) -> None:
    """List recent conversation sessions."""
    from ._sessions import history_command

    asyncio.run(history_command(limit))


@app.command()
def resume(
    session_id: str = typer.Argument(None, help="Session ID to resume (default: most recent)"),
) -> None:
    """Resume a previous conversation session."""
    from ._sessions import resume_command

    asyncio.run(resume_command(session_id))


@app.command()
def index(
    path: str = typer.Argument(".", help="Directory to index"),
    project: str = typer.Option(None, "--project", "-p", help="Project name (defaults to directory name)"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-index all files (ignore hashes)"),
) -> None:
    """Index a directory for RAG — walk, chunk, embed, store."""
    from ._index import index_command

    asyncio.run(index_command(path, project, force))


@app.command()
def draft(
    prompt: str = typer.Argument(..., help="What to build or generate"),
    project: str = typer.Option(None, "--project", "-p", help="Project for RAG context"),
    show_stages: bool = typer.Option(False, "--show-stages", "-s", help="Show draft and critique"),
) -> None:
    """Draft → Critique → Refine pipeline — two-model orchestration."""
    from ._draft import draft_command

    asyncio.run(draft_command(prompt, project, show_stages))


@app.command()
def run(
    prompt: str = typer.Argument(..., help="What code to generate and execute"),
    retries: int = typer.Option(3, "--retries", "-r", help="Max retry attempts on failure"),
    timeout: float = typer.Option(30.0, "--timeout", "-t", help="Execution timeout in seconds"),
    project: str = typer.Option(None, "--project", "-p", help="Project for RAG context"),
) -> None:
    """Generate code, execute it, and self-correct on errors."""
    from ._run import run_command

    asyncio.run(run_command(prompt, retries, timeout, project))


@app.command()
def serve(
    cwd: str = typer.Option(None, "--cwd", help="Working directory for MCP tools"),
) -> None:
    """Start Forge as an MCP server (stdio transport)."""
    from forge.mcp_server import run as mcp_run

    mcp_run(cwd=cwd)
