"""Verify specific gaps found in the trace analysis.

Targeted tests for:
1. ESPN-style content: long enough to pass threshold but useless (no data, just headlines)
2. JS-heavy sites returning shell HTML
3. The search budget being exceeded
4. Knowledge questions using web unnecessarily

Usage: uv run python tests/manual/test_verify_gaps.py
"""

import asyncio
import json
import os
import time
from pathlib import Path

from rich.console import Console

os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:3456/v1")

from forge.agent.loop import create_agent, AGENT_SYSTEM
from forge.agent.deps import AgentDeps
from pydantic_ai.usage import UsageLimits
from pydantic_ai.messages import (
    ModelResponse,
    ModelRequest,
    ToolCallPart,
    ToolReturnPart,
    TextPart,
)


def count_tool_calls(messages: list) -> dict:
    """Count tool calls by name."""
    counts = {}
    calls_detail = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    counts[part.tool_name] = counts.get(part.tool_name, 0) + 1
                    args = part.args
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    calls_detail.append((part.tool_name, args))
    return counts, calls_detail


def check_retries(messages: list) -> list[str]:
    """Check if the model retried URLs that returned errors."""
    errored_urls = set()
    retries = []

    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == "web_fetch":
                    content = part.content if isinstance(part.content, str) else ""
                    if content.startswith("Error"):
                        # Extract URL from the tool return — look for the URL in the error
                        # Error format: "Error: HTTP 401 fetching URL. Do NOT..."
                        for word in content.split():
                            if word.startswith("http"):
                                url = word.rstrip(".,;")
                                errored_urls.add(url)
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_name == "web_fetch":
                    args = part.args
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    url = args.get("url", "") if isinstance(args, dict) else ""
                    if url in errored_urls:
                        retries.append(url)

    return retries


async def test_gap(name: str, prompt: str, agent, cwd: Path, console: Console, display: Console):
    """Run a single gap test."""
    display.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    display.print(f"[bold cyan]GAP TEST: {name}[/bold cyan]")
    display.print(f"[dim]Prompt: {prompt}[/dim]")

    deps = AgentDeps(cwd=cwd, console=console)
    start = time.time()

    try:
        result = await agent.run(
            prompt,
            deps=deps,
            usage_limits=UsageLimits(request_limit=15),
        )
        elapsed = time.time() - start
        messages = result.all_messages()
        counts, details = count_tool_calls(messages)
        retries = check_retries(messages)
        usage = result.usage()

        display.print(f"[bold]Time:[/bold] {elapsed:.1f}s | [bold]Requests:[/bold] {usage.requests}")
        display.print(f"[bold]Tool calls:[/bold] {counts}")

        display.print(f"[bold]Call sequence:[/bold]")
        for tool_name, args in details:
            if tool_name == "web_search":
                display.print(f"  [yellow]web_search[/yellow] → {args.get('query', '?')!r}")
            elif tool_name == "web_fetch":
                display.print(f"  [blue]web_fetch[/blue] → {args.get('url', '?')!r}")
            else:
                display.print(f"  [dim]{tool_name}[/dim]")

        if retries:
            display.print(f"  [red]RETRIED ERRORED URLs: {retries}[/red]")
        else:
            display.print(f"  [green]No URL retries detected[/green]")

        display.print(f"\n[bold]Answer ({len(result.output)} chars):[/bold]")
        display.print(result.output[:500])
        if len(result.output) > 500:
            display.print("...")

        return {
            "name": name,
            "counts": counts,
            "details": details,
            "retries": retries,
            "requests": usage.requests,
            "answer_len": len(result.output),
            "elapsed": elapsed,
        }
    except Exception as e:
        display.print(f"[red]ERROR: {e}[/red]")
        return {"name": name, "error": str(e)}


async def main():
    console = Console(stderr=True)
    display = Console()
    cwd = Path.cwd()
    agent = create_agent(system=AGENT_SYSTEM, cwd=cwd)

    results = []

    # Gap 1: Knowledge question that shouldn't need web
    r = await test_gap(
        "knowledge_no_web",
        "What year did World War 2 end?",
        agent, cwd, console, display,
    )
    results.append(r)

    # Gap 2: Knowledge question that the model previously searched for
    r = await test_gap(
        "knowledge_circumnavigation",
        "Who was the first person to circumnavigate the globe?",
        agent, cwd, console, display,
    )
    results.append(r)

    # Gap 3: Simple factual question
    r = await test_gap(
        "knowledge_capital",
        "What is the capital of France?",
        agent, cwd, console, display,
    )
    results.append(r)

    # Gap 4: Multi-part question — does the model stay within budget?
    r = await test_gap(
        "multi_part_budget",
        "What's the current temperature in London and what causes the northern lights?",
        agent, cwd, console, display,
    )
    results.append(r)

    # Gap 5: Query where snippets should suffice but model might over-fetch
    r = await test_gap(
        "snippet_sufficient",
        "How tall is Mount Everest?",
        agent, cwd, console, display,
    )
    results.append(r)

    # Gap 6: Query with a known-blocked site in search results
    r = await test_gap(
        "blocked_sites",
        "Best hiking trails near San Diego",
        agent, cwd, console, display,
    )
    results.append(r)

    # Summary
    display.print(f"\n\n[bold]{'='*60}[/bold]")
    display.print("[bold]GAP ANALYSIS SUMMARY[/bold]")
    display.print(f"[bold]{'='*60}[/bold]\n")

    for r in results:
        if "error" in r:
            display.print(f"[red]{r['name']}: ERROR - {r['error']}[/red]")
            continue

        issues = []
        web_searches = r["counts"].get("web_search", 0)
        web_fetches = r["counts"].get("web_fetch", 0)

        if r["name"].startswith("knowledge") and (web_searches > 0 or web_fetches > 0):
            issues.append(f"Used web for knowledge question ({web_searches}S/{web_fetches}F)")
        if web_searches > 2:
            issues.append(f"Excess searches: {web_searches}")
        if web_fetches > 3:
            issues.append(f"Excess fetches: {web_fetches}")
        if r["retries"]:
            issues.append(f"Retried errored URLs: {r['retries']}")

        status = "[red]ISSUES[/red]" if issues else "[green]OK[/green]"
        display.print(
            f"{status} {r['name']}: {r['requests']} req, "
            f"{r['counts'].get('web_search', 0)}S/{r['counts'].get('web_fetch', 0)}F, "
            f"{r['elapsed']:.1f}s, {r['answer_len']} chars"
        )
        for issue in issues:
            display.print(f"  [yellow]⚠ {issue}[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
