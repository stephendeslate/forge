"""Manual test: run diverse prompts through the agent and report results.

Usage: uv run python tests/manual/test_diverse_prompts.py
"""

import asyncio
import os
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:3456/v1")

from forge.agent.loop import create_agent, AGENT_SYSTEM
from forge.agent.deps import AgentDeps
from pydantic_ai.usage import UsageLimits


PROMPTS = {
    # --- Web research ---
    "web/news": "What are the biggest tech news stories today?",
    "web/sports": "What were the latest NBA scores?",
    "web/science": "What is the current status of the Artemis moon program?",
    "web/finance": "What is the current price of Bitcoin?",
    "web/health": "What are the CDC's current COVID recommendations?",
    "web/local": "Best restaurants in downtown San Diego",

    # --- Coding / file tasks ---
    "code/read": "Read pyproject.toml and tell me the project name and version",
    "code/explain": "Explain what src/forge/agent/tools.py does in 3 sentences",
    "code/search": "Find all files that import httpx in this project",
    "code/structure": "Show me the project directory structure",

    # --- General knowledge (no tools needed) ---
    "knowledge/math": "What is the derivative of x^3 * sin(x)?",
    "knowledge/history": "Who was the first person to circumnavigate the globe?",
    "knowledge/explain": "Explain the difference between TCP and UDP in simple terms",

    # --- Ambiguous / edge cases ---
    "edge/vague": "help",
    "edge/short": "ping",
    "edge/mixed": "What's the weather in Tokyo and also explain what a monad is",
    "edge/error_prone": "Fetch the contents of https://httpstat.us/503",
}


async def run_single(prompt: str, agent, cwd: Path, console: Console) -> dict:
    """Run a single prompt and return metrics."""
    deps = AgentDeps(cwd=cwd, console=console)
    start = time.time()

    try:
        result = await agent.run(
            prompt,
            deps=deps,
            usage_limits=UsageLimits(request_limit=15),
        )
        elapsed = time.time() - start
        output = result.output
        usage = result.usage()

        return {
            "status": "ok",
            "answer": output,
            "requests": usage.requests,
            "tokens_in": usage.request_tokens,
            "tokens_out": usage.response_tokens,
            "elapsed": elapsed,
            "answer_len": len(output),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "status": f"ERROR: {type(e).__name__}: {e}",
            "answer": "",
            "requests": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "elapsed": elapsed,
            "answer_len": 0,
        }


async def main():
    console = Console(stderr=True)
    display = Console()  # stdout for results
    cwd = Path.cwd()
    agent = create_agent(system=AGENT_SYSTEM, cwd=cwd)

    results = {}
    total = len(PROMPTS)

    for i, (category, prompt) in enumerate(PROMPTS.items(), 1):
        display.print(f"\n[bold cyan][{i}/{total}] {category}[/bold cyan]: {prompt}")
        result = await run_single(prompt, agent, cwd, console)
        results[category] = result

        # Show quick summary
        if result["status"] == "ok":
            display.print(
                f"  [green]OK[/green] | {result['requests']} requests | "
                f"{result['elapsed']:.1f}s | {result['tokens_out']} tokens out | "
                f"{result['answer_len']} chars"
            )
            # Show first 200 chars of answer
            preview = result["answer"][:200].replace("\n", " ")
            if len(result["answer"]) > 200:
                preview += "..."
            display.print(f"  [dim]{preview}[/dim]")
        else:
            display.print(f"  [red]{result['status']}[/red] ({result['elapsed']:.1f}s)")

    # Summary table
    display.print("\n")
    table = Table(title="Test Results Summary", show_lines=True)
    table.add_column("Category", style="cyan", width=20)
    table.add_column("Status", width=8)
    table.add_column("Requests", justify="right", width=8)
    table.add_column("Time", justify="right", width=8)
    table.add_column("Tokens Out", justify="right", width=10)
    table.add_column("Answer Length", justify="right", width=12)

    issues = []
    for category, r in results.items():
        status_str = "[green]OK[/green]" if r["status"] == "ok" else "[red]FAIL[/red]"
        table.add_row(
            category,
            status_str,
            str(r["requests"]),
            f"{r['elapsed']:.1f}s",
            str(r["tokens_out"]),
            str(r["answer_len"]),
        )

        # Flag potential issues
        if r["status"] != "ok":
            issues.append(f"{category}: {r['status']}")
        elif r["requests"] > 8:
            issues.append(f"{category}: excessive requests ({r['requests']})")
        elif r["answer_len"] < 20:
            issues.append(f"{category}: suspiciously short answer ({r['answer_len']} chars)")

    display.print(table)

    # Issue summary
    if issues:
        display.print(Panel("\n".join(issues), title="[yellow]Potential Issues[/yellow]", border_style="yellow"))
    else:
        display.print("[bold green]All tests passed — no issues detected.[/bold green]")

    # Print full answers for review
    display.print("\n[bold]═══ Full Answers ═══[/bold]\n")
    for category, r in results.items():
        if r["status"] == "ok":
            display.print(Panel(
                r["answer"],
                title=f"[cyan]{category}[/cyan] ({r['requests']} req, {r['elapsed']:.1f}s)",
                border_style="dim",
            ))


if __name__ == "__main__":
    asyncio.run(main())
