"""Manual test: trace every tool call for each prompt to find behavioral gaps.

Usage: uv run python tests/manual/test_trace_prompts.py
"""

import asyncio
import os
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:3456/v1")

from forge.agent.loop import create_agent, AGENT_SYSTEM
from forge.agent.deps import AgentDeps
from pydantic_ai.usage import UsageLimits
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    TextPart,
)


PROMPTS = {
    "web/news": "What are the biggest tech news stories today?",
    "web/sports": "What were the latest NBA scores?",
    "web/science": "What is the current status of the Artemis moon program?",
    "web/finance": "What is the current price of Bitcoin?",
    "web/health": "What are the CDC's current COVID recommendations?",
    "web/local": "Best restaurants in downtown San Diego",
    "code/read": "Read pyproject.toml and tell me the project name and version",
    "code/explain": "Explain what src/forge/agent/tools.py does in 3 sentences",
    "code/search": "Find all files that import httpx in this project",
    "code/structure": "Show me the project directory structure",
    "knowledge/math": "What is the derivative of x^3 * sin(x)?",
    "knowledge/history": "Who was the first person to circumnavigate the globe?",
    "knowledge/explain": "Explain the difference between TCP and UDP in simple terms",
    "edge/vague": "help",
    "edge/short": "ping",
    "edge/mixed": "What's the weather in Tokyo and also explain what a monad is",
    "edge/error_prone": "Fetch the contents of https://httpstat.us/503",
}


def extract_trace(messages: list) -> list[dict]:
    """Extract a structured trace of tool calls and responses from message history."""
    trace = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    args = part.args
                    if isinstance(args, str):
                        import json
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    trace.append({
                        "type": "tool_call",
                        "tool": part.tool_name,
                        "args_summary": _summarize_args(part.tool_name, args),
                    })
                elif isinstance(part, TextPart):
                    text = part.content.strip()
                    if text:
                        trace.append({
                            "type": "text",
                            "preview": text[:150] + ("..." if len(text) > 150 else ""),
                            "length": len(text),
                        })
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    content = part.content
                    if isinstance(content, str):
                        trace.append({
                            "type": "tool_result",
                            "tool": part.tool_name,
                            "result_preview": content[:200] + ("..." if len(content) > 200 else ""),
                            "result_length": len(content),
                            "has_error": content.startswith("Error") or "Warning:" in content[:50],
                        })

    return trace


def _summarize_args(tool_name: str, args) -> str:
    if not isinstance(args, dict):
        return str(args)[:100]
    if tool_name == "web_search":
        return f"query={args.get('query', '?')!r}"
    elif tool_name == "web_fetch":
        return f"url={args.get('url', '?')!r}"
    elif tool_name == "read_file":
        return f"path={args.get('file_path', '?')!r}"
    elif tool_name == "search_code":
        return f"pattern={args.get('pattern', '?')!r} path={args.get('path', '.')!r}"
    elif tool_name == "list_files":
        return f"pattern={args.get('pattern', '**/*')!r} path={args.get('path', '.')!r}"
    elif tool_name == "run_command":
        return f"cmd={args.get('command', '?')!r}"
    elif tool_name == "write_file":
        return f"path={args.get('file_path', '?')!r}"
    elif tool_name == "edit_file":
        return f"path={args.get('file_path', '?')!r}"
    return str(args)[:100]


async def run_traced(prompt: str, agent, cwd: Path, console: Console) -> dict:
    """Run a single prompt and return trace + metrics."""
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
        trace = extract_trace(messages)
        usage = result.usage()

        return {
            "status": "ok",
            "answer": result.output,
            "trace": trace,
            "requests": usage.requests,
            "tokens_in": usage.request_tokens,
            "tokens_out": usage.response_tokens,
            "elapsed": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "status": f"ERROR: {type(e).__name__}: {e}",
            "answer": "",
            "trace": [],
            "requests": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "elapsed": elapsed,
        }


def analyze_trace(category: str, result: dict) -> list[str]:
    """Analyze a trace for potential issues. Returns list of findings."""
    findings = []
    trace = result["trace"]

    # Count tool calls by type
    tool_calls = [t for t in trace if t["type"] == "tool_call"]
    tool_results = [t for t in trace if t["type"] == "tool_result"]
    errors = [t for t in tool_results if t.get("has_error")]

    web_searches = [t for t in tool_calls if t["tool"] == "web_search"]
    web_fetches = [t for t in tool_calls if t["tool"] == "web_fetch"]
    file_reads = [t for t in tool_calls if t["tool"] == "read_file"]
    code_searches = [t for t in tool_calls if t["tool"] == "search_code"]
    file_lists = [t for t in tool_calls if t["tool"] == "list_files"]
    commands = [t for t in tool_calls if t["tool"] == "run_command"]

    # Check: web search used for knowledge questions that don't need it
    is_knowledge = category.startswith("knowledge/")
    if is_knowledge and (web_searches or web_fetches):
        findings.append(
            f"UNNECESSARY_WEB: Knowledge question used web tools "
            f"({len(web_searches)} searches, {len(web_fetches)} fetches) — "
            f"should answer from training data"
        )

    # Check: excessive web fetches
    if len(web_fetches) > 3:
        findings.append(
            f"EXCESS_FETCH: {len(web_fetches)} web_fetch calls (budget is 2-3)"
        )

    # Check: multiple web searches
    if len(web_searches) > 2:
        findings.append(
            f"EXCESS_SEARCH: {len(web_searches)} web_search calls"
        )

    # Check: errors not respected (fetch after error on same URL)
    error_urls = set()
    for t in trace:
        if t["type"] == "tool_result" and t.get("has_error"):
            # Extract URL from result preview
            preview = t.get("result_preview", "")
            if "fetching" in preview:
                url_start = preview.find("fetching ") + 9
                url = preview[url_start:].split(".")[0] + preview[url_start:].split(".")[0]
                error_urls.add(preview[url_start:url_start+50])
    for t in trace:
        if t["type"] == "tool_call" and t["tool"] == "web_fetch":
            url = t["args_summary"]
            for err_url in error_urls:
                if err_url[:30] in url:
                    findings.append(f"RETRY_AFTER_ERROR: Fetched URL that previously errored: {url[:80]}")

    # Check: suspiciously short answer
    if result["status"] == "ok" and len(result["answer"]) < 20 and category not in ("edge/short", "edge/vague"):
        findings.append(f"SHORT_ANSWER: Only {len(result['answer'])} chars")

    # Check: answer mentions raw tool output or apologizes excessively
    answer = result.get("answer", "").lower()
    if "search results for:" in answer:
        findings.append("RAW_DUMP: Answer contains raw search result headers")
    if "fetched:" in answer and "─" * 10 in result.get("answer", ""):
        findings.append("RAW_DUMP: Answer contains raw fetch output")

    # Check: used run_command for something that has a dedicated tool
    for t in commands:
        cmd = t["args_summary"].lower()
        if "grep" in cmd or "rg " in cmd:
            findings.append(f"WRONG_TOOL: Used run_command for grep instead of search_code: {t['args_summary']}")
        if "find " in cmd:
            findings.append(f"WRONG_TOOL: Used run_command for find instead of list_files: {t['args_summary']}")
        if "cat " in cmd:
            findings.append(f"WRONG_TOOL: Used run_command for cat instead of read_file: {t['args_summary']}")

    # Check: very high total requests
    if result["requests"] > 8:
        findings.append(f"HIGH_REQUESTS: {result['requests']} total requests")

    return findings


async def main():
    console = Console(stderr=True)
    display = Console()
    cwd = Path.cwd()
    agent = create_agent(system=AGENT_SYSTEM, cwd=cwd)

    all_findings = {}
    total = len(PROMPTS)

    for i, (category, prompt) in enumerate(PROMPTS.items(), 1):
        display.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        display.print(f"[bold cyan][{i}/{total}] {category}[/bold cyan]: {prompt}")
        display.print(f"[bold cyan]{'='*60}[/bold cyan]")

        result = await run_traced(prompt, agent, cwd, console)

        # Print trace
        display.print(f"\n[bold]Tool Trace:[/bold] ({result['requests']} requests, {result['elapsed']:.1f}s)")
        for j, t in enumerate(result["trace"]):
            if t["type"] == "tool_call":
                display.print(f"  [yellow]→ {t['tool']}[/yellow]({t['args_summary']})")
            elif t["type"] == "tool_result":
                err = " [red]ERROR[/red]" if t.get("has_error") else ""
                display.print(f"  [dim]← {t['tool']}[/dim] ({t['result_length']} chars){err}")
                if t.get("has_error"):
                    display.print(f"    [red]{t['result_preview'][:120]}[/red]")
            elif t["type"] == "text":
                display.print(f"  [green]✎ text[/green] ({t['length']} chars): {t['preview'][:100]}")

        # Print answer preview
        display.print(f"\n[bold]Answer[/bold] ({len(result.get('answer', ''))} chars):")
        answer = result.get("answer", "")
        if len(answer) > 300:
            display.print(f"  {answer[:300]}...")
        else:
            display.print(f"  {answer}")

        # Analyze
        findings = analyze_trace(category, result)
        all_findings[category] = findings
        if findings:
            for f in findings:
                display.print(f"  [yellow]⚠ {f}[/yellow]")
        else:
            display.print(f"  [green]✓ No issues[/green]")

    # Final summary
    display.print(f"\n\n[bold]{'='*60}[/bold]")
    display.print("[bold]FINDINGS SUMMARY[/bold]")
    display.print(f"[bold]{'='*60}[/bold]\n")

    any_issues = False
    for category, findings in all_findings.items():
        if findings:
            any_issues = True
            display.print(f"[cyan]{category}[/cyan]:")
            for f in findings:
                display.print(f"  [yellow]⚠ {f}[/yellow]")

    if not any_issues:
        display.print("[bold green]No issues found across all prompts.[/bold green]")
    else:
        # Categorize findings
        finding_types = {}
        for category, findings in all_findings.items():
            for f in findings:
                ftype = f.split(":")[0]
                finding_types.setdefault(ftype, []).append(f"{category}: {f}")

        display.print(f"\n[bold]By finding type:[/bold]")
        for ftype, items in sorted(finding_types.items()):
            display.print(f"\n  [yellow]{ftype}[/yellow] ({len(items)} occurrences):")
            for item in items:
                display.print(f"    {item}")


if __name__ == "__main__":
    asyncio.run(main())
