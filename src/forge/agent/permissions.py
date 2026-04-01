"""Permission system for agent tool calls."""

from __future__ import annotations

import asyncio
from enum import Enum

from rich.console import Console


class PermissionPolicy(Enum):
    AUTO = "auto"  # Reads auto-allowed, writes/commands prompt
    ASK = "ask"  # Prompt for everything
    YOLO = "yolo"  # Allow everything without prompting


SAFE_TOOLS = {"read_file", "search_code", "list_files", "web_search", "web_fetch"}
DANGEROUS_TOOLS = {"write_file", "edit_file", "run_command"}


async def check_permission(
    console: Console,
    policy: PermissionPolicy,
    tool_name: str,
    args: dict,
) -> bool:
    """Check if a tool call is allowed under the current policy.

    Returns True if allowed, False if denied.
    """
    if policy == PermissionPolicy.YOLO:
        return True

    if policy == PermissionPolicy.AUTO and tool_name in SAFE_TOOLS:
        return True

    # ASK policy, or dangerous tool under AUTO — prompt user
    summary = _summarize_call(tool_name, args)
    return await _prompt_user(console, tool_name, summary)


async def _prompt_user(console: Console, tool_name: str, summary: str) -> bool:
    """Ask the user whether to allow a tool call."""
    style = "yellow" if tool_name in {"write_file", "edit_file"} else "red"
    console.print(f"\n[{style}]Allow {tool_name}?[/{style}] {summary}")

    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None, lambda: input("[y]es / [n]o > ").strip().lower()
        )
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]Denied.[/dim]")
        return False

    return response in ("y", "yes", "")


def _summarize_call(tool_name: str, args: dict) -> str:
    """Create a short human-readable summary of a tool call."""
    if tool_name == "write_file":
        path = args.get("file_path", "?")
        content = args.get("content", "")
        lines = content.count("\n") + 1
        return f"[dim]{path}[/dim] ({lines} lines)"
    elif tool_name == "edit_file":
        path = args.get("file_path", "?")
        return f"[dim]{path}[/dim]"
    elif tool_name == "run_command":
        cmd = args.get("command", "?")
        return f"[dim]$ {cmd}[/dim]"
    elif tool_name in ("read_file", "list_files"):
        path = args.get("file_path", args.get("path", "?"))
        return f"[dim]{path}[/dim]"
    elif tool_name == "search_code":
        pattern = args.get("pattern", "?")
        return f"[dim]/{pattern}/[/dim]"
    elif tool_name == "web_search":
        query = args.get("query", "?")
        return f"[dim]🔍 {query}[/dim]"
    elif tool_name == "web_fetch":
        url = args.get("url", "?")
        return f"[dim]🌐 {url}[/dim]"
    return ""
