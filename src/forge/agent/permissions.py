"""Permission system for agent tool calls.

Supports a 5-stage permission waterfall:
  1. Deny rules  → BLOCK
  2. Ask rules   → prompt user
  3. Allow rules → ALLOW
  4. Mode-based  → YOLO/AUTO/ASK fallback
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console


class PermissionPolicy(Enum):
    AUTO = "auto"  # Reads auto-allowed, writes/commands prompt
    ASK = "ask"  # Prompt for everything
    YOLO = "yolo"  # Allow everything without prompting


SAFE_TOOLS = {"read_file", "search_code", "list_files", "web_search", "web_fetch"}
DANGEROUS_TOOLS = {"write_file", "edit_file", "run_command"}


# ---------------------------------------------------------------------------
# Rule-based permission matching
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PermissionRule:
    """A parsed permission rule like 'run_command(git:*)'."""

    tool: str
    match_type: str = "any"  # "any", "exact", "prefix"
    match_value: str = ""


@dataclass
class PermissionRuleSet:
    """Ordered sets of allow/deny/ask rules."""

    allow: list[PermissionRule] = field(default_factory=list)
    deny: list[PermissionRule] = field(default_factory=list)
    ask: list[PermissionRule] = field(default_factory=list)


def parse_permission_rule(rule_str: str) -> PermissionRule:
    """Parse a rule string into a PermissionRule.

    Formats:
      "tool_name"             → match any invocation of tool_name
      "tool_name(exact_val)"  → match exact subject value
      "tool_name(prefix:*)"   → match subject starting with prefix
    """
    rule_str = rule_str.strip()
    if "(" not in rule_str:
        return PermissionRule(tool=rule_str)

    tool, _, rest = rule_str.partition("(")
    pattern = rest.rstrip(")")
    tool = tool.strip()

    if pattern.endswith(":*"):
        return PermissionRule(tool=tool, match_type="prefix", match_value=pattern[:-2])
    return PermissionRule(tool=tool, match_type="exact", match_value=pattern)


def extract_permission_subject(tool_name: str, args: dict) -> str:
    """Extract the subject string for rule matching from tool args."""
    if tool_name == "run_command":
        return args.get("command", "")
    if tool_name in ("write_file", "edit_file", "read_file", "list_files"):
        return args.get("file_path", args.get("path", ""))
    if tool_name == "web_fetch":
        return args.get("url", "")
    if tool_name == "web_search":
        return args.get("query", "")
    return ""


def _matches_rule(rule: PermissionRule, tool_name: str, subject: str) -> bool:
    """Check if a rule matches a tool invocation."""
    if rule.tool != tool_name:
        return False
    if rule.match_type == "any":
        return True
    if rule.match_type == "exact":
        return subject == rule.match_value
    if rule.match_type == "prefix":
        return subject.startswith(rule.match_value)
    return False


def authorize(
    tool_name: str,
    args: dict,
    policy: PermissionPolicy,
    rules: PermissionRuleSet,
) -> str:
    """5-stage permission waterfall. Returns 'allow', 'block', or 'ask'.

    Stage 1: deny rules → block
    Stage 2: ask rules  → ask
    Stage 3: allow rules → allow
    Stage 4: mode-based fallback (YOLO/AUTO/ASK)
    """
    subject = extract_permission_subject(tool_name, args)

    # Stage 1: deny rules
    for rule in rules.deny:
        if _matches_rule(rule, tool_name, subject):
            return "block"

    # Stage 2: ask rules
    for rule in rules.ask:
        if _matches_rule(rule, tool_name, subject):
            return "ask"

    # Stage 3: allow rules
    for rule in rules.allow:
        if _matches_rule(rule, tool_name, subject):
            return "allow"

    # Stage 4: mode-based fallback
    if policy == PermissionPolicy.YOLO:
        return "allow"
    if policy == PermissionPolicy.AUTO and tool_name in SAFE_TOOLS:
        return "allow"
    if policy == PermissionPolicy.AUTO and tool_name in DANGEROUS_TOOLS:
        return "ask"
    if policy == PermissionPolicy.ASK:
        return "ask"

    return "ask"


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

    loop = asyncio.get_running_loop()
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
