"""Command safety — blocklist hook and path boundary enforcement."""

from __future__ import annotations

import re
from pathlib import Path

from forge.agent.hooks import HookAction, HookResult, PreToolUse
from forge.log import get_logger

logger = get_logger(__name__)

# File tools that take a file_path argument
FILE_TOOLS = frozenset({"read_file", "write_file", "edit_file"})


# ── Pure check functions (reusable outside hooks, e.g. in MCP server) ──

def check_command_blocklist(command: str) -> str | None:
    """Check a command against the sandbox blocklist.

    Returns an error message if blocked, None if allowed.
    """
    from forge.config import settings

    if not settings.sandbox.enabled:
        return None

    segments = _split_command_segments(command)
    check_targets = [command] + [s for s in segments if s != command]

    for target in check_targets:
        for pattern in settings.sandbox.blocked_patterns:
            try:
                if re.search(pattern, target):
                    detail = f"Matched segment: `{target}`\n" if target != command else ""
                    return (
                        f"Command blocked by safety policy: `{command}`\n"
                        f"{detail}"
                        f"Matched pattern: `{pattern}`"
                    )
            except re.error:
                pass

    # Infinite loop detection
    if re.search(r"\bwhile\s+true\s*;?\s*do\b", command) or re.search(
        r"\bfor\s*\(\s*;\s*;\s*\)", command
    ):
        return f"Blocked: infinite loop detected in `{command[:80]}`."

    return None


def check_path_boundary(file_path: str, cwd: Path) -> str | None:
    """Check if a file path is within allowed boundaries.

    Returns an error message if outside boundaries, None if allowed.
    """
    from forge.config import settings

    if not settings.sandbox.enabled or not settings.sandbox.restrict_paths:
        return None

    p = Path(file_path)
    if not p.is_absolute():
        p = (cwd / p).resolve()
    else:
        p = p.resolve()

    resolved_cwd = cwd.resolve()
    allowed_roots = [resolved_cwd, Path("/tmp")]
    for extra in settings.sandbox.allowed_paths:
        allowed_roots.append(Path(extra).resolve())

    for root in allowed_roots:
        try:
            p.relative_to(root)
            return None
        except ValueError:
            continue

    return (
        f"Path outside allowed directories: `{p}`\n"
        f"Allowed: {resolved_cwd}, /tmp"
        + (f", {', '.join(settings.sandbox.allowed_paths)}" if settings.sandbox.allowed_paths else "")
    )


def _split_command_segments(command: str) -> list[str]:
    """Split a compound command into individual segments.

    Splits on &&, ||, ;, | while respecting quoted strings.
    Returns each segment stripped.
    """
    # Split on unquoted shell operators: ;, &&, ||, |
    # Walk character by character to respect quotes
    segments: list[str] = []
    current: list[str] = []
    i = 0
    in_single = False
    in_double = False

    while i < len(command):
        c = command[i]

        # Track quote state
        if c == "'" and not in_double:
            in_single = not in_single
            current.append(c)
            i += 1
            continue
        if c == '"' and not in_single:
            in_double = not in_double
            current.append(c)
            i += 1
            continue

        # Only split when outside quotes
        if not in_single and not in_double:
            # Check for && or ||
            if i + 1 < len(command) and command[i : i + 2] in ("&&", "||"):
                seg = "".join(current).strip()
                if seg:
                    segments.append(seg)
                current = []
                i += 2
                continue
            # Check for ; or |
            if c in (";", "|"):
                seg = "".join(current).strip()
                if seg:
                    segments.append(seg)
                current = []
                i += 1
                continue

        current.append(c)
        i += 1

    # Last segment
    seg = "".join(current).strip()
    if seg:
        segments.append(seg)

    return segments if segments else [command]


def make_command_blocklist_handler():
    """Create a PreToolUse handler that blocks dangerous commands.

    Delegates core blocklist/infinite-loop checks to ``check_command_blocklist``
    (shared with MCP server), then adds hook-only extras: logging, sleep
    detection, warn patterns.
    """

    async def _handler(event: PreToolUse) -> HookResult:
        from forge.config import settings

        if not settings.sandbox.enabled:
            return HookResult()

        if event.tool_name != "run_command":
            return HookResult()

        command = event.args.get("command", "")
        if not command:
            return HookResult()

        # Core blocklist + infinite loop check (shared with MCP server)
        block_msg = check_command_blocklist(command)
        if block_msg:
            logger.warning("Blocked command: %s", command)
            return HookResult(
                action=HookAction.BLOCK,
                message=(
                    block_msg + "\n"
                    "This command could cause irreversible damage. "
                    "If you need to run it, ask the user to do it manually."
                ),
            )

        # Hook-only: sleep detection — check each segment for sleep >= 10s
        segments = _split_command_segments(command)
        check_targets = [command] + [s for s in segments if s != command]
        for target in check_targets:
            sleep_match = re.search(r"\bsleep\s+(\d+)", target)
            if sleep_match and int(sleep_match.group(1)) >= 10:
                return HookResult(
                    action=HookAction.BLOCK,
                    message=(
                        f"Blocked: `sleep {sleep_match.group(1)}` — long sleeps waste agent time. "
                        "Use a shorter duration or a different approach."
                    ),
                )

        # Hook-only: warn patterns (log only, don't block)
        for segment in segments:
            for pattern in settings.sandbox.warn_patterns:
                try:
                    if re.search(pattern, segment):
                        logger.info("Warn-flagged command: %s (pattern: %s)", command, pattern)
                except re.error:
                    pass

        return HookResult()

    return _handler


def make_write_command_detector(deps):
    """Create a PreToolUse handler that detects write-via-command patterns.

    Detects sed -i, awk -i inplace, perl -i in run_command and sets
    a flag on deps so the permission handler can escalate to write-level permission.

    Args:
        deps: AgentDeps instance — the _write_escalated flag is set on it.
    """
    WRITE_CMD_PATTERNS = [
        re.compile(r"\bsed\s+.*-i\b"),
        re.compile(r"\bawk\s+.*-i\s+inplace\b"),
        re.compile(r"\bperl\s+.*-i\b"),
    ]

    async def _handler(event: PreToolUse) -> HookResult:
        if event.tool_name != "run_command":
            return HookResult()
        command = event.args.get("command", "")
        for pattern in WRITE_CMD_PATTERNS:
            if pattern.search(command):
                deps._write_escalated = True
                return HookResult()
        # Reset flag for non-write commands
        deps._write_escalated = False
        return HookResult()

    return _handler


def make_path_boundary_handler(cwd: Path):
    """Create a PreToolUse handler that restricts file tool paths.

    Delegates core boundary check to ``check_path_boundary`` (shared with
    MCP server), then adds logging and user-facing guidance.
    """

    async def _handler(event: PreToolUse) -> HookResult:
        if event.tool_name not in FILE_TOOLS:
            return HookResult()

        file_path = event.args.get("file_path", "")
        if not file_path:
            return HookResult()

        block_msg = check_path_boundary(file_path, cwd)
        if block_msg:
            logger.warning("Path outside boundary: %s (cwd: %s)", file_path, cwd)
            return HookResult(
                action=HookAction.BLOCK,
                message=block_msg + "\nUse paths within the project directory or /tmp.",
            )

        return HookResult()

    return _handler
