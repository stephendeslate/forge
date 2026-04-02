"""Command safety — blocklist hook and path boundary enforcement."""

from __future__ import annotations

import re
from pathlib import Path

from forge.agent.hooks import HookAction, HookResult, PreToolUse
from forge.log import get_logger

logger = get_logger(__name__)

# File tools that take a file_path argument
FILE_TOOLS = frozenset({"read_file", "write_file", "edit_file"})


def make_command_blocklist_handler():
    """Create a PreToolUse handler that blocks dangerous commands.

    Reads patterns from SandboxSettings at call time so config changes
    take effect without restarting.
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

        # Check blocked patterns
        for pattern in settings.sandbox.blocked_patterns:
            try:
                if re.search(pattern, command):
                    logger.warning("Blocked command: %s (pattern: %s)", command, pattern)
                    return HookResult(
                        action=HookAction.BLOCK,
                        message=(
                            f"Command blocked by safety policy: `{command}`\n"
                            f"Matched pattern: `{pattern}`\n"
                            "This command could cause irreversible damage. "
                            "If you need to run it, ask the user to do it manually."
                        ),
                    )
            except re.error:
                logger.debug("Invalid sandbox pattern: %s", pattern)

        # Check warn patterns (log only, don't block)
        for pattern in settings.sandbox.warn_patterns:
            try:
                if re.search(pattern, command):
                    logger.info("Warn-flagged command: %s (pattern: %s)", command, pattern)
            except re.error:
                pass

        return HookResult()

    return _handler


def make_path_boundary_handler(cwd: Path):
    """Create a PreToolUse handler that restricts file tool paths.

    Ensures file operations stay within the working directory, /tmp,
    or explicitly allowed paths.
    """

    async def _handler(event: PreToolUse) -> HookResult:
        from forge.config import settings

        if not settings.sandbox.enabled or not settings.sandbox.restrict_paths:
            return HookResult()

        if event.tool_name not in FILE_TOOLS:
            return HookResult()

        file_path = event.args.get("file_path", "")
        if not file_path:
            return HookResult()

        # Resolve the path
        p = Path(file_path)
        if not p.is_absolute():
            p = (cwd / p).resolve()
        else:
            p = p.resolve()

        resolved_cwd = cwd.resolve()

        # Build allowed roots
        allowed_roots = [resolved_cwd, Path("/tmp")]
        for extra in settings.sandbox.allowed_paths:
            allowed_roots.append(Path(extra).resolve())

        # Check if path is within any allowed root
        for root in allowed_roots:
            try:
                p.relative_to(root)
                return HookResult()
            except ValueError:
                continue

        logger.warning("Path outside boundary: %s (cwd: %s)", p, resolved_cwd)
        return HookResult(
            action=HookAction.BLOCK,
            message=(
                f"Path outside allowed directories: `{p}`\n"
                f"Allowed: {resolved_cwd}, /tmp"
                + (f", {', '.join(settings.sandbox.allowed_paths)}" if settings.sandbox.allowed_paths else "")
                + "\nUse paths within the project directory or /tmp."
            ),
        )

    return _handler
