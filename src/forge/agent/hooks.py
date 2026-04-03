"""Event/hook system for agent observability and extensibility."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from forge.log import get_logger

if TYPE_CHECKING:
    from forge.agent.deps import AgentDeps

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionStart:
    session_id: str
    cwd: str
    permission: str


@dataclass(frozen=True)
class SessionEnd:
    session_id: str
    message_count: int


@dataclass(frozen=True)
class UserPromptSubmit:
    """Blocking event — handlers can reject or modify the prompt."""
    prompt: str


@dataclass(frozen=True)
class TurnStart:
    turn_number: int
    prompt: str


@dataclass(frozen=True)
class TurnEnd:
    turn_number: int
    tool_call_count: int
    elapsed: float
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass(frozen=True)
class PreToolUse:
    """Blocking event — handlers can ALLOW, BLOCK, or MODIFY the call."""
    tool_name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class PostToolUse:
    tool_name: str
    args: dict[str, Any]
    result: Any
    elapsed: float


@dataclass(frozen=True)
class PostToolUseFailure:
    tool_name: str
    args: dict[str, Any]
    error: BaseException


@dataclass(frozen=True)
class Stop:
    reason: str


# Set of event types that block execution (sequential, first non-ALLOW wins)
BLOCKING_EVENTS = frozenset({UserPromptSubmit, PreToolUse})

# All event types for reference
EVENT_TYPES = (
    SessionStart, SessionEnd, UserPromptSubmit,
    TurnStart, TurnEnd,
    PreToolUse, PostToolUse, PostToolUseFailure,
    Stop,
)


# ---------------------------------------------------------------------------
# Hook action / result
# ---------------------------------------------------------------------------

class HookEscalation(Exception):
    """Raised by hook handlers to escalate beyond the hook system.

    Not caught by HookRegistry.check() — propagates to the agent loop.
    """


class HookAction(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"


@dataclass
class HookResult:
    action: HookAction = HookAction.ALLOW
    message: str = ""
    modified_args: dict[str, Any] | None = None


# Type alias for hook handlers
HookHandler = Callable[..., HookResult | Awaitable[HookResult] | None | Awaitable[None]]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class HookRegistry:
    """Registry of event handlers, keyed by event type."""

    _handlers: dict[type, list[tuple[int, HookHandler]]] = field(
        default_factory=dict, init=False,
    )

    def on(self, event_type: type, handler: HookHandler, *, priority: int = 0) -> None:
        """Register a handler for an event type. Lower priority runs first."""
        bucket = self._handlers.setdefault(event_type, [])
        bucket.append((priority, handler))
        bucket.sort(key=lambda t: t[0])

    async def emit(self, event: object) -> None:
        """Fire a notification event (non-blocking). All handlers run concurrently."""
        handlers = self._handlers.get(type(event), [])
        if not handlers:
            return

        async def _run(handler: HookHandler) -> None:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except HookEscalation:
                raise
            except Exception:
                logger.debug("Hook handler error on %s", type(event).__name__, exc_info=True)

        await asyncio.gather(*[_run(h) for _, h in handlers])

    def get_handlers(self, event_type: type) -> list[tuple[int, HookHandler]]:
        """Return a copy of handlers for the given event type."""
        return list(self._handlers.get(event_type, []))

    async def check(self, event: object) -> HookResult:
        """Fire a blocking event. Handlers run sequentially; first non-ALLOW wins."""
        handlers = self._handlers.get(type(event), [])
        for _, handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, HookResult) and result.action != HookAction.ALLOW:
                    return result
            except HookEscalation:
                raise
            except Exception:
                logger.debug("Hook handler error on %s", type(event).__name__, exc_info=True)
        return HookResult()


# ---------------------------------------------------------------------------
# with_hooks wrapper
# ---------------------------------------------------------------------------

def with_hooks(fn: Callable) -> Callable:
    """Wrap a tool function with PreToolUse/PostToolUse hook emission.

    Applied at Tool registration time. When deps.hook_registry is None,
    the original function passes through unchanged.
    """
    sig = inspect.signature(fn)
    tool_name = fn.__name__

    async def _wrapped(*args: Any, **kwargs: Any) -> Any:
        # Bind args to get parameter names
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # First arg is always ctx (RunContext)
        ctx = bound.arguments.get("ctx") or next(iter(bound.arguments.values()))
        deps: AgentDeps = ctx.deps

        registry: HookRegistry | None = getattr(deps, "hook_registry", None)
        if registry is None:
            # Passthrough — no hooks configured
            return await fn(*args, **kwargs)

        # Build args dict (excluding ctx)
        tool_args = {k: v for k, v in bound.arguments.items() if k != "ctx"}

        # PreToolUse check (blocking)
        from pydantic_ai import ModelRetry

        pre_event = PreToolUse(tool_name=tool_name, args=tool_args)
        hook_result = await registry.check(pre_event)

        if hook_result.action == HookAction.BLOCK:
            raise ModelRetry(hook_result.message or f"{tool_name} blocked by hook")

        if hook_result.action == HookAction.MODIFY and hook_result.modified_args:
            # Update kwargs with modified args
            for k, v in hook_result.modified_args.items():
                if k in bound.arguments:
                    bound.arguments[k] = v

        # Execute
        start = time.monotonic()
        try:
            result = await fn(*bound.args, **bound.kwargs)
            elapsed = time.monotonic() - start

            # PostToolUse (non-blocking)
            await registry.emit(PostToolUse(
                tool_name=tool_name,
                args=tool_args,
                result=result,
                elapsed=elapsed,
            ))

            # Check for post-tool feedback injected by hooks (e.g. syntax errors)
            feedback = getattr(deps, "_post_tool_feedback", None)
            if feedback and isinstance(result, str):
                result = f"{result}\n\n{feedback}"
                deps._post_tool_feedback = None

            return result
        except Exception as exc:
            # PostToolUseFailure (non-blocking)
            await registry.emit(PostToolUseFailure(
                tool_name=tool_name,
                args=tool_args,
                error=exc,
            ))
            raise

    # Preserve function metadata for pydantic-ai Tool introspection
    import functools

    functools.update_wrapper(_wrapped, fn)
    # Explicitly copy signature so pydantic-ai can introspect parameters
    _wrapped.__signature__ = sig  # type: ignore[attr-defined]

    return _wrapped


# ---------------------------------------------------------------------------
# Built-in permission hook
# ---------------------------------------------------------------------------

async def permission_hook(event: PreToolUse, deps: AgentDeps) -> HookResult:
    """Built-in PreToolUse handler that enforces permission policy.

    Replaces inline check_permission() calls in write_file/edit_file/run_command.
    """
    from forge.agent.permissions import (
        SAFE_TOOLS,
        PermissionPolicy,
        _prompt_user,
        _summarize_call,
    )

    policy = deps.permission

    if policy == PermissionPolicy.YOLO:
        return HookResult()

    if policy == PermissionPolicy.AUTO and event.tool_name in SAFE_TOOLS:
        return HookResult()

    # ASK policy, or dangerous tool under AUTO — prompt user
    summary = _summarize_call(event.tool_name, event.args)
    allowed = await _prompt_user(deps.console, event.tool_name, summary)

    if allowed:
        return HookResult()
    return HookResult(action=HookAction.BLOCK, message="Permission denied by user.")


def make_permission_handler(deps: AgentDeps) -> HookHandler:
    """Create a permission hook handler bound to specific deps."""
    async def _handler(event: PreToolUse) -> HookResult:
        return await permission_hook(event, deps)
    return _handler
