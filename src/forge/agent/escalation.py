"""Model escalation — auto-switch from fast to heavy model on trouble signals.

Accumulates weighted signals from hooks. When cumulative weight crosses a
threshold, switches deps.model_override from fast to heavy. Sticky for the
session — user can revert with /model fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from forge.log import get_logger

if TYPE_CHECKING:
    from rich.console import Console

    from forge.agent.deps import AgentDeps

logger = get_logger(__name__)


# Default signal weights
SIGNAL_WEIGHTS = {
    "tool_failure": 1.0,
    "circuit_breaker": 3.0,
    "empty_response": 2.0,
    "high_tool_ratio": 1.5,
}


@dataclass
class EscalationSignal:
    kind: str
    weight: float
    turn: int


class ModelEscalator:
    """Accumulates trouble signals and auto-escalates from fast to heavy model."""

    def __init__(
        self,
        deps: AgentDeps,
        console: Console,
        *,
        threshold: float = 5.0,
    ) -> None:
        self._deps = deps
        self._console = console
        self._threshold = threshold
        self._signals: list[EscalationSignal] = []
        self._escalated: bool = False

    @property
    def is_active(self) -> bool:
        """Only active when on fast model and not yet escalated."""
        from forge.config import settings

        if self._escalated:
            return False
        return self._deps.model_override == settings.ollama.fast_model

    @property
    def escalated(self) -> bool:
        return self._escalated

    @property
    def total_weight(self) -> float:
        return sum(s.weight for s in self._signals)

    def add_signal(self, kind: str, weight: float, turn: int) -> None:
        """Add a trouble signal. Auto-escalates if threshold is crossed."""
        if not self.is_active:
            return

        self._signals.append(EscalationSignal(kind=kind, weight=weight, turn=turn))
        logger.debug(
            "Escalation signal: %s (%.1f), total: %.1f/%.1f",
            kind, weight, self.total_weight, self._threshold,
        )

        if self.total_weight >= self._threshold:
            self._escalate()

    def _escalate(self) -> None:
        """Switch from fast to heavy model."""
        from forge.config import settings

        self._escalated = True
        self._deps.model_override = None  # None = heavy (default)
        self._console.print(
            f"[yellow]Auto-escalating to {settings.ollama.heavy_model} "
            f"(accumulated trouble weight: {self.total_weight:.1f}/{self._threshold:.1f})[/yellow]"
        )
        logger.info(
            "Model escalated: %d signals, total weight %.1f",
            len(self._signals), self.total_weight,
        )

    def reset(self) -> None:
        """Reset escalation state. Called when user manually switches to fast."""
        self._signals.clear()
        self._escalated = False


def wire_escalation(
    escalator: ModelEscalator,
    deps: AgentDeps,
    turn_counter_ref: list[int],
) -> None:
    """Register escalation hooks on deps.hook_registry.

    turn_counter_ref is a mutable list[int] holding the current turn number,
    so the hooks can reference it.
    """
    from forge.agent.hooks import PostToolUseFailure, TurnEnd

    registry = deps.hook_registry
    if registry is None:
        return

    async def _on_tool_failure(event: PostToolUseFailure) -> None:
        escalator.add_signal("tool_failure", SIGNAL_WEIGHTS["tool_failure"], turn_counter_ref[0])

    async def _on_turn_end(event: TurnEnd) -> None:
        # Detect empty response: no tool calls and no meaningful output
        if event.tool_call_count == 0:
            escalator.add_signal("empty_response", SIGNAL_WEIGHTS["empty_response"], event.turn_number)
        # Detect high tool ratio: many tool calls with no text output
        elif event.tool_call_count >= 8:
            escalator.add_signal("high_tool_ratio", SIGNAL_WEIGHTS["high_tool_ratio"], event.turn_number)

    registry.on(PostToolUseFailure, _on_tool_failure)
    registry.on(TurnEnd, _on_turn_end)
