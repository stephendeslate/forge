"""Task classification + model routing — heuristic-based, no LLM call needed."""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.models.base import ModelBackend


class Route(str, Enum):
    HEAVY = "heavy"
    FAST = "fast"
    NPU = "npu"


# Keywords that signal heavy model usage
_HEAVY_KEYWORDS = re.compile(
    r"\b(write|implement|refactor|debug|fix|create|build|design|architect|optimize|analyze)\b",
    re.IGNORECASE,
)

# Keywords that signal lightweight tasks
_FAST_KEYWORDS = re.compile(
    r"\b(what is|explain|summarize|define|list|how does|translate|convert)\b",
    re.IGNORECASE,
)


def classify(prompt: str, *, force: Route | None = None, has_npu: bool = False) -> Route:
    """Classify a prompt to determine which model should handle it."""
    if force is not None:
        return force

    # Short prompts (< 30 chars) are usually quick questions → NPU if available
    if len(prompt) < 30 and not _HEAVY_KEYWORDS.search(prompt):
        return Route.NPU if has_npu else Route.FAST

    # Check for heavy keywords first (code generation, debugging)
    if _HEAVY_KEYWORDS.search(prompt):
        return Route.HEAVY

    # Check for light keywords
    if _FAST_KEYWORDS.search(prompt):
        return Route.FAST

    # Default: heavy for longer prompts, fast for shorter
    return Route.HEAVY if len(prompt) > 200 else Route.FAST


class ModelRouter:
    """Routes prompts to the appropriate model backend."""

    def __init__(
        self,
        heavy: ModelBackend,
        fast: ModelBackend,
        npu: ModelBackend | None = None,
    ) -> None:
        self._backends = {
            Route.HEAVY: heavy,
            Route.FAST: fast,
        }
        if npu is not None:
            self._backends[Route.NPU] = npu

    def get_backend(self, route: Route) -> ModelBackend:
        if route == Route.NPU and Route.NPU not in self._backends:
            return self._backends[Route.FAST]  # fallback
        return self._backends[route]

    def route(self, prompt: str, *, force: Route | None = None) -> tuple[Route, ModelBackend]:
        has_npu = Route.NPU in self._backends
        route = classify(prompt, force=force, has_npu=has_npu)
        return route, self.get_backend(route)
