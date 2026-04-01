"""Model backend protocol — all backends implement this interface."""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol for model backends (Ollama, NPU, etc.)."""

    @property
    def name(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    async def generate(self, prompt: str, *, system: str = "") -> str:
        """Generate a complete response."""
        ...

    async def stream(self, prompt: str, *, system: str = "") -> AsyncIterator[str]:
        """Stream response tokens."""
        ...
