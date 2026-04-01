"""Ollama model backend using Pydantic AI."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from forge.config import settings

# Ollama can take a while to load models, especially the 84GB heavy model
_OLLAMA_TIMEOUT = ModelSettings(timeout=300)


def _ensure_ollama_env() -> None:
    """Set OLLAMA_BASE_URL env var so pydantic-ai's OllamaProvider picks it up.

    Pydantic AI uses the OpenAI SDK under the hood, which needs the /v1 suffix.
    """
    if "OLLAMA_BASE_URL" not in os.environ:
        base = settings.ollama.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        os.environ["OLLAMA_BASE_URL"] = base


class OllamaBackend:
    """Wraps a Pydantic AI Agent with an Ollama model for generation and streaming."""

    def __init__(self, model_name: str, label: str = "ollama") -> None:
        _ensure_ollama_env()
        self._model_name = model_name
        self._label = label
        self._model_id = f"ollama:{model_name}"
        self._agent = Agent(model=self._model_id)

    @property
    def name(self) -> str:
        return self._label

    @property
    def model_id(self) -> str:
        return self._model_name

    async def generate(self, prompt: str, *, system: str = "") -> str:
        agent = self._agent if not system else Agent(model=self._model_id, instructions=system)
        result = await agent.run(prompt, model_settings=_OLLAMA_TIMEOUT)
        return result.output

    async def stream(self, prompt: str, *, system: str = "") -> AsyncIterator[str]:
        agent = self._agent if not system else Agent(model=self._model_id, instructions=system)
        async with agent.run_stream(prompt, model_settings=_OLLAMA_TIMEOUT) as stream:
            async for chunk in stream.stream_text(delta=True):
                yield chunk


def get_heavy_backend() -> OllamaBackend:
    return OllamaBackend(settings.ollama.heavy_model, label="gpu-heavy")


def get_fast_backend() -> OllamaBackend:
    return OllamaBackend(settings.ollama.fast_model, label="gpu-fast")
