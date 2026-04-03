"""Ollama model backend using Pydantic AI."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from forge.config import settings


def _model_settings(timeout: int = 300, num_ctx: int | None = None) -> ModelSettings:
    """Build ModelSettings with num_ctx passed through to Ollama via extra_body."""
    ctx = num_ctx or settings.agent.num_ctx
    return ModelSettings(
        timeout=timeout,
        extra_body={"options": {"num_ctx": ctx}, "keep_alive": -1},
    )


# Legacy alias — callers that just need the default timeout + num_ctx
_OLLAMA_TIMEOUT = _model_settings()


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
        self._agents: dict[str, Agent] = {"": self._agent}

    @property
    def name(self) -> str:
        return self._label

    @property
    def model_id(self) -> str:
        return self._model_name

    def _get_agent(self, system: str) -> Agent:
        if system not in self._agents:
            self._agents[system] = Agent(model=self._model_id, instructions=system)
        return self._agents[system]

    async def generate(self, prompt: str, *, system: str = "") -> str:
        agent = self._get_agent(system)
        result = await agent.run(prompt, model_settings=_OLLAMA_TIMEOUT)
        return result.output

    async def stream(self, prompt: str, *, system: str = "") -> AsyncIterator[str]:
        agent = self._get_agent(system)
        async with agent.run_stream(prompt, model_settings=_OLLAMA_TIMEOUT) as stream:
            async for chunk in stream.stream_text(delta=True):
                yield chunk


class OllamaMonitor:
    """Tracks Ollama model state via /api/ps and manages lifecycle."""

    def __init__(self, base_url: str | None = None):
        self._base_url = (base_url or settings.ollama.base_url).rstrip("/")
        self._session: "httpx.AsyncClient | None" = None

    async def _client(self) -> "httpx.AsyncClient":
        import httpx

        if self._session is None:
            self._session = httpx.AsyncClient(base_url=self._base_url, timeout=10)
        return self._session

    async def list_loaded(self) -> list[dict]:
        """Call GET /api/ps, return list of loaded model dicts."""
        client = await self._client()
        resp = await client.get("/api/ps")
        resp.raise_for_status()
        data = resp.json()
        return data.get("models", [])

    async def is_loaded(self, model: str) -> bool:
        """Check if a specific model is currently loaded."""
        models = await self.list_loaded()
        return any(m.get("name", "").startswith(model) for m in models)

    async def preload(self, model: str, num_ctx: int | None = None) -> bool:
        """POST /api/generate with empty prompt to preload a model.

        Returns True if model loaded successfully.
        """
        import httpx

        client = await self._client()
        body: dict = {"model": model, "prompt": "", "stream": False, "keep_alive": -1}
        if num_ctx:
            body["options"] = {"num_ctx": num_ctx}
        try:
            resp = await client.post("/api/generate", json=body, timeout=120)
            return resp.status_code == 200
        except httpx.TimeoutException:
            return False

    async def health_check(self) -> bool:
        """GET /api/ps — returns True if Ollama is responsive."""
        try:
            client = await self._client()
            resp = await client.get("/api/ps")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the httpx session."""
        if self._session:
            await self._session.aclose()
            self._session = None


_heavy_backend: OllamaBackend | None = None
_fast_backend: OllamaBackend | None = None


def get_heavy_backend() -> OllamaBackend:
    global _heavy_backend
    if _heavy_backend is None:
        _heavy_backend = OllamaBackend(settings.ollama.heavy_model, label="gpu-heavy")
    return _heavy_backend


def get_fast_backend() -> OllamaBackend:
    global _fast_backend
    if _fast_backend is None:
        _fast_backend = OllamaBackend(settings.ollama.fast_model, label="gpu-fast")
    return _fast_backend
