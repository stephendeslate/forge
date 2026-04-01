"""NPU model backend via FastFlowLM (OpenAI-compatible API)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from forge.config import settings


class NPUBackend:
    """NPU backend using FastFlowLM's OpenAI-compatible chat completions API."""

    def __init__(self) -> None:
        self._base_url = settings.npu.base_url.rstrip("/")
        self._model = settings.npu.model
        self._timeout = settings.npu.timeout

    @property
    def name(self) -> str:
        return "npu"

    @property
    def model_id(self) -> str:
        return self._model

    async def generate(self, prompt: str, *, system: str = "") -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json={"model": self._model, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def stream(self, prompt: str, *, system: str = "") -> AsyncIterator[str]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json={"model": self._model, "messages": messages, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content

    async def is_available(self) -> bool:
        """Check if the NPU endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/models")
                return resp.status_code == 200
        except Exception:
            return False


def get_npu_backend() -> NPUBackend | None:
    """Return an NPUBackend if NPU is enabled, else None."""
    if settings.npu.enabled:
        return NPUBackend()
    return None
