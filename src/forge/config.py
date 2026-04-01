"""Configuration via pydantic-settings — env vars + TOML file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_CONFIG_DIR = Path.home() / ".config" / "forge"
_CONFIG_FILE = _CONFIG_DIR / "config.toml"


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORGE_OLLAMA_")

    base_url: str = "http://127.0.0.1:11434"
    heavy_model: str = "qwen3-coder-next:q8_0"
    fast_model: str = "qwen3.5:4b"
    embed_model: str = "nomic-embed-text-v2-moe"


class NPUSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORGE_NPU_")

    enabled: bool = False
    base_url: str = "http://127.0.0.1:52625/v1"
    model: str = "llama-3.2-3b"
    timeout: int = 60


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FORGE_",
        toml_file=str(_CONFIG_FILE) if _CONFIG_FILE.exists() else None,
        env_nested_delimiter="__",
    )

    default_route: Literal["heavy", "fast", "auto"] = "auto"
    streaming: bool = True
    max_history: int = Field(default=50, description="Max conversation turns kept in memory")
    persist_history: bool = Field(default=True, description="Persist conversations to PostgreSQL")

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    npu: NPUSettings = Field(default_factory=NPUSettings)

    @classmethod
    def ensure_config_dir(cls) -> Path:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return _CONFIG_DIR


settings = Settings()
