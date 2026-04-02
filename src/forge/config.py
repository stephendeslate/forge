"""Configuration via pydantic-settings — env vars + TOML file."""

from __future__ import annotations

import getpass
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_USER = getpass.getuser()


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


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORGE_DB_")

    dsn: str = f"postgresql://{_DEFAULT_USER}@/forge?host=/var/run/postgresql&port=5433"
    pool_min: int = 2
    pool_max: int = 10
    connect_timeout: float = 10.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


class SearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORGE_SEARCH_")

    searxng_urls: list[str] = ["http://localhost:8888", "http://localhost:8080"]
    ddg_enabled: bool = True


class HookSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORGE_HOOKS_")

    pre_tool_use: list[str] = Field(default_factory=list, description="Shell commands to run before tool use")
    post_tool_use: list[str] = Field(default_factory=list, description="Shell commands to run after tool use")
    user_prompt_submit: list[str] = Field(default_factory=list, description="Shell commands on prompt submit")


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
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    hooks: HookSettings = Field(default_factory=HookSettings)

    @classmethod
    def ensure_config_dir(cls) -> Path:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return _CONFIG_DIR


settings = Settings()
