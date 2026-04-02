"""Configuration via pydantic-settings — env vars + TOML file."""

from __future__ import annotations

import getpass
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource

_DEFAULT_USER = getpass.getuser()


_CONFIG_DIR = Path.home() / ".config" / "forge"
_CONFIG_FILE = _CONFIG_DIR / "config.toml"


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORGE_OLLAMA_")

    base_url: str = "http://127.0.0.1:11434"
    heavy_model: str = "qwen3-coder-next:q8_0"
    fast_model: str = "qwen3.5:4b"
    embed_model: str = "nomic-embed-text-v2-moe"
    vision_model: str = ""


class NPUSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORGE_NPU_")

    enabled: bool = False
    base_url: str = "http://127.0.0.1:52625/v1"
    model: str = "llama3.2:3b"
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


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORGE_AGENT_")

    num_ctx: int = Field(default=131072, description="Ollama num_ctx — context window size in tokens")
    token_budget: int = Field(default=120000, description="Max tokens before auto-compaction triggers")
    compaction_threshold: float = Field(default=0.85, description="Fraction of token_budget that triggers auto-compaction")
    compaction_input_limit: int = Field(default=32000, description="Max chars of conversation text fed to the summarizer")
    request_limit: int = Field(default=25, description="Max LLM requests per agent turn")
    run_command_timeout: float = Field(default=120.0, description="Default timeout for run_command tool (seconds)")
    run_command_stdout_limit: int = Field(default=50000, description="Max stdout chars captured from run_command")
    run_command_stderr_limit: int = Field(default=20000, description="Max stderr chars captured from run_command")
    search_max_matches: int = Field(default=200, description="Max ripgrep matches in search_code")
    list_files_limit: int = Field(default=500, description="Max files returned by list_files")
    rag_max_tokens: int = Field(default=12000, description="Max tokens of RAG context injected into prompts")
    delegate_model: str = Field(default="", description="Model for sub-agent delegation (empty = use heavy model)")
    compaction_model: str = Field(default="", description="Model for compaction summarization (empty = use fast model)")
    preload_model: bool = Field(default=True, description="Preload the heavy model on agent startup")

    # Circuit breaker
    cb_enabled: bool = Field(default=True, description="Enable tool-call loop detection")
    cb_identical: int = Field(default=3, description="Identical call threshold before warning")
    cb_failures: int = Field(default=3, description="Consecutive failure threshold")
    cb_oscillation_window: int = Field(default=3, description="A-B oscillation cycle count")
    cb_post_warning_grace: int = Field(default=2, description="Extra calls allowed after warning before trip")
    cb_history_size: int = Field(default=20, description="Tool call history buffer size")

    # Model escalation
    auto_escalation: bool = Field(default=True, description="Auto-escalate from fast to heavy on trouble")
    escalation_threshold: float = Field(default=5.0, description="Cumulative signal weight to trigger escalation")


DEFAULT_BLOCKED_PATTERNS = [
    r"\brm\s+(-[rfv]*\s+)?/",          # rm -rf /
    r"\bsudo\b",                         # sudo anything
    r"\bchmod\s+777\b",                  # chmod 777
    r"\bmkfs\b",                         # format disk
    r"\bdd\s+.*of=/",                    # dd to device
    r"\bcurl\b.*\|\s*(bash|sh|zsh)\b",   # curl pipe to shell
    r"\bwget\b.*\|\s*(bash|sh|zsh)\b",   # wget pipe to shell
    r">\s*/etc/",                         # redirect to /etc
    r">\s*~/\.ssh/",                      # redirect to .ssh
    r"\bgit\s+push\s+.*--force\b",       # force push
    r"\bgit\s+reset\s+--hard\b",         # hard reset
]

DEFAULT_WARN_PATTERNS = [
    r"\brm\s+-r",                        # recursive delete (non-root)
    r"\bkill\s+-9\b",                    # kill -9
    r"\bpkill\b",                        # pkill
]


class SandboxSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORGE_SANDBOX_")

    enabled: bool = True
    blocked_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_BLOCKED_PATTERNS))
    warn_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_WARN_PATTERNS))
    restrict_paths: bool = True
    allowed_paths: list[str] = Field(default_factory=list, description="Extra allowed paths beyond cwd + /tmp")


class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORGE_MEMORY_")

    max_memories: int = Field(default=50, description="Max memories per project before pruning")
    similarity_threshold: float = Field(default=0.92, description="Cosine similarity threshold for dedup")
    max_merges: int = Field(default=5, description="Max merge operations per prune cycle")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FORGE_",
        toml_file=str(_CONFIG_FILE) if _CONFIG_FILE.exists() else None,
        env_nested_delimiter="__",
    )

    default_route: Literal["heavy", "fast", "auto"] = "auto"
    streaming: bool = True
    max_history: int = Field(default=50, description="Max conversation turns kept in memory (code mode)")
    persist_history: bool = Field(default=True, description="Persist conversations to PostgreSQL")

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    npu: NPUSettings = Field(default_factory=NPUSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    hooks: HookSettings = Field(default_factory=HookSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        toml_source = TomlConfigSettingsSource(settings_cls, toml_file=str(_CONFIG_FILE) if _CONFIG_FILE.exists() else None)
        return (init_settings, env_settings, toml_source, file_secret_settings)

    @classmethod
    def ensure_config_dir(cls) -> Path:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return _CONFIG_DIR


settings = Settings()
