"""MCP server configuration discovery and loading."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from forge.log import get_logger

logger = get_logger(__name__)

MCP_CONFIG_NAME = "mcp.json"


def find_mcp_configs(cwd: Path) -> list[Path]:
    """Return MCP config paths in priority order (project-local first, then global)."""
    paths: list[Path] = []
    project_config = cwd / ".forge" / MCP_CONFIG_NAME
    if project_config.is_file():
        paths.append(project_config)
    global_config = Path.home() / ".config" / "forge" / MCP_CONFIG_NAME
    if global_config.is_file():
        paths.append(global_config)
    return paths


def _expand_env_vars(obj: object) -> object:
    """Recursively expand ${VAR} references in string values."""
    if isinstance(obj, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            obj,
        )
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    return obj


def load_all_mcp_servers(cwd: Path) -> list:
    """Load and merge MCP servers from all config files.

    Project-local (.forge/mcp.json) overrides global (~/.config/forge/mcp.json)
    on name collision. Environment variables in ${VAR} syntax are expanded.

    Returns a list of pydantic-ai MCP server objects.
    """
    from pydantic_ai.mcp import load_mcp_servers

    configs = find_mcp_configs(cwd)
    if not configs:
        return []

    servers_by_name: dict[str, object] = {}

    # Load in reverse priority order (global first, project overrides)
    for config_path in reversed(configs):
        try:
            # Expand env vars before passing to pydantic-ai
            raw = json.loads(config_path.read_text())
            expanded = _expand_env_vars(raw)
            # Write expanded config to a temp location for load_mcp_servers
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                json.dump(expanded, tmp)
                tmp_path = tmp.name
            try:
                for server in load_mcp_servers(tmp_path):
                    servers_by_name[server.id] = server
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to load MCP config: %s", config_path, exc_info=True)

    return list(servers_by_name.values())
