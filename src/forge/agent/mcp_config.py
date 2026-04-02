"""MCP server configuration discovery and loading."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from forge.log import get_logger

logger = get_logger(__name__)

MCP_CONFIG_NAME = "mcp.json"


def _default_mcp_servers() -> dict:
    """Built-in MCP servers available if their runtime exists."""
    servers: dict = {}
    # Playwright browser — available if npx is on PATH
    if shutil.which("npx"):
        servers["browser"] = {
            "command": "npx",
            "args": ["@playwright/mcp", "--headless"],
        }
    return servers


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


def load_all_mcp_servers(cwd: Path) -> list:
    """Load and merge MCP servers from all config files.

    Priority (lowest to highest): built-in defaults → global → project-local.
    Setting a server name to ``false`` in any config disables it.
    Environment variables in ``${VAR}`` syntax are expanded by pydantic-ai natively.

    Returns a list of pydantic-ai MCP server objects.
    """
    from pydantic_ai.mcp import load_mcp_servers

    # Start with built-in defaults
    merged: dict = _default_mcp_servers()

    configs = find_mcp_configs(cwd)

    # Load in reverse priority order (global first, then project overrides)
    for config_path in reversed(configs):
        try:
            raw = json.loads(config_path.read_text())
            mcp_servers = raw.get("mcpServers", {})
            for name, value in mcp_servers.items():
                if value is False:
                    # Explicit disable — remove from merged
                    merged.pop(name, None)
                else:
                    merged[name] = value
        except Exception:
            logger.warning("Failed to load MCP config: %s", config_path, exc_info=True)

    if not merged:
        return []

    # Write merged config to temp file for pydantic-ai's load_mcp_servers
    # (which handles ${VAR} expansion natively — no need to pre-expand)
    import tempfile

    config_data = {"mcpServers": merged}
    servers: list = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as tmp:
            json.dump(config_data, tmp)
            tmp_path = tmp.name
        try:
            servers = list(load_mcp_servers(tmp_path))
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        logger.warning("Failed to load merged MCP config", exc_info=True)

    return servers
