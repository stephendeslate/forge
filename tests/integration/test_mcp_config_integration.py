"""Integration tests — MCP config discovery and merging."""

import json

import pytest

from forge.agent.mcp_config import find_mcp_configs, load_all_mcp_servers


class TestMCPConfigDiscovery:
    def test_project_overrides_global(self, tmp_path, monkeypatch):
        """Project-local config takes priority over global."""
        # Global config
        global_dir = tmp_path / "global_home" / ".config" / "forge"
        global_dir.mkdir(parents=True)
        global_config = global_dir / "mcp.json"
        global_config.write_text(json.dumps({
            "mcpServers": {
                "global-server": {"command": "global-cmd", "args": []},
                "shared": {"command": "global-shared", "args": []},
            }
        }))

        # Project-local config
        project_dir = tmp_path / "project" / ".forge"
        project_dir.mkdir(parents=True)
        project_config = project_dir / "mcp.json"
        project_config.write_text(json.dumps({
            "mcpServers": {
                "shared": {"command": "project-shared", "args": []},
                "project-only": {"command": "proj-cmd", "args": []},
            }
        }))

        cwd = tmp_path / "project"

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "global_home")

        configs = find_mcp_configs(cwd)
        # Project first, then global
        assert len(configs) == 2
        assert "project" in str(configs[0])

    def test_disable_server_with_false(self, tmp_path, monkeypatch):
        """Setting server to false excludes it from merged config."""
        global_dir = tmp_path / "home" / ".config" / "forge"
        global_dir.mkdir(parents=True)
        (global_dir / "mcp.json").write_text(json.dumps({
            "mcpServers": {
                "browser": {"command": "npx", "args": ["@playwright/mcp"]},
            }
        }))

        project_dir = tmp_path / "proj" / ".forge"
        project_dir.mkdir(parents=True)
        (project_dir / "mcp.json").write_text(json.dumps({
            "mcpServers": {
                "browser": False,
            }
        }))

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
        # Patch _default_mcp_servers to return nothing (avoid npx detection)
        monkeypatch.setattr("forge.agent.mcp_config._default_mcp_servers", lambda: {})

        servers = load_all_mcp_servers(tmp_path / "proj")
        # browser was disabled by project config
        assert len(servers) == 0

    def test_invalid_json_graceful(self, tmp_path, monkeypatch):
        """Malformed JSON doesn't crash, just logs and continues."""
        project_dir = tmp_path / "proj" / ".forge"
        project_dir.mkdir(parents=True)
        (project_dir / "mcp.json").write_text("{invalid json!!!")

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "fake_home")
        monkeypatch.setattr("forge.agent.mcp_config._default_mcp_servers", lambda: {})

        # Should not raise
        servers = load_all_mcp_servers(tmp_path / "proj")
        assert servers == []
