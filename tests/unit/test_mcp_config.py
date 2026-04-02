"""Tests for MCP server configuration discovery and loading."""

import json
from pathlib import Path
from unittest.mock import patch

from forge.agent.mcp_config import (
    find_mcp_configs,
    load_all_mcp_servers,
)


def _no_npx(_name):
    """Stub shutil.which to hide npx so built-in browser default is suppressed."""
    return None


def test_find_configs_none(tmp_path):
    """No config files → empty list."""
    assert find_mcp_configs(tmp_path) == []


def test_find_configs_global_only(tmp_path, monkeypatch):
    """Only global config exists → returns it."""
    fake_home = tmp_path / "home"
    config_dir = fake_home / ".config" / "forge"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "mcp.json"
    config_file.write_text("{}")

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    # Use a cwd with no project config
    cwd = tmp_path / "project"
    cwd.mkdir()

    result = find_mcp_configs(cwd)
    assert len(result) == 1
    assert result[0] == config_file


def test_find_configs_project_only(tmp_path, monkeypatch):
    """Only project config exists → returns it."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    cwd = tmp_path / "project"
    forge_dir = cwd / ".forge"
    forge_dir.mkdir(parents=True)
    config_file = forge_dir / "mcp.json"
    config_file.write_text("{}")

    result = find_mcp_configs(cwd)
    assert len(result) == 1
    assert result[0] == config_file


def test_find_configs_both(tmp_path, monkeypatch):
    """Both configs exist → project first, then global."""
    fake_home = tmp_path / "home"
    global_dir = fake_home / ".config" / "forge"
    global_dir.mkdir(parents=True)
    global_config = global_dir / "mcp.json"
    global_config.write_text("{}")

    monkeypatch.setattr(Path, "home", lambda: fake_home)

    cwd = tmp_path / "project"
    project_dir = cwd / ".forge"
    project_dir.mkdir(parents=True)
    project_config = project_dir / "mcp.json"
    project_config.write_text("{}")

    result = find_mcp_configs(cwd)
    assert len(result) == 2
    assert result[0] == project_config  # project first
    assert result[1] == global_config  # global second


@patch("forge.agent.mcp_config.shutil.which", _no_npx)
def test_load_all_empty(tmp_path):
    """No configs and no npx → empty server list."""
    assert load_all_mcp_servers(tmp_path) == []


@patch("forge.agent.mcp_config.shutil.which", _no_npx)
def test_load_servers_from_config(tmp_path, monkeypatch):
    """Load servers from a valid config file."""
    fake_home = tmp_path / "home"
    config_dir = fake_home / ".config" / "forge"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "mcp.json"
    config_file.write_text(json.dumps({
        "mcpServers": {
            "test-echo": {
                "command": "echo",
                "args": ["hello"],
            }
        }
    }))
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    servers = load_all_mcp_servers(tmp_path)
    assert len(servers) == 1
    assert servers[0].id == "test-echo"
    assert servers[0].command == "echo"


@patch("forge.agent.mcp_config.shutil.which", _no_npx)
def test_load_servers_project_overrides_global(tmp_path, monkeypatch):
    """Project config overrides global on name collision."""
    fake_home = tmp_path / "home"
    global_dir = fake_home / ".config" / "forge"
    global_dir.mkdir(parents=True)
    (global_dir / "mcp.json").write_text(json.dumps({
        "mcpServers": {
            "myserver": {"command": "global-cmd", "args": []},
            "global-only": {"command": "only-global", "args": []},
        }
    }))

    cwd = tmp_path / "project"
    project_dir = cwd / ".forge"
    project_dir.mkdir(parents=True)
    (project_dir / "mcp.json").write_text(json.dumps({
        "mcpServers": {
            "myserver": {"command": "project-cmd", "args": []},
        }
    }))

    monkeypatch.setattr(Path, "home", lambda: fake_home)

    servers = load_all_mcp_servers(cwd)
    by_id = {s.id: s for s in servers}
    assert len(by_id) == 2
    assert by_id["myserver"].command == "project-cmd"  # project wins
    assert by_id["global-only"].command == "only-global"  # global-only preserved


def test_default_browser_discovered_when_npx_exists(tmp_path, monkeypatch):
    """Built-in browser server is included when npx is on PATH."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    with patch("forge.agent.mcp_config.shutil.which", return_value="/usr/bin/npx"):
        servers = load_all_mcp_servers(tmp_path)

    by_id = {s.id: s for s in servers}
    assert "browser" in by_id
    assert by_id["browser"].command == "npx"


@patch("forge.agent.mcp_config.shutil.which", return_value="/usr/bin/npx")
def test_disable_builtin_server(_, tmp_path, monkeypatch):
    """Setting a server to false disables it."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    cwd = tmp_path / "project"
    project_dir = cwd / ".forge"
    project_dir.mkdir(parents=True)
    (project_dir / "mcp.json").write_text(json.dumps({
        "mcpServers": {
            "browser": False,
        }
    }))

    servers = load_all_mcp_servers(cwd)
    by_id = {s.id: s for s in servers}
    assert "browser" not in by_id
