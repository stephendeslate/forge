"""Tests for MCP server configuration discovery and loading."""

import json
from pathlib import Path

from forge.agent.mcp_config import (
    _expand_env_vars,
    find_mcp_configs,
    load_all_mcp_servers,
)


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


def test_load_all_empty(tmp_path):
    """No configs → empty server list."""
    assert load_all_mcp_servers(tmp_path) == []


def test_expand_env_vars_basic(monkeypatch):
    """${VAR} references are expanded from environment."""
    monkeypatch.setenv("MY_TOKEN", "secret123")
    assert _expand_env_vars("Bearer ${MY_TOKEN}") == "Bearer secret123"


def test_expand_env_vars_missing():
    """Missing env vars are left as-is."""
    assert _expand_env_vars("${NONEXISTENT_VAR_XYZ}") == "${NONEXISTENT_VAR_XYZ}"


def test_expand_env_vars_nested(monkeypatch):
    """Recursively expands in dicts and lists."""
    monkeypatch.setenv("KEY", "val")
    result = _expand_env_vars({"env": {"TOKEN": "${KEY}"}, "args": ["--key=${KEY}"]})
    assert result == {"env": {"TOKEN": "val"}, "args": ["--key=val"]}


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
