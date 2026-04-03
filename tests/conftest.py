"""Shared test fixtures."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.hooks import HookRegistry, make_permission_handler
from forge.agent.permissions import PermissionPolicy
from forge.agent.sandbox import make_command_blocklist_handler, make_path_boundary_handler


@pytest.fixture
def tmp_cwd(tmp_path):
    """A temporary working directory for agent tools."""
    return tmp_path


@pytest.fixture
def console():
    """A Rich Console that captures output (no terminal)."""
    return Console(file=None, force_terminal=False, no_color=True)


@pytest.fixture
def agent_deps(tmp_cwd, console):
    """AgentDeps with YOLO permissions for testing (no prompts)."""
    return AgentDeps(cwd=tmp_cwd, console=console, permission=PermissionPolicy.YOLO)


@pytest.fixture
def mock_ctx(agent_deps):
    """Mock RunContext[AgentDeps] for tool functions."""
    ctx = MagicMock()
    ctx.deps = agent_deps
    return ctx


@pytest.fixture
def sample_files(tmp_cwd):
    """Create a set of sample files for testing."""
    py_file = tmp_cwd / "hello.py"
    py_file.write_text('def greet(name):\n    return f"Hello, {name}!"\n')

    txt_file = tmp_cwd / "notes.txt"
    txt_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")

    sub = tmp_cwd / "src" / "utils"
    sub.mkdir(parents=True)
    (sub / "helper.py").write_text("import os\n\ndef helper():\n    pass\n")
    (sub / "__init__.py").write_text("")

    return tmp_cwd


# ---------------------------------------------------------------------------
# Hooked fixtures (permission hook via HookRegistry)
# ---------------------------------------------------------------------------


@pytest.fixture
def hook_registry():
    """Fresh HookRegistry for testing."""
    return HookRegistry()


@pytest.fixture
def hooked_deps(tmp_cwd, console, hook_registry):
    """AgentDeps with AUTO permissions + permission hook on a real HookRegistry."""
    deps = AgentDeps(
        cwd=tmp_cwd,
        console=console,
        permission=PermissionPolicy.AUTO,
        hook_registry=hook_registry,
    )
    hook_registry.on(
        __import__("forge.agent.hooks", fromlist=["PreToolUse"]).PreToolUse,
        make_permission_handler(deps),
    )
    return deps


@pytest.fixture
def hooked_ctx(hooked_deps):
    """Mock RunContext with hooked deps (AUTO + permission hook)."""
    ctx = MagicMock()
    ctx.deps = hooked_deps
    return ctx


@pytest.fixture
def sandboxed_deps(tmp_cwd, console):
    """AgentDeps with YOLO + sandbox blocklist + path boundary + permission hooks.

    Mirrors production wiring: sandbox at priority -50, permission at 0.
    """
    from forge.agent.hooks import PreToolUse

    registry = HookRegistry()
    deps = AgentDeps(
        cwd=tmp_cwd,
        console=console,
        permission=PermissionPolicy.YOLO,
        hook_registry=registry,
    )
    registry.on(PreToolUse, make_command_blocklist_handler(), priority=-50)
    registry.on(PreToolUse, make_path_boundary_handler(tmp_cwd), priority=-50)
    registry.on(PreToolUse, make_permission_handler(deps), priority=0)
    return deps


@pytest.fixture
def sandboxed_ctx(sandboxed_deps):
    """Mock RunContext with sandboxed deps."""
    ctx = MagicMock()
    ctx.deps = sandboxed_deps
    return ctx
