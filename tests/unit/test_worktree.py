"""Tests for git worktree management."""

import atexit
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.agent.worktree import (
    WorktreeInfo,
    _run_git,
    create_worktree,
    get_git_root,
    is_git_repo,
    remove_worktree,
)


def _init_git_repo(path: Path) -> None:
    """Initialize a minimal git repo with one commit."""
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(path), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(path), capture_output=True, check=True,
    )
    (path / "README.md").write_text("# test\n")
    subprocess.run(["git", "add", "."], cwd=str(path), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(path), capture_output=True, check=True,
    )


class TestRunGit:
    def test_returns_completed_process(self, tmp_path):
        _init_git_repo(tmp_path)
        result = _run_git(["status"], tmp_path)
        assert result.returncode == 0

    def test_fails_on_bad_command(self, tmp_path):
        result = _run_git(["not-a-command"], tmp_path)
        assert result.returncode != 0


class TestIsGitRepo:
    def test_true_for_git_repo(self, tmp_path):
        _init_git_repo(tmp_path)
        assert is_git_repo(tmp_path) is True

    def test_false_for_non_repo(self, tmp_path):
        assert is_git_repo(tmp_path) is False


class TestGetGitRoot:
    def test_returns_root(self, tmp_path):
        _init_git_repo(tmp_path)
        root = get_git_root(tmp_path)
        assert root == tmp_path.resolve()

    def test_from_subdirectory(self, tmp_path):
        _init_git_repo(tmp_path)
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        root = get_git_root(sub)
        assert root == tmp_path.resolve()

    def test_raises_if_not_repo(self, tmp_path):
        with pytest.raises(RuntimeError, match="Not a git repository"):
            get_git_root(tmp_path)


class TestWorktreeInfo:
    def test_fields(self):
        info = WorktreeInfo(
            name="test",
            path=Path("/tmp/wt"),
            branch="forge/worktree-test",
            base_dir=Path("/tmp/repo"),
        )
        assert info.name == "test"
        assert info.path == Path("/tmp/wt")
        assert info.branch == "forge/worktree-test"
        assert info._atexit_registered is False

    def test_register_atexit(self):
        info = WorktreeInfo(
            name="t", path=Path("/tmp/wt"), branch="b", base_dir=Path("/tmp")
        )
        with patch.object(atexit, "register") as mock_reg:
            info.register_atexit()
            assert mock_reg.called
            assert info._atexit_registered is True

    def test_register_atexit_idempotent(self):
        info = WorktreeInfo(
            name="t", path=Path("/tmp/wt"), branch="b", base_dir=Path("/tmp")
        )
        with patch.object(atexit, "register") as mock_reg:
            info.register_atexit()
            info.register_atexit()
            assert mock_reg.call_count == 1

    def test_unregister_atexit(self):
        info = WorktreeInfo(
            name="t", path=Path("/tmp/wt"), branch="b", base_dir=Path("/tmp")
        )
        with patch.object(atexit, "register"):
            info.register_atexit()
        with patch.object(atexit, "unregister") as mock_unreg:
            info.unregister_atexit()
            assert mock_unreg.called
            assert info._atexit_registered is False

    def test_unregister_noop_if_not_registered(self):
        info = WorktreeInfo(
            name="t", path=Path("/tmp/wt"), branch="b", base_dir=Path("/tmp")
        )
        # Should not raise
        info.unregister_atexit()


class TestCreateWorktree:
    def test_creates_worktree_with_name(self, tmp_path):
        _init_git_repo(tmp_path)
        info = create_worktree(tmp_path, name="test-wt")
        try:
            assert info.name == "test-wt"
            assert info.branch == "forge/worktree-test-wt"
            assert info.path.exists()
            assert info.base_dir == tmp_path.resolve()
            assert info._atexit_registered is True
            # Verify it's a valid git worktree
            result = _run_git(["status"], info.path)
            assert result.returncode == 0
        finally:
            remove_worktree(info)

    def test_creates_worktree_auto_name(self, tmp_path):
        _init_git_repo(tmp_path)
        info = create_worktree(tmp_path)
        try:
            assert info.name.startswith("forge-")
            assert len(info.name) == len("forge-") + 8
            assert info.path.exists()
        finally:
            remove_worktree(info)

    def test_raises_if_not_git_repo(self, tmp_path):
        with pytest.raises(RuntimeError, match="Not a git repository"):
            create_worktree(tmp_path, name="fail")

    def test_worktree_path_structure(self, tmp_path):
        _init_git_repo(tmp_path)
        info = create_worktree(tmp_path, name="structured")
        try:
            expected = tmp_path.resolve() / ".forge" / "worktrees" / "structured"
            assert info.path == expected
        finally:
            remove_worktree(info)


class TestRemoveWorktree:
    def test_removes_worktree_and_branch(self, tmp_path):
        _init_git_repo(tmp_path)
        info = create_worktree(tmp_path, name="to-remove")
        assert info.path.exists()

        remove_worktree(info)

        assert not info.path.exists()
        assert info._atexit_registered is False
        # Branch should be gone
        result = _run_git(["branch", "--list", info.branch], tmp_path)
        assert info.branch not in result.stdout

    def test_handles_already_removed(self, tmp_path):
        _init_git_repo(tmp_path)
        info = create_worktree(tmp_path, name="double-rm")
        remove_worktree(info)
        # Second remove should not raise
        remove_worktree(info)
