"""Integration tests — git worktree lifecycle using real repos in tmp_path."""

import subprocess
from pathlib import Path

import pytest

from forge.agent.worktree import create_worktree, is_git_repo, remove_worktree


@pytest.fixture
def git_repo(tmp_path):
    """Initialize a real git repo with one commit."""
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"],
        capture_output=True, check=True,
    )
    # Create initial commit (git worktree requires at least one commit)
    (tmp_path / "README.md").write_text("# Test")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "."],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return tmp_path


class TestWorktreeLifecycle:
    def test_create_and_remove(self, git_repo):
        info = create_worktree(git_repo, name="test-wt")
        assert info.path.exists()
        assert info.branch == "forge/worktree-test-wt"
        # Worktree should contain the same file
        assert (info.path / "README.md").exists()

        remove_worktree(info)
        assert not info.path.exists()

    def test_worktree_under_forge_dir(self, git_repo):
        info = create_worktree(git_repo, name="check-path")
        try:
            assert ".forge/worktrees/check-path" in str(info.path)
        finally:
            remove_worktree(info)

    def test_custom_name(self, git_repo):
        info = create_worktree(git_repo, name="my-feature")
        try:
            assert info.name == "my-feature"
            assert info.branch == "forge/worktree-my-feature"
        finally:
            remove_worktree(info)

    def test_not_git_repo_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Not a git repository"):
            create_worktree(tmp_path, name="fail")
