"""Git worktree management for isolated agent sessions."""

from __future__ import annotations

import atexit
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from forge.log import get_logger

logger = get_logger(__name__)


@dataclass
class WorktreeInfo:
    """Tracks an active git worktree created for an agent session."""

    name: str
    path: Path
    branch: str
    base_dir: Path
    _atexit_registered: bool = field(default=False, repr=False)

    def register_atexit(self) -> None:
        """Register atexit handler for crash safety."""
        if self._atexit_registered:
            return

        def _cleanup() -> None:
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(self.path)],
                    cwd=str(self.base_dir),
                    capture_output=True,
                    timeout=10,
                )
                logger.debug("Atexit: removed worktree %s", self.path)
            except Exception:
                logger.debug("Atexit: failed to remove worktree", exc_info=True)

        self._cleanup_fn = _cleanup  # type: ignore[attr-defined]
        atexit.register(_cleanup)
        self._atexit_registered = True

    def unregister_atexit(self) -> None:
        """Unregister atexit handler (user chose to keep worktree)."""
        if self._atexit_registered and hasattr(self, "_cleanup_fn"):
            atexit.unregister(self._cleanup_fn)  # type: ignore[attr-defined]
            self._atexit_registered = False


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a git command and return the result."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
    )


def is_git_repo(path: Path) -> bool:
    """Check if path is inside a git repository."""
    result = _run_git(["rev-parse", "--is-inside-work-tree"], path)
    return result.returncode == 0


def get_git_root(path: Path) -> Path:
    """Get the root directory of the git repository."""
    result = _run_git(["rev-parse", "--show-toplevel"], path)
    if result.returncode != 0:
        raise RuntimeError(f"Not a git repository: {path}")
    return Path(result.stdout.strip())


def create_worktree(base_dir: Path, name: str | None = None) -> WorktreeInfo:
    """Create an isolated git worktree for agent work.

    Args:
        base_dir: Directory inside the git repository.
        name: Optional name for the worktree. Auto-generated if None.

    Returns:
        WorktreeInfo with path and branch details.

    Raises:
        RuntimeError: If not in a git repo or worktree creation fails.
    """
    if not is_git_repo(base_dir):
        raise RuntimeError(f"Not a git repository: {base_dir}")

    git_root = get_git_root(base_dir)
    name = name or f"forge-{uuid4().hex[:8]}"
    worktree_path = git_root / ".forge" / "worktrees" / name
    branch = f"forge/worktree-{name}"

    # Ensure parent directory exists
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    result = _run_git(
        ["worktree", "add", "-b", branch, str(worktree_path), "HEAD"],
        git_root,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create worktree: {result.stderr.strip()}")

    info = WorktreeInfo(
        name=name,
        path=worktree_path,
        branch=branch,
        base_dir=git_root,
    )
    info.register_atexit()
    return info


def remove_worktree(info: WorktreeInfo) -> None:
    """Remove a git worktree and its branch.

    Args:
        info: WorktreeInfo from create_worktree.
    """
    info.unregister_atexit()

    # Remove worktree
    result = _run_git(
        ["worktree", "remove", "--force", str(info.path)],
        info.base_dir,
    )
    if result.returncode != 0:
        logger.warning("Failed to remove worktree: %s", result.stderr.strip())

    # Soft-delete branch (warn on unmerged)
    result = _run_git(["branch", "-d", info.branch], info.base_dir)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "not fully merged" in stderr:
            logger.warning(
                "Branch %s has unmerged changes — keeping it. "
                "Delete manually with: git branch -D %s",
                info.branch,
                info.branch,
            )
        else:
            logger.debug("Branch delete note: %s", stderr)


async def prompt_worktree_cleanup(info: WorktreeInfo) -> bool:
    """Ask user whether to keep the worktree.

    Returns True if user wants to keep it.
    """
    import asyncio

    from prompt_toolkit import prompt as pt_prompt

    print(f"\nWorktree: {info.path}")
    print(f"Branch:   {info.branch}")

    try:
        answer = await asyncio.get_running_loop().run_in_executor(
            None, lambda: pt_prompt("Keep worktree? [y/N] ")
        )
        return answer.strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False
