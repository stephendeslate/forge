"""Async subprocess utilities — avoid blocking the event loop."""

from __future__ import annotations

import asyncio
import subprocess


async def async_run(
    cmd: list[str] | str,
    *,
    capture_output: bool = True,
    text: bool = True,
    timeout: float = 30.0,
    cwd: str | None = None,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess without blocking the event loop.

    API mirrors subprocess.run() but executes in a thread pool executor.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            cmd,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            cwd=cwd,
            shell=shell,
        ),
    )
