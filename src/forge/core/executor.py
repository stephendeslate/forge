"""Code execution with retry loop and self-correction.

Extracts code from model output, runs it in a subprocess, and feeds errors
back to the model for correction.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path

from forge.models.base import ModelBackend
from forge.prompts.refine import EXECUTOR_SYSTEM, FIX_ERROR_SYSTEM


@dataclass
class ExecutionResult:
    code: str
    stdout: str
    stderr: str
    returncode: int
    attempt: int

    @property
    def success(self) -> bool:
        return self.returncode == 0


def extract_code(text: str) -> str | None:
    """Extract the first fenced code block from model output."""
    # Match ```python ... ``` or ``` ... ```
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


async def execute_code(
    code: str,
    *,
    timeout: float = 30.0,
    cwd: str | None = None,
) -> ExecutionResult:
    """Execute Python code in a subprocess.

    WARNING: This runs with full host privileges — no sandboxing is applied.

    The code is written to a temp file and run with `python`, capturing
    stdout and stderr. Enforces a timeout.
    """
    if os.environ.get("FORGE_EXECUTOR_DISABLED"):
        raise RuntimeError("Code execution is disabled (FORGE_EXECUTOR_DISABLED is set)")

    # Use project-local tmp dir if cwd is provided, else system tmp
    tmp_dir = None
    if cwd:
        forge_tmp = Path(cwd) / ".forge" / "tmp"
        forge_tmp.mkdir(parents=True, exist_ok=True)
        tmp_dir = str(forge_tmp)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tmp_dir,
    ) as f:
        f.write(code)
        script_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            # Kill entire process group for clean cleanup
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                proc.kill()
            await proc.communicate()
            return ExecutionResult(
                code=code,
                stdout="",
                stderr=f"Execution timed out after {timeout}s",
                returncode=-1,
                attempt=0,
            )

        return ExecutionResult(
            code=code,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            returncode=proc.returncode or 0,
            attempt=0,
        )
    finally:
        Path(script_path).unlink(missing_ok=True)


async def run_with_retry(
    prompt: str,
    backend: ModelBackend,
    *,
    max_retries: int = 3,
    timeout: float = 30.0,
    context: str = "",
) -> list[ExecutionResult]:
    """Generate code, execute it, and retry on failure with self-correction.

    Returns list of all execution attempts (last one is the final result).
    """
    system = EXECUTOR_SYSTEM
    if context:
        system = f"{EXECUTOR_SYSTEM}\n\n{context}"

    results: list[ExecutionResult] = []

    # Initial generation
    response = await backend.generate(prompt, system=system)
    code = extract_code(response)

    if not code:
        results.append(ExecutionResult(
            code=response,
            stdout="",
            stderr="Could not extract code block from model output.",
            returncode=-1,
            attempt=1,
        ))
        return results

    for attempt in range(1, max_retries + 1):
        result = await execute_code(code, timeout=timeout)
        result.attempt = attempt
        results.append(result)

        if result.success:
            return results

        # Self-correction: feed error back to model
        fix_prompt = (
            f"## Original Request\n{prompt}\n\n"
            f"## Failing Script\n```python\n{code}\n```\n\n"
            f"## Error Output\n```\n{result.stderr or result.stdout}\n```"
        )
        response = await backend.generate(fix_prompt, system=FIX_ERROR_SYSTEM)
        fixed_code = extract_code(response)

        if not fixed_code:
            # Model didn't produce a code block — give up
            break

        code = fixed_code

    return results
