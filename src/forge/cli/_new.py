"""forge new — create a new project from an idea."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ._helpers import console


async def new_command(
    idea: str,
    output: str | None,
    stack: str | None,
    model: str | None,
    auto: bool,
    yolo: bool,
) -> None:
    """Create a new project directory and launch an agent session to build it."""
    from forge.prompts.new_project import build_initial_prompt, build_new_project_system, slugify

    # 1. Resolve output directory
    project_name = slugify(idea)
    output_dir = Path(output) if output else Path.cwd() / project_name
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Creating project in[/bold] {output_dir}")

    # 2. Init git repo (silently, skip if already a repo)
    git_dir = output_dir / ".git"
    if not git_dir.exists():
        subprocess.run(
            ["git", "init", "-q"],
            cwd=output_dir,
            check=True,
            capture_output=True,
        )
        console.print("[dim]Initialized git repository.[/dim]")

    # 3. Change to project directory — agent works relative to it
    os.chdir(output_dir)

    # 4. Build system prompt and initial prompt
    system = build_new_project_system(stack)
    initial_prompt = build_initial_prompt(idea, stack, auto)

    # 5. Launch agent session — the agent loop IS the build engine
    from forge.agent.loop import agent_repl
    from forge.agent.permissions import PermissionPolicy

    policy = PermissionPolicy.YOLO if yolo else PermissionPolicy.AUTO
    console.print(
        f"[dim]Stack: {stack or 'auto'} | "
        f"Model: {model or 'default'} | "
        f"Mode: {'auto' if auto else 'interactive'}[/dim]\n"
    )

    await agent_repl(
        initial_prompt=initial_prompt,
        permission=policy,
        system=system,
    )
