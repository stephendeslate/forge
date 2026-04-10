"""forge new — create a new project from an idea."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ._helpers import console


def _detect_existing_project(output_dir: Path) -> dict | None:
    """Check if the output directory already has project files.

    Returns a dict with 'summary' string and 'needs_verify' bool,
    or None if the directory is empty/new.
    """
    if not output_dir.exists():
        return None

    source_files = [
        f for f in list(output_dir.rglob("*.ts")) + list(output_dir.rglob("*.tsx"))
        if "node_modules" not in str(f) and "dist" not in str(f)
    ]
    test_files = [
        f for f in source_files
        if ".spec." in f.name or ".test." in f.name
    ]

    has_build_plan = (output_dir / "BUILD_PLAN.md").exists()
    has_claude_md = (output_dir / "CLAUDE.md").exists()
    has_package = (output_dir / "package.json").exists()

    if not (has_build_plan or has_package or source_files):
        return None

    parts = []
    if has_build_plan:
        parts.append("BUILD_PLAN.md")
    if has_claude_md:
        parts.append("CLAUDE.md")
    if has_package:
        parts.append("package.json")
    if source_files:
        parts.append(f"{len(source_files)} TypeScript files")
    if test_files:
        parts.append(f"{len(test_files)} test files")

    # Project has code but no tests → needs verification, not more building
    needs_verify = len(source_files) > 10 and len(test_files) == 0

    return {"summary": ", ".join(parts), "needs_verify": needs_verify}


# ---------------------------------------------------------------------------
# Planning providers — cascade: Claude → Gemini → local
# ---------------------------------------------------------------------------


def _build_plan_prompt(idea: str, stack_guidance: str) -> str:
    """Build the prompt for generating BUILD_PLAN.md."""
    return (
        f"Generate a comprehensive BUILD_PLAN.md for this project:\n\n{idea}\n"
        f"{stack_guidance}\n\n"
        "The BUILD_PLAN.md must include:\n"
        "- Project overview and tech stack\n"
        "- Architecture: data model (entities with fields, types, and relationships), "
        "key user flows, module/directory structure\n"
        "- Key commands: EVERY npm/pnpm script that will exist in package.json files\n"
        "- Phased build order: 3-5 phases, each with 2-5 features, in dependency order\n"
        "- Test strategy and integration checkpoints\n\n"
        "Output ONLY the markdown content. No preamble, no explanation."
    )


def _build_claude_md_prompt() -> str:
    """Build the prompt for generating CLAUDE.md from BUILD_PLAN.md."""
    return (
        "Read the BUILD_PLAN.md in the current directory. Generate a CLAUDE.md "
        "that is STRICTLY DERIVED from it.\n\n"
        "Include:\n"
        "- What the project is (one paragraph — must match BUILD_PLAN overview)\n"
        "- Tech stack and key commands — ONLY commands from BUILD_PLAN\n"
        "- Architecture rules and conventions — matching BUILD_PLAN\n"
        "- Planned project structure (directory tree matching BUILD_PLAN modules)\n"
        "- Domain concepts (entity names and flows matching BUILD_PLAN EXACTLY)\n"
        "- No-go zones and constraints\n\n"
        "CRITICAL: Do NOT invent commands, packages, or features not in BUILD_PLAN.md.\n\n"
        "Output ONLY the markdown content. No preamble, no explanation."
    )


async def _plan_with_claude(idea: str, stack_guidance: str, output_dir: Path) -> bool:
    """Generate specs using Claude CLI (`claude -p`). Returns True on success."""
    if not shutil.which("claude"):
        return False

    console.print("[bold cyan]Planning with Claude[/bold cyan] (claude -p)")

    plan_prompt = (
        f"Generate project specs for this idea:\n\n{idea}\n{stack_guidance}\n\n"
        f"Do the following:\n"
        f"1. Write BUILD_PLAN.md with ALL of these sections:\n"
        f"   - Project overview and tech stack\n"
        f"   - Architecture: data model (entities with fields, types, relationships), "
        f"key user flows, module/directory structure\n"
        f"   - Key commands: EVERY npm/pnpm script that will exist in package.json files\n"
        f"   - Infrastructure requirements:\n"
        f"     * Guards (e.g., JwtAuthGuard)\n"
        f"     * Global pipes (ValidationPipe config)\n"
        f"     * Global filters (AllExceptionsFilter that catches Prisma P2002/P2025 errors)\n"
        f"     * Validation strategy (how input validation works — Zod or class-validator)\n"
        f"     * Error handling (constraint violations → 409, not found → 404)\n"
        f"     * Config/env vars (list all required env vars)\n"
        f"   - Shared package contract: what types/schemas live in packages/shared "
        f"and which apps import them\n"
        f"   - Phased build order: 3-5 phases, 2-5 features each, in dependency order\n"
        f"   - Test strategy and integration checkpoints\n"
        f"2. Read the BUILD_PLAN.md you just wrote.\n"
        f"3. Write CLAUDE.md strictly derived from BUILD_PLAN.md with: overview, "
        f"tech stack and commands (ONLY from BUILD_PLAN), architecture rules, "
        f"project structure, domain concepts, no-go zones.\n\n"
        f"Do not invent commands or features in CLAUDE.md that aren't in BUILD_PLAN.md."
    )

    try:
        console.print("[dim]  Generating specs (this may take a minute)...[/dim]")
        result = subprocess.run(
            ["claude", "-p", plan_prompt, "--dangerously-skip-permissions"],
            cwd=output_dir, capture_output=True, text=True, timeout=300,
        )

        # Check if files were written
        build_plan = output_dir / "BUILD_PLAN.md"
        claude_md = output_dir / "CLAUDE.md"

        if build_plan.exists() and build_plan.stat().st_size > 200:
            lines = len(build_plan.read_text().splitlines())
            console.print(f"[green]  ✓[/green] BUILD_PLAN.md ({lines} lines)")
        else:
            console.print("[yellow]  BUILD_PLAN.md not generated.[/yellow]")
            return False

        if claude_md.exists() and claude_md.stat().st_size > 100:
            lines = len(claude_md.read_text().splitlines())
            console.print(f"[green]  ✓[/green] CLAUDE.md ({lines} lines)")
        else:
            console.print("[yellow]  CLAUDE.md not generated — will create during build.[/yellow]")

        return True

    except subprocess.TimeoutExpired:
        console.print("[yellow]  Claude planning timed out.[/yellow]")
        return False
    except (FileNotFoundError, OSError) as e:
        console.print(f"[yellow]  Claude CLI error: {e}[/yellow]")
        return False


async def _plan_with_gemini(idea: str, stack_guidance: str, output_dir: Path) -> bool:
    """Generate specs using Gemini API via pydantic-ai. Returns True on success."""
    from forge.agent.gemini import _ensure_api_key, get_gemini_model_settings

    api_key = _ensure_api_key()
    if not api_key:
        return False

    from forge.config import settings
    model_name = f"google-gla:{settings.gemini.model}"
    model_settings = get_gemini_model_settings() or {}

    console.print(f"[bold cyan]Planning with Gemini[/bold cyan] ({settings.gemini.model})")

    try:
        from pydantic_ai import Agent

        # Step 1: BUILD_PLAN.md
        prompt = _build_plan_prompt(idea, stack_guidance)
        agent = Agent(model=model_name, instructions="You are a software architect. Output only markdown.")
        import asyncio
        result = await agent.run(prompt, model_settings=model_settings)
        content = result.output.strip()

        if not content or len(content) < 200:
            console.print("[yellow]  Gemini returned insufficient content.[/yellow]")
            return False

        (output_dir / "BUILD_PLAN.md").write_text(content)
        console.print(f"[green]  ✓[/green] BUILD_PLAN.md ({len(content.splitlines())} lines)")

        # Step 2: CLAUDE.md — include BUILD_PLAN as context
        prompt2 = f"Here is BUILD_PLAN.md:\n\n{content}\n\n---\n\n{_build_claude_md_prompt()}"
        result2 = await agent.run(prompt2, model_settings=model_settings)
        content2 = result2.output.strip()

        if content2 and len(content2) >= 100:
            (output_dir / "CLAUDE.md").write_text(content2)
            console.print(f"[green]  ✓[/green] CLAUDE.md ({len(content2.splitlines())} lines)")
        else:
            console.print("[yellow]  CLAUDE.md insufficient — will generate during build.[/yellow]")

    except Exception as e:
        console.print(f"[yellow]  Gemini error: {e}[/yellow]")
        return False

    return True


async def _plan_specs(idea: str, stack: str | None, output_dir: Path) -> bool:
    """Generate BUILD_PLAN.md + CLAUDE.md using the best available model.

    Cascade: Claude CLI → Gemini API → (return False, let local model handle it)
    """
    from forge.prompts.new_project import STACK_PRESETS

    stack_guidance = ""
    if stack and stack in STACK_PRESETS:
        stack_guidance = f"\nTech stack:\n{STACK_PRESETS[stack]}"
    elif stack:
        stack_guidance = f"\nUse the {stack} tech stack."

    # Try Claude first (highest quality, uses OAuth — no API key needed)
    if await _plan_with_claude(idea, stack_guidance, output_dir):
        return True

    # Fall back to Gemini (good quality, needs API key)
    if await _plan_with_gemini(idea, stack_guidance, output_dir):
        return True

    # Both unavailable — local model will handle planning in the agent loop
    console.print("[dim]No cloud model available for planning. Local model will handle everything.[/dim]")
    return False


async def _verify_with_claude(output_dir: Path) -> bool:
    """Run Claude CLI to verify and complete the project (write missing tests, fix gaps).

    Returns True if Claude was available and ran, False otherwise.
    """
    if not shutil.which("claude"):
        return False

    console.print("\n[bold cyan]Verification with Claude[/bold cyan] (claude -p)")

    verify_prompt = (
        "You are verifying and completing a NestJS + Next.js project. "
        "Read BUILD_PLAN.md to understand what was planned. "
        "Then systematically check and fix ALL gaps:\n\n"
        "## 1. Module Wiring (check FIRST — most common failure)\n"
        "Every NestJS module that injects a service from another module MUST import that module.\n"
        "- Read every *.module.ts file\n"
        "- If any module's service injects PrismaService but the module doesn't import PrismaModule, fix it\n"
        "- Verify by building: `cd apps/api && npx nest build`\n\n"
        "## 2. Runtime Boot (CRITICAL)\n"
        "- Build the API: `cd apps/api && npx nest build`\n"
        "- Start the API: `cd apps/api && timeout 8 node dist/main.js 2>&1 || true`\n"
        "- If it crashes with DI errors, fix module imports and retry\n"
        "- If it crashes with missing packages, install them and retry\n"
        "- It MUST print 'Nest application successfully started' before proceeding\n\n"
        "## 3. Infrastructure (main.ts)\n"
        "Ensure main.ts has ALL of the following:\n"
        "- `app.enableCors({ origin: process.env.CORS_ORIGIN || 'http://localhost:3000', credentials: true })`\n"
        "- `app.useGlobalPipes(new ValidationPipe({ whitelist: true, transform: true }))`\n"
        "- `app.useGlobalFilters(new AllExceptionsFilter())` — create this filter if it doesn't exist. "
        "It must catch PrismaClientKnownRequestError (P2002 → 409, P2025 → 404) and return clean JSON.\n"
        "- `app.enableShutdownHooks()`\n"
        "- Port from env with default 3001 (NOT 3000)\n"
        "- .env in .gitignore\n\n"
        "## 4. Security\n"
        "- Auth endpoints MUST NOT return passwordHash. Use Prisma select to exclude it.\n"
        "- JWT secret MUST NOT have a hardcoded fallback in source code. Throw on missing JWT_SECRET.\n"
        "- Ensure .env DATABASE_URL port matches docker-compose.yml port mapping\n\n"
        "## 5. Error Handling\n"
        "- Services must NOT use find-then-create/update patterns. Use direct create/update with "
        "try/catch on PrismaClientKnownRequestError (P2002 for unique constraint → ConflictException, "
        "P2025 for not found → NotFoundException).\n"
        "- Every list endpoint must support pagination (take/skip with defaults)\n\n"
        "## 6. Shared Package Contract\n"
        "- Read packages/shared/src/ — if Zod schemas or types exist there, verify the API actually "
        "imports and uses them. If the API defines its own inline types that duplicate shared types, "
        "replace them with imports from @app/shared (or whatever the package name is).\n"
        "- Verify the frontend imports types from the shared package for API response shapes.\n\n"
        "## 7. Frontend-Backend Contract\n"
        "- Read every frontend page that fetches from the API.\n"
        "- Verify the response field names the frontend destructures EXACTLY match what the API returns.\n"
        "- If the frontend reads `stats.total` but the API returns `totalTasks`, fix ONE side to match.\n"
        "- Verify the auth flow uses ONE token strategy consistently (either httpOnly cookies with "
        "server-side reading OR localStorage with client-side reading — NOT both).\n\n"
        "## 8. Tests\n"
        "- Write unit tests for EVERY service file. Use vitest with mocked Prisma.\n"
        "- Test both happy paths AND error paths (not found, duplicate, invalid input).\n"
        "- Run `pnpm test` — fix until all tests pass.\n\n"
        "## 9. Functional Smoke Test (do this LAST)\n"
        "Start the API and run this exact sequence. If any step fails, fix the code and retry:\n"
        "```\n"
        "# Build and start\n"
        "cd apps/api && npx nest build && PORT=3001 node dist/main.js &\n"
        "sleep 3\n"
        "# 1. Register\n"
        "curl -s -X POST http://localhost:3001/auth/register -H 'Content-Type: application/json' "
        "-d '{\"email\":\"smoke@test.com\",\"password\":\"Test1234!\",\"name\":\"Smoke\"}'\n"
        "# 2. Login (get token)\n"
        "TOKEN=$(curl -s -X POST http://localhost:3001/auth/login -H 'Content-Type: application/json' "
        "-d '{\"email\":\"smoke@test.com\",\"password\":\"Test1234!\"}' | jq -r '.access_token')\n"
        "# 3. Create task\n"
        "curl -s -X POST http://localhost:3001/tasks -H 'Content-Type: application/json' "
        "-H \"Authorization: Bearer $TOKEN\" -d '{\"title\":\"Test task\",\"priority\":\"HIGH\"}'\n"
        "# 4. List tasks (should return 1)\n"
        "curl -s http://localhost:3001/tasks -H \"Authorization: Bearer $TOKEN\"\n"
        "# 5. Check dashboard (should show count > 0)\n"
        "curl -s http://localhost:3001/dashboard/stats -H \"Authorization: Bearer $TOKEN\"\n"
        "# Kill server\n"
        "kill %1 2>/dev/null\n"
        "```\n"
        "Every step must return the expected status code (201 for register, 200 for login with token, "
        "201 for task create, 200 for list with data, 200 for dashboard with counts > 0).\n"
        "If ANY step fails, read the error, fix the code, rebuild, and re-run the smoke test.\n\n"
        "## 10. Cleanup & Commit\n"
        "- Remove .bak files, compiled .js/.d.ts in source dirs\n"
        "- Ensure .env is in .gitignore\n"
        "- Commit: `git add -A && git commit -m 'fix: verification - infrastructure, security, tests, smoke test'`\n\n"
        "Do not ask questions. Fix everything systematically in order. Commit when done."
    )

    try:
        console.print("[dim]  Running verification (this may take a few minutes)...[/dim]")
        result = subprocess.run(
            ["claude", "-p", verify_prompt, "--dangerously-skip-permissions"],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )

        if result.returncode == 0:
            output_lines = result.stdout.strip().count('\n') + 1
            console.print(f"[green]  ✓[/green] Verification complete ({output_lines} lines of output)")
            return True
        else:
            console.print(f"[yellow]  Claude returned exit code {result.returncode}[/yellow]")
            return True  # Still ran, may have made partial progress

    except subprocess.TimeoutExpired:
        console.print("[yellow]  Claude verification timed out (10 min). Partial progress may exist.[/yellow]")
        return True
    except (FileNotFoundError, OSError) as e:
        console.print(f"[yellow]  Claude CLI error: {e}[/yellow]")
        return False


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


def _reset_directory(output_dir: Path) -> None:
    """Wipe the output directory to a clean state, preserving only .git history.

    Removes all files and directories except .git/, then resets the git index.
    This guarantees a clean environment with no stale code, broken node_modules,
    or leftover config from previous runs.
    """
    if not output_dir.exists():
        return

    for item in output_dir.iterdir():
        if item.name == ".git":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # Reset git staging area (keep commit history for reference)
    subprocess.run(
        ["git", "reset", "HEAD", "--quiet"],
        cwd=output_dir, capture_output=True,
    )

    console.print("[yellow]  Directory wiped clean (git history preserved).[/yellow]")


async def new_command(
    idea: str,
    output: str | None,
    stack: str | None,
    model: str | None,
    auto: bool,
    yolo: bool,
    clean: bool = False,
) -> None:
    """Create a new project directory and launch an agent session to build it."""
    from forge.prompts.new_project import (
        build_continue_prompt,
        build_initial_prompt,
        build_new_project_system,
        build_verify_prompt,
        slugify,
    )

    # 1. Resolve output directory
    project_name = slugify(idea)
    output_dir = Path(output) if output else Path.cwd() / project_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Init git repo
    git_dir = output_dir / ".git"
    if not git_dir.exists():
        subprocess.run(["git", "init", "-q"], cwd=output_dir, check=True, capture_output=True)

    # 3. Clean if requested — guaranteed fresh start
    if clean:
        console.print("[bold]Resetting environment...[/bold]")
        _reset_directory(output_dir)

    # 4. Detect existing project or generate specs
    existing = _detect_existing_project(output_dir)

    if existing:
        console.print(f"[bold]Continuing project in[/bold] {output_dir}")
        console.print(f"[dim]Found: {existing['summary']}[/dim]")
        if existing["needs_verify"]:
            console.print("[cyan]Project has code but missing tests — running verification.[/cyan]")
            # Use Claude for verification (surgical, high-quality)
            claude_verified = await _verify_with_claude(output_dir)
            if claude_verified:
                # Claude handled verification — use verify prompt for local model follow-up
                initial_prompt = build_verify_prompt(idea, stack, auto)
            else:
                # No Claude — local model does verification
                initial_prompt = build_verify_prompt(idea, stack, auto)
        else:
            initial_prompt = build_continue_prompt(idea, stack, auto)
    else:
        console.print(f"[bold]Creating project in[/bold] {output_dir}")

        # Planning phase: Claude → Gemini → local (cascade)
        cloud_planned = await _plan_specs(idea, stack, output_dir)

        if cloud_planned:
            # Specs done by cloud model — agent starts building from them
            initial_prompt = build_continue_prompt(idea, stack, auto)
        else:
            # Local model handles everything
            initial_prompt = build_initial_prompt(idea, stack, auto)

    # 4. Change to project directory
    os.chdir(output_dir)

    # 5. Launch agent session
    from forge.agent.loop import agent_repl
    from forge.agent.permissions import PermissionPolicy

    system = build_new_project_system(stack)
    policy = PermissionPolicy.YOLO if yolo else PermissionPolicy.AUTO

    # Determine which planner was used for display
    planner = "cloud" if (output_dir / "BUILD_PLAN.md").exists() and not existing else "local"
    console.print(
        f"[dim]Plan: {planner} | Build: local | Stack: {stack or 'auto'} | "
        f"Mode: {'auto' if auto else 'interactive'}[/dim]\n"
    )

    await agent_repl(
        initial_prompt=initial_prompt,
        permission=policy,
        system=system,
        headless=auto,
    )
