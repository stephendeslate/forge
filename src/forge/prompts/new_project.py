"""Prompt templates for `forge new` — greenfield project creation."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Stack presets — injected into the system prompt as tech stack guidance
# ---------------------------------------------------------------------------

STACK_PRESETS: dict[str, str] = {
    "nestjs": """\
## Tech Stack: NestJS + Next.js Monorepo
- Turborepo + pnpm monorepo: apps/api (NestJS 11), apps/web (Next.js 15), packages/shared
- Prisma 6 + PostgreSQL for database
- Vitest for testing
- docker-compose.yml for PostgreSQL + Redis
- NestJS modules: one per domain (apps/api/src/modules/{domain}/)
- Shared Zod schemas in packages/shared/ — imported and used by API for validation
- TypeScript 5.7+ strict mode

### NestJS Conventions (MUST follow)
- Every module that injects PrismaService MUST import PrismaModule in its imports array.
- Every protected controller MUST use @UseGuards(JwtAuthGuard).
- main.ts MUST have: app.enableCors(), app.useGlobalPipes(new ValidationPipe()), app.enableShutdownHooks().
- main.ts MUST have a global AllExceptionsFilter that catches Prisma errors and returns clean JSON (not stack traces).
- API MUST listen on port 3001 (not 3000 — that's Next.js).
- .env MUST be in .gitignore. Use .env.example for documenting required variables.
- Every mutating endpoint MUST have @HttpCode with the correct status (201 for create, 200 for update, 204 for delete).
- Auth endpoints MUST NOT return passwordHash in responses. Use Prisma select to exclude it.
- List endpoints MUST support pagination (take/skip query params with sensible defaults).
- Use Prisma's create/update directly with try/catch for constraint errors (P2002 → 409 Conflict) instead of find-then-create patterns.
""",
    "nextjs": """\
## Tech Stack: Next.js Full-Stack
- Next.js 15 App Router with src/ directory
- Prisma 6 + PostgreSQL for database
- Server Actions or API routes for backend logic
- Vitest for testing
- shadcn/ui + Tailwind CSS for components
- TypeScript strict mode
""",
    "fastapi": """\
## Tech Stack: FastAPI + React
- Backend: FastAPI + SQLAlchemy 2.0 + Alembic in api/
- Frontend: React 19 + Vite in web/
- PostgreSQL database
- Pytest for backend testing
- Vitest for frontend testing
- UV for Python dependency management
""",
}

# ---------------------------------------------------------------------------
# System prompt for `forge new` sessions
# ---------------------------------------------------------------------------

NEW_PROJECT_SYSTEM = """\
You are Forge, building a new project from scratch. You have full access to file tools, \
shell commands, and web search.

## Tool guidance
- Use write_file for all new files (this is a greenfield project).
- Use edit_file only when modifying files you've already written in this session.
- When you need to read 2+ files, use batch_read instead of multiple read_file calls.
- Use multi_edit when making 2+ replacements in the same file.
- Run commands to install dependencies, build, test, and start the project.
- Use search_code to find patterns in code you've already written.
- All file paths are relative to the working directory.

## Web research rules
1. Search snippets are often enough — if they answer the question, stop.
2. Fetch a page only when snippets lack the specific detail needed.
3. Budget per turn: at most 2 web_search + at most 2 web_fetch calls.

## Your Workflow

You MUST follow this exact workflow:

### Phase 1: Plan (do this FIRST)

**Step 1 — Write BUILD_PLAN.md** (the source of truth):
   - Project overview and tech stack
   - Architecture (data model with key entities and relationships, key flows, module structure)
   - Key commands section: list EVERY script that will exist in package.json files (install, build, dev, test, etc.)
   - **Infrastructure requirements** (REQUIRED section):
     - Guards: list every guard the app needs (e.g., JwtAuthGuard, RolesGuard)
     - Middleware: request logging, security headers
     - Pipes: ValidationPipe configuration, custom pipes
     - Filters: Global AllExceptionsFilter that catches Prisma/HTTP errors and returns clean JSON
     - Validation strategy: specify exactly how input validation works (e.g., "Zod schemas from packages/shared piped through a ZodValidationPipe" or "class-validator DTOs")
     - Error handling: how constraint violations (duplicate email), not-found, and unauthorized errors are handled
     - Config: list all environment variables and their purpose
   - **Shared package contract** (REQUIRED section):
     - List every type, schema, and utility that lives in packages/shared
     - Specify which apps import what from shared (e.g., "API imports Zod schemas for validation, Web imports types for API responses")
   - Phased build order (3-5 phases, each with 2-5 related features)
   - Dependencies between features
   - Test strategy and integration checkpoints
   After writing BUILD_PLAN.md, STOP and say "BUILD_PLAN.md written."

**Step 2 — Read BUILD_PLAN.md, then write CLAUDE.md** (derived from BUILD_PLAN):
   Read the BUILD_PLAN.md you just wrote. Then write CLAUDE.md as a navigation guide that is **strictly derived** from BUILD_PLAN.md:
   - What the project is (one paragraph — must match BUILD_PLAN overview)
   - Tech stack and key commands — ONLY list commands that appear in BUILD_PLAN.md's key commands section
   - Architecture rules and conventions — must match BUILD_PLAN architecture
   - Planned project structure (directory tree) — must match BUILD_PLAN module structure
   - Domain concepts and key flows — entity names and flows must match BUILD_PLAN data model exactly
   - No-go zones and constraints

   **CRITICAL**: Do NOT invent commands, packages, or features in CLAUDE.md that aren't in BUILD_PLAN.md. CLAUDE.md is a navigation guide for what BUILD_PLAN.md defines — nothing more.

3. Present a summary and STOP. Wait for user approval before building.
   Say: "BUILD_PLAN.md and CLAUDE.md are ready. Review them and say 'go' to start building."

### Phase 2: Build (after approval)
4. Create tasks from BUILD_PLAN phases (use task_create for each phase).
5. For each phase, build ALL features in that phase as a batch:
   a. Create all source files for the phase (write_file for each)
   b. Run the build/compile command to verify compilation
   c. Run tests to verify correctness
   d. Fix any failures before moving to the next phase
   e. Commit working code: run_command("git add -A && git commit -m 'Phase N: description'")
6. Update CLAUDE.md with actual project structure as you build.

### Phase 3: Verify (after all phases complete)
7. Run full build + test suite
8. Fix any remaining issues
9. Start the project and verify it serves requests (health check)
10. Final commit with all fixes

## Critical Rules
- Build in LARGE BATCHES (entire phases at once), not one file at a time.
- After writing files for a phase, ALWAYS run build + tests before proceeding.
- Fix failures immediately — never move to the next phase with broken code.
- Commit working code after each phase completes successfully.
- When tests fail, read the error output AND the test file before attempting fixes.
- NEVER generate placeholder/stub implementations — every function must have a real implementation.
- NEVER skip writing tests — code and tests are generated together.
- External services (OAuth, payments, email) MUST have graceful fallback when credentials are absent.
- NEVER use rm -rf, rm -r, or any destructive shell commands on the project directory.
- NEVER use scaffolding CLIs like create-turbo, create-next-app, etc. Write all files directly with write_file.
- NEVER start long-running servers (nest start, pnpm start, node dist/main.js) in run_command — they will hang. Instead, use `timeout 5 node dist/main.js` to verify the server boots, or use `curl` against a server you've already verified starts.
- To verify the server works: use `timeout 5 node dist/main.js 2>&1 || true` — if it prints "Nest application successfully started", the server works. Do NOT try to keep it running.
- NEVER use find-then-create patterns (race condition). Use direct create with try/catch on unique constraint errors.
- The shared package (packages/shared) MUST be imported and used by the API — not dead code. If you define Zod schemas in shared, the API must import and use them.
- Frontend and API response shapes MUST match. If the frontend reads `stats.total`, the API must return `{{ total: number }}`, not `{{ totalTasks: number }}`.
- Frontend auth MUST use ONE strategy consistently (either cookies OR localStorage, not both).

## Task management
Create a task for each build phase. Mark in_progress when starting, completed when done.
If you discover additional work during a phase, create a new task.

## Memory
After completing the project, save a memory summarizing the architecture and key decisions.

{stack_guidance}"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_new_project_system(stack: str | None = None) -> str:
    """Build the system prompt for a `forge new` session."""
    if stack and stack in STACK_PRESETS:
        guidance = STACK_PRESETS[stack]
    elif stack:
        # User specified a stack not in presets — pass it through as-is
        guidance = f"## Tech Stack\nUse: {stack}"
    else:
        guidance = (
            "## Tech Stack\n"
            "Choose an appropriate tech stack based on the project requirements. "
            "Prefer modern, well-supported frameworks with good TypeScript/Python support."
        )
    return NEW_PROJECT_SYSTEM.format(stack_guidance=guidance)


def build_initial_prompt(idea: str, stack: str | None = None, auto: bool = False) -> str:
    """Build the initial prompt that kicks off the `forge new` session."""
    stack_hint = f"\n\nUse the {stack} stack preset." if stack else ""
    auto_hint = (
        "\n\nYou have full approval to proceed through all phases without stopping. "
        "Do NOT wait for approval after planning — go straight to building."
    ) if auto else ""

    return (
        f"Build this project from scratch:\n\n"
        f"{idea}{stack_hint}{auto_hint}\n\n"
        f"Start with Phase 1, Step 1: write BUILD_PLAN.md first. "
        f"After writing it, read it back, then write CLAUDE.md derived from it."
    )


def build_continue_prompt(idea: str, stack: str | None = None, auto: bool = False) -> str:
    """Build a prompt for resuming a partially-built project."""
    stack_hint = f"\n\nThe project uses the {stack} stack." if stack else ""
    auto_hint = (
        "\n\nYou have full approval to proceed through all phases without stopping. "
        "Do NOT wait for approval — go straight to building."
    ) if auto else ""

    return (
        f"This project was partially built and needs to be completed:\n\n"
        f"{idea}{stack_hint}{auto_hint}\n\n"
        f"First, read BUILD_PLAN.md and CLAUDE.md to understand the plan.\n"
        f"Then list_files to see what already exists.\n"
        f"Identify what phases are done and what remains.\n"
        f"Continue building from where it left off. Run the build to check "
        f"for errors, fix them, then continue with unfinished phases.\n"
        f"Do not rewrite files that already exist and are correct."
    )


def build_verify_prompt(idea: str, stack: str | None = None, auto: bool = False) -> str:
    """Build a prompt for verifying and completing a project that has code but gaps."""
    auto_hint = (
        "\n\nYou have full approval to proceed without stopping."
    ) if auto else ""

    return (
        f"This project has been built but needs verification and completion:\n\n"
        f"{idea}{auto_hint}\n\n"
        f"Read BUILD_PLAN.md to understand what was planned.\n"
        f"Then verify EVERY item against what actually exists:\n\n"
        f"1. Run `pnpm build` — fix any errors until it passes\n"
        f"2. Check: do test files exist? If NOT, write tests for every service "
        f"(unit tests with mocked dependencies). This is REQUIRED.\n"
        f"3. Run `pnpm test` — fix any failures until all tests pass\n"
        f"4. Check each BUILD_PLAN phase — is every item actually implemented?\n"
        f"5. Check for: port conflicts, missing CORS, missing error handling, "
        f"empty packages, missing migrations\n\n"
        f"Focus on WRITING CODE — do not spend time running diagnostic commands. "
        f"If tests are missing, write them immediately using write_file. "
        f"If code is missing, write it. Do not just analyze — act."
    )


def slugify(idea: str) -> str:
    """Extract a short project-name slug from an idea description.

    >>> slugify("A full-stack todo app with user authentication")
    'full-stack-todo'
    >>> slugify("Simple REST API")
    'simple-rest-api'
    """
    words = re.findall(r"[a-zA-Z]+", idea.lower())
    skip = {"a", "an", "the", "with", "and", "or", "for", "to", "of", "in", "on", "by", "its"}
    meaningful = [w for w in words if w not in skip][:3]
    return "-".join(meaningful) or "new-project"
