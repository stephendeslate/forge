# Forge — Local AI Orchestration

## Quick Reference

- **Language:** Python 3.12, managed with `uv`
- **Entry point:** `forge.cli:app` (typer)
- **Config:** `~/.config/forge/config.toml` + env vars (`FORGE_*`)
- **Models:** Ollama (GPU, ROCm) — heavy: `qwen3-coder-next:q8_0` (131K ctx), fast: `qwen3.5:4b` (8K ctx via `fast_num_ctx`); NPU (FastFlowLM) — `llama-3.2-3b`; Gemini (cloud) — `gemini-2.5-flash` for planning/critique/recovery

## Commands

```bash
forge                           # Interactive REPL (defaults to agent mode)
forge agent                     # Agentic coding mode (tool-using REPL)
forge agent "read X and fix Y"  # Agentic mode with initial prompt
forge agent --yolo              # Skip permission prompts for writes/commands
forge agent --ask               # Prompt for every tool call
forge agent --worktree          # Run in isolated git worktree (auto-named)
forge agent --worktree --worktree-name feat  # Named worktree
forge ask "question"            # One-shot query (auto-routes)
forge ask --gpu "complex"       # Force heavy model
forge ask --fast "quick"        # Force fast model
forge ask --npu "hello"         # Force NPU model (FastFlowLM)
forge code                      # REPL with RAG context from indexed project
forge code "explain X" -p proj  # One-shot with RAG from specific project
forge index .                   # Index cwd for RAG
forge index /path -p myproject  # Index specific directory
forge index . --force           # Re-index all files (ignore hashes)
forge draft "build X"           # Draft → Critique → Refine pipeline
forge draft "X" --show-stages   # Show all pipeline stages
forge run "write a script..."   # Generate + execute + self-correct
forge status                    # Show models, DB, config
forge history                   # List recent conversation sessions
forge history --limit 50        # List more sessions
forge resume                    # Resume most recent session
forge resume <session_id>       # Resume a specific session (prefix match)
forge serve                     # Start as MCP server (stdio transport)
forge serve --cwd /path         # MCP server rooted at specific directory
```

### Agent REPL Commands

```
/help         — show all commands and keyboard shortcuts
/compact      — force context compaction (shows before/after token counts)
/tokens       — show current token estimate
/plan <task>  — plan before executing (two-phase: plan → review → execute)
/plan-status  — show active plan
/think        — toggle extended thinking (qwen3 /think tag)
/tasks        — show task list with status and dependencies
/memory       — show memory stats or search memories
/forget <id>  — delete a memory by numeric ID
/exemplars    — list captured cloud model exemplars (delete with /exemplars delete <id>)
/mcp          — list connected MCP servers
/checkpoint   — save conversation checkpoint [name]
/restore <n>  — restore to named checkpoint
/checkpoints  — list saved checkpoints
/index        — index/reindex project for RAG search
/worktree [n] — create isolated git worktree (optional name)
/cd <dir>     — change working directory (reloads agent context)
/cwd          — show current working directory
/model <name> — switch model mid-session
/exit         — exit the REPL
```

### Keyboard Shortcuts

- **Ctrl-O** — toggle status line visibility
- **Ctrl-R** — toggle tool result visibility
- **Ctrl-C** — interrupt current generation
- **Ctrl-D** — exit REPL

## Architecture

```
CLI (typer) → Router (heuristic) → OllamaBackend (pydantic-ai)
                                     ├── gpu-heavy: qwen3-coder-next:q8_0
                                     ├── gpu-fast:  qwen3.5:4b
                                     └── npu:       llama-3.2-3b (FastFlowLM, :52625)

forge index → walk → tree-sitter chunk → embed (nomic-v2-moe) → pgvector store
forge code  → embed query → pgvector search → inject context → generate
forge agent → pydantic-ai Agent(tools=[...]) → agentic loop
               ├── tools: read/write/edit/run/search/web_search/web_fetch
               ├── task tools: task_create/task_update/task_list/task_get
               ├── memory tools: save_memory/recall_memories
               ├── exemplar learning: cloud successes → few-shot injection
               ├── MCP: external tool servers via .forge/mcp.json
               └── hooks: pre/post tool use events, permission enforcement
forge serve → MCP server (stdio) — exposes Forge tools to external clients
               ├── file tools: read/write/edit/search/list/run_command
               ├── forge_ask: local model inference (heavy/fast/npu)
               ├── forge_rag_search: semantic code search
               └── forge_memory_recall/save: cross-session memory
```

- **Router** uses keyword heuristics (no LLM call) to classify prompts; auto-routes short simple prompts to NPU when enabled
- **Pydantic AI** wraps Ollama via OpenAI-compatible API (`OLLAMA_BASE_URL` env var with `/v1` suffix)
- **Streaming** via `Agent.run_stream()` → `stream_text(delta=True)` → Rich Live display
- **Agent mode** uses `Agent.run()` with `event_stream_handler` for tool-calling loop + streaming display
- **Hooks** event-driven system (`PreToolUse`, `PostToolUse`, `SessionStart/End`, etc.) with priority ordering and block/allow semantics
- **Tasks** in-memory store with `PENDING → IN_PROGRESS → COMPLETED` lifecycle, dependency tracking, system prompt injection
- **Memory** cross-session persistence via pgvector embeddings — categories: `feedback`, `project`, `user`, `reference`; semantic recall + auto-pruning
- **MCP** discovers `.forge/mcp.json` (project-local) and `~/.config/forge/mcp.json` (global); `${VAR}` expanded by pydantic-ai natively; project overrides global; built-in Playwright browser server auto-discovered if npx is on PATH; set server to `false` in config to disable
- **Context compaction** 3-tier: truncate tool results → summarize task sequences → full LLM summarization with domain-aware prompt; auto-triggers at 80% token budget
- **Worktrees** git worktree isolation via `--worktree` flag or `/worktree` command; atexit crash safety, cleanup prompt on exit
- **RAG** tree-sitter AST chunking → nomic-embed-text-v2-moe (768d) → pgvector cosine search
- **Sandbox** command blocklist hook (`sandbox.py`) blocks dangerous commands (sudo, rm -rf /, curl|sh, etc.) and path boundary enforcement restricts file tools to cwd + /tmp; configurable via `SandboxSettings`
- **Multimodal** `@path/to/image.png` syntax in REPL input attaches images via `BinaryContent`; auto-routes to `vision_model` if configured; `read_file` detects images and returns metadata
- **Exemplar learning** cloud model successes (recovery, planning, critique) are stored in `exemplars` table with embeddings; retrieved as few-shot context for local model via system prompt injection; outcome tracking via exponential moving average
- **MCP server** `forge serve` exposes file tools + `forge_ask` (local inference), `forge_rag_search`, `forge_memory_recall/save`; stdio transport; lazy DB init; designed for use as Claude Code MCP backend
- **Database** PostgreSQL on Unix socket (port 5433), pgvector 0.6.0 with `vector` type (not halfvec)

## Key Implementation Notes

- pydantic-ai v1.75+ uses `Agent(model="ollama:model_name")` — no `OllamaModel` class
- Must set `OLLAMA_BASE_URL` env var (with `/v1` suffix) before creating agents
- `instructions` parameter replaces `system_prompt` in Agent constructor
- Timeout set to 300s for model loading (84GB model takes ~30s to load)
- pgvector 0.6.0 doesn't support `halfvec` — using `vector` type with string casting (no `register_vector`)
- DB connects via Unix socket: `postgresql://stephen@/forge?host=/var/run/postgresql&port=5433`
- tree-sitter 0.25+ requires individual language packages (`tree-sitter-python`, etc.), not `tree-sitter-languages`
- Conversation persistence: `sessions` table (metadata) + `conversations` table (messages), async fire-and-forget writes
- Disable persistence with `FORGE_PERSIST_HISTORY=false`
- MCP config format: `{"mcpServers": {"name": {"command": "...", "args": [...], "env": {...}}}}` — same schema as Claude Desktop; set `"name": false` to disable a built-in server
- Playwright browser MCP auto-discovered if `npx` is on PATH (headless mode); MCP tools are passed to sub-agents via `toolsets` param
- Memory stored in `memories` table with 768d embeddings; recalled via cosine similarity; auto-prunes at 50 entries
- Exemplars stored in `exemplars` table with 768d embeddings; ranked by outcome_score + similarity; auto-prunes at 100 per project
- Task store is in-memory per session, serialized to DB for resume; injected into system prompt via `to_prompt()`
- Hook registry: `HookRegistry.on(EventType, handler)` with priority ordering; `@with_hooks` decorator wraps tools
- Sandbox hooks registered at priority -50 (before permission hook at 0); command blocklist + path boundary enforcement
- Multimodal: `parse_multimodal_input()` extracts `@path.png` refs → `BinaryContent.from_path()`; `ModelMessagesTypeAdapter` used for serialization (supports base64 bytes)
- Vision model configured via `ollama.vision_model` in config.toml (e.g. `gemma3:12b`); empty = disabled
- `fastmcp` is a direct dependency (used by `mcp_server.py`); `mcp` removed (transitive via pydantic-ai)
- Executor uses `start_new_session=True` + process group kill; temp files in `.forge/tmp/` when cwd is set

## Testing

```bash
uv run pytest tests/unit/ -x -q              # Run all unit tests
uv run forge --version
uv run forge status
uv run forge ask --fast "test"
uv run forge ask --gpu "test"
uv run forge index .                           # Index current project
uv run forge code "explain the router" --fast  # RAG-augmented query
uv run forge agent "read pyproject.toml and tell me the project name"
uv run forge agent                             # Interactive agentic REPL
uv run forge agent --yolo "find all files importing typer"  # No permission prompts
uv run forge agent --worktree                  # Agent in isolated worktree
```

## Phases

- [x] Phase 0: Environment setup (Ollama models, project init)
- [x] Phase 1: MVP (CLI, REPL, streaming, two-model routing)
- [x] Phase 2: RAG (pgvector, tree-sitter chunking, embeddings, index + code commands)
- [x] Phase 3: Draft-Verify-Refine pipeline + code executor
- [x] Phase 4: Conversation persistence (PostgreSQL sessions + messages, history/resume commands)
- [x] Phase 5: NPU integration (FastFlowLM) — httpx backend, auto-routing, `/npu` + `--npu` CLI
- [x] Phase 6: Agentic coding mode — tool-using agent loop (read, write, edit, search, run commands)
- [x] Phase 7A: Hooks, task tracking, cross-session memory — event system, in-memory tasks with dependencies, pgvector semantic memory
- [x] Phase 7B: Worktrees + improved compaction — git worktree isolation, 3-tier context compaction with domain-aware summarization
- [x] Phase 7C: MCP integration — discover and connect external MCP tool servers via pydantic-ai
- [x] Phase 8: Conversation checkpoints + RAG auto-index — named save/restore within sessions, staleness detection, post-write reindex hooks, /index command
- [x] Phase 9: Sandboxing, multimodal, web browsing, dependency fixes — command blocklist + path boundaries, @image input with vision routing, Playwright MCP auto-discovery, fastmcp direct dep, MCP in sub-agents
- [x] Phase 10: Remove max mode, enhanced MCP server, exemplar learning — removed Claude Code CLI wrapper (double-agent problem), added forge_ask/rag/memory MCP tools for Claude Code integration, exemplar learning system captures cloud model successes for local model few-shot injection
