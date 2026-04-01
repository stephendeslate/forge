# Forge — Local AI Orchestration

## Quick Reference

- **Language:** Python 3.12, managed with `uv`
- **Entry point:** `forge.cli:app` (typer)
- **Config:** `~/.config/forge/config.toml` + env vars (`FORGE_*`)
- **Models:** Ollama (GPU) — heavy: `qwen3-coder-next:q8_0`, fast: `qwen3.5:4b`; NPU (FastFlowLM) — `llama-3.2-3b`

## Commands

```bash
forge                           # Interactive REPL
forge agent                     # Agentic coding mode (tool-using REPL)
forge agent "read X and fix Y"  # Agentic mode with initial prompt
forge agent --yolo              # Skip permission prompts for writes/commands
forge agent --ask               # Prompt for every tool call
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
```

## Architecture

```
CLI (typer) → Router (heuristic) → OllamaBackend (pydantic-ai)
                                     ├── gpu-heavy: qwen3-coder-next:q8_0
                                     ├── gpu-fast:  qwen3.5:4b
                                     └── npu:       llama-3.2-3b (FastFlowLM, :52625)

forge index → walk → tree-sitter chunk → embed (nomic-v2-moe) → pgvector store
forge code  → embed query → pgvector search → inject context → generate
forge agent → pydantic-ai Agent(tools=[...]) → agentic loop (read/write/edit/run/search)
```

- **Router** uses keyword heuristics (no LLM call) to classify prompts; auto-routes short simple prompts to NPU when enabled
- **Pydantic AI** wraps Ollama via OpenAI-compatible API (`OLLAMA_BASE_URL` env var with `/v1` suffix)
- **Streaming** via `Agent.run_stream()` → `stream_text(delta=True)` → Rich Live display
- **Agent mode** uses `Agent.run()` with `event_stream_handler` for tool-calling loop + streaming display
- **RAG** tree-sitter AST chunking → nomic-embed-text-v2-moe (768d) → pgvector cosine search
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

## Testing

```bash
uv run forge --version
uv run forge status
uv run forge ask --fast "test"
uv run forge ask --gpu "test"
uv run forge index .                           # Index current project
uv run forge code "explain the router" --fast  # RAG-augmented query
uv run forge agent "read pyproject.toml and tell me the project name"
uv run forge agent                             # Interactive agentic REPL
uv run forge agent --yolo "find all files importing typer"  # No permission prompts
```

## Phases

- [x] Phase 0: Environment setup (Ollama models, project init)
- [x] Phase 1: MVP (CLI, REPL, streaming, two-model routing)
- [x] Phase 2: RAG (pgvector, tree-sitter chunking, embeddings, index + code commands)
- [x] Phase 3: Draft-Verify-Refine pipeline + code executor
- [x] Phase 4: Conversation persistence (PostgreSQL sessions + messages, history/resume commands)
- [x] Phase 5: NPU integration (FastFlowLM) — httpx backend, auto-routing, `/npu` + `--npu` CLI
- [x] Phase 6: Agentic coding mode — tool-using agent loop (read, write, edit, search, run commands)
