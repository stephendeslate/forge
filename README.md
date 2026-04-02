# Forge

Local AI orchestration CLI — smart model routing, RAG, agentic coding, and self-correction pipelines. Runs entirely on your hardware via [Ollama](https://ollama.com).

## Features

- **Smart routing** — automatically routes prompts to the right model (heavy GPU, fast GPU, or NPU) based on complexity
- **Agentic coding mode** — tool-using agent loop with file read/write/edit, search, command execution, web browsing, task tracking, and cross-session memory
- **RAG** — tree-sitter AST chunking, nomic-embed-text embeddings, pgvector cosine search
- **Draft-Verify-Refine** — multi-stage generation pipeline with self-critique
- **Code executor** — generate, run, and self-correct scripts
- **MCP integration** — connect external tool servers (Playwright, custom servers) via standard MCP protocol
- **Conversation persistence** — session history with checkpoint/restore
- **Git worktree isolation** — run agents in isolated worktrees for safe experimentation
- **Sandboxing** — command blocklist and path boundary enforcement
- **Multimodal input** — attach images with `@path/to/image.png` syntax

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) with your preferred models
- PostgreSQL with pgvector (optional, for RAG and memory)

## Install

```bash
git clone https://github.com/stephendeslate/forge.git
cd forge
uv sync
```

For RAG support:

```bash
uv sync --extra rag
```

## Quick Start

```bash
# One-shot question
forge ask "explain this error"

# Interactive agentic coding
forge agent

# Agentic mode with initial prompt
forge agent "read the config and summarize it"

# Skip permission prompts
forge agent --yolo

# Index a project for RAG, then query it
forge index .
forge code "how does the router work?"

# Draft-verify-refine pipeline
forge draft "build a CLI that does X"

# Generate and run a script
forge run "write a script that finds large files"

# Check status
forge status
```

## Configuration

Forge reads from `~/.config/forge/config.toml` and `FORGE_*` environment variables.

```toml
[ollama]
base_url = "http://localhost:11434"
heavy_model = "qwen3-coder-next:q8_0"
fast_model = "qwen3.5:4b"
```

## Architecture

```
CLI (typer) → Router (heuristic) → OllamaBackend (pydantic-ai)
                                     ├── gpu-heavy model
                                     ├── gpu-fast model
                                     └── npu model (optional)

forge agent → pydantic-ai Agent → agentic tool loop
               ├── file tools: read, write, edit, search, glob
               ├── system tools: run commands, web search, web fetch
               ├── workflow tools: tasks, memory, checkpoints
               └── MCP: external tool servers
```

## License

MIT
