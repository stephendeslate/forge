# Claude Code Parity Report for Forge

> Comprehensive analysis of patterns from Claude Code's BashTool source, the Claw Code
> Rust port (~/projects/claude-code-parity/), and ccunpacked.dev — cross-referenced against
> Forge's current implementation. Only verified, concrete findings included.

---

## Table of Contents

1. [Output Handling](#1-output-handling)
2. [Agent Loop](#2-agent-loop)
3. [Context Management & Compaction](#3-context-management--compaction)
4. [System Prompt Architecture](#4-system-prompt-architecture)
5. [Tool System](#5-tool-system)
6. [Permission & Security](#6-permission--security)
7. [Sub-Agent Architecture](#7-sub-agent-architecture)
8. [Session & Persistence](#8-session--persistence)
9. [UI & Rendering](#9-ui--rendering)
10. [Memory System](#10-memory-system)
11. [Prioritized Recommendations](#11-prioritized-recommendations)

---

## 1. Output Handling

### 1.1 Head+Tail Truncation

**Claude Code (BashTool.tsx):** Keeps first N + last N characters of command output with
`[...truncated middle...]` marker. Ensures error messages at the end of output are never lost.
Large output is persisted to disk (`MAX_PERSISTED_SIZE = 64MB`) with only a preview sent to
the model (`maxResultSizeChars = 30_000`).

**Claw Code (tools/lib.rs):** Display truncation keeps first lines + "output truncated for
display; full result preserved in session" notice. Model always sees full output.

**Forge (tools.py):** Front-only truncation — `stdout[:stdout_limit]`, `stderr[:stderr_limit]`.
Critical errors at the end of long output are silently dropped.

**Gap:** Forge loses the most important part of output. A compiler that prints 10K lines of
warnings before the actual error will have the error cut off.

**Fix:** Replace `stdout[:limit]` with head+tail:
```python
def head_tail_truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n\n[...truncated middle...]\n\n" + text[-half:]
```

### 1.2 Output Size Kill

**Claude Code (BashTool.tsx):** Kills processes that produce more than 64MB of output.

**Forge:** No size guard. A command like `cat /dev/urandom | base64` will buffer unbounded
memory until OOM or timeout.

**Fix:** Track total bytes read in the stdout/stderr read loop. Kill the process group when
cumulative output exceeds a configurable limit (e.g., 5MB for local models with smaller context).

### 1.3 Streaming Command Output

**Claude Code (BashTool.tsx):** `runShellCommand` is an async generator yielding progress
updates every `PROGRESS_THRESHOLD_MS = 2000ms`. The user sees partial output while the
command runs.

**Claw Code (main.rs):** Tool progress heartbeat thread fires at regular intervals, showing
`"... heartbeat ... Ns elapsed"` for long-running tools.

**Forge (tools.py:run_command):** Blocks completely until the process exits. The status line
shows elapsed time but zero output. Users stare at a spinner for up to 120s.

**Gap:** This is Forge's biggest UX gap for agentic coding. Commands like `npm install`,
`cargo build`, and `pytest` produce streaming output that helps users understand progress.

**Fix:** Refactor `run_command` to yield partial output chunks via the PostToolUse hook or
a new streaming callback. The status line (`status.py`) could show the last line of output
as its detail text.

### 1.4 Auto-Backgrounding

**Claude Code (BashTool.tsx):** Tracks an `ASSISTANT_BLOCKING_BUDGET_MS = 15_000`. Commands
exceeding this budget automatically move to background execution, returning a partial result
with a task ID. The model can check on the task later.

**Forge:** No auto-backgrounding. Long commands block the entire agent loop. The model can't
make progress on other parts of its plan while waiting for `make` to finish.

**Fix:** After a configurable threshold (e.g., 15s), capture current output, return it with
a background task ID, and continue the process in the background. Wire into the existing
`TaskStore` for tracking.

---

## 2. Agent Loop

### 2.1 Stop Condition

**Claw Code (conversation.rs:356):** Loop breaks when the assistant returns NO `ToolUse`
blocks. No explicit `stop_reason` check — purely based on absence of tool calls. Hard guard
via `max_iterations` (defaults to `usize::MAX`).

**Forge (loop.py):** Uses pydantic-ai's `agent.run()` which handles the loop internally.
Stop is controlled by `UsageLimits(request_limit=settings.agent.request_limit)` plus
pydantic-ai's built-in stop logic (model returns text without tool calls).

**Status:** Forge's approach is equivalent. The `request_limit` serves as the hard guard.
No change needed.

### 2.2 API Retry Logic

**Claw Code (anthropic.rs:397-459):** Explicit retry with exponential backoff — 200ms initial,
2s max, 2 retries. Retryable: HTTP 408/409/429/500/502/503/504 + connect/timeout errors.
Non-retryable: auth, JSON, SSE parse errors.

**Forge (loop.py):** Relies on pydantic-ai's built-in retry (`retries=3`) and httpx's
transport-level retry. No explicit backoff tuning. Connection errors to Ollama are caught
in `_handle_agent_error()` but not retried — the turn fails.

**Gap:** Forge doesn't retry on Ollama 5xx errors or connection drops during model loading.
The 84GB model sometimes causes timeouts when loading. pydantic-ai's retry is for tool-level
`ModelRetry`, not transport-level retries.

**Fix:** Add an httpx `Transport(retries=2)` to the Ollama client and/or wrap `_run_with_status()`
in a retry loop for transient connection errors (model loading, OOM recovery).

### 2.3 Parallel Tool Execution

**Claw Code (conversation.rs:360):** Sequential only — tool calls iterated one by one. All
results available before next API call.

**Forge (tools.py):** Also sequential via pydantic-ai. Tools marked `sequential=True`
(writes/exec) enforce ordering; read tools could theoretically run in parallel but don't.

**Status:** Neither implementation parallelizes tool calls. This is a pydantic-ai limitation.
Not worth pursuing now.

---

## 3. Context Management & Compaction

### 3.1 Compaction Strategy

**Claw Code (compact.rs):** Structured XML summary with: scope stats (message count, tool
calls, errors), tools mentioned, recent user requests, pending work (keyword scan for "todo",
"next", "pending"), key files (path extraction from content, capped at 8), current work
description, timeline. Supports iterative compaction — merges "Previously compacted" with
"Newly compacted" context.

**Forge (context.py):** Three-tier: (1) truncate tool results + strip binary, (2) group
messages into task sequences and summarize each via fast LLM, (3) full LLM summarization.
Fallback to mechanical truncation.

**Comparison:** Forge's approach is actually more sophisticated in tier 2 (per-task-sequence
summarization preserves more structure than a single summary). But Forge lacks:
- **Iterative summary merging** — re-compacting loses the previous summary entirely
- **Structured metadata extraction** — key files, tools used, pending work aren't extracted
  as structured data, only preserved if the LLM summary includes them
- **Keyword-based pending work detection** — Claw Code scans for "todo"/"next"/"pending"
  without needing an LLM call

**Applicable patterns:**
1. Add iterative summary merging: when compacting an already-compacted conversation, preserve
   the old summary prefix
2. Extract key files and tools used as structured data before the LLM summary, inject them
   as a header that survives re-compaction
3. Add keyword-based pending work detection as a cheap pre-pass

### 3.2 Auto-Compaction Trigger

**Claw Code (conversation.rs:507-530):** Triggers when cumulative API-reported input tokens
exceed 100,000 (`DEFAULT_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD`). Configurable via env var.
Uses API-reported tokens, not local estimates.

**Forge (loop.py):** Triggers when `len(message_history) > 40` AND estimated tokens exceed
`budget * compaction_threshold`. Uses local character-based estimation (3.2-3.8 chars/token).

**Gap:** Forge's trigger is based on message count (fragile — 40 one-line messages vs 40
multi-file edits are very different) AND local estimates (less accurate than API-reported).
However, Forge targets local Ollama models which don't always report token usage consistently,
so API-reported tokens may not be reliable.

**Fix:** Use API-reported tokens when available (Ollama does report them in responses), fall
back to estimates. Drop the `len > 40` guard — it's redundant when you have token estimates.

### 3.3 Token Estimation

**Claw Code (compact.rs:392):** Simple `text.len() / 4 + 1` per block. Used only for sizing
compaction output, not for triggering it.

**Forge (context.py):** More nuanced — 3.2 chars/token for code, 3.8 for prose, with
per-message and per-tool overhead. Code detection via newline/brace density.

**Status:** Forge's estimation is already better. No change needed.

### 3.4 Dynamic Boundary for Prompt Caching

**Claw Code (prompt.rs):** `__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__` separates the static system
prompt (identity, tools, behavior) from dynamic content (environment, CLAUDE.md, config).
Everything above the boundary is prompt-cached by the Anthropic API across turns.

**Forge:** No prompt caching optimization. The entire system prompt (including static tool
descriptions) is rebuilt and re-sent every turn.

**Applicability:** Limited. Forge uses Ollama, which doesn't support Anthropic-style prompt
caching. Ollama has its own KV cache, but it's automatic and doesn't benefit from boundary
markers. **Not applicable to Forge's current backend.**

---

## 4. System Prompt Architecture

### 4.1 Prompt Assembly

**Claw Code (prompt.rs):** Builder pattern with strict section ordering: intro, output_style,
system, doing_tasks, actions, DYNAMIC_BOUNDARY, environment, project_context,
instruction_files (budgeted), config, append_sections.

**Forge (loop.py + agent.py):** System prompt is a string constant `AGENT_SYSTEM` (~85 lines)
plus `build_project_context(cwd)` which reads `.forge/instructions.md`, CLAUDE.md, detects
git state, lists directory structure.

**Differences:**
- Forge has no instruction file budgeting — a massive CLAUDE.md is injected verbatim,
  potentially consuming significant context
- Forge has no deduplication of instruction files from parent directories
- Forge doesn't walk parent directories for CLAUDE.md files (only checks cwd)

**Applicable patterns:**
1. **Instruction file budgeting** — cap CLAUDE.md injection at e.g., 4000 chars per file,
   12000 total, with `[truncated]` marker
2. **Parent directory walk** — discover and include CLAUDE.md from parent directories up to
   project root (stop at `.git`), not just cwd
3. **Deduplication** — if a parent and child CLAUDE.md have identical content, include only once

### 4.2 Task Injection

**Claw Code (tools/lib.rs, TodoWrite):** Todos injected into system prompt via `to_prompt()`.
Includes a `verification_nudge_needed` flag — when all todos are completed and none contain
"verif", the system nudges the model to verify its work.

**Forge (task_store.py):** `to_prompt()` renders tasks as markdown checklist. No verification
nudge.

**Applicable pattern:** Add verification nudge — when all tasks are marked completed, append
a note like "All tasks marked complete. Consider verifying the changes work as expected."

---

## 5. Tool System

### 5.1 Display-Only Truncation

**Claw Code (main.rs:4480-4600):** Tool results are truncated FOR DISPLAY ONLY. The model
always sees the full result. Constants: `READ_DISPLAY_MAX_LINES=80`,
`READ_DISPLAY_MAX_CHARS=6000`, `TOOL_OUTPUT_DISPLAY_MAX_LINES=60`,
`TOOL_OUTPUT_DISPLAY_MAX_CHARS=4000`. A dim notice says "output truncated for display; full
result preserved in session."

**Forge (render.py):** Tool results truncated at 500 chars for display — much more aggressive.
And separately, tools.py truncates output before the model sees it too (stdout_limit for
run_command, 2000 lines for read_file).

**Status:** Forge already uses separate display vs model truncation limits. The 500-char
display limit is tighter than Claw Code's 4000-6000 chars, but this is appropriate for
smaller terminal windows (SSH from phone). No change needed unless users complain.

### 5.2 Tool-Specific Result Formatting

**Claw Code (main.rs:4431-4679):** Each tool type has a dedicated formatter:
- Bash: `$ command` with dark background, stdout normal, stderr red (color 203)
- Read: `file icon + path (lines X-Y of Z)`
- Edit: diff preview with red (removed) and green (added) lines
- Glob/Grep: search icon with pattern

**Forge (render.py):** Generic Panel rendering for all tools. Color varies by category
(green=safe, yellow=write, red=exec, blue=other). Args shown but not formatted per-tool.

**Applicable pattern:** Per-tool result formatting. Highest value:
1. **Edit results** — show colored diff (red/green) instead of raw text
2. **Bash results** — show stderr in red, stdout normal
3. **Read results** — show line range info

### 5.3 Sleep Pattern Detection

**Claude Code (BashTool.tsx):** `detectBlockedSleepPattern()` blocks commands like
`sleep 60 && curl ...` that waste time and tokens.

**Forge (sandbox.py):** No sleep detection. The blocklist can be extended but currently
doesn't catch `sleep` abuse.

**Fix:** Add `r"^\s*sleep\s+\d{2,}"` to warn_patterns (not block — `sleep 1` between
retries is valid).

### 5.4 Semantic Exit Code Interpretation

**Claude Code (BashTool.tsx):** `interpretCommandResult()` knows that `grep` returning 1
means "no matches" (not an error), `diff` returning 1 means "files differ", etc.

**Forge:** Exit code > 0 is always treated as an error in the result formatting.

**Fix:** Small lookup table for common commands where non-zero exit codes are informational:
```python
INFORMATIONAL_EXIT_CODES = {
    "grep": {1: "no matches found"},
    "diff": {1: "files differ"},
    "test": {1: "condition is false"},
}
```

---

## 6. Permission & Security

### 6.1 Permission Rule Syntax

**Claw Code (permissions.rs:342-393):** Rules like `bash(git:*)` match tool name + input
prefix. Supports exact match (`bash(npm test)`), prefix match (`bash(git:*)`), and wildcard
(`bash(*)`). Input extracted from JSON: `command` for bash, `file_path` for file tools,
`url` for web tools.

**Forge (permissions.py):** Three modes only (YOLO/AUTO/ASK). No per-tool or per-input rules.
No allow/deny lists.

**Gap:** Forge can't express "auto-allow git commands but ask for npm commands" or "allow
reads from /etc but not writes." It's all-or-nothing per permission class.

**Applicable pattern:** Add rule-based permissions:
```toml
[permissions]
allow = ["read_file(*)", "search_code(*)", "run_command(git:*)"]
deny = ["run_command(rm:*)", "run_command(sudo:*)"]
ask = ["run_command(*)"]
```

This would be a significant improvement to the `AUTO` mode, making it much more useful
for day-to-day work without going full `YOLO`.

### 6.2 Compound Command Security Parsing

**Claude Code (BashTool.tsx):** `parseForSecurity()` decomposes compound commands
(`cmd1 && cmd2`, `cmd1 | cmd2`, `$(cmd)`, `` `cmd` ``) and checks each component against
permission rules. This prevents `echo harmless && rm -rf /` from bypassing the blocklist.

**Claw Code (PARITY.md):** Explicitly notes this as missing — 18 upstream BashTool modules
not ported, including `bashSecurity.ts`.

**Forge (sandbox.py):** Regex blocklist checks the entire command string. `rm -rf /` is
caught, but `echo x && rm -rf /` is NOT because the regex matches against the whole string,
not individual components.

**Gap:** This is a real security hole. Easy evasion: `true; rm -rf /`, `echo | rm -rf /`,
`$(rm -rf /)`.

**Fix (pragmatic):** Split commands on `&&`, `||`, `;`, `|` and check each component. This
catches 90% of evasion. Full shell parsing (handling quotes, subshells, heredocs) is much
harder and probably not worth it for local models that rarely try to evade.

### 6.3 Path Boundary on run_command

**Forge (sandbox.py):** Path boundary enforces `FILE_TOOLS` (read/write/edit) but NOT
`run_command`. A model can run `cat /etc/shadow` even when path boundary restricts
`read_file` to cwd.

**Claw Code (PARITY.md):** Also notes this as a gap — path validation for bash commands
is listed as missing.

**Claude Code (BashTool.tsx):** Applies path checks to file arguments extracted from bash
commands.

**Fix (pragmatic):** For `run_command`, extract obvious file path arguments and check them
against the path boundary. This won't catch everything (piped commands, variable expansion)
but catches the common case of `cat /path/outside/cwd`.

### 6.4 Linux Namespace Sandboxing

**Claw Code (sandbox.rs:211-262):** Uses `unshare --user --map-root-user --mount --ipc
--pid --uts --fork` with optional `--net` for network isolation. Sets HOME and TMPDIR to
workspace-local dirs.

**Forge:** No process-level sandboxing. The sandbox is entirely advisory (regex blocklist +
path checks).

**Gap:** This is a defense-in-depth measure. Forge's approach is adequate for local models
(which are less likely to attempt malicious actions than cloud models receiving untrusted
input), but namespace isolation would be valuable for `--yolo` mode where all permissions
are bypassed.

**Applicability:** Medium priority. Would require Linux-specific code and testing. The
`unshare` approach is the right one — no Docker/container dependency needed.

---

## 7. Sub-Agent Architecture

### 7.1 Tool Whitelisting per Sub-Agent Type

**Claw Code (tools/lib.rs:2335-2414):** Each sub-agent type gets an explicit tool allowlist:

| Type | Tools | Key Restrictions |
|------|-------|------------------|
| Explore | read, glob, grep, web | NO bash, write, edit |
| Plan | read, glob, grep, web, TodoWrite | NO bash, write, edit |
| Verification | bash, read, grep, web | NO write, edit |
| Coder | bash, read, write, edit, glob, grep | NO web, NO recursive Agent |
| Default | everything except Agent recursion | Full access minus spawning |

Enforced by `SubagentToolExecutor` that checks `allowed_tools.contains(name)` before dispatch.

**Forge (tools.py:delegate):** Sub-agents get the same tools as the parent. No restriction.
The `coder` sub-agent (local model) gets write tools, bash, web search — everything.

**Gap:** This is both a security and quality issue. A research sub-agent shouldn't be able
to write files. A coder sub-agent shouldn't spawn recursive sub-agents.

**Fix:** Define tool sets per sub-agent type:
```python
SUBAGENT_TOOLS = {
    "coder": ["read_file", "write_file", "edit_file", "run_command", "search_code", "list_files"],
    "general-purpose": ALL_TOOLS - {"delegate", "delegate_parallel"},
    "reviewer": ["read_file", "search_code", "list_files", "run_command"],
}
```

### 7.2 Coordinator Mode (Parallel Workers in Worktrees)

**ccunpacked.dev + Claw Code:** A lead agent decomposes a task, spawns parallel workers in
isolated git worktrees, collects and merges results.

**Forge:** Has both worktrees (`--worktree`) and sub-agent delegation (`delegate_parallel`)
but they're not combined. Sub-agents don't run in worktrees. `delegate_parallel` runs 2-4
sub-agents in the same working directory.

**Applicable pattern:** Wire worktree creation into `delegate_parallel` so each parallel
sub-agent gets its own worktree. On completion, the coordinator reviews diffs and merges.
This requires:
1. Auto-creating worktrees for each parallel sub-agent
2. Passing the worktree path as the sub-agent's cwd
3. Collecting diffs from each worktree on completion
4. Cleaning up worktrees after merge

This is a significant feature but builds entirely on existing Forge infrastructure.

### 7.3 Sub-Agent Permission Escalation

**Claw Code (tools/lib.rs:2416-2421):** Sub-agents run with `DangerFullAccess` — no user
prompting. This is safe because the tool set is pre-restricted.

**Forge (tools.py:delegate):** Sub-agents inherit the parent's permission mode. In `ASK`
mode, sub-agents would prompt the user for every tool call — which is impractical for
background work.

**Fix:** Sub-agents should auto-allow their whitelisted tools without prompting. This is
safe when combined with tool whitelisting (7.1).

### 7.4 File-Based Agent Persistence

**Claw Code (tools/lib.rs):** Sub-agents write markdown output files and JSON manifests to
disk, enabling post-hoc inspection independent of conversation logs.

**Forge:** Sub-agent results exist only in the conversation history. No way to inspect what
a sub-agent did after the session.

**Applicable pattern:** Write sub-agent manifest + output to `.forge/agents/` on completion.
Useful for debugging delegation failures.

---

## 8. Session & Persistence

### 8.1 Incremental JSONL Persistence

**Claw Code (session.rs):** Messages appended to JSONL files as they arrive — crash-safe
incremental writes. File rotation at 256KB with max 3 rotations. Legacy JSON format
supported for backwards compatibility.

**Forge (loop.py):** Sessions persisted to PostgreSQL via async fire-and-forget writes.
Full message history serialized via `ModelMessagesTypeAdapter` on each save.

**Comparison:** Forge's PostgreSQL approach is more robust (ACID, queryable) but heavier.
The JSONL approach is simpler and crash-safe without a database.

**Status:** Forge's approach is fine. PostgreSQL is already a dependency for pgvector
(embeddings, memory, RAG). No reason to add a parallel JSONL system.

### 8.2 Session Forking

**Claw Code (session.rs):** `SessionFork { parent_session_id, branch_name }` tracks lineage
when sessions are branched.

**Forge:** No session forking. Resume always continues linearly.

**Applicable pattern:** When creating a worktree or branching work, fork the session with a
parent pointer. This enables branching a conversation ("try approach A in one session,
approach B in another") with shared history up to the fork point.

---

## 9. UI & Rendering

### 9.1 Markdown-Safe Streaming

**Claw Code (render.rs):** `MarkdownStreamState` buffers incoming deltas and only renders
when `find_stream_safe_boundary()` finds a safe split point (empty line or closed code fence).
Prevents partial markdown from producing rendering artifacts.

**Forge (render.py):** Streams directly into Rich `Live` display with `Markdown()` rendering.
No safe-boundary detection. Partial code blocks or tables can produce flicker or broken
rendering.

**Gap:** Forge can show garbled output when a code fence is mid-stream (the markdown renderer
sees an unclosed fence and renders it incorrectly until the closing fence arrives).

**Fix:** Buffer incoming text and only update the Rich Live display at safe boundaries
(blank lines, closed fences). This is ~30 lines of logic.

### 9.2 Box-Drawing Tool Display

**Claw Code (main.rs:4431-4679):** Tool calls and results rendered with Unicode box-drawing
characters (`--- name ---`, `$ command` with background color). Per-tool icons and formatting.

**Forge (render.py):** Uses Rich `Panel` with borders and titles. Already looks good.

**Status:** Forge's Rich Panel approach is equivalent or better than manual box drawing.
No change needed.

### 9.3 Diff Visualization for Edit Results

**Claw Code (main.rs:4541-4550, 4661-4679):** Edit tool results show colored diffs — red for
removed lines, green for added lines. Uses structured patch data.

**Forge (render.py):** Edit results shown as plain text. No diff coloring.

**Fix:** When rendering `edit_file` results, parse the old/new strings and show a colored
unified diff. Rich supports `Syntax` highlighting and ANSI colors natively.

### 9.4 Command Classification for UI

**Claude Code (BashTool.tsx):** `isSearchOrReadBashCommand()` and `isSilentBashCommand()`
classify commands for display collapsing. Search/read results are collapsible. Silent
commands show "Done" instead of "(No output)".

**Forge:** All tool results displayed identically regardless of command type.

**Applicable pattern:** Classify common commands:
```python
SILENT_COMMANDS = {"mkdir", "touch", "chmod", "cp", "mv", "ln", "git add", "git checkout"}
READ_COMMANDS = {"cat", "head", "tail", "grep", "find", "ls", "tree", "git log", "git diff"}
```
Show "Done" for silent commands with no output. Collapse read command output by default.

### 9.5 Levenshtein Suggestions for Unknown Commands

**Claw Code (commands/lib.rs):** Unknown slash commands trigger Levenshtein distance matching
against all known commands + aliases. Top 3 suggestions within distance 4, with prefix bonus.
Shows "Did you mean /X, /Y, /Z?"

**Forge (commands.py):** Unknown commands just show "Unknown command: /X" with no suggestions.

**Fix:** Add Levenshtein matching. Python has `difflib.get_close_matches()` built in:
```python
from difflib import get_close_matches
suggestions = get_close_matches(cmd, COMMANDS.keys(), n=3, cutoff=0.6)
```

---

## 10. Memory System

### 10.1 Post-Session Memory Consolidation (Auto-Dream)

**ccunpacked.dev:** Between sessions, the AI reviews what happened and organizes what it
learned. Not implemented in Claw Code but documented as a planned feature.

**Forge:** Memories are saved explicitly during sessions via `save_memory` tool. No
between-session consolidation.

**Applicable pattern:** Add a session-end hook that:
1. Scans the conversation for architectural decisions, corrections, and gotchas
2. Auto-generates `project` or `feedback` memories from them
3. Runs after session save, before exit

This fits naturally into Forge's hook system as a `SessionEnd` handler.

### 10.2 Memory Deduplication

**Forge (memory.py):** No deduplication before saving. Relies on auto-pruning to eventually
clean up duplicates.

**Fix:** Before saving a new memory, check cosine similarity against existing memories.
If similarity > 0.9, merge or skip. The embedding infrastructure already exists.

### 10.3 TodoWrite Verification Nudge

**Claw Code (tools/lib.rs):** When all todos are completed and none contain "verif", returns
a nudge: verify your work before considering the task done.

**Forge (task_store.py):** No verification nudge.

**Fix:** In `TaskStore.to_prompt()`, when all tasks are COMPLETED, append:
"All tasks marked complete. Verify the changes work as expected before finishing."

---

## 11. Prioritized Recommendations

### Tier 1: Quick Wins (< 1 day each, high impact)

| # | Pattern | Source | Effort | Impact |
|---|---------|--------|--------|--------|
| 1 | Head+tail output truncation | BashTool.tsx | 30 min | High — stops losing error output |
| 2 | Output size kill guard | BashTool.tsx | 30 min | High — prevents OOM |
| 3 | Sleep pattern detection | BashTool.tsx | 15 min | Low — prevents token waste |
| 4 | Semantic exit code table | BashTool.tsx | 30 min | Medium — reduces model confusion |
| 5 | Levenshtein command suggestions | Claw commands | 15 min | Low — better UX |
| 6 | Task verification nudge | Claw TodoWrite | 15 min | Medium — improves task completion quality |
| 7 | Compound command splitting for sandbox | BashTool.tsx | 1 hr | High — closes security hole |
| 8 | Instruction file budgeting (cap CLAUDE.md) | Claw prompt.rs | 1 hr | Medium — prevents context waste |

### Tier 2: Medium Effort (1-3 days each, high impact)

| # | Pattern | Source | Effort | Impact |
|---|---------|--------|--------|--------|
| 9 | Streaming command output to status line | BashTool.tsx | 1 day | High — biggest UX gap |
| 10 | Per-tool result formatting (diffs, stderr color) | Claw main.rs | 1 day | Medium — better readability |
| 11 | Markdown-safe streaming boundaries | Claw render.rs | 0.5 day | Medium — fixes rendering glitch |
| 12 | Rule-based permissions (allow/deny/ask per tool+input) | Claw permissions.rs | 2 days | High — makes AUTO mode usable |
| 13 | Sub-agent tool whitelisting | Claw tools/lib.rs | 1 day | Medium — security + quality |
| 14 | Auto-backgrounding long commands | BashTool.tsx | 2 days | High — unblocks agent loop |
| 15 | Iterative compaction summary merging | Claw compact.rs | 1 day | Medium — preserves context across compactions |
| 16 | Post-session memory consolidation | ccunpacked.dev | 1 day | Medium — automatic learning |

### Tier 3: Larger Effort (3+ days, strategic)

| # | Pattern | Source | Effort | Impact |
|---|---------|--------|--------|--------|
| 17 | Coordinator mode (parallel worktree agents) | ccunpacked.dev | 3-5 days | High — parallel complex tasks |
| 18 | Daemon mode (`forge agent --bg` via tmux) | ccunpacked.dev | 2 days | Medium — background agents |
| 19 | Namespace sandboxing (unshare) | Claw sandbox.rs | 3 days | Medium — defense in depth |
| 20 | API retry with backoff for Ollama | Claw anthropic.rs | 1 day | Medium — resilience |

### Not Applicable to Forge

| Pattern | Reason |
|---------|--------|
| Prompt cache dynamic boundary | Ollama doesn't support Anthropic-style prompt caching |
| PTY allocation | Complex, marginal benefit for headless SSH workflow |
| Sed edit simulation in bash | Forge's `edit_file` tool handles this |
| `<claude-code-hint/>` protocol | Requires controlling CLI tools that produce output |
| Image detection in bash output | Niche; Forge handles input images already |
| PowerShell/REPL tools | Linux-only project |
| Team tools | Single-user local tool |
| OAuth login flow | Local Ollama doesn't need auth |
| Model-specific pricing | Free local inference |

---

## Appendix: Source File Map

### Claude Code (BashTool.tsx — provided by user)
- `runShellCommand` async generator with progress yielding
- `PROGRESS_THRESHOLD_MS = 2000`, `ASSISTANT_BLOCKING_BUDGET_MS = 15000`
- `MAX_PERSISTED_SIZE = 64MB`, `maxResultSizeChars = 30000`
- `parseForSecurity()` compound command decomposition
- `detectBlockedSleepPattern()` sleep abuse prevention
- `interpretCommandResult()` semantic exit codes
- `isSearchOrReadBashCommand()` / `isSilentBashCommand()` classification

### Claw Code Rust Port (~/projects/claude-code-parity/claw-code-parity/rust/)
- `crates/runtime/src/conversation.rs` — agent loop (1679 LOC)
- `crates/runtime/src/compact.rs` — compaction with structured summaries (689 LOC)
- `crates/runtime/src/permissions.rs` — 5-level permission hierarchy (675 LOC)
- `crates/runtime/src/hooks.rs` — shell-based hooks with exit code protocol (987 LOC)
- `crates/runtime/src/sandbox.rs` — namespace isolation via unshare (364 LOC)
- `crates/runtime/src/prompt.rs` — system prompt builder with dynamic boundary (795 LOC)
- `crates/runtime/src/session.rs` — JSONL persistence with rotation (1239 LOC)
- `crates/tools/src/lib.rs` — all tools + sub-agent architecture (5632 LOC)
- `crates/rusty-claude-cli/src/main.rs` — REPL + rendering (9720 LOC)
- `crates/rusty-claude-cli/src/render.rs` — markdown streaming + syntax highlighting (797 LOC)
- `crates/api/src/providers/anthropic.rs` — retry with exponential backoff
- `crates/api/src/sse.rs` — SSE frame parser

### ccunpacked.dev
- Coordinator mode: lead agent + parallel worktree workers
- Auto-Dream: between-session memory consolidation
- Daemon mode: `--bg` via tmux for background sessions
- UDS Inbox: Unix domain socket inter-session messaging
- Kairos: persistent mode with autonomous actions (not implemented anywhere)

### Forge (~/projects/forge/src/forge/)
- `agent/tools.py` — 956 LOC, all tool implementations
- `agent/loop.py` — 701 LOC, agent loop + REPL
- `agent/context.py` — 486 LOC, 3-tier compaction
- `agent/render.py` — 353 LOC, Rich-based rendering
- `agent/hooks.py` — 305 LOC, event system
- `agent/status.py` — 237 LOC, ANSI status line
- `agent/task_store.py` — 170 LOC, in-memory tasks
- `agent/sandbox.py` — 121 LOC, regex blocklist + path boundary
- `agent/permissions.py` — 84 LOC, 3 modes
- `agent/memory.py` — 82 LOC, pgvector memory
- `agent/commands.py` — 572 LOC, 24 slash commands
