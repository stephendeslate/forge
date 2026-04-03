"""Tests for context extraction functions used in compaction."""

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from forge.agent.context import (
    _collect_key_files,
    _collect_recent_requests,
    _collect_tools_used,
    _extract_preservable_refs,
    _extract_prior_summary,
    _infer_pending_work,
    _strip_tag_blocks,
)


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _response(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _tool_response(name: str, text: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args="{}", tool_call_id="tc1")])


class TestStripTagBlocks:
    def test_strips_analysis_tag(self):
        text = "before <analysis>inner content</analysis> after"
        assert _strip_tag_blocks(text, "analysis") == "before  after"

    def test_strips_multiline(self):
        text = "start\n<analysis>\nline1\nline2\n</analysis>\nend"
        assert _strip_tag_blocks(text, "analysis") == "start\n\nend"

    def test_no_tag_unchanged(self):
        text = "no tags here"
        assert _strip_tag_blocks(text, "analysis") == "no tags here"

    def test_empty_tag(self):
        text = "before <analysis></analysis> after"
        assert _strip_tag_blocks(text, "analysis") == "before  after"

    def test_multiple_tags(self):
        text = "<analysis>a</analysis> middle <analysis>b</analysis>"
        assert _strip_tag_blocks(text, "analysis") == "middle"

    def test_different_tag_name(self):
        text = "keep <think>thought</think> this"
        assert _strip_tag_blocks(text, "think") == "keep  this"


class TestCollectKeyFiles:
    def test_finds_python_files(self):
        msgs = [_response("Modified src/main.py and tests/test_main.py")]
        files = _collect_key_files(msgs)
        assert "src/main.py" in files
        assert "tests/test_main.py" in files

    def test_finds_toml_files(self):
        msgs = [_response("Read pyproject.toml")]
        files = _collect_key_files(msgs)
        assert "pyproject.toml" in files

    def test_deduplicates(self):
        msgs = [
            _response("Read config.toml"),
            _response("Modified config.toml"),
        ]
        files = _collect_key_files(msgs)
        assert files.count("config.toml") == 1

    def test_empty_messages(self):
        assert _collect_key_files([]) == []

    def test_no_files_found(self):
        msgs = [_response("Just some text without file paths")]
        assert _collect_key_files(msgs) == []

    def test_sorted_output(self):
        msgs = [_response("z.py a.py m.py")]
        files = _collect_key_files(msgs)
        assert files == sorted(files)


class TestInferPendingWork:
    def test_finds_todo(self):
        msgs = [_response("TODO: fix the login bug")]
        pending = _infer_pending_work(msgs)
        assert any("TODO" in p for p in pending)

    def test_finds_next(self):
        msgs = [_response("next step: write tests")]
        pending = _infer_pending_work(msgs)
        assert len(pending) >= 1

    def test_finds_blocked(self):
        msgs = [_response("blocked on API key")]
        pending = _infer_pending_work(msgs)
        assert len(pending) >= 1

    def test_no_pending(self):
        msgs = [_response("Everything is complete and working.")]
        assert _infer_pending_work(msgs) == []

    def test_only_scans_last_5(self):
        old_msgs = [_response("TODO: old task")] * 6
        recent = [_response("All done")]
        # Only last 5 are scanned; 6th oldest is excluded
        msgs = old_msgs + recent
        pending = _infer_pending_work(msgs)
        # The old TODO might still be in last 5, but "All done" shouldn't match
        assert not any("All done" in p for p in pending)

    def test_deduplicates(self):
        msgs = [
            _response("TODO: same task"),
            _response("TODO: same task"),
        ]
        pending = _infer_pending_work(msgs)
        assert pending.count("TODO: same task") == 1

    def test_max_10(self):
        msgs = [_response("\n".join(f"TODO: task {i}" for i in range(20)))]
        pending = _infer_pending_work(msgs)
        assert len(pending) <= 10


class TestCollectToolsUsed:
    def test_finds_tools(self):
        msgs = [
            ModelResponse(parts=[
                ToolCallPart(tool_name="read_file", args="{}", tool_call_id="tc1"),
                ToolCallPart(tool_name="write_file", args="{}", tool_call_id="tc2"),
            ]),
        ]
        tools = _collect_tools_used(msgs)
        assert "read_file" in tools
        assert "write_file" in tools

    def test_deduplicates(self):
        msgs = [
            ModelResponse(parts=[
                ToolCallPart(tool_name="read_file", args="{}", tool_call_id="tc1"),
            ]),
            ModelResponse(parts=[
                ToolCallPart(tool_name="read_file", args="{}", tool_call_id="tc2"),
            ]),
        ]
        tools = _collect_tools_used(msgs)
        assert tools.count("read_file") == 1

    def test_sorted(self):
        msgs = [
            ModelResponse(parts=[
                ToolCallPart(tool_name="write_file", args="{}", tool_call_id="tc1"),
                ToolCallPart(tool_name="edit_file", args="{}", tool_call_id="tc2"),
                ToolCallPart(tool_name="read_file", args="{}", tool_call_id="tc3"),
            ]),
        ]
        tools = _collect_tools_used(msgs)
        assert tools == sorted(tools)

    def test_ignores_non_response(self):
        msgs = [_user("hello")]
        assert _collect_tools_used(msgs) == []

    def test_empty(self):
        assert _collect_tools_used([]) == []


class TestCollectRecentRequests:
    def test_collects_up_to_3(self):
        msgs = [_user(f"request {i}") for i in range(5)]
        requests = _collect_recent_requests(msgs)
        assert len(requests) == 3

    def test_most_recent_first(self):
        msgs = [_user("first"), _user("second"), _user("third")]
        requests = _collect_recent_requests(msgs)
        assert requests[0] == "third"
        assert requests[1] == "second"

    def test_truncates_to_100_chars(self):
        msgs = [_user("x" * 200)]
        requests = _collect_recent_requests(msgs)
        assert len(requests[0]) == 100

    def test_empty(self):
        assert _collect_recent_requests([]) == []


class TestExtractPreservableRefs:
    def test_produces_structured_output(self):
        msgs = [
            _user("Fix src/main.py"),
            ModelResponse(parts=[
                ToolCallPart(tool_name="read_file", args="{}", tool_call_id="tc1"),
            ]),
            _response("TODO: write tests"),
        ]
        result = _extract_preservable_refs(msgs)
        assert "## Preserved References" in result
        assert "src/main.py" in result
        assert "read_file" in result

    def test_empty_messages_returns_empty(self):
        assert _extract_preservable_refs([]) == ""


class TestExtractPriorSummary:
    def test_separates_compacted_messages(self):
        msgs = [
            _user("[Context compacted — 5 messages summarized]\nSummary here"),
            _user("New question"),
            _response("Answer"),
        ]
        prior, remaining = _extract_prior_summary(msgs)
        assert prior is not None
        assert "Summary here" in prior
        assert len(remaining) == 2

    def test_compacted_task_format(self):
        msgs = [
            _user("[Compacted task: fix the bug]\nDid stuff"),
            _user("Next question"),
        ]
        prior, remaining = _extract_prior_summary(msgs)
        assert prior is not None
        assert "fix the bug" in prior

    def test_no_prior_summary(self):
        msgs = [_user("hello"), _response("hi")]
        prior, remaining = _extract_prior_summary(msgs)
        assert prior is None
        assert remaining is msgs  # Same reference

    def test_multiple_compacted(self):
        msgs = [
            _user("[Context compacted — first]"),
            _user("[Compacted task: second]"),
            _user("Fresh message"),
        ]
        prior, remaining = _extract_prior_summary(msgs)
        assert prior is not None
        assert "first" in prior
        assert "second" in prior
        assert len(remaining) == 1
