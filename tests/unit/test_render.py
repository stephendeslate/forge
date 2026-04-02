"""Tests for render helper functions."""

from forge.agent.render import (
    _tool_style,
    _format_tool_args,
    _truncate,
    _split_thinking,
)


class TestToolStyle:
    def test_safe_tools_green(self):
        assert _tool_style("read_file") == "green"
        assert _tool_style("search_code") == "green"
        assert _tool_style("list_files") == "green"

    def test_write_tools_yellow(self):
        assert _tool_style("write_file") == "yellow"
        assert _tool_style("edit_file") == "yellow"

    def test_exec_tools_red(self):
        assert _tool_style("run_command") == "red"

    def test_unknown_tools_blue(self):
        assert _tool_style("some_new_tool") == "blue"
        assert _tool_style("") == "blue"


class TestFormatToolArgs:
    def test_none_args(self):
        assert _format_tool_args(None) == ""

    def test_dict_args(self):
        result = _format_tool_args({"file_path": "test.py", "content": "hello"})
        assert "file_path" in result
        assert "test.py" in result

    def test_string_json_args(self):
        result = _format_tool_args('{"command": "ls"}')
        assert "command" in result
        assert "ls" in result

    def test_invalid_json_string(self):
        result = _format_tool_args("not json")
        assert result == "not json"

    def test_long_values_truncated(self):
        long_val = "x" * 200
        result = _format_tool_args({"content": long_val})
        assert "..." in result

    def test_empty_dict(self):
        assert _format_tool_args({}) == ""


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", max_len=100) == "hello"

    def test_long_text_truncated(self):
        text = "x" * 5000
        result = _truncate(text, max_len=100)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_exact_limit(self):
        text = "x" * 500
        assert _truncate(text) == text

    def test_one_over_limit(self):
        text = "x" * 501
        assert "truncated" in _truncate(text)


class TestSplitThinking:
    def test_no_thinking_tags(self):
        thinking, visible = _split_thinking("hello world")
        assert thinking == ""
        assert visible == "hello world"

    def test_complete_thinking_block(self):
        thinking, visible = _split_thinking("<think>plan this</think>actual response")
        assert thinking == "plan this"
        assert visible == "actual response"

    def test_thinking_only(self):
        thinking, visible = _split_thinking("<think>just thinking</think>")
        assert thinking == "just thinking"
        assert visible == ""

    def test_partial_thinking_unclosed(self):
        """Streaming scenario: <think> opened but not yet closed."""
        thinking, visible = _split_thinking("<think>still thinking...")
        assert thinking == "still thinking..."
        assert visible == ""

    def test_text_before_thinking(self):
        thinking, visible = _split_thinking("prefix <think>thought</think> suffix")
        assert thinking == "thought"
        assert visible == "prefix  suffix"

    def test_multiple_thinking_blocks(self):
        raw = "<think>first</think>middle<think>second</think>end"
        thinking, visible = _split_thinking(raw)
        assert "first" in thinking
        assert "second" in thinking
        assert "middle" in visible
        assert "end" in visible

    def test_case_insensitive(self):
        thinking, visible = _split_thinking("<THINK>thought</THINK>visible")
        assert thinking == "thought"
        assert visible == "visible"

    def test_empty_string(self):
        thinking, visible = _split_thinking("")
        assert thinking == ""
        assert visible == ""

    def test_empty_thinking_block(self):
        thinking, visible = _split_thinking("<think></think>content")
        assert thinking == ""
        assert visible == "content"

    def test_multiline_thinking(self):
        raw = "<think>line 1\nline 2\nline 3</think>response"
        thinking, visible = _split_thinking(raw)
        assert "line 1\nline 2\nline 3" == thinking
        assert visible == "response"
