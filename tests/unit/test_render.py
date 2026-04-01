"""Tests for render helper functions."""

from forge.agent.render import _tool_style, _format_tool_args, _truncate


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
        text = "x" * 2000
        assert _truncate(text) == text

    def test_one_over_limit(self):
        text = "x" * 2001
        assert "truncated" in _truncate(text)
