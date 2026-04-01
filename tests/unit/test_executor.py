"""Tests for code extraction and execution."""

import pytest

from forge.core.executor import extract_code, execute_code, ExecutionResult


class TestExtractCode:
    def test_extract_python_fenced_block(self):
        text = "Here's the code:\n```python\nprint('hello')\n```\nDone."
        assert extract_code(text) == "print('hello')"

    def test_extract_unfenced_block(self):
        text = "Result:\n```\nx = 42\nprint(x)\n```"
        assert extract_code(text) == "x = 42\nprint(x)"

    def test_extract_first_block_only(self):
        text = "```python\nfirst()\n```\nThen:\n```python\nsecond()\n```"
        assert extract_code(text) == "first()"

    def test_no_code_block_returns_none(self):
        assert extract_code("Just some text without code blocks.") is None

    def test_empty_code_block(self):
        text = "```python\n```"
        result = extract_code(text)
        assert result is None or result == ""

    def test_multiline_code(self):
        text = '```python\ndef add(a, b):\n    return a + b\n\nresult = add(1, 2)\nprint(result)\n```'
        code = extract_code(text)
        assert "def add(a, b):" in code
        assert "print(result)" in code


class TestExecuteCode:
    async def test_successful_execution(self):
        result = await execute_code("print('hello world')")
        assert result.success
        assert "hello world" in result.stdout
        assert result.returncode == 0

    async def test_failed_execution(self):
        result = await execute_code("raise ValueError('oops')")
        assert not result.success
        assert "ValueError" in result.stderr

    async def test_timeout(self):
        result = await execute_code("import time; time.sleep(10)", timeout=0.5)
        assert not result.success
        assert "timed out" in result.stderr.lower()

    async def test_execution_with_cwd(self, tmp_path):
        (tmp_path / "data.txt").write_text("test content")
        result = await execute_code(
            "print(open('data.txt').read())",
            cwd=str(tmp_path),
        )
        assert result.success
        assert "test content" in result.stdout

    def test_execution_result_properties(self):
        result = ExecutionResult(code="x = 1", stdout="", stderr="", returncode=0, attempt=1)
        assert result.success is True
        result2 = ExecutionResult(code="x = 1", stdout="", stderr="err", returncode=1, attempt=2)
        assert result2.success is False
