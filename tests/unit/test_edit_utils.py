"""Tests for the multi-layer edit matching engine."""

import pytest

from forge.agent.edit_utils import (
    EditMatchError,
    find_and_replace,
    _normalize_ws,
    _find_line_block,
)


class TestExactMatch:
    def test_exact_single_occurrence(self):
        text = "hello world\nfoo bar\n"
        new, method, warning = find_and_replace(text, "foo bar", "baz qux")
        assert method == "exact"
        assert warning == ""
        assert new == "hello world\nbaz qux\n"

    def test_exact_multiple_occurrences_raises(self):
        text = "x = 1\nx = 1\n"
        with pytest.raises(EditMatchError, match="appears 2 times"):
            find_and_replace(text, "x = 1", "x = 2")

    def test_exact_preserves_surrounding(self):
        text = "before\ntarget\nafter\n"
        new, method, _ = find_and_replace(text, "target", "replaced")
        assert new == "before\nreplaced\nafter\n"
        assert method == "exact"


class TestWhitespaceNormalized:
    def test_extra_spaces(self):
        text = "def foo(  x,  y  ):\n    return x + y\n"
        old = "def foo(x, y):\n    return x + y\n"
        new, method, warning = find_and_replace(text, old, "def foo(a, b):\n    return a + b\n")
        assert method == "whitespace_normalized"
        assert "normalizing whitespace" in warning
        assert "def foo(a, b)" in new

    def test_trailing_whitespace_mismatch(self):
        text = "line one  \nline two\n"
        old = "line one\nline two\n"
        new, method, _ = find_and_replace(text, old, "replaced\n")
        assert method == "whitespace_normalized"
        assert "replaced" in new

    def test_tabs_vs_spaces(self):
        text = "if True:\n\t\treturn 1\n"
        old = "if True:\n        return 1\n"
        new, method, _ = find_and_replace(text, old, "if True:\n    return 2\n")
        assert method == "whitespace_normalized"
        assert "return 2" in new


class TestFuzzyLineMatch:
    def test_minor_differences(self):
        text = "def hello():\n    msg = 'hello world'\n    print(msg)\n    return msg\n"
        # Model gets it slightly wrong
        old = "def hello():\n    msg = 'hello wrld'\n    print(msg)\n    return msg\n"
        new_text = "def hello():\n    msg = 'goodbye'\n    print(msg)\n    return msg\n"
        new, method, warning = find_and_replace(text, old, new_text)
        assert method == "fuzzy_line"
        assert "Fuzzy matched" in warning
        assert "goodbye" in new

    def test_below_threshold_raises(self):
        text = "completely different content\nnothing similar\n"
        old = "def foo():\n    return 42\n"
        with pytest.raises(EditMatchError):
            find_and_replace(text, old, "replacement\n")

    def test_single_line_skips_fuzzy(self):
        """Single-line old_text should not try fuzzy matching."""
        text = "alpha\nbeta\ngamma\n"
        with pytest.raises(EditMatchError):
            find_and_replace(text, "totally_wrong", "replacement")

    def test_ambiguous_match_rejected(self):
        """When two regions match similarly, fuzzy should reject for safety."""
        text = "def foo():\n    return 1\n\ndef bar():\n    return 1\n"
        old = "def baz():\n    return 1\n"
        with pytest.raises(EditMatchError):
            find_and_replace(text, old, "def baz():\n    return 2\n")


class TestDiagnostics:
    def test_single_line_diagnostic(self):
        text = "alpha = 1\nbeta = 2\ngamma = 3\n"
        with pytest.raises(EditMatchError, match="Closest match") as exc_info:
            find_and_replace(text, "betta = 2", "delta = 4")
        assert "line" in str(exc_info.value).lower()

    def test_multi_line_diagnostic(self):
        text = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        old = "def foo():\n    return 99\n"
        with pytest.raises(EditMatchError, match="Closest match") as exc_info:
            find_and_replace(text, old, "replaced\n")
        assert "Read the file" in str(exc_info.value)

    def test_completely_unrelated_text(self):
        text = "hello world\n"
        with pytest.raises(EditMatchError, match="not found"):
            find_and_replace(text, "zzz_no_match_zzz\nanother line\n", "x\n")


class TestNormalizeWs:
    def test_collapse_multiple_spaces(self):
        assert _normalize_ws("a   b   c") == "a b c"

    def test_strip_trailing(self):
        assert _normalize_ws("hello   \n  world  ") == "hello\n world"

    def test_normalize_newlines(self):
        assert _normalize_ws("a\r\nb\rc") == "a\nb\nc"


class TestFindLineBlock:
    def test_finds_block(self):
        haystack = ["a", "b", "c", "d"]
        assert _find_line_block(haystack, ["b", "c"]) == 1

    def test_not_found(self):
        haystack = ["a", "b", "c"]
        assert _find_line_block(haystack, ["x", "y"]) is None

    def test_empty_needle(self):
        assert _find_line_block(["a", "b"], []) is None

    def test_at_start(self):
        assert _find_line_block(["a", "b", "c"], ["a", "b"]) == 0

    def test_at_end(self):
        assert _find_line_block(["a", "b", "c"], ["b", "c"]) == 1
