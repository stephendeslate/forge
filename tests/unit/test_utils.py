"""Tests for agent utility functions."""

from forge.agent.utils import head_tail_truncate


class TestHeadTailTruncate:
    def test_short_text_unchanged(self):
        assert head_tail_truncate("hello", 100) == "hello"

    def test_exact_limit_unchanged(self):
        text = "x" * 100
        assert head_tail_truncate(text, 100) == text

    def test_one_over_limit_truncated(self):
        text = "x" * 101
        result = head_tail_truncate(text, 100)
        assert "truncated" in result

    def test_head_tail_preserved(self):
        # 26 chars: abcdefghijklmnopqrstuvwxyz
        text = "AAAA" + "x" * 100 + "ZZZZ"  # 108 chars
        result = head_tail_truncate(text, 20)
        # head=10, tail=10 by default ratio=0.5
        assert result.startswith("AAAA")
        assert result.endswith("ZZZZ")
        assert "truncated" in result

    def test_custom_ratio(self):
        text = "A" * 10 + "B" * 90  # 100 chars
        result = head_tail_truncate(text, 50, ratio=0.8)
        # head = 40, tail = 10
        head_part = result.split("\n\n")[0]
        assert len(head_part) == 40  # 40 A's
        assert head_part == "A" * 10 + "B" * 30

    def test_dropped_count_accurate(self):
        text = "x" * 1000
        result = head_tail_truncate(text, 100)
        # dropped = 1000 - 100 = 900
        assert "900" in result

    def test_empty_string(self):
        assert head_tail_truncate("", 100) == ""

    def test_zero_max_chars(self):
        result = head_tail_truncate("hello", 0)
        assert "truncated" in result
