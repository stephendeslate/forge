"""Exclude manual tests from pytest collection.

These tests require a running Ollama instance and are meant to be run
directly: uv run python tests/manual/test_verify_gaps.py
"""

collect_ignore_glob = ["test_*.py"]
