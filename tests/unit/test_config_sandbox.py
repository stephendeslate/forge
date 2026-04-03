"""Tests for SandboxSettings rule fields."""

from forge.config import SandboxSettings, DEFAULT_BLOCKED_PATTERNS, DEFAULT_WARN_PATTERNS


class TestSandboxSettings:
    def test_defaults(self):
        s = SandboxSettings()
        assert s.enabled is True
        assert s.restrict_paths is True
        assert len(s.blocked_patterns) == len(DEFAULT_BLOCKED_PATTERNS)
        assert len(s.warn_patterns) == len(DEFAULT_WARN_PATTERNS)

    def test_rule_fields_default_empty(self):
        s = SandboxSettings()
        assert s.allow_rules == []
        assert s.deny_rules == []
        assert s.ask_rules == []

    def test_rule_fields_accept_values(self):
        s = SandboxSettings(
            allow_rules=["run_command(git:*)"],
            deny_rules=["run_command(sudo:*)"],
            ask_rules=["write_file"],
        )
        assert len(s.allow_rules) == 1
        assert s.allow_rules[0] == "run_command(git:*)"
        assert s.deny_rules[0] == "run_command(sudo:*)"
        assert s.ask_rules[0] == "write_file"

    def test_allowed_paths_default_empty(self):
        s = SandboxSettings()
        assert s.allowed_paths == []
