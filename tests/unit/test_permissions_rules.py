"""Tests for the rule-based permission system (parse, match, authorize)."""

from forge.agent.permissions import (
    PermissionPolicy,
    PermissionRule,
    PermissionRuleSet,
    authorize,
    extract_permission_subject,
    parse_permission_rule,
    _matches_rule,
)


class TestParsePermissionRule:
    def test_simple_tool_name(self):
        rule = parse_permission_rule("run_command")
        assert rule.tool == "run_command"
        assert rule.match_type == "any"
        assert rule.match_value == ""

    def test_exact_match(self):
        rule = parse_permission_rule("run_command(git status)")
        assert rule.tool == "run_command"
        assert rule.match_type == "exact"
        assert rule.match_value == "git status"

    def test_prefix_match(self):
        rule = parse_permission_rule("run_command(git:*)")
        assert rule.tool == "run_command"
        assert rule.match_type == "prefix"
        assert rule.match_value == "git"

    def test_whitespace_stripped(self):
        rule = parse_permission_rule("  write_file  ")
        assert rule.tool == "write_file"
        assert rule.match_type == "any"

    def test_prefix_with_path(self):
        rule = parse_permission_rule("write_file(/home/user/safe/:*)")
        assert rule.tool == "write_file"
        assert rule.match_type == "prefix"
        assert rule.match_value == "/home/user/safe/"


class TestExtractPermissionSubject:
    def test_run_command(self):
        assert extract_permission_subject("run_command", {"command": "ls -la"}) == "ls -la"

    def test_write_file(self):
        assert extract_permission_subject("write_file", {"file_path": "/tmp/x.py"}) == "/tmp/x.py"

    def test_edit_file(self):
        assert extract_permission_subject("edit_file", {"file_path": "src/main.py"}) == "src/main.py"

    def test_read_file(self):
        assert extract_permission_subject("read_file", {"file_path": "README.md"}) == "README.md"

    def test_list_files_path_key(self):
        assert extract_permission_subject("list_files", {"path": "/src"}) == "/src"

    def test_web_fetch(self):
        assert extract_permission_subject("web_fetch", {"url": "https://example.com"}) == "https://example.com"

    def test_web_search(self):
        assert extract_permission_subject("web_search", {"query": "python docs"}) == "python docs"

    def test_unknown_tool(self):
        assert extract_permission_subject("some_tool", {"x": 1}) == ""

    def test_missing_key(self):
        assert extract_permission_subject("run_command", {}) == ""


class TestMatchesRule:
    def test_any_matches_any_invocation(self):
        rule = PermissionRule(tool="run_command", match_type="any")
        assert _matches_rule(rule, "run_command", "anything") is True

    def test_any_wrong_tool(self):
        rule = PermissionRule(tool="run_command", match_type="any")
        assert _matches_rule(rule, "write_file", "anything") is False

    def test_exact_matches(self):
        rule = PermissionRule(tool="run_command", match_type="exact", match_value="git status")
        assert _matches_rule(rule, "run_command", "git status") is True

    def test_exact_no_match(self):
        rule = PermissionRule(tool="run_command", match_type="exact", match_value="git status")
        assert _matches_rule(rule, "run_command", "git push") is False

    def test_prefix_matches(self):
        rule = PermissionRule(tool="run_command", match_type="prefix", match_value="git")
        assert _matches_rule(rule, "run_command", "git status") is True
        assert _matches_rule(rule, "run_command", "git push origin main") is True

    def test_prefix_no_match(self):
        rule = PermissionRule(tool="run_command", match_type="prefix", match_value="git")
        assert _matches_rule(rule, "run_command", "npm install") is False

    def test_unknown_match_type(self):
        rule = PermissionRule(tool="run_command", match_type="regex", match_value=".*")
        assert _matches_rule(rule, "run_command", "anything") is False


class TestAuthorize:
    def _empty_rules(self):
        return PermissionRuleSet()

    def test_yolo_allows_everything(self):
        result = authorize("run_command", {"command": "rm -rf /"}, PermissionPolicy.YOLO, self._empty_rules())
        assert result == "allow"

    def test_auto_allows_safe_tools(self):
        result = authorize("read_file", {"file_path": "x"}, PermissionPolicy.AUTO, self._empty_rules())
        assert result == "allow"

    def test_auto_asks_for_dangerous_tools(self):
        result = authorize("write_file", {"file_path": "x"}, PermissionPolicy.AUTO, self._empty_rules())
        assert result == "ask"

    def test_ask_asks_for_everything(self):
        result = authorize("read_file", {"file_path": "x"}, PermissionPolicy.ASK, self._empty_rules())
        assert result == "ask"

    def test_deny_rule_overrides_yolo(self):
        """Deny rules take precedence over YOLO mode."""
        rules = PermissionRuleSet(
            deny=[PermissionRule(tool="run_command", match_type="prefix", match_value="rm")]
        )
        result = authorize("run_command", {"command": "rm -rf /"}, PermissionPolicy.YOLO, rules)
        assert result == "block"

    def test_allow_rule_overrides_auto_ask(self):
        """Allow rules can whitelist normally-prompted tools."""
        rules = PermissionRuleSet(
            allow=[PermissionRule(tool="run_command", match_type="prefix", match_value="git")]
        )
        result = authorize("run_command", {"command": "git status"}, PermissionPolicy.AUTO, rules)
        assert result == "allow"

    def test_ask_rule_overrides_allow(self):
        """Ask rules take priority over allow rules (stage 2 before stage 3)."""
        rules = PermissionRuleSet(
            allow=[PermissionRule(tool="run_command", match_type="any")],
            ask=[PermissionRule(tool="run_command", match_type="exact", match_value="git push")],
        )
        result = authorize("run_command", {"command": "git push"}, PermissionPolicy.YOLO, rules)
        assert result == "ask"

    def test_deny_overrides_ask_and_allow(self):
        """Deny (stage 1) beats ask (stage 2) and allow (stage 3)."""
        rules = PermissionRuleSet(
            allow=[PermissionRule(tool="run_command", match_type="any")],
            ask=[PermissionRule(tool="run_command", match_type="any")],
            deny=[PermissionRule(tool="run_command", match_type="prefix", match_value="sudo")],
        )
        result = authorize("run_command", {"command": "sudo rm"}, PermissionPolicy.YOLO, rules)
        assert result == "block"

    def test_unmatched_tool_falls_through(self):
        """Rules for different tools don't match."""
        rules = PermissionRuleSet(
            deny=[PermissionRule(tool="write_file", match_type="any")]
        )
        result = authorize("run_command", {"command": "ls"}, PermissionPolicy.YOLO, rules)
        assert result == "allow"

    def test_auto_unknown_tool_asks(self):
        """Unknown tools under AUTO policy default to ask."""
        result = authorize("unknown_tool", {}, PermissionPolicy.AUTO, self._empty_rules())
        assert result == "ask"
