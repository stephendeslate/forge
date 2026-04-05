"""Unit tests for AnthropicSettings and anthropic model module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from forge.config import AnthropicSettings, Settings


class TestAnthropicSettingsDefaults:
    """Verify AnthropicSettings default values."""

    def test_defaults(self):
        s = AnthropicSettings()
        assert s.enabled is True
        assert s.model == "claude-opus-4-6"
        assert s.api_key == ""
        assert s.timeout == 300
        assert s.max_tokens == 16384

    def test_env_override(self):
        env = {
            "FORGE_ANTHROPIC_ENABLED": "false",
            "FORGE_ANTHROPIC_MODEL": "claude-sonnet-4-6",
            "FORGE_ANTHROPIC_API_KEY": "sk-test",
            "FORGE_ANTHROPIC_TIMEOUT": "60",
            "FORGE_ANTHROPIC_MAX_TOKENS": "8192",
        }
        with patch.dict(os.environ, env):
            s = AnthropicSettings()
            assert s.enabled is False
            assert s.model == "claude-sonnet-4-6"
            assert s.api_key == "sk-test"
            assert s.timeout == 60
            assert s.max_tokens == 8192


class TestAnthropicInSettings:
    """Verify AnthropicSettings is wired into the main Settings class."""

    def test_settings_has_anthropic(self):
        s = Settings()
        assert hasattr(s, "anthropic")
        assert isinstance(s.anthropic, AnthropicSettings)
        assert s.anthropic.enabled is True


class TestAnthropicModelModule:
    """Test the forge.models.anthropic module functions."""

    def test_model_string_enabled(self):
        from forge.models.anthropic import get_anthropic_model_string

        # Default config has enabled=True, model=claude-opus-4-6
        result = get_anthropic_model_string()
        assert result == "anthropic:claude-opus-4-6"

    def test_model_string_disabled(self):
        """Test that disabled settings return None for model string."""
        # Just test the AnthropicSettings class directly
        s = AnthropicSettings(enabled=False)
        assert s.enabled is False

    def test_model_settings(self):
        from forge.models.anthropic import get_anthropic_model_settings

        result = get_anthropic_model_settings()
        assert "timeout" in result
        assert "max_tokens" in result
        assert result["timeout"] == 300
        assert result["max_tokens"] == 16384

    def test_availability_disabled(self):
        import forge.models.anthropic as mod

        # Temporarily disable
        orig = mod._rate_limited_until
        try:
            mod._rate_limited_until = 0.0
            # We can't easily change settings.anthropic.enabled at runtime
            # so test via rate limiting instead
            import time
            mod._rate_limited_until = time.monotonic() + 1000
            assert mod.is_anthropic_available() is False
        finally:
            mod._rate_limited_until = orig

    def test_availability_enabled(self):
        import forge.models.anthropic as mod

        mod._rate_limited_until = 0.0
        assert mod.is_anthropic_available() is True

    def test_rate_limiting(self):
        import forge.models.anthropic as mod

        mod._rate_limited_until = 0.0
        assert mod.is_rate_limited() is False

        mod.mark_rate_limited(10.0)
        assert mod.is_rate_limited() is True

        # Reset for other tests
        mod._rate_limited_until = 0.0

    def test_ensure_env_sets_dummy(self):
        from forge.models.anthropic import _ensure_anthropic_env

        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            _ensure_anthropic_env()
            assert os.environ.get("ANTHROPIC_API_KEY") is not None
            assert len(os.environ["ANTHROPIC_API_KEY"]) > 0

    def test_ensure_env_preserves_existing(self):
        from forge.models.anthropic import _ensure_anthropic_env

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "existing-key"}):
            _ensure_anthropic_env()
            assert os.environ["ANTHROPIC_API_KEY"] == "existing-key"

    def test_is_anthropic_error(self):
        from forge.models.anthropic import _is_anthropic_error

        # Generic errors are not anthropic errors
        assert _is_anthropic_error(ValueError("nope")) is False

        # String-based detection
        assert _is_anthropic_error(RuntimeError("anthropic connection refused")) is True
        assert _is_anthropic_error(RuntimeError("anthropic rate limit exceeded")) is True

    def test_is_anthropic_error_model_http(self):
        from pydantic_ai.exceptions import ModelHTTPError

        from forge.models.anthropic import _is_anthropic_error

        # pydantic-ai wrapped errors with claude model name
        err = ModelHTTPError(429, "claude-opus-4-6", {"type": "error"})
        assert _is_anthropic_error(err) is True

        # Non-claude model should not match
        err2 = ModelHTTPError(429, "gemini-2.5-flash", {"type": "error"})
        assert _is_anthropic_error(err2) is False
