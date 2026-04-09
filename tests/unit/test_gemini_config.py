"""Unit tests for GeminiSettings in forge.config."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from forge.config import GeminiSettings, Settings


class TestGeminiSettingsDefaults:
    """Verify GeminiSettings default values."""

    def test_defaults(self):
        s = GeminiSettings()
        assert s.enabled is True
        assert s.model == "gemini-2.5-flash"
        assert s.fallback_model == "gemini-2.0-flash"
        assert s.api_key == ""
        assert s.timeout == 120
        assert s.critique_model == "gemini-2.5-flash"

    def test_env_override(self):
        env = {
            "FORGE_GEMINI_ENABLED": "true",
            "FORGE_GEMINI_MODEL": "gemini-custom",
            "FORGE_GEMINI_FALLBACK_MODEL": "gemini-flash-custom",
            "FORGE_GEMINI_API_KEY": "AIzaEnvTest",
            "FORGE_GEMINI_TIMEOUT": "60",
        }
        with patch.dict(os.environ, env):
            s = GeminiSettings()
            assert s.enabled is True
            assert s.model == "gemini-custom"
            assert s.fallback_model == "gemini-flash-custom"
            assert s.api_key == "AIzaEnvTest"
            assert s.timeout == 60


class TestGeminiInSettings:
    """Verify GeminiSettings is wired into the main Settings class."""

    def test_settings_has_gemini(self):
        s = Settings()
        assert hasattr(s, "gemini")
        assert isinstance(s.gemini, GeminiSettings)
        assert s.gemini.enabled is True
