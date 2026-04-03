"""Unit tests for forge.agent.gemini module."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from forge.agent.gemini import (
    _ensure_api_key,
    get_gemini_model_string,
    get_gemini_model_settings,
    is_gemini_available,
)

# The gemini module does lazy `from forge.config import settings` inside each
# function, so we patch forge.config.settings (the canonical location).
_SETTINGS_PATH = "forge.config.settings"


class TestEnsureApiKey:
    """Tests for _ensure_api_key()."""

    def test_returns_none_when_no_key(self):
        with patch(_SETTINGS_PATH) as mock_settings:
            mock_settings.gemini.api_key = ""
            env = os.environ.copy()
            env.pop("GOOGLE_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                assert _ensure_api_key() is None

    def test_returns_key_from_config(self):
        with patch(_SETTINGS_PATH) as mock_settings:
            mock_settings.gemini.api_key = "AIzaTest123"
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GOOGLE_API_KEY", None)
                key = _ensure_api_key()
                assert key == "AIzaTest123"
                assert os.environ["GOOGLE_API_KEY"] == "AIzaTest123"
        # Cleanup
        os.environ.pop("GOOGLE_API_KEY", None)

    def test_returns_key_from_env(self):
        with patch(_SETTINGS_PATH) as mock_settings:
            mock_settings.gemini.api_key = ""
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "EnvKey456"}):
                key = _ensure_api_key()
                assert key == "EnvKey456"

    def test_does_not_overwrite_existing_env(self):
        with patch(_SETTINGS_PATH) as mock_settings:
            mock_settings.gemini.api_key = "ConfigKey"
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "ExistingKey"}):
                _ensure_api_key()
                # Should NOT overwrite existing env var
                assert os.environ["GOOGLE_API_KEY"] == "ExistingKey"


class TestGetGeminiModelString:
    """Tests for get_gemini_model_string()."""

    def test_returns_none_without_key(self):
        with patch("forge.agent.gemini._ensure_api_key", return_value=None):
            assert get_gemini_model_string() is None

    def test_returns_primary_model(self):
        with patch("forge.agent.gemini._ensure_api_key", return_value="key"), \
             patch(_SETTINGS_PATH) as mock_settings:
            mock_settings.gemini.model = "gemini-2.5-pro"
            mock_settings.gemini.fallback_model = "gemini-2.0-flash"
            result = get_gemini_model_string(fallback=False)
            assert result == "google-gla:gemini-2.5-pro"

    def test_returns_fallback_model(self):
        with patch("forge.agent.gemini._ensure_api_key", return_value="key"), \
             patch(_SETTINGS_PATH) as mock_settings:
            mock_settings.gemini.model = "gemini-2.5-pro"
            mock_settings.gemini.fallback_model = "gemini-2.0-flash"
            result = get_gemini_model_string(fallback=True)
            assert result == "google-gla:gemini-2.0-flash"


class TestGetGeminiModelSettings:
    """Tests for get_gemini_model_settings()."""

    def test_returns_timeout_dict(self):
        with patch(_SETTINGS_PATH) as mock_settings:
            mock_settings.gemini.timeout = 120
            result = get_gemini_model_settings()
            assert result == {"timeout": 120}

    def test_returns_none_when_timeout_zero(self):
        with patch(_SETTINGS_PATH) as mock_settings:
            mock_settings.gemini.timeout = 0
            result = get_gemini_model_settings()
            assert result is None


class TestIsGeminiAvailable:
    """Tests for is_gemini_available()."""

    def test_returns_false_when_cloud_disabled(self):
        deps = MagicMock()
        deps.cloud_reasoning_enabled = False
        assert is_gemini_available(deps) is False

    def test_returns_false_when_no_key(self):
        deps = MagicMock()
        deps.cloud_reasoning_enabled = True
        with patch("forge.agent.gemini._ensure_api_key", return_value=None):
            assert is_gemini_available(deps) is False

    def test_returns_true_when_enabled_and_key_present(self):
        deps = MagicMock()
        deps.cloud_reasoning_enabled = True
        with patch("forge.agent.gemini._ensure_api_key", return_value="key"):
            assert is_gemini_available(deps) is True
