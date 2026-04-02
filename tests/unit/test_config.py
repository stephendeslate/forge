"""Tests for configuration settings."""

import os
from unittest.mock import patch

from forge.config import Settings, OllamaSettings, NPUSettings


class TestOllamaSettings:
    def test_defaults(self):
        s = OllamaSettings()
        assert s.base_url == "http://127.0.0.1:11434"
        assert s.heavy_model
        assert s.fast_model
        assert s.embed_model

    def test_env_override(self):
        with patch.dict(os.environ, {"FORGE_OLLAMA_BASE_URL": "http://custom:9999"}):
            s = OllamaSettings()
            assert s.base_url == "http://custom:9999"


class TestNPUSettings:
    def test_defaults(self):
        s = NPUSettings()
        assert s.enabled is False
        assert "52625" in s.base_url
        assert s.timeout == 60

    def test_env_enable(self):
        with patch.dict(os.environ, {"FORGE_NPU_ENABLED": "true"}):
            s = NPUSettings()
            assert s.enabled is True


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.default_route == "auto"
        assert s.streaming is True
        assert s.max_history == 50
        assert s.persist_history is True
        assert isinstance(s.ollama, OllamaSettings)
        assert isinstance(s.npu, NPUSettings)
        assert s.agent.num_ctx == 131072
        assert s.agent.token_budget == 120000

    def test_env_override_streaming(self):
        with patch.dict(os.environ, {"FORGE_STREAMING": "false"}):
            s = Settings()
            assert s.streaming is False

    def test_ensure_config_dir(self):
        Settings.ensure_config_dir()
