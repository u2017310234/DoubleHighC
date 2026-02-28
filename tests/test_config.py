"""Tests for configuration utilities."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch


class TestGetDbConfig:
    """Tests for database configuration loading."""

    def test_returns_none_when_missing_vars(self):
        with patch.dict(os.environ, {}, clear=True):
            from src.config import get_db_config
            assert get_db_config() is None

    def test_returns_config_when_all_vars_present(self):
        env = {
            "DAILY_DATABASE": "testdb",
            "DAILY_USER": "testuser",
            "DAILY_PASSWORD": "testpass",
            "DAILY_HOST": "localhost",
        }
        with patch.dict(os.environ, env, clear=False):
            from src.config import get_db_config
            config = get_db_config()
            assert config is not None
            assert config["dbname"] == "testdb"
            assert config["user"] == "testuser"
            assert config["port"] == "5432"  # default

    def test_uses_custom_port(self):
        env = {
            "DAILY_DATABASE": "testdb",
            "DAILY_USER": "testuser",
            "DAILY_PASSWORD": "testpass",
            "DAILY_HOST": "localhost",
            "DAILY_PORT": "5433",
        }
        with patch.dict(os.environ, env, clear=False):
            from src.config import get_db_config
            config = get_db_config()
            assert config["port"] == "5433"


class TestLoadPrompt:
    """Tests for prompt file loading."""

    def test_loads_existing_file(self, tmp_path):
        prompt_file = tmp_path / "test.md"
        prompt_file.write_text("Test prompt content", encoding="utf-8")

        from src.config import load_prompt
        content = load_prompt(prompt_file)
        assert content == "Test prompt content"

    def test_raises_on_missing_file(self):
        from src.config import load_prompt
        with pytest.raises(FileNotFoundError, match="提示词文件不存在"):
            load_prompt("/nonexistent/path/prompt.md")
