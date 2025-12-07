"""Tests for loader configuration module."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nerve.loader.config import Config, _load_dotenv


class TestConfig:
    """Tests for Config dataclass."""

    def test_config_creation_with_required_fields(self) -> None:
        """Test creating Config with required fields."""
        config = Config(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
        )
        assert config.neo4j_uri == "bolt://localhost:7687"
        assert config.neo4j_user == "neo4j"
        assert config.neo4j_password == "password"

    def test_config_default_values(self) -> None:
        """Test Config default values."""
        config = Config(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
        )
        assert config.ncbi_api_key is None
        assert config.disgenet_api_key is None
        assert config.cosmic_email is None
        assert config.cosmic_password is None
        assert config.data_dir == Path("./data")
        assert config.cosmic_version == "v103"
        assert config.reactome_species == "Homo sapiens"
        assert config.batch_size == 5000
        assert config.max_retries == 3
        assert config.retry_initial_delay == 2.0
        assert config.retry_max_delay == 60.0

    def test_has_cosmic_credentials_true(self) -> None:
        """Test has_cosmic_credentials when both are set."""
        config = Config(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            cosmic_email="test@example.com",
            cosmic_password="cosmic_pass",
        )
        assert config.has_cosmic_credentials() is True

    def test_has_cosmic_credentials_false_missing_email(self) -> None:
        """Test has_cosmic_credentials when email is missing."""
        config = Config(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            cosmic_password="cosmic_pass",
        )
        assert config.has_cosmic_credentials() is False

    def test_has_cosmic_credentials_false_missing_password(self) -> None:
        """Test has_cosmic_credentials when password is missing."""
        config = Config(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            cosmic_email="test@example.com",
        )
        assert config.has_cosmic_credentials() is False

    def test_has_ncbi_api_key_true(self) -> None:
        """Test has_ncbi_api_key when set."""
        config = Config(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            ncbi_api_key="test_key",
        )
        assert config.has_ncbi_api_key() is True

    def test_has_ncbi_api_key_false(self) -> None:
        """Test has_ncbi_api_key when not set."""
        config = Config(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
        )
        assert config.has_ncbi_api_key() is False

    def test_has_disgenet_api_key_true(self) -> None:
        """Test has_disgenet_api_key when set."""
        config = Config(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            disgenet_api_key="test_key",
        )
        assert config.has_disgenet_api_key() is True

    def test_has_disgenet_api_key_false(self) -> None:
        """Test has_disgenet_api_key when not set."""
        config = Config(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
        )
        assert config.has_disgenet_api_key() is False


class TestConfigFromEnv:
    """Tests for Config.from_env() class method."""

    def test_from_env_with_required_vars(self) -> None:
        """Test loading config from environment variables."""
        env = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Config.from_env(env_file=Path("/nonexistent/.env"))
            assert config.neo4j_uri == "bolt://localhost:7687"
            assert config.neo4j_user == "neo4j"
            assert config.neo4j_password == "secret"

    def test_from_env_missing_neo4j_uri(self) -> None:
        """Test error when NEO4J_URI is missing."""
        env = {
            "NEO4J_PASSWORD": "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="NEO4J_URI"):
                Config.from_env(env_file=Path("/nonexistent/.env"))

    def test_from_env_missing_neo4j_password(self) -> None:
        """Test error when NEO4J_PASSWORD is missing."""
        env = {
            "NEO4J_URI": "bolt://localhost:7687",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
                Config.from_env(env_file=Path("/nonexistent/.env"))

    def test_from_env_default_user(self) -> None:
        """Test default NEO4J_USER value."""
        env = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_PASSWORD": "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Config.from_env(env_file=Path("/nonexistent/.env"))
            assert config.neo4j_user == "neo4j"

    def test_from_env_with_optional_vars(self) -> None:
        """Test loading optional environment variables."""
        env = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_PASSWORD": "secret",
            "NCBI_API_KEY": "ncbi_key",
            "DISGENET_API_KEY": "disgenet_key",
            "COSMIC_EMAIL": "test@example.com",
            "COSMIC_PASSWORD": "cosmic_pass",
            "DATA_DIR": "/custom/data",
            "COSMIC_VERSION": "v104",
            "REACTOME_SPECIES": "Mus musculus",
            "NEO4J_BATCH_SIZE": "10000",
            "MAX_RETRIES": "5",
            "RETRY_INITIAL_DELAY": "3.0",
            "RETRY_MAX_DELAY": "120.0",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Config.from_env(env_file=Path("/nonexistent/.env"))
            assert config.ncbi_api_key == "ncbi_key"
            assert config.disgenet_api_key == "disgenet_key"
            assert config.cosmic_email == "test@example.com"
            assert config.cosmic_password == "cosmic_pass"
            assert config.data_dir == Path("/custom/data")
            assert config.cosmic_version == "v104"
            assert config.reactome_species == "Mus musculus"
            assert config.batch_size == 10000
            assert config.max_retries == 5
            assert config.retry_initial_delay == 3.0
            assert config.retry_max_delay == 120.0


class TestLoadDotenv:
    """Tests for _load_dotenv helper function."""

    def test_load_dotenv_basic(self, tmp_path: Path) -> None:
        """Test loading basic .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=test_value\nANOTHER_VAR=another_value\n")

        # Clear existing vars to test loading
        with patch.dict(os.environ, {}, clear=True):
            _load_dotenv(env_file)
            assert os.environ.get("TEST_VAR") == "test_value"
            assert os.environ.get("ANOTHER_VAR") == "another_value"

    def test_load_dotenv_ignores_comments(self, tmp_path: Path) -> None:
        """Test that comments are ignored."""
        env_file = tmp_path / ".env"
        env_file.write_text("# This is a comment\nTEST_VAR=value\n#IGNORED=value\n")

        with patch.dict(os.environ, {}, clear=True):
            _load_dotenv(env_file)
            assert os.environ.get("TEST_VAR") == "value"
            assert os.environ.get("IGNORED") is None

    def test_load_dotenv_ignores_empty_lines(self, tmp_path: Path) -> None:
        """Test that empty lines are ignored."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=value\n\n\nANOTHER=val\n")

        with patch.dict(os.environ, {}, clear=True):
            _load_dotenv(env_file)
            assert os.environ.get("TEST_VAR") == "value"
            assert os.environ.get("ANOTHER") == "val"

    def test_load_dotenv_strips_quotes(self, tmp_path: Path) -> None:
        """Test that quotes are stripped from values."""
        env_file = tmp_path / ".env"
        env_file.write_text("DOUBLE=\"double quoted\"\nSINGLE='single quoted'\n")

        with patch.dict(os.environ, {}, clear=True):
            _load_dotenv(env_file)
            assert os.environ.get("DOUBLE") == "double quoted"
            assert os.environ.get("SINGLE") == "single quoted"

    def test_load_dotenv_does_not_override_existing(self, tmp_path: Path) -> None:
        """Test that existing env vars are not overridden."""
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING=from_file\n")

        with patch.dict(os.environ, {"EXISTING": "from_env"}, clear=True):
            _load_dotenv(env_file)
            assert os.environ.get("EXISTING") == "from_env"

    def test_load_dotenv_handles_missing_file(self) -> None:
        """Test that missing file doesn't raise error."""
        # Should not raise
        _load_dotenv(Path("/nonexistent/path/.env"))
