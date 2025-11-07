"""Comprehensive unit tests for configuration."""

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_loads_defaults() -> None:
    """Test that settings load with default values."""
    # Create settings with minimal env
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
    )
    assert settings.openai_api_key == "test-key"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.openai_embedding_dimensions == 1536
    assert settings.openai_timeout == 300
    assert settings.log_level == "INFO"
    assert settings.max_file_size_mb == 10
    assert settings.postgres_user == "ai"
    assert settings.postgres_password == "ai"
    assert settings.postgres_db == "ai"
    assert settings.postgres_host == "postgres"
    assert settings.postgres_port == 5432


def test_settings_validation_passes() -> None:
    """Test that valid settings pass validation."""
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        openai_embedding_model="text-embedding-3-small",
        openai_embedding_dimensions=1536,
        openai_timeout=300,
    )
    settings.validate_openai_config()  # Should not raise


def test_settings_validation_fails_missing_api_key() -> None:
    """Test that missing API key raises error."""
    settings = Settings(
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_embedding_model="text-embedding-3-small",
        openai_embedding_dimensions=1536,
    )
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        settings.validate_openai_config()


def test_settings_validation_fails_missing_model() -> None:
    """Test that missing model raises error."""
    settings = Settings(
        openai_api_key="test-key",
        openai_model="",
        openai_embedding_model="text-embedding-3-small",
        openai_embedding_dimensions=1536,
    )
    with pytest.raises(ValueError, match="OPENAI_MODEL is required"):
        settings.validate_openai_config()


def test_settings_validation_fails_invalid_timeout() -> None:
    """Test that invalid timeout raises error."""
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        openai_embedding_model="text-embedding-3-small",
        openai_embedding_dimensions=1536,
        openai_timeout=-1,
    )
    with pytest.raises(ValueError, match="OPENAI_TIMEOUT must be positive"):
        settings.validate_openai_config()


def test_settings_validation_fails_missing_embedding_model() -> None:
    """Test that missing embedding model raises error."""
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        openai_embedding_model="",
        openai_embedding_dimensions=1536,
    )
    with pytest.raises(ValueError, match="OPENAI_EMBEDDING_MODEL is required"):
        settings.validate_openai_config()


def test_settings_validation_fails_missing_embedding_dimensions() -> None:
    """Test that missing embedding dimensions raises error."""
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        openai_embedding_model="text-embedding-3-small",
        openai_embedding_dimensions=0,
    )
    with pytest.raises(ValueError, match="OPENAI_EMBEDDING_DIMENSIONS is required"):
        settings.validate_openai_config()


def test_get_postgres_url_default() -> None:
    """Test get_postgres_url with default values."""
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
    )
    url = settings.get_postgres_url()
    # Check default values are used
    assert "ai" in url  # default user/password/db
    assert "postgres" in url  # default host
    assert "5432" in url  # default port


def test_get_postgres_url_custom() -> None:
    """Test get_postgres_url with custom URL."""
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        postgres_url="postgresql+psycopg://custom:url@host:5432/db",
    )
    url = settings.get_postgres_url()
    assert url == "postgresql+psycopg://custom:url@host:5432/db"


def test_get_max_file_size_bytes() -> None:
    """Test get_max_file_size_bytes method."""
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        max_file_size_mb=10,
    )
    assert settings.get_max_file_size_bytes() == 10 * 1024 * 1024


def test_get_max_file_size_bytes_custom() -> None:
    """Test get_max_file_size_bytes with custom size."""
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        max_file_size_mb=20,
    )
    assert settings.get_max_file_size_bytes() == 20 * 1024 * 1024


def test_get_allowed_extensions() -> None:
    """Test get_allowed_extensions method."""
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        allowed_file_extensions=".pdf",
    )
    extensions = settings.get_allowed_extensions()
    assert isinstance(extensions, set)
    assert ".pdf" in extensions
    assert len(extensions) == 1  # Only PDF allowed


def test_get_allowed_extensions_default() -> None:
    """Test get_allowed_extensions with default value (PDF-only)."""
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
    )
    extensions = settings.get_allowed_extensions()
    assert isinstance(extensions, set)
    assert ".pdf" in extensions
    assert len(extensions) == 1  # Only PDF allowed by default
    # Verify no other extensions
    assert ".csv" not in extensions
    assert ".json" not in extensions


def test_get_allowed_extensions_strips_whitespace() -> None:
    """Test get_allowed_extensions strips whitespace."""
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        allowed_file_extensions=".pdf ",
    )
    extensions = settings.get_allowed_extensions()
    assert ".pdf" in extensions


def test_get_allowed_extensions_lowercase() -> None:
    """Test get_allowed_extensions converts to lowercase."""
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        allowed_file_extensions=".PDF",
    )
    extensions = settings.get_allowed_extensions()
    assert ".pdf" in extensions


def test_settings_custom_values() -> None:
    """Test settings with custom values."""
    settings = Settings(
        openai_api_key="custom-key",
        openai_model="gpt-4",
        openai_embedding_model="text-embedding-3-large",
        openai_embedding_dimensions=3072,
        openai_timeout=600,
        max_file_size_mb=20,
        log_level="DEBUG",
        postgres_user="custom_user",
        postgres_password="custom_pass",
        postgres_db="custom_db",
        postgres_host="custom_host",
        postgres_port=5433,
    )
    assert settings.openai_api_key == "custom-key"
    assert settings.openai_model == "gpt-4"
    assert settings.openai_embedding_model == "text-embedding-3-large"
    assert settings.openai_embedding_dimensions == 3072
    assert settings.openai_timeout == 600
    assert settings.max_file_size_mb == 20
    assert settings.log_level == "DEBUG"
    assert settings.postgres_user == "custom_user"
    assert settings.postgres_password == "custom_pass"
    assert settings.postgres_db == "custom_db"
    assert settings.postgres_host == "custom_host"
    assert settings.postgres_port == 5433

