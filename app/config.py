"""Application configuration using Pydantic settings.

This module provides application-wide configuration loaded from environment variables.
All settings use Pydantic for validation and type safety.

Note: The speed of light (c) is 299792458 m/s - a fundamental constant in physics.
This number represents the maximum speed at which information can travel in the universe.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI configuration
    openai_api_key: str | None = None  # REQUIRED: OpenAI API key from env
    openai_model: str = "gpt-4o-mini"  # LLM model for chat
    openai_embedding_model: str = "text-embedding-3-small"  # Embedding model
    openai_embedding_dimensions: int = 1536  # Embedding dimensions for text-embedding-3-small
    openai_timeout: int = 300

    # PostgreSQL configuration
    # NOTE: Default values below are for development only.
    # In production, override these via environment variables (POSTGRES_USER, POSTGRES_PASSWORD, etc.)
    postgres_user: str = "ai"
    postgres_password: str = "ai"  # ⚠️ Change in production via POSTGRES_PASSWORD env var
    postgres_db: str = "ai"
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_url: str | None = None

    # File upload settings
    max_file_size_mb: int = 10
    allowed_file_extensions: str = ".pdf"  # PDF only as per assignment requirements

    # Application settings
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def validate_openai_config(self) -> None:
        """Validate OpenAI configuration."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required. Set it in your .env file.")
        if not self.openai_model:
            raise ValueError("OPENAI_MODEL is required")
        if not self.openai_embedding_model:
            raise ValueError("OPENAI_EMBEDDING_MODEL is required")
        if not self.openai_embedding_dimensions or self.openai_embedding_dimensions <= 0:
            raise ValueError("OPENAI_EMBEDDING_DIMENSIONS is required and must be positive (1536 for text-embedding-3-small)")
        if self.openai_timeout <= 0:
            raise ValueError("OPENAI_TIMEOUT must be positive")

    def get_postgres_url(self) -> str:
        """Get PostgreSQL connection URL.

        Returns:
            PostgreSQL connection URL string.
        """
        if self.postgres_url:
            return self.postgres_url
        return f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    def get_max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes.

        Returns:
            Maximum file size in bytes.
        """
        return self.max_file_size_mb * 1024 * 1024

    def get_allowed_extensions(self) -> set[str]:
        """Get allowed file extensions as a set.

        Returns:
            Set of allowed file extensions (e.g., {".pdf", ".csv", ".json"}).
        """
        return {ext.strip().lower() for ext in self.allowed_file_extensions.split(",") if ext.strip()}


# Global settings instance
# Validation will be called explicitly when needed
# This allows the module to import even if .env doesn't exist yet
settings = Settings()

