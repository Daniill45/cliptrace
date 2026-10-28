"""Validated environment-based application configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings validated before any dependency is initialized."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app_name: str = "cliptrace-backend"
    app_env: Literal["development", "test", "production"] = "development"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://cliptrace:cliptrace@db:5432/cliptrace"

    upload_dir: Path = Path("data/uploads")
    max_upload_size_bytes: int = 524_288_000
    max_video_duration_seconds: float = 14_400
    min_video_duration_seconds: float = 1.0
    upload_chunk_size_bytes: int = 1_048_576
    temp_file_max_age_seconds: int = 86_400

    ffprobe_binary: str = "ffprobe"
    ffprobe_timeout_seconds: float = 30

    mock_result_start_seconds: float = 5.0
    mock_result_duration_seconds: float = 5.0
    min_fragment_duration_seconds: float = 5.0
    search_result_padding_seconds: float = 1.0
    search_engine: Literal["mock", "semantic"] = "mock"

    ml_pipeline_version: str = "speech-e5-v1"
    ml_worker_poll_seconds: float = 2.0
    ml_worker_lease_seconds: int = 900
    ml_worker_max_attempts: int = 3
    ml_model_cache_dir: Path = Path("data/models")

    whisper_model: str = "small"
    whisper_model_revision: str = "536b0662742c02347bc0e980a01041f333bce120"
    whisper_device: Literal["cpu", "cuda"] = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str | None = None

    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_model_revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    embedding_device: Literal["cpu"] = "cpu"
    embedding_batch_size: int = 32

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    @field_validator("api_v1_prefix")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        """Require a normalized absolute URL prefix."""
        if not value.startswith("/") or value.endswith("/"):
            raise ValueError("API_V1_PREFIX must start with '/' and must not end with '/'")
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Restrict log levels to standard names."""
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL is not supported")
        return normalized

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, values: list[str]) -> list[str]:
        """Accept only scheme-and-authority HTTP origins."""
        for origin in values:
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
                or parsed.username
                or parsed.password
            ):
                raise ValueError(f"Invalid CORS origin: {origin!r}")
            if origin.endswith("/"):
                raise ValueError(f"CORS origin must not end with '/': {origin!r}")
        return values

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        """Validate limits whose correctness depends on multiple settings."""
        positive_values = {
            "MAX_UPLOAD_SIZE_BYTES": self.max_upload_size_bytes,
            "MAX_VIDEO_DURATION_SECONDS": self.max_video_duration_seconds,
            "MIN_VIDEO_DURATION_SECONDS": self.min_video_duration_seconds,
            "UPLOAD_CHUNK_SIZE_BYTES": self.upload_chunk_size_bytes,
            "TEMP_FILE_MAX_AGE_SECONDS": self.temp_file_max_age_seconds,
            "FFPROBE_TIMEOUT_SECONDS": self.ffprobe_timeout_seconds,
            "ML_WORKER_POLL_SECONDS": self.ml_worker_poll_seconds,
            "ML_WORKER_LEASE_SECONDS": self.ml_worker_lease_seconds,
            "ML_WORKER_MAX_ATTEMPTS": self.ml_worker_max_attempts,
            "EMBEDDING_BATCH_SIZE": self.embedding_batch_size,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"Settings must be positive: {', '.join(invalid)}")
        if self.upload_chunk_size_bytes > self.max_upload_size_bytes:
            raise ValueError("UPLOAD_CHUNK_SIZE_BYTES must not exceed MAX_UPLOAD_SIZE_BYTES")
        if self.min_video_duration_seconds > self.max_video_duration_seconds:
            raise ValueError(
                "MIN_VIDEO_DURATION_SECONDS must not exceed MAX_VIDEO_DURATION_SECONDS"
            )
        if self.mock_result_start_seconds < 0:
            raise ValueError("MOCK_RESULT_START_SECONDS must be non-negative")
        if self.mock_result_duration_seconds <= 0:
            raise ValueError("MOCK_RESULT_DURATION_SECONDS must be positive")
        if self.min_fragment_duration_seconds <= 0:
            raise ValueError("MIN_FRAGMENT_DURATION_SECONDS must be positive")
        if self.search_result_padding_seconds < 0:
            raise ValueError("SEARCH_RESULT_PADDING_SECONDS must be non-negative")
        if self.mock_result_duration_seconds < self.min_fragment_duration_seconds:
            raise ValueError(
                "MOCK_RESULT_DURATION_SECONDS must be at least MIN_FRAGMENT_DURATION_SECONDS"
            )
        if (
            self.mock_result_start_seconds + self.mock_result_duration_seconds
            > self.max_video_duration_seconds
        ):
            raise ValueError("Mock result cannot fit within MAX_VIDEO_DURATION_SECONDS")
        if self.app_env == "production" and "*" in self.cors_origins:
            raise ValueError("Wildcard CORS origin is forbidden in production")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the immutable process configuration."""
    return Settings()
