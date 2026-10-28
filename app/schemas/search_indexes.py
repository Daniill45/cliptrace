"""Public schemas for asynchronous semantic-search indexing."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class SearchIndexStatusResponse(BaseModel):
    """Current state of the independently retryable search index."""

    video_id: UUID
    status: Literal["pending", "processing", "ready", "failed"]
    attempts: int = Field(ge=0)
    fragment_count: int = Field(ge=0)
    pipeline_version: str | None
    asr_model_version: str | None
    embedding_model_version: str | None
    error_code: str | None
    error_message: str | None
    created_at: datetime | None
    updated_at: datetime | None
    started_at: datetime | None
    indexed_at: datetime | None
