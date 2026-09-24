"""Search request and response schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def trim_query(cls, value: str) -> str:
        normalized = value.strip()
        if not 1 <= len(normalized) <= 500:
            raise ValueError("query must contain 1 to 500 characters after trim")
        return normalized


class SearchResult(BaseModel):
    rank: int
    start_seconds: float
    end_seconds: float
    start_timecode: str
    end_timecode: str
    score: float | None
    preview_text: str | None
    content_url: str


class SearchResponse(BaseModel):
    search_id: UUID
    video_id: UUID
    query: str
    mode: str
    engine_version: str
    results: list[SearchResult]
    created_at: datetime
