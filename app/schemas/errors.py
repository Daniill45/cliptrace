"""Stable API error schemas."""

from pydantic import BaseModel, Field


class ErrorData(BaseModel):
    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorData
