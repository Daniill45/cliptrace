"""Video API schemas and ORM-to-public mapping."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.db.models.video import Video


class VideoMedia(BaseModel):
    duration_seconds: float
    width: int
    height: int
    fps: float | None
    video_codec: str
    video_profile: str | None
    pixel_format: str
    audio_codec: str


class VideoError(BaseModel):
    code: str
    message: str


class VideoLinks(BaseModel):
    self: str
    content: str
    search: str


class VideoResponse(BaseModel):
    id: UUID
    title: str | None
    original_filename: str
    status: str
    content_type: str
    size_bytes: int
    media: VideoMedia
    error: VideoError | None
    links: VideoLinks
    created_at: datetime
    updated_at: datetime
    ready_at: datetime | None


class VideoListResponse(BaseModel):
    items: list[VideoResponse]
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    total: int = Field(ge=0)


def video_to_response(video: Video, *, api_prefix: str) -> VideoResponse:
    """Map a persistence object to the stable public schema."""
    base = f"{api_prefix}/videos/{video.id}"
    error = None
    if video.error_code and video.error_message:
        error = VideoError(code=video.error_code, message=video.error_message)
    return VideoResponse(
        id=video.id,
        title=video.title,
        original_filename=video.original_filename,
        status=video.status.value,
        content_type=video.content_type,
        size_bytes=video.size_bytes,
        media=VideoMedia(
            duration_seconds=video.duration_seconds,
            width=video.width,
            height=video.height,
            fps=video.fps,
            video_codec=video.video_codec,
            video_profile=video.video_profile,
            pixel_format=video.pixel_format,
            audio_codec=video.audio_codec,
        ),
        error=error,
        links=VideoLinks(self=base, content=f"{base}/content", search=f"{base}/search"),
        created_at=video.created_at,
        updated_at=video.updated_at,
        ready_at=video.ready_at,
    )
