"""Video metadata persistence model."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class VideoStatus(StrEnum):
    READY = "ready"
    FAILED = "failed"


class Video(Base):
    """Persisted media-asset metadata; physical bytes live in storage."""

    __tablename__ = "videos"
    __table_args__ = (
        CheckConstraint("size_bytes > 0", name="ck_videos_size_positive"),
        CheckConstraint("duration_seconds > 0", name="ck_videos_duration_positive"),
        CheckConstraint("width > 0", name="ck_videos_width_positive"),
        CheckConstraint("height > 0", name="ck_videos_height_positive"),
        CheckConstraint("fps IS NULL OR fps > 0", name="ck_videos_fps_positive"),
        CheckConstraint("status IN ('ready', 'failed')", name="ck_videos_status"),
        CheckConstraint(
            "(status = 'ready' AND duration_seconds IS NOT NULL AND ready_at IS NOT NULL "
            "AND error_code IS NULL AND error_message IS NULL) OR status = 'failed'",
            name="ck_videos_status_fields",
        ),
        Index("ix_videos_created_at_desc", text("created_at DESC")),
        Index("ix_videos_status", "status"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_codec: Mapped[str] = mapped_column(String(50), nullable=False)
    video_profile: Mapped[str | None] = mapped_column(String(50), nullable=True)
    pixel_format: Mapped[str] = mapped_column(String(30), nullable=False)
    audio_codec: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[VideoStatus] = mapped_column(
        Enum(
            VideoStatus,
            native_enum=False,
            create_constraint=False,
            values_callable=lambda enum_type: [item.value for item in enum_type],
            length=30,
        ),
        nullable=False,
    )
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=lambda: datetime.now().astimezone(),
    )
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
