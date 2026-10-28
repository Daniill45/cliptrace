"""Persistence models for asynchronous semantic-search indexing."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

EMBEDDING_DIMENSIONS = 384


class SearchIndexStatus(StrEnum):
    """Lifecycle of a video's independently retryable search index."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class VideoSearchIndex(Base):
    """One-to-one job and state record for indexing a video."""

    __tablename__ = "video_search_indexes"
    __table_args__ = (
        CheckConstraint("attempts >= 0", name="ck_video_search_indexes_attempts_nonnegative"),
        CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')",
            name="ck_video_search_indexes_status",
        ),
        CheckConstraint(
            "(status = 'processing' AND worker_id IS NOT NULL "
            "AND lease_expires_at IS NOT NULL) OR "
            "(status <> 'processing' AND worker_id IS NULL "
            "AND lease_expires_at IS NULL)",
            name="ck_video_search_indexes_lease_fields",
        ),
        CheckConstraint(
            "(status IN ('ready', 'failed') AND finished_at IS NOT NULL) OR "
            "(status IN ('pending', 'processing') AND finished_at IS NULL)",
            name="ck_video_search_indexes_finished_at",
        ),
        CheckConstraint(
            "(status = 'failed' AND error_code IS NOT NULL AND error_message IS NOT NULL) OR "
            "(status <> 'failed' AND error_code IS NULL AND error_message IS NULL)",
            name="ck_video_search_indexes_failed_error",
        ),
        CheckConstraint(
            "(status = 'pending' AND started_at IS NULL) OR "
            "(status <> 'pending' AND started_at IS NOT NULL)",
            name="ck_video_search_indexes_started_at",
        ),
        CheckConstraint(
            "status <> 'ready' OR (pipeline_version IS NOT NULL "
            "AND asr_model_version IS NOT NULL AND embedding_model_version IS NOT NULL)",
            name="ck_video_search_indexes_ready_versions",
        ),
        Index(
            "ix_video_search_indexes_claim",
            "status",
            "lease_expires_at",
            "updated_at",
        ),
    )

    video_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="CASCADE"),
        primary_key=True,
    )
    status: Mapped[SearchIndexStatus] = mapped_column(
        Enum(
            SearchIndexStatus,
            native_enum=False,
            create_constraint=False,
            values_callable=lambda enum_type: [item.value for item in enum_type],
            length=30,
        ),
        nullable=False,
        default=SearchIndexStatus.PENDING,
        server_default=SearchIndexStatus.PENDING.value,
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    worker_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    asr_model_version: Mapped[str | None] = mapped_column(String(200), nullable=True)
    embedding_model_version: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=lambda: datetime.now().astimezone(),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class VideoFragment(Base):
    """A timestamped transcript fragment and its dense retrieval vector."""

    __tablename__ = "video_fragments"
    __table_args__ = (
        CheckConstraint("position >= 0", name="ck_video_fragments_position_nonnegative"),
        CheckConstraint("start_seconds >= 0", name="ck_video_fragments_start_nonnegative"),
        CheckConstraint("end_seconds > start_seconds", name="ck_video_fragments_time_order"),
        CheckConstraint(
            "length(btrim(transcript)) > 0", name="ck_video_fragments_transcript_nonempty"
        ),
        UniqueConstraint("video_id", "position", name="uq_video_fragments_video_position"),
        Index("ix_video_fragments_video_time", "video_id", "start_seconds"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    video_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    transcript: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
