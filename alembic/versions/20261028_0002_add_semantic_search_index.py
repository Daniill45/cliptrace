"""Add semantic-search indexing state and transcript fragments.

Revision ID: 20261028_0002
Revises: 20260805_0001
Create Date: 2026-10-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20261028_0002"
down_revision: str | None = "20260805_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "video_search_indexes",
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=30), server_default="pending", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("worker_id", sa.String(length=200), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.String(length=1000), nullable=True),
        sa.Column("pipeline_version", sa.String(length=100), nullable=True),
        sa.Column("asr_model_version", sa.String(length=200), nullable=True),
        sa.Column("embedding_model_version", sa.String(length=200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("attempts >= 0", name="ck_video_search_indexes_attempts_nonnegative"),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')",
            name="ck_video_search_indexes_status",
        ),
        sa.CheckConstraint(
            "(status = 'processing' AND worker_id IS NOT NULL "
            "AND lease_expires_at IS NOT NULL) OR "
            "(status <> 'processing' AND worker_id IS NULL "
            "AND lease_expires_at IS NULL)",
            name="ck_video_search_indexes_lease_fields",
        ),
        sa.CheckConstraint(
            "(status IN ('ready', 'failed') AND finished_at IS NOT NULL) OR "
            "(status IN ('pending', 'processing') AND finished_at IS NULL)",
            name="ck_video_search_indexes_finished_at",
        ),
        sa.CheckConstraint(
            "(status = 'failed' AND error_code IS NOT NULL AND error_message IS NOT NULL) OR "
            "(status <> 'failed' AND error_code IS NULL AND error_message IS NULL)",
            name="ck_video_search_indexes_failed_error",
        ),
        sa.CheckConstraint(
            "(status = 'pending' AND started_at IS NULL) OR "
            "(status <> 'pending' AND started_at IS NOT NULL)",
            name="ck_video_search_indexes_started_at",
        ),
        sa.CheckConstraint(
            "status <> 'ready' OR (pipeline_version IS NOT NULL "
            "AND asr_model_version IS NOT NULL AND embedding_model_version IS NOT NULL)",
            name="ck_video_search_indexes_ready_versions",
        ),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("video_id"),
    )
    op.create_index(
        "ix_video_search_indexes_claim",
        "video_search_indexes",
        ["status", "lease_expires_at", "updated_at"],
    )

    op.create_table(
        "video_fragments",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("transcript", sa.Text(), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=True),
        sa.Column("embedding", Vector(dim=384), nullable=False),
        sa.CheckConstraint("position >= 0", name="ck_video_fragments_position_nonnegative"),
        sa.CheckConstraint("start_seconds >= 0", name="ck_video_fragments_start_nonnegative"),
        sa.CheckConstraint("end_seconds > start_seconds", name="ck_video_fragments_time_order"),
        sa.CheckConstraint(
            "length(btrim(transcript)) > 0", name="ck_video_fragments_transcript_nonempty"
        ),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("video_id", "position", name="uq_video_fragments_video_position"),
    )
    op.create_index(
        "ix_video_fragments_video_time",
        "video_fragments",
        ["video_id", "start_seconds"],
    )
    op.execute(
        "INSERT INTO video_search_indexes (video_id, status, attempts) "
        "SELECT id, 'pending', 0 FROM videos"
    )


def downgrade() -> None:
    op.drop_index("ix_video_fragments_video_time", table_name="video_fragments")
    op.drop_table("video_fragments")
    op.drop_index("ix_video_search_indexes_claim", table_name="video_search_indexes")
    op.drop_table("video_search_indexes")
    # Keep the database-level extension: it may have existed before this application.
