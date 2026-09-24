"""Create videos metadata table.

Revision ID: 20260805_0001
Revises:
Create Date: 2026-08-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260805_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "videos",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=True),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("storage_key", sa.String(length=500), nullable=False),
        sa.Column("content_type", sa.String(length=100), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("video_codec", sa.String(length=50), nullable=False),
        sa.Column("video_profile", sa.String(length=50), nullable=True),
        sa.Column("pixel_format", sa.String(length=30), nullable=False),
        sa.Column("audio_codec", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.String(length=500), nullable=True),
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
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("size_bytes > 0", name="ck_videos_size_positive"),
        sa.CheckConstraint("duration_seconds > 0", name="ck_videos_duration_positive"),
        sa.CheckConstraint("width > 0", name="ck_videos_width_positive"),
        sa.CheckConstraint("height > 0", name="ck_videos_height_positive"),
        sa.CheckConstraint("fps IS NULL OR fps > 0", name="ck_videos_fps_positive"),
        sa.CheckConstraint("status IN ('ready', 'failed')", name="ck_videos_status"),
        sa.CheckConstraint(
            "(status = 'ready' AND duration_seconds IS NOT NULL AND ready_at IS NOT NULL "
            "AND error_code IS NULL AND error_message IS NULL) OR status = 'failed'",
            name="ck_videos_status_fields",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_key"),
    )
    op.create_index("ix_videos_created_at_desc", "videos", [sa.text("created_at DESC")])
    op.create_index("ix_videos_status", "videos", ["status"])


def downgrade() -> None:
    op.drop_index("ix_videos_status", table_name="videos")
    op.drop_index("ix_videos_created_at_desc", table_name="videos")
    op.drop_table("videos")
