"""Add lexical transcript retrieval and enqueue semantic reindexing.

Revision ID: 20261112_0003
Revises: 20261028_0002
Create Date: 2026-11-12
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20261112_0003"
down_revision: str | None = "20261028_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_video_fragments_transcript_fts "
        "ON video_fragments USING GIN "
        "(to_tsvector('simple'::regconfig, transcript))"
    )
    _enqueue_all_indexes()


def downgrade() -> None:
    op.execute("DROP INDEX ix_video_fragments_transcript_fts")
    # Fixed-window fragments cannot be reconstructed by a schema rollback. Enqueue
    # every video so the worker matching the downgraded code rebuilds them safely.
    _enqueue_all_indexes()


def _enqueue_all_indexes() -> None:
    op.execute(
        "UPDATE video_search_indexes SET "
        "status = 'pending', attempts = 0, worker_id = NULL, "
        "lease_expires_at = NULL, error_code = NULL, error_message = NULL, "
        "pipeline_version = NULL, asr_model_version = NULL, "
        "embedding_model_version = NULL, started_at = NULL, finished_at = NULL, "
        "updated_at = CURRENT_TIMESTAMP"
    )
