"""Persistence operations for semantic-search jobs and transcript fragments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

from sqlalchemy import delete, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.search import (
    EMBEDDING_DIMENSIONS,
    SearchIndexStatus,
    VideoFragment,
    VideoSearchIndex,
)


class FragmentWrite(Protocol):
    """Structural write contract implemented by pipeline fragment values."""

    @property
    def position(self) -> int: ...

    @property
    def start_seconds(self) -> float: ...

    @property
    def end_seconds(self) -> float: ...

    @property
    def transcript(self) -> str: ...

    @property
    def embedding(self) -> Sequence[float]: ...

    @property
    def language(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class FragmentInput:
    """Validated-on-write payload produced by the indexing pipeline."""

    position: int
    start_seconds: float
    end_seconds: float
    transcript: str
    embedding: Sequence[float]
    language: str | None = None


@dataclass(frozen=True, slots=True)
class FragmentMatch:
    """A fragment ranked by exact cosine similarity within one video."""

    fragment_id: int
    video_id: UUID
    position: int
    start_seconds: float
    end_seconds: float
    transcript: str
    language: str | None
    score: float


class SearchIndexLeaseLostError(RuntimeError):
    """Raised when a stale worker tries to finish a reclaimed indexing job."""


class SearchIndexRepository:
    """Search-index repository; callers own commit and rollback boundaries."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_pending(self, video_id: UUID) -> VideoSearchIndex:
        """Create the initial pending index row for a newly persisted video."""
        index = VideoSearchIndex(video_id=video_id, status=SearchIndexStatus.PENDING)
        self._session.add(index)
        await self._session.flush()
        return index

    async def upsert_pending(self, video_id: UUID) -> VideoSearchIndex:
        """Idempotently enqueue or re-enqueue a video without removing its old fragments."""
        current_time = datetime.now(UTC)
        statement = (
            pg_insert(VideoSearchIndex)
            .values(
                video_id=video_id,
                status=SearchIndexStatus.PENDING.value,
                attempts=0,
                updated_at=current_time,
            )
            .on_conflict_do_update(
                index_elements=[VideoSearchIndex.video_id],
                set_={
                    "status": SearchIndexStatus.PENDING.value,
                    "attempts": 0,
                    "worker_id": None,
                    "lease_expires_at": None,
                    "error_code": None,
                    "error_message": None,
                    "pipeline_version": None,
                    "asr_model_version": None,
                    "embedding_model_version": None,
                    "updated_at": current_time,
                    "started_at": None,
                    "finished_at": None,
                },
            )
        )
        await self._session.execute(statement)
        await self._session.flush()
        return await self._get_required(video_id, populate_existing=True)

    async def get(self, video_id: UUID) -> VideoSearchIndex | None:
        """Return current indexing state for a video, if indexing was enqueued."""
        return await self._session.get(VideoSearchIndex, video_id)

    async def get_for_search(self, video_id: UUID) -> VideoSearchIndex | None:
        """Lock index state for a consistent readiness check and fragment query."""
        result = await self._session.execute(
            select(VideoSearchIndex)
            .where(VideoSearchIndex.video_id == video_id)
            .with_for_update(read=True)
        )
        return result.scalar_one_or_none()

    async def count_fragments(self, video_id: UUID) -> int:
        """Count stored fragments for status and diagnostics endpoints."""
        count = await self._session.scalar(
            select(func.count())
            .select_from(VideoFragment)
            .where(VideoFragment.video_id == video_id)
        )
        return int(count or 0)

    async def claim_next(
        self,
        *,
        worker_id: str,
        lease_duration: timedelta,
        now: datetime | None = None,
    ) -> VideoSearchIndex | None:
        """Claim one pending or expired job using a non-blocking row lock."""
        normalized_worker_id = worker_id.strip()
        if not normalized_worker_id:
            raise ValueError("worker_id must not be blank")
        if lease_duration.total_seconds() <= 0:
            raise ValueError("lease_duration must be positive")

        current_time = self._normalize_time(now)
        result = await self._session.execute(
            select(VideoSearchIndex)
            .where(
                or_(
                    VideoSearchIndex.status == SearchIndexStatus.PENDING,
                    (
                        (VideoSearchIndex.status == SearchIndexStatus.PROCESSING)
                        & (VideoSearchIndex.lease_expires_at <= current_time)
                    ),
                )
            )
            .order_by(
                VideoSearchIndex.updated_at.asc(),
                VideoSearchIndex.video_id.asc(),
            )
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        index = result.scalar_one_or_none()
        if index is None:
            return None

        index.status = SearchIndexStatus.PROCESSING
        index.attempts += 1
        index.worker_id = normalized_worker_id
        index.lease_expires_at = current_time + lease_duration
        index.error_code = None
        index.error_message = None
        index.started_at = current_time
        index.finished_at = None
        index.updated_at = current_time
        await self._session.flush()
        return index

    async def renew_lease(
        self,
        video_id: UUID,
        *,
        worker_id: str,
        lease_duration: timedelta,
        now: datetime | None = None,
    ) -> bool:
        """Extend a live lease; return false if it has expired or changed owner."""
        if lease_duration.total_seconds() <= 0:
            raise ValueError("lease_duration must be positive")
        current_time = self._normalize_time(now)
        index = await self._get_for_update(video_id)
        if not self._owns_live_lease(index, worker_id=worker_id, now=current_time):
            return False
        assert index is not None
        index.lease_expires_at = current_time + lease_duration
        index.updated_at = current_time
        await self._session.flush()
        return True

    async def requeue(
        self,
        video_id: UUID,
        *,
        worker_id: str,
        now: datetime | None = None,
    ) -> VideoSearchIndex:
        """Return an interrupted live claim to pending without resetting attempts."""
        current_time = self._normalize_time(now)
        index = await self._require_live_claim(video_id, worker_id=worker_id, now=current_time)
        index.status = SearchIndexStatus.PENDING
        index.worker_id = None
        index.lease_expires_at = None
        index.started_at = None
        index.updated_at = current_time
        await self._session.flush()
        return index

    async def replace_fragments(
        self,
        video_id: UUID,
        fragments: Sequence[FragmentWrite],
    ) -> int:
        """Replace all fragments in the caller's transaction, preserving rollback safety."""
        normalized = [self._normalize_fragment(fragment) for fragment in fragments]
        positions = [fragment.position for fragment in normalized]
        if len(positions) != len(set(positions)):
            raise ValueError("fragment positions must be unique within a video")

        await self._session.execute(delete(VideoFragment).where(VideoFragment.video_id == video_id))
        self._session.add_all(
            [
                VideoFragment(
                    video_id=video_id,
                    position=fragment.position,
                    start_seconds=fragment.start_seconds,
                    end_seconds=fragment.end_seconds,
                    transcript=fragment.transcript,
                    language=fragment.language,
                    embedding=list(fragment.embedding),
                )
                for fragment in normalized
            ]
        )
        await self._session.flush()
        return len(normalized)

    async def mark_ready(
        self,
        video_id: UUID,
        *,
        worker_id: str,
        pipeline_version: str,
        asr_model_version: str,
        embedding_model_version: str,
        now: datetime | None = None,
    ) -> VideoSearchIndex:
        """Finish a live claim and expose the newly written fragment set."""
        current_time = self._normalize_time(now)
        index = await self._require_live_claim(video_id, worker_id=worker_id, now=current_time)
        index.status = SearchIndexStatus.READY
        index.worker_id = None
        index.lease_expires_at = None
        index.error_code = None
        index.error_message = None
        index.pipeline_version = self._require_text(pipeline_version, field="pipeline_version")
        index.asr_model_version = self._require_text(asr_model_version, field="asr_model_version")
        index.embedding_model_version = self._require_text(
            embedding_model_version, field="embedding_model_version"
        )
        index.finished_at = current_time
        index.updated_at = current_time
        await self._session.flush()
        return index

    async def mark_failed(
        self,
        video_id: UUID,
        *,
        worker_id: str,
        error_code: str,
        error_message: str,
        now: datetime | None = None,
    ) -> VideoSearchIndex:
        """Finish a live claim as failed while retaining the last fragment set."""
        current_time = self._normalize_time(now)
        index = await self._require_live_claim(video_id, worker_id=worker_id, now=current_time)
        index.status = SearchIndexStatus.FAILED
        index.worker_id = None
        index.lease_expires_at = None
        index.error_code = self._require_text(error_code, field="error_code")
        index.error_message = self._require_text(error_message, field="error_message")
        index.finished_at = current_time
        index.updated_at = current_time
        await self._session.flush()
        return index

    async def complete(
        self,
        video_id: UUID,
        *,
        worker_id: str,
        fragments: Sequence[FragmentWrite],
        pipeline_version: str,
        asr_model_version: str,
        embedding_model_version: str,
        now: datetime | None = None,
    ) -> VideoSearchIndex:
        """Atomically replace fragments and mark a live indexing claim ready."""
        current_time = self._normalize_time(now)
        await self._require_live_claim(video_id, worker_id=worker_id, now=current_time)
        await self.replace_fragments(video_id, fragments)
        return await self.mark_ready(
            video_id,
            worker_id=worker_id,
            pipeline_version=pipeline_version,
            asr_model_version=asr_model_version,
            embedding_model_version=embedding_model_version,
            now=current_time,
        )

    async def search(
        self,
        video_id: UUID,
        query_embedding: Sequence[float],
        limit: int,
    ) -> Sequence[FragmentMatch]:
        """Run exact cosine search over fragments belonging to one video."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        embedding = list(self._normalize_embedding(query_embedding))
        distance = VideoFragment.embedding.cosine_distance(embedding)
        score = (1.0 - distance).label("score")
        rows = (
            await self._session.execute(
                select(VideoFragment, score)
                .join(
                    VideoSearchIndex,
                    VideoSearchIndex.video_id == VideoFragment.video_id,
                )
                .where(VideoFragment.video_id == video_id)
                .where(VideoSearchIndex.status == SearchIndexStatus.READY)
                .order_by(distance.asc(), VideoFragment.position.asc())
                .limit(limit)
            )
        ).all()
        return [
            FragmentMatch(
                fragment_id=fragment.id,
                video_id=fragment.video_id,
                position=fragment.position,
                start_seconds=fragment.start_seconds,
                end_seconds=fragment.end_seconds,
                transcript=fragment.transcript,
                language=fragment.language,
                score=float(similarity),
            )
            for fragment, similarity in rows
        ]

    async def _get_required(
        self, video_id: UUID, *, populate_existing: bool = False
    ) -> VideoSearchIndex:
        statement = select(VideoSearchIndex).where(VideoSearchIndex.video_id == video_id)
        if populate_existing:
            statement = statement.execution_options(populate_existing=True)
        result = await self._session.execute(statement)
        index = result.scalar_one_or_none()
        if index is None:
            raise LookupError(f"search index does not exist for video {video_id}")
        return index

    async def _get_for_update(self, video_id: UUID) -> VideoSearchIndex | None:
        result = await self._session.execute(
            select(VideoSearchIndex).where(VideoSearchIndex.video_id == video_id).with_for_update()
        )
        return result.scalar_one_or_none()

    async def _require_live_claim(
        self,
        video_id: UUID,
        *,
        worker_id: str,
        now: datetime,
    ) -> VideoSearchIndex:
        index = await self._get_for_update(video_id)
        if not self._owns_live_lease(index, worker_id=worker_id, now=now):
            raise SearchIndexLeaseLostError(
                f"worker {worker_id!r} no longer owns a live lease for video {video_id}"
            )
        assert index is not None
        return index

    @staticmethod
    def _owns_live_lease(
        index: VideoSearchIndex | None,
        *,
        worker_id: str,
        now: datetime,
    ) -> bool:
        return bool(
            index is not None
            and index.status == SearchIndexStatus.PROCESSING
            and index.worker_id == worker_id.strip()
            and index.lease_expires_at is not None
            and index.lease_expires_at > now
        )

    @staticmethod
    def _normalize_fragment(fragment: FragmentWrite) -> FragmentInput:
        if fragment.position < 0:
            raise ValueError("fragment position must be nonnegative")
        if not math.isfinite(fragment.start_seconds) or fragment.start_seconds < 0:
            raise ValueError("fragment start_seconds must be finite and nonnegative")
        if (
            not math.isfinite(fragment.end_seconds)
            or fragment.end_seconds <= fragment.start_seconds
        ):
            raise ValueError("fragment end_seconds must be finite and greater than start_seconds")
        transcript = fragment.transcript.strip()
        if not transcript:
            raise ValueError("fragment transcript must not be blank")
        language = fragment.language.strip() if fragment.language is not None else None
        if language == "":
            language = None
        return FragmentInput(
            position=fragment.position,
            start_seconds=float(fragment.start_seconds),
            end_seconds=float(fragment.end_seconds),
            transcript=transcript,
            embedding=SearchIndexRepository._normalize_embedding(fragment.embedding),
            language=language,
        )

    @staticmethod
    def _normalize_embedding(embedding: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(value) for value in embedding)
        if len(values) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"embedding must contain exactly {EMBEDDING_DIMENSIONS} values, got {len(values)}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("embedding values must be finite")
        return values

    @staticmethod
    def _normalize_time(value: datetime | None) -> datetime:
        current_time = value or datetime.now(UTC)
        if current_time.tzinfo is None or current_time.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        return current_time

    @staticmethod
    def _require_text(value: str, *, field: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field} must not be blank")
        return normalized
