"""Application services and adapters for semantic-search index state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import (
    SearchIndexFailedError,
    SearchIndexNotReadyError,
    VideoNotFoundError,
    VideoNotReadyError,
)
from app.db.models.search import SearchIndexStatus
from app.db.models.video import VideoStatus
from app.repositories.search import FragmentMatch, SearchIndexRepository
from app.repositories.videos import VideoRepository


@dataclass(frozen=True, slots=True)
class SearchIndexSnapshot:
    """Persistence-neutral state returned to the API layer."""

    video_id: UUID
    status: SearchIndexStatus
    attempts: int
    fragment_count: int
    pipeline_version: str | None
    asr_model_version: str | None
    embedding_model_version: str | None
    error_code: str | None
    error_message: str | None
    created_at: datetime | None
    updated_at: datetime | None
    started_at: datetime | None
    indexed_at: datetime | None


class SearchIndexService:
    """Read and re-enqueue a video's independent ML index."""

    def __init__(
        self,
        *,
        session: AsyncSession,
        videos: VideoRepository,
        indexes: SearchIndexRepository,
        enabled: bool,
    ) -> None:
        self._session = session
        self._videos = videos
        self._indexes = indexes
        self._enabled = enabled

    async def get(self, video_id: UUID) -> SearchIndexSnapshot:
        video = await self._videos.get(video_id)
        if video is None:
            raise VideoNotFoundError
        if not self._enabled:
            return SearchIndexSnapshot(
                video_id=video.id,
                status=SearchIndexStatus.READY,
                attempts=0,
                fragment_count=0,
                pipeline_version="mock",
                asr_model_version=None,
                embedding_model_version=None,
                error_code=None,
                error_message=None,
                created_at=video.created_at,
                updated_at=video.updated_at,
                started_at=None,
                indexed_at=video.ready_at,
            )
        index = await self._indexes.get(video_id)
        if index is None:
            return SearchIndexSnapshot(
                video_id=video.id,
                status=SearchIndexStatus.PENDING,
                attempts=0,
                fragment_count=0,
                pipeline_version=None,
                asr_model_version=None,
                embedding_model_version=None,
                error_code=None,
                error_message=None,
                created_at=None,
                updated_at=None,
                started_at=None,
                indexed_at=None,
            )
        return SearchIndexSnapshot(
            video_id=index.video_id,
            status=index.status,
            attempts=index.attempts,
            fragment_count=await self._indexes.count_fragments(video_id),
            pipeline_version=index.pipeline_version,
            asr_model_version=index.asr_model_version,
            embedding_model_version=index.embedding_model_version,
            error_code=index.error_code,
            error_message=index.error_message,
            created_at=index.created_at,
            updated_at=index.updated_at,
            started_at=index.started_at,
            indexed_at=index.finished_at if index.status is SearchIndexStatus.READY else None,
        )

    async def reindex(self, video_id: UUID) -> SearchIndexSnapshot:
        if not self._enabled:
            return await self.get(video_id)
        async with self._session.begin():
            video = await self._videos.get_for_update(video_id)
            if video is None:
                raise VideoNotFoundError
            if video.status is not VideoStatus.READY:
                raise VideoNotReadyError
            await self._indexes.upsert_pending(video_id)
        return await self.get(video_id)


class ReadySearchIndexStore:
    """Expose only ready fragment sets to the online semantic engine."""

    def __init__(self, repository: SearchIndexRepository) -> None:
        self._repository = repository

    async def search(
        self,
        *,
        video_id: UUID,
        query_embedding: Sequence[float],
        limit: int,
    ) -> Sequence[FragmentMatch]:
        index = await self._repository.get_for_search(video_id)
        if index is None:
            raise SearchIndexNotReadyError(SearchIndexStatus.PENDING.value)
        if index.status is SearchIndexStatus.FAILED:
            raise SearchIndexFailedError
        if index.status is not SearchIndexStatus.READY:
            raise SearchIndexNotReadyError(index.status.value)
        return await self._repository.search(video_id, query_embedding, limit)
