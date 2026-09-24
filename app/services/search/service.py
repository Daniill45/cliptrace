"""Application-level search orchestration and result invariant checks."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.core.errors import InternalApplicationError, VideoNotFoundError, VideoNotReadyError
from app.db.models.video import VideoStatus
from app.repositories.videos import VideoRepository
from app.schemas.search import SearchResponse, SearchResult
from app.services.media.storage import LocalStorage
from app.services.search.base import FragmentSearchEngine, VideoDescriptor
from app.utils.timecodes import format_timecode


class SearchService:
    """Search one ready, physically present video through an injected engine."""

    def __init__(
        self,
        *,
        repository: VideoRepository,
        storage: LocalStorage,
        engine: FragmentSearchEngine,
        api_prefix: str,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._engine = engine
        self._api_prefix = api_prefix

    async def search(
        self,
        *,
        video_id: UUID,
        query: str,
        limit: int,
    ) -> SearchResponse:
        """Validate resource readiness and map engine hits to the public result DTO."""
        video = await self._repository.get(video_id)
        if video is None:
            raise VideoNotFoundError
        if video.status is not VideoStatus.READY:
            raise VideoNotReadyError
        await self._storage.verify(video.storage_key, video.size_bytes)
        descriptor = VideoDescriptor(
            id=video.id,
            duration_seconds=video.duration_seconds,
            storage_key=video.storage_key,
        )
        try:
            hits = list(await self._engine.search(video=descriptor, query=query, limit=limit))
        except ValueError as exc:
            raise InternalApplicationError from exc
        results: list[SearchResult] = []
        for rank, hit in enumerate(hits, start=1):
            if (
                not math.isfinite(hit.start_seconds)
                or not math.isfinite(hit.end_seconds)
                or hit.start_seconds < 0
                or hit.start_seconds >= hit.end_seconds
                or hit.end_seconds > video.duration_seconds
            ):
                raise InternalApplicationError
            results.append(
                SearchResult(
                    rank=rank,
                    start_seconds=hit.start_seconds,
                    end_seconds=hit.end_seconds,
                    start_timecode=format_timecode(hit.start_seconds),
                    end_timecode=format_timecode(hit.end_seconds),
                    score=hit.score,
                    preview_text=hit.preview_text,
                    content_url=f"{self._api_prefix}/videos/{video.id}/content",
                )
            )
        return SearchResponse(
            search_id=uuid4(),
            video_id=video.id,
            query=query,
            mode=self._engine.mode,
            engine_version=self._engine.version,
            results=results,
            created_at=datetime.now(UTC),
        )
