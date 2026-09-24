from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.db.models.video import VideoStatus
from app.services.search.base import SearchHit, VideoDescriptor
from app.services.search.service import SearchService


class FixedRepository:
    def __init__(self, video: SimpleNamespace) -> None:
        self.video = video

    async def get(self, video_id: UUID) -> SimpleNamespace | None:
        return self.video if video_id == self.video.id else None


class PresentStorage:
    async def verify(self, storage_key: str, expected_size: int) -> int:
        assert storage_key == "video.mp4"
        return expected_size


class RelevanceOrderedEngine:
    mode = "ml"
    version = "test-ranked-v1"

    async def search(
        self,
        *,
        video: VideoDescriptor,
        query: str,
        limit: int,
    ) -> list[SearchHit]:
        del video, query, limit
        return [
            SearchHit(50.0, 55.0, 0.9, "best"),
            SearchHit(10.0, 15.0, 0.8, "second"),
        ]


@pytest.mark.asyncio
async def test_search_preserves_engine_order_when_assigning_public_ranks() -> None:
    video_id = uuid4()
    video = SimpleNamespace(
        id=video_id,
        status=VideoStatus.READY,
        storage_key="video.mp4",
        size_bytes=1,
        duration_seconds=100.0,
    )
    service = SearchService(
        repository=FixedRepository(video),  # type: ignore[arg-type]
        storage=PresentStorage(),  # type: ignore[arg-type]
        engine=RelevanceOrderedEngine(),
        api_prefix="/api/v1",
    )

    response = await service.search(video_id=video_id, query="moment", limit=2)

    assert [
        (result.rank, result.preview_text, result.start_seconds, result.score)
        for result in response.results
    ] == [
        (1, "best", 50.0, 0.9),
        (2, "second", 10.0, 0.8),
    ]
