"""Explicit deterministic mock fragment search engine."""

from app.services.search.base import SearchHit, VideoDescriptor


class MockFixedTimestampSearchEngine:
    """Return one configured interval without inspecting video contents."""

    mode = "mock"
    version = "mock-fixed-v2"

    def __init__(
        self,
        *,
        start_seconds: float,
        duration_seconds: float,
        min_fragment_duration_seconds: float,
    ) -> None:
        if start_seconds < 0 or duration_seconds <= 0 or min_fragment_duration_seconds <= 0:
            raise ValueError("Mock search timing must be positive and start must be non-negative")
        if duration_seconds < min_fragment_duration_seconds:
            raise ValueError("Mock duration is shorter than the minimum fragment")
        self._start = start_seconds
        self._duration = duration_seconds
        self._minimum = min_fragment_duration_seconds

    async def search(
        self,
        *,
        video: VideoDescriptor,
        query: str,
        limit: int,
    ) -> list[SearchHit]:
        del query, limit
        end = self._start + self._duration
        if video.duration_seconds < end:
            raise ValueError("Video cannot contain the configured mock fragment")
        return [
            SearchHit(
                start_seconds=self._start,
                end_seconds=end,
                score=None,
                preview_text=None,
            )
        ]
