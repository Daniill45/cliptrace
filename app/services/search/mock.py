"""Explicit deterministic mock fragment search engine."""

import math

from app.services.search.base import SearchHit, VideoDescriptor


class MockFixedTimestampSearchEngine:
    """Return one configured interval without inspecting video contents."""

    mode = "mock"
    version = "mock-fixed-v3"

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
        if not math.isfinite(video.duration_seconds) or video.duration_seconds <= 0:
            raise ValueError("Video duration must be positive and finite")
        end = min(self._start + self._duration, video.duration_seconds)
        duration = min(self._duration, video.duration_seconds)
        start = max(0.0, end - duration)
        return [
            SearchHit(
                start_seconds=start,
                end_seconds=end,
                score=None,
                preview_text=None,
            )
        ]
