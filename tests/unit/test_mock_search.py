from uuid import uuid4

import pytest

from app.services.search.base import VideoDescriptor
from app.services.search.mock import MockFixedTimestampSearchEngine


@pytest.fixture
def engine() -> MockFixedTimestampSearchEngine:
    return MockFixedTimestampSearchEngine(
        start_seconds=5.0,
        duration_seconds=5.0,
        min_fragment_duration_seconds=5.0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [10.0, 12.0, 100.0])
async def test_mock_search_interval(
    engine: MockFixedTimestampSearchEngine, duration: float
) -> None:
    descriptor = VideoDescriptor(id=uuid4(), duration_seconds=duration, storage_key="internal")
    first = await engine.search(video=descriptor, query="one", limit=1)
    second = await engine.search(video=descriptor, query="completely different", limit=20)
    assert first == second
    assert len(first) == 1
    assert first[0].start_seconds == 5.0
    assert first[0].end_seconds == 10.0
    assert first[0].score is None


def test_mock_search_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        MockFixedTimestampSearchEngine(
            start_seconds=5,
            duration_seconds=0.5,
            min_fragment_duration_seconds=1,
        )


@pytest.mark.asyncio
async def test_mock_search_rejects_video_that_cannot_fit_fragment(
    engine: MockFixedTimestampSearchEngine,
) -> None:
    descriptor = VideoDescriptor(id=uuid4(), duration_seconds=9.99, storage_key="internal")
    with pytest.raises(ValueError):
        await engine.search(video=descriptor, query="q", limit=5)
