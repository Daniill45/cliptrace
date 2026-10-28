from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest

from app.services.ml.contracts import Embedding
from app.services.search.base import VideoDescriptor
from app.services.search.semantic import SemanticFragmentSearchEngine, StoredFragmentMatch


class FakeQueryEmbedder:
    model_id = "fake-e5-v1"
    dimension = 3

    def __init__(self, embedding: Embedding = (1.0, 0.0, 0.0)) -> None:
        self.embedding = embedding
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> Embedding:
        self.queries.append(text)
        return self.embedding


class FakeFragmentStore:
    def __init__(self, matches: Sequence[StoredFragmentMatch]) -> None:
        self.matches = matches
        self.calls: list[tuple[UUID, tuple[float, ...], int]] = []

    async def search(
        self,
        *,
        video_id: UUID,
        query_embedding: Sequence[float],
        limit: int,
    ) -> Sequence[StoredFragmentMatch]:
        self.calls.append((video_id, tuple(query_embedding), limit))
        return self.matches


@pytest.mark.asyncio
async def test_semantic_engine_preserves_rank_and_pads_with_video_clamping() -> None:
    video = VideoDescriptor(id=uuid4(), duration_seconds=60.0, storage_key="video.mp4")
    embedder = FakeQueryEmbedder()
    store = FakeFragmentStore(
        (
            StoredFragmentMatch(50.0, 60.0, " best   result ", 0.91),
            StoredFragmentMatch(-2.0, 3.0, "second", 0.82),
            StoredFragmentMatch(20.0, 25.0, "third", 0.73),
        )
    )
    engine = SemanticFragmentSearchEngine(
        store=store,
        query_embedder=embedder,
        padding_seconds=2.0,
    )

    hits = await engine.search(video=video, query="  find   this  ", limit=2)

    assert embedder.queries == ["find this"]
    assert store.calls == [(video.id, (1.0, 0.0, 0.0), 2)]
    assert engine.mode == "semantic-transcript"
    assert engine.version == "semantic/fake-e5-v1"
    assert [(hit.start_seconds, hit.end_seconds, hit.preview_text, hit.score) for hit in hits] == [
        (48.0, 60.0, "best result", 0.91),
        (0.0, 5.0, "second", 0.82),
    ]


@pytest.mark.asyncio
async def test_semantic_engine_rejects_bad_query_embedding_or_store_interval() -> None:
    video = VideoDescriptor(id=uuid4(), duration_seconds=60.0, storage_key="video.mp4")
    store = FakeFragmentStore((StoredFragmentMatch(5.0, 5.0, "bad", 0.5),))
    wrong_dimension = SemanticFragmentSearchEngine(
        store=store,
        query_embedder=FakeQueryEmbedder((1.0, 0.0)),
    )
    with pytest.raises(ValueError, match="dimension"):
        await wrong_dimension.search(video=video, query="query", limit=1)

    invalid_interval = SemanticFragmentSearchEngine(
        store=store,
        query_embedder=FakeQueryEmbedder(),
    )
    with pytest.raises(ValueError, match="interval"):
        await invalid_interval.search(video=video, query="query", limit=1)


@pytest.mark.asyncio
async def test_semantic_engine_rejects_blank_query_before_calling_dependencies() -> None:
    embedder = FakeQueryEmbedder()
    store = FakeFragmentStore(())
    engine = SemanticFragmentSearchEngine(store=store, query_embedder=embedder)

    with pytest.raises(ValueError, match="blank"):
        await engine.search(
            video=VideoDescriptor(id=uuid4(), duration_seconds=10.0, storage_key="video.mp4"),
            query="   ",
            limit=1,
        )

    assert embedder.queries == []
    assert store.calls == []
