from __future__ import annotations

import math
from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest

from app.services.ml.contracts import Embedding
from app.services.search.base import VideoDescriptor
from app.services.search.hybrid import (
    HybridFragmentSearchEngine,
    RetrievedFragmentCandidates,
    StoredFragmentCandidate,
    reciprocal_rank_fuse,
)


class FakeQueryEmbedder:
    model_id = "fake-e5-v1"
    dimension = 3

    def __init__(self, embedding: Embedding = (1.0, 0.0, 0.0)) -> None:
        self.embedding = embedding
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> Embedding:
        self.queries.append(text)
        return self.embedding


class FakeHybridStore:
    def __init__(self, candidates: RetrievedFragmentCandidates) -> None:
        self.candidates = candidates
        self.calls: list[tuple[UUID, str, tuple[float, ...], int, int]] = []

    async def retrieve(
        self,
        *,
        video_id: UUID,
        query_text: str,
        query_embedding: Sequence[float],
        dense_limit: int,
        lexical_limit: int,
    ) -> RetrievedFragmentCandidates:
        self.calls.append(
            (
                video_id,
                query_text,
                tuple(query_embedding),
                dense_limit,
                lexical_limit,
            )
        )
        return self.candidates


class FakeReranker:
    model_id = "fake-reranker-v1"

    def __init__(
        self,
        *,
        scores_by_document: dict[str, float] | None = None,
        fixed_scores: Sequence[float] | None = None,
    ) -> None:
        self._scores_by_document = scores_by_document or {}
        self._fixed_scores = tuple(fixed_scores) if fixed_scores is not None else None
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        normalized_documents = tuple(documents)
        self.calls.append((query, normalized_documents))
        if self._fixed_scores is not None:
            return self._fixed_scores
        return tuple(self._scores_by_document[document] for document in normalized_documents)


def _candidate(
    fragment_id: int,
    *,
    position: int | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    transcript: str | None = None,
    score: float = 0.0,
) -> StoredFragmentCandidate:
    start = float(fragment_id * 10) if start_seconds is None else start_seconds
    end = start + 5.0 if end_seconds is None else end_seconds
    return StoredFragmentCandidate(
        fragment_id=fragment_id,
        position=fragment_id - 1 if position is None else position,
        start_seconds=start,
        end_seconds=end,
        transcript=transcript or f"fragment {fragment_id}",
        score=score,
    )


def _engine(
    *,
    store: FakeHybridStore,
    embedder: FakeQueryEmbedder | None = None,
    reranker: FakeReranker | None = None,
    **kwargs: float | int,
) -> HybridFragmentSearchEngine:
    return HybridFragmentSearchEngine(
        store=store,
        query_embedder=embedder or FakeQueryEmbedder(),
        reranker=reranker or FakeReranker(),
        **kwargs,
    )


def test_rrf_fuses_dense_and_lexical_ranks_without_mixing_raw_score_scales() -> None:
    dense = (
        _candidate(1, score=-1e100),
        _candidate(2, score=1e100),
    )
    lexical = (
        _candidate(2, score=-1e90),
        _candidate(3, score=1e90),
    )

    fused = reciprocal_rank_fuse(dense, lexical, rrf_k=60.0)
    rescaled = reciprocal_rank_fuse(
        tuple(
            _candidate(
                item.fragment_id,
                position=item.position,
                start_seconds=item.start_seconds,
                end_seconds=item.end_seconds,
                transcript=item.transcript,
                score=-item.score,
            )
            for item in dense
        ),
        tuple(
            _candidate(
                item.fragment_id,
                position=item.position,
                start_seconds=item.start_seconds,
                end_seconds=item.end_seconds,
                transcript=item.transcript,
                score=-item.score,
            )
            for item in lexical
        ),
        rrf_k=60.0,
    )

    assert [item.fragment_id for item in fused] == [2, 1, 3]
    assert [item.rrf_score for item in fused] == pytest.approx(
        [1.0 / 62.0 + 1.0 / 61.0, 1.0 / 61.0, 1.0 / 62.0]
    )
    assert [item.fragment_id for item in rescaled] == [2, 1, 3]
    assert [item.rrf_score for item in rescaled] == pytest.approx(
        [item.rrf_score for item in fused]
    )


def test_rrf_rejects_duplicate_or_inconsistent_fragment_identities() -> None:
    duplicate = _candidate(1)
    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank_fuse((duplicate, duplicate), (), rrf_k=60.0)

    with pytest.raises(ValueError, match="inconsistent"):
        reciprocal_rank_fuse(
            (_candidate(1, transcript="dense text"),),
            (_candidate(1, transcript="different lexical text"),),
            rrf_k=60.0,
        )


@pytest.mark.asyncio
async def test_hybrid_engine_uses_reranker_order_after_rrf() -> None:
    video = VideoDescriptor(id=uuid4(), duration_seconds=60.0, storage_key="video.mp4")
    candidates = (
        _candidate(1, transcript="first-stage first"),
        _candidate(2, transcript="reranker winner"),
        _candidate(3, transcript="reranker second"),
    )
    store = FakeHybridStore(RetrievedFragmentCandidates(dense=candidates, lexical=()))
    embedder = FakeQueryEmbedder()
    reranker = FakeReranker(
        scores_by_document={
            "first-stage first": -2.0,
            "reranker winner": 3.0,
            "reranker second": 1.0,
        }
    )
    engine = _engine(store=store, embedder=embedder, reranker=reranker, padding_seconds=0.0)

    hits = await engine.search(video=video, query="  find   this ", limit=3)

    assert embedder.queries == ["find this"]
    assert store.calls == [(video.id, "find this", (1.0, 0.0, 0.0), 40, 40)]
    assert reranker.calls == [
        (
            "find this",
            ("first-stage first", "reranker winner", "reranker second"),
        )
    ]
    assert engine.mode == "hybrid-transcript"
    assert engine.version == "hybrid/fake-e5-v1+fake-reranker-v1"
    assert [hit.preview_text for hit in hits] == [
        "reranker winner",
        "reranker second",
        "first-stage first",
    ]
    assert [hit.score for hit in hits] == pytest.approx(
        [1.0 / (1.0 + math.exp(-3.0)), 1.0 / (1.0 + math.exp(-1.0)), 1.0 / (1.0 + math.exp(2.0))]
    )


@pytest.mark.asyncio
async def test_hybrid_engine_applies_nms_before_padding_and_backfills_limit() -> None:
    video = VideoDescriptor(id=uuid4(), duration_seconds=60.0, storage_key="video.mp4")
    candidates = (
        _candidate(1, start_seconds=10.0, end_seconds=20.0, transcript="winner"),
        _candidate(2, start_seconds=11.0, end_seconds=19.0, transcript="overlap"),
        _candidate(3, start_seconds=20.5, end_seconds=25.0, transcript="backfill"),
        _candidate(4, start_seconds=30.0, end_seconds=35.0, transcript="unused"),
    )
    store = FakeHybridStore(RetrievedFragmentCandidates(dense=candidates, lexical=()))
    reranker = FakeReranker(
        scores_by_document={
            "winner": 4.0,
            "overlap": 3.0,
            "backfill": 2.0,
            "unused": 1.0,
        }
    )
    engine = _engine(
        store=store,
        reranker=reranker,
        temporal_iou_threshold=0.5,
        padding_seconds=2.0,
    )

    hits = await engine.search(video=video, query="query", limit=2)

    assert [hit.preview_text for hit in hits] == ["winner", "backfill"]
    assert [(hit.start_seconds, hit.end_seconds) for hit in hits] == [
        (8.0, 22.0),
        (18.5, 27.0),
    ]


@pytest.mark.asyncio
async def test_hybrid_engine_returns_empty_without_calling_reranker() -> None:
    video = VideoDescriptor(id=uuid4(), duration_seconds=60.0, storage_key="video.mp4")
    store = FakeHybridStore(RetrievedFragmentCandidates(dense=(), lexical=()))
    reranker = FakeReranker()
    engine = _engine(store=store, reranker=reranker)

    hits = await engine.search(video=video, query="nothing found", limit=5)

    assert hits == ()
    assert reranker.calls == []


@pytest.mark.asyncio
async def test_hybrid_engine_validates_query_and_dependency_outputs() -> None:
    video = VideoDescriptor(id=uuid4(), duration_seconds=60.0, storage_key="video.mp4")
    empty_store = FakeHybridStore(RetrievedFragmentCandidates(dense=(), lexical=()))
    embedder = FakeQueryEmbedder()
    engine = _engine(store=empty_store, embedder=embedder)

    with pytest.raises(ValueError, match="blank"):
        await engine.search(video=video, query="   ", limit=1)
    assert embedder.queries == []
    assert empty_store.calls == []

    wrong_dimension = _engine(
        store=empty_store,
        embedder=FakeQueryEmbedder((1.0, 0.0)),
    )
    with pytest.raises(ValueError, match="dimension"):
        await wrong_dimension.search(video=video, query="query", limit=1)

    candidate = _candidate(1)
    populated_store = FakeHybridStore(RetrievedFragmentCandidates(dense=(candidate,), lexical=()))
    bad_count = _engine(
        store=populated_store,
        reranker=FakeReranker(fixed_scores=()),
    )
    with pytest.raises(ValueError, match="count"):
        await bad_count.search(video=video, query="query", limit=1)

    non_finite = _engine(
        store=populated_store,
        reranker=FakeReranker(fixed_scores=(float("nan"),)),
    )
    with pytest.raises(ValueError, match="non-finite"):
        await non_finite.search(video=video, query="query", limit=1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dense_candidate_limit": 0}, "limits"),
        ({"rrf_k": float("nan")}, "RRF"),
        ({"temporal_iou_threshold": 0.0}, "IoU"),
        ({"padding_seconds": -1.0}, "padding"),
    ],
)
def test_hybrid_engine_rejects_invalid_configuration(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    store = FakeHybridStore(RetrievedFragmentCandidates(dense=(), lexical=()))

    with pytest.raises(ValueError, match=message):
        _engine(store=store, **kwargs)
