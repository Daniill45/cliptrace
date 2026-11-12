"""Hybrid transcript retrieval, cross-encoder reranking, and temporal deduplication."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.services.ml.contracts import CandidateReranker, QueryEmbedder
from app.services.search.base import SearchHit, VideoDescriptor


class RankedFragment(Protocol):
    """Store-neutral fragment returned by either first-stage retriever."""

    @property
    def fragment_id(self) -> int: ...

    @property
    def position(self) -> int: ...

    @property
    def start_seconds(self) -> float: ...

    @property
    def end_seconds(self) -> float: ...

    @property
    def transcript(self) -> str: ...

    @property
    def score(self) -> float: ...


@dataclass(frozen=True, slots=True)
class StoredFragmentCandidate:
    """Convenient candidate value for adapters and tests."""

    fragment_id: int
    position: int
    start_seconds: float
    end_seconds: float
    transcript: str
    score: float


@dataclass(frozen=True, slots=True)
class RetrievedFragmentCandidates:
    """Independent relevance-ordered outputs of dense and lexical retrieval."""

    dense: Sequence[RankedFragment]
    lexical: Sequence[RankedFragment]


class HybridFragmentStore(Protocol):
    """Retrieve dense and lexical candidate lists from one ready video index."""

    async def retrieve(
        self,
        *,
        video_id: UUID,
        query_text: str,
        query_embedding: Sequence[float],
        dense_limit: int,
        lexical_limit: int,
    ) -> RetrievedFragmentCandidates: ...


@dataclass(frozen=True, slots=True)
class FusedFragmentCandidate:
    """One unique fragment with a rank-only reciprocal-rank-fusion score."""

    fragment_id: int
    position: int
    start_seconds: float
    end_seconds: float
    transcript: str
    rrf_score: float
    best_rank: int


@dataclass(frozen=True, slots=True)
class _RerankedFragment:
    candidate: FusedFragmentCandidate
    raw_score: float
    normalized_score: float


class HybridFragmentSearchEngine:
    """Run dense and lexical retrieval, RRF, cross-encoder reranking, and NMS."""

    mode = "hybrid-transcript"

    def __init__(
        self,
        *,
        store: HybridFragmentStore,
        query_embedder: QueryEmbedder,
        reranker: CandidateReranker,
        dense_candidate_limit: int = 40,
        lexical_candidate_limit: int = 40,
        rerank_candidate_limit: int = 30,
        rrf_k: float = 60.0,
        temporal_iou_threshold: float = 0.5,
        padding_seconds: float = 1.0,
    ) -> None:
        limits = (dense_candidate_limit, lexical_candidate_limit, rerank_candidate_limit)
        if any(value <= 0 for value in limits):
            raise ValueError("Hybrid candidate limits must be positive")
        if not math.isfinite(rrf_k) or rrf_k <= 0:
            raise ValueError("RRF k must be positive and finite")
        if not math.isfinite(temporal_iou_threshold) or not 0 < temporal_iou_threshold <= 1:
            raise ValueError("Temporal IoU threshold must be in (0, 1]")
        if not math.isfinite(padding_seconds) or padding_seconds < 0:
            raise ValueError("Search padding must be finite and non-negative")
        self._store = store
        self._query_embedder = query_embedder
        self._reranker = reranker
        self._dense_candidate_limit = dense_candidate_limit
        self._lexical_candidate_limit = lexical_candidate_limit
        self._rerank_candidate_limit = rerank_candidate_limit
        self._rrf_k = rrf_k
        self._temporal_iou_threshold = temporal_iou_threshold
        self._padding = padding_seconds

    @property
    def version(self) -> str:
        return f"hybrid/{self._query_embedder.model_id}+{self._reranker.model_id}"

    async def search(
        self,
        *,
        video: VideoDescriptor,
        query: str,
        limit: int,
    ) -> Sequence[SearchHit]:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("Search query must not be blank")
        if limit <= 0:
            raise ValueError("Search limit must be positive")
        if not math.isfinite(video.duration_seconds) or video.duration_seconds <= 0:
            raise ValueError("Video duration must be positive and finite")

        query_embedding = await self._query_embedder.embed_query(normalized_query)
        _validate_query_embedding(query_embedding, self._query_embedder.dimension)
        retrieved = await self._store.retrieve(
            video_id=video.id,
            query_text=normalized_query,
            query_embedding=query_embedding,
            dense_limit=self._dense_candidate_limit,
            lexical_limit=self._lexical_candidate_limit,
        )
        fused = reciprocal_rank_fuse(
            retrieved.dense,
            retrieved.lexical,
            rrf_k=self._rrf_k,
        )[: self._rerank_candidate_limit]
        if not fused:
            return ()

        raw_scores = tuple(
            await self._reranker.score(
                normalized_query,
                tuple(candidate.transcript for candidate in fused),
            )
        )
        if len(raw_scores) != len(fused):
            raise ValueError("Reranker score count does not match the candidate count")
        reranked: list[_RerankedFragment] = []
        for candidate, raw_score in zip(fused, raw_scores, strict=True):
            if not math.isfinite(raw_score):
                raise ValueError("Reranker returned a non-finite score")
            _validate_fragment(candidate, video.duration_seconds)
            reranked.append(
                _RerankedFragment(
                    candidate=candidate,
                    raw_score=raw_score,
                    normalized_score=_sigmoid(raw_score),
                )
            )
        reranked.sort(
            key=lambda item: (
                -item.raw_score,
                -item.candidate.rrf_score,
                item.candidate.best_rank,
                item.candidate.position,
                item.candidate.fragment_id,
            )
        )
        selected = temporal_non_maximum_suppression(
            reranked,
            limit=limit,
            iou_threshold=self._temporal_iou_threshold,
        )
        return tuple(
            SearchHit(
                start_seconds=max(0.0, item.candidate.start_seconds - self._padding),
                end_seconds=min(
                    video.duration_seconds,
                    item.candidate.end_seconds + self._padding,
                ),
                score=item.normalized_score,
                preview_text=" ".join(item.candidate.transcript.split()) or None,
            )
            for item in selected
        )


def reciprocal_rank_fuse(
    dense: Sequence[RankedFragment],
    lexical: Sequence[RankedFragment],
    *,
    rrf_k: float,
) -> tuple[FusedFragmentCandidate, ...]:
    """Fuse ranks without mixing incomparable cosine and FTS score scales."""
    if not math.isfinite(rrf_k) or rrf_k <= 0:
        raise ValueError("RRF k must be positive and finite")
    candidates: dict[int, RankedFragment] = {}
    scores: dict[int, float] = {}
    best_ranks: dict[int, int] = {}
    for ranked_list in (dense, lexical):
        seen: set[int] = set()
        for rank, match in enumerate(ranked_list, start=1):
            if match.fragment_id in seen:
                raise ValueError("Retriever returned a duplicate fragment identity")
            seen.add(match.fragment_id)
            _validate_store_match(match)
            existing = candidates.get(match.fragment_id)
            if existing is not None and not _same_fragment(existing, match):
                raise ValueError("Retrievers returned inconsistent data for one fragment")
            candidates[match.fragment_id] = match
            scores[match.fragment_id] = scores.get(match.fragment_id, 0.0) + 1.0 / (rrf_k + rank)
            best_ranks[match.fragment_id] = min(best_ranks.get(match.fragment_id, rank), rank)

    fused = [
        FusedFragmentCandidate(
            fragment_id=fragment_id,
            position=match.position,
            start_seconds=match.start_seconds,
            end_seconds=match.end_seconds,
            transcript=match.transcript,
            rrf_score=scores[fragment_id],
            best_rank=best_ranks[fragment_id],
        )
        for fragment_id, match in candidates.items()
    ]
    fused.sort(
        key=lambda candidate: (
            -candidate.rrf_score,
            candidate.best_rank,
            candidate.position,
            candidate.fragment_id,
        )
    )
    return tuple(fused)


def temporal_non_maximum_suppression(
    candidates: Sequence[_RerankedFragment],
    *,
    limit: int,
    iou_threshold: float,
) -> tuple[_RerankedFragment, ...]:
    """Greedily remove lower-ranked fragments that cover the same time interval."""
    if limit <= 0:
        raise ValueError("NMS limit must be positive")
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("NMS IoU threshold must be in (0, 1]")
    selected: list[_RerankedFragment] = []
    for candidate in candidates:
        interval = (candidate.candidate.start_seconds, candidate.candidate.end_seconds)
        if any(
            _temporal_iou(
                interval,
                (item.candidate.start_seconds, item.candidate.end_seconds),
            )
            >= iou_threshold
            for item in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _validate_query_embedding(embedding: Sequence[float], expected_dimension: int) -> None:
    if expected_dimension <= 0 or len(embedding) != expected_dimension:
        raise ValueError("Query embedding has an unexpected dimension")
    if any(not math.isfinite(value) for value in embedding):
        raise ValueError("Query embedding contains a non-finite value")


def _validate_store_match(match: RankedFragment) -> None:
    if match.fragment_id <= 0 or match.position < 0:
        raise ValueError("Retriever returned an invalid fragment identity")
    if not all(
        math.isfinite(value) for value in (match.start_seconds, match.end_seconds, match.score)
    ):
        raise ValueError("Retriever returned non-finite values")
    if match.start_seconds < 0 or match.start_seconds >= match.end_seconds:
        raise ValueError("Retriever returned an invalid time interval")
    if not " ".join(match.transcript.split()):
        raise ValueError("Retriever returned a blank transcript")


def _validate_fragment(candidate: FusedFragmentCandidate, video_duration_seconds: float) -> None:
    if candidate.start_seconds >= video_duration_seconds:
        raise ValueError("Retriever returned a fragment outside the video")


def _same_fragment(left: RankedFragment, right: RankedFragment) -> bool:
    return (
        left.position == right.position
        and left.start_seconds == right.start_seconds
        and left.end_seconds == right.end_seconds
        and left.transcript == right.transcript
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


def _temporal_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = (left[1] - left[0]) + (right[1] - right[0]) - intersection
    return intersection / union if union > 0 else 0.0
