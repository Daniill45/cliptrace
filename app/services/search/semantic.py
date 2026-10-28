"""Online semantic fragment search backed by an abstract vector store."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.services.ml.contracts import QueryEmbedder
from app.services.search.base import SearchHit, VideoDescriptor


class FragmentMatch(Protocol):
    """Read-only shape expected from a relevance-ordered fragment store."""

    @property
    def start_seconds(self) -> float: ...

    @property
    def end_seconds(self) -> float: ...

    @property
    def transcript(self) -> str: ...

    @property
    def score(self) -> float: ...


@dataclass(frozen=True, slots=True)
class StoredFragmentMatch:
    """Convenient store-neutral match value for adapters and tests."""

    start_seconds: float
    end_seconds: float
    transcript: str
    score: float


class FragmentStore(Protocol):
    """Return vector matches in descending relevance order."""

    async def search(
        self,
        *,
        video_id: UUID,
        query_embedding: Sequence[float],
        limit: int,
    ) -> Sequence[FragmentMatch]: ...


class SemanticFragmentSearchEngine:
    """Adapt query embedding and vector retrieval to FragmentSearchEngine."""

    mode = "semantic-transcript"

    def __init__(
        self,
        *,
        store: FragmentStore,
        query_embedder: QueryEmbedder,
        padding_seconds: float = 1.0,
    ) -> None:
        if not math.isfinite(padding_seconds) or padding_seconds < 0:
            raise ValueError("Search padding must be finite and non-negative")
        self._store = store
        self._query_embedder = query_embedder
        self._padding = padding_seconds

    @property
    def version(self) -> str:
        return f"semantic/{self._query_embedder.model_id}"

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
        matches = await self._store.search(
            video_id=video.id,
            query_embedding=query_embedding,
            limit=limit,
        )

        hits: list[SearchHit] = []
        for match in matches[:limit]:
            if not all(
                math.isfinite(value)
                for value in (match.start_seconds, match.end_seconds, match.score)
            ):
                raise ValueError("Fragment store returned non-finite values")
            bounded_start = max(0.0, min(match.start_seconds, video.duration_seconds))
            bounded_end = max(0.0, min(match.end_seconds, video.duration_seconds))
            if bounded_start >= bounded_end:
                raise ValueError("Fragment store returned an invalid time interval")
            preview = " ".join(match.transcript.split())
            hits.append(
                SearchHit(
                    start_seconds=max(0.0, bounded_start - self._padding),
                    end_seconds=min(video.duration_seconds, bounded_end + self._padding),
                    score=match.score,
                    preview_text=preview or None,
                )
            )
        return tuple(hits)


def _validate_query_embedding(embedding: Sequence[float], expected_dimension: int) -> None:
    if expected_dimension <= 0 or len(embedding) != expected_dimension:
        raise ValueError("Query embedding has an unexpected dimension")
    if any(not math.isfinite(value) for value in embedding):
        raise ValueError("Query embedding contains a non-finite value")
