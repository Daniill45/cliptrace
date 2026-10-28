"""Offline transcript-to-vector indexing orchestration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from app.services.ml.chunking import TranscriptChunker
from app.services.ml.contracts import IndexedFragment, SpeechTranscriber, TextEmbedder
from app.services.search.base import VideoDescriptor


class TranscriptIndexingService:
    """Transcribe one verified video and produce persistence-ready fragments."""

    def __init__(
        self,
        *,
        transcriber: SpeechTranscriber,
        embedder: TextEmbedder,
        chunker: TranscriptChunker,
    ) -> None:
        self._transcriber = transcriber
        self._embedder = embedder
        self._chunker = chunker

    @property
    def version(self) -> str:
        return f"{self._transcriber.model_id}+{self._embedder.model_id}"

    @property
    def transcriber_model_id(self) -> str:
        return self._transcriber.model_id

    @property
    def embedding_model_id(self) -> str:
        return self._embedder.model_id

    async def index(
        self,
        *,
        video: VideoDescriptor,
        media_path: Path,
    ) -> tuple[IndexedFragment, ...]:
        """Return ordered fragments; persistence and job state remain caller-owned."""
        transcription = await self._transcriber.transcribe(media_path)
        chunks = self._chunker.chunk(
            transcription.segments,
            video_duration_seconds=video.duration_seconds,
            language=transcription.language,
        )
        if not chunks:
            return ()

        embeddings = tuple(
            await self._embedder.embed_documents(tuple(chunk.text for chunk in chunks))
        )
        if len(embeddings) != len(chunks):
            raise ValueError("Embedding count does not match transcript chunk count")
        _validate_embeddings(embeddings, expected_dimension=self._embedder.dimension)

        return tuple(
            IndexedFragment(
                position=position,
                start_seconds=chunk.start_seconds,
                end_seconds=chunk.end_seconds,
                transcript=chunk.text,
                embedding=embedding,
                language=chunk.language,
            )
            for position, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        )


def _validate_embeddings(
    embeddings: Sequence[Sequence[float]],
    *,
    expected_dimension: int,
) -> None:
    if expected_dimension <= 0:
        raise ValueError("Embedding dimension must be positive")
    for embedding in embeddings:
        if len(embedding) != expected_dimension:
            raise ValueError(
                f"Expected embedding dimension {expected_dimension}, received {len(embedding)}"
            )
        if any(not math.isfinite(value) for value in embedding):
            raise ValueError("Embedding contains a non-finite value")
