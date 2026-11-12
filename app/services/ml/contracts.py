"""Framework-neutral contracts and values for transcript indexing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

Embedding = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One timestamped unit emitted by speech recognition."""

    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class Transcription:
    """Speech-recognition output before temporal chunking."""

    segments: tuple[TranscriptSegment, ...]
    language: str | None = None
    language_probability: float | None = None


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    """A bounded transcript window that will become one search fragment."""

    start_seconds: float
    end_seconds: float
    text: str
    language: str | None = None


class TranscriptChunkingStrategy(Protocol):
    """Convert timestamped transcript units into persistence-ready text windows."""

    @property
    def version(self) -> str: ...

    def chunk(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        video_duration_seconds: float,
        language: str | None = None,
    ) -> tuple[TranscriptChunk, ...]: ...


@dataclass(frozen=True, slots=True)
class IndexedFragment:
    """Persistence-neutral fragment produced by the offline indexing pipeline."""

    position: int
    start_seconds: float
    end_seconds: float
    transcript: str
    embedding: Embedding
    language: str | None = None


class SpeechTranscriber(Protocol):
    """Convert a local media asset into timestamped speech units."""

    @property
    def model_id(self) -> str: ...

    async def transcribe(self, media_path: Path) -> Transcription: ...


class QueryEmbedder(Protocol):
    """Encode a user query into the shared retrieval vector space."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed_query(self, text: str) -> Embedding: ...


class TextEmbedder(QueryEmbedder, Protocol):
    """Encode both indexed passages and online queries."""

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Embedding]: ...


class CandidateReranker(Protocol):
    """Score query-document pairs after first-stage retrieval."""

    @property
    def model_id(self) -> str: ...

    async def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...
