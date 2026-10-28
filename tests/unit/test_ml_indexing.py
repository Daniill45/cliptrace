from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pytest

from app.services.ml.chunking import TranscriptChunker
from app.services.ml.contracts import Embedding, Transcription, TranscriptSegment
from app.services.ml.indexing import TranscriptIndexingService
from app.services.search.base import VideoDescriptor


class FakeTranscriber:
    model_id = "fake-asr-v1"

    def __init__(self, transcription: Transcription) -> None:
        self.transcription = transcription
        self.paths: list[Path] = []

    async def transcribe(self, media_path: Path) -> Transcription:
        self.paths.append(media_path)
        return self.transcription


class FakeEmbedder:
    model_id = "fake-embedding-v1"
    dimension = 3

    def __init__(self, embeddings: Sequence[Embedding]) -> None:
        self.embeddings = embeddings
        self.document_batches: list[tuple[str, ...]] = []

    async def embed_query(self, text: str) -> Embedding:
        del text
        return (1.0, 0.0, 0.0)

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Embedding]:
        self.document_batches.append(tuple(texts))
        return self.embeddings


@pytest.mark.asyncio
async def test_indexing_service_runs_pipeline_and_returns_ordered_payloads(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "video.mp4"
    transcriber = FakeTranscriber(
        Transcription(
            segments=(
                TranscriptSegment(0.0, 4.0, "first"),
                TranscriptSegment(9.0, 11.0, "second"),
                TranscriptSegment(14.0, 18.0, "third"),
            ),
            language="ru",
        )
    )
    embedder = FakeEmbedder(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    service = TranscriptIndexingService(
        transcriber=transcriber,
        embedder=embedder,
        chunker=TranscriptChunker(target_duration_seconds=12.0, overlap_seconds=2.0),
    )
    video = VideoDescriptor(id=uuid4(), duration_seconds=20.0, storage_key="video.mp4")

    fragments = await service.index(video=video, media_path=media_path)

    assert transcriber.paths == [media_path]
    assert embedder.document_batches == [("first second", "second third")]
    assert service.version == "fake-asr-v1+fake-embedding-v1"
    assert [fragment.position for fragment in fragments] == [0, 1]
    assert [(fragment.start_seconds, fragment.end_seconds) for fragment in fragments] == [
        (0.0, 12.0),
        (10.0, 20.0),
    ]
    assert [fragment.embedding for fragment in fragments] == [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    assert all(fragment.language == "ru" for fragment in fragments)


@pytest.mark.asyncio
async def test_indexing_service_does_not_call_embedder_for_empty_transcript(
    tmp_path: Path,
) -> None:
    transcriber = FakeTranscriber(Transcription(segments=()))
    embedder = FakeEmbedder(())
    service = TranscriptIndexingService(
        transcriber=transcriber,
        embedder=embedder,
        chunker=TranscriptChunker(),
    )

    fragments = await service.index(
        video=VideoDescriptor(id=uuid4(), duration_seconds=20.0, storage_key="video.mp4"),
        media_path=tmp_path / "video.mp4",
    )

    assert fragments == ()
    assert embedder.document_batches == []


@pytest.mark.asyncio
async def test_indexing_service_rejects_embedding_count_or_dimension_mismatch(
    tmp_path: Path,
) -> None:
    transcriber = FakeTranscriber(Transcription(segments=(TranscriptSegment(0.0, 2.0, "speech"),)))
    video = VideoDescriptor(id=uuid4(), duration_seconds=20.0, storage_key="video.mp4")

    count_service = TranscriptIndexingService(
        transcriber=transcriber,
        embedder=FakeEmbedder(()),
        chunker=TranscriptChunker(),
    )
    with pytest.raises(ValueError, match="count"):
        await count_service.index(video=video, media_path=tmp_path / "video.mp4")

    dimension_service = TranscriptIndexingService(
        transcriber=transcriber,
        embedder=FakeEmbedder(((1.0, 2.0),)),  # type: ignore[arg-type]
        chunker=TranscriptChunker(),
    )
    with pytest.raises(ValueError, match="dimension"):
        await dimension_service.index(video=video, media_path=tmp_path / "video.mp4")
