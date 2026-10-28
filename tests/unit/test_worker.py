from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import app.worker as worker_module
from app.core.config import Settings
from app.db.models.video import Video
from app.services.media.storage import LocalStorage
from app.services.ml.contracts import IndexedFragment
from app.services.ml.indexing import TranscriptIndexingService
from app.services.search.base import VideoDescriptor
from app.worker import TranscriptIndexWorker


@dataclass(frozen=True, slots=True)
class FakeClaim:
    video_id: UUID
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class FakeVideo:
    id: UUID
    duration_seconds: float = 12.0
    storage_key: str = "video.mp4"
    size_bytes: int = 123


@dataclass(frozen=True, slots=True)
class CompleteCall:
    video_id: UUID
    worker_id: str
    fragments: tuple[IndexedFragment, ...]
    pipeline_version: str
    asr_model_version: str
    embedding_model_version: str


@dataclass(frozen=True, slots=True)
class FailureCall:
    video_id: UUID
    worker_id: str
    error_code: str
    error_message: str


class FakeTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc, traceback
        return False


class FakeSession:
    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc, traceback
        return False

    def begin(self) -> FakeTransaction:
        return FakeTransaction()


class FakeSessionFactory:
    def __call__(self) -> FakeSession:
        return FakeSession()


class FakeSearchIndexRepository:
    def __init__(
        self,
        claim: FakeClaim,
        *,
        mark_failed_error: Exception | None = None,
    ) -> None:
        self.claim = claim
        self.mark_failed_error = mark_failed_error
        self.complete_calls: list[CompleteCall] = []
        self.failure_calls: list[FailureCall] = []
        self.requeue_calls: list[tuple[UUID, str]] = []

    async def claim_next(self, *, worker_id: str, lease_duration: object) -> FakeClaim:
        del worker_id, lease_duration
        return self.claim

    async def complete(
        self,
        video_id: UUID,
        *,
        worker_id: str,
        fragments: tuple[IndexedFragment, ...],
        pipeline_version: str,
        asr_model_version: str,
        embedding_model_version: str,
    ) -> None:
        self.complete_calls.append(
            CompleteCall(
                video_id=video_id,
                worker_id=worker_id,
                fragments=fragments,
                pipeline_version=pipeline_version,
                asr_model_version=asr_model_version,
                embedding_model_version=embedding_model_version,
            )
        )

    async def mark_failed(
        self,
        video_id: UUID,
        *,
        worker_id: str,
        error_code: str,
        error_message: str,
    ) -> None:
        self.failure_calls.append(
            FailureCall(
                video_id=video_id,
                worker_id=worker_id,
                error_code=error_code,
                error_message=error_message,
            )
        )
        if self.mark_failed_error is not None:
            raise self.mark_failed_error

    async def requeue(self, video_id: UUID, *, worker_id: str) -> None:
        self.requeue_calls.append((video_id, worker_id))


class FakeVideoRepository:
    def __init__(self, video: FakeVideo) -> None:
        self.video = video

    async def get(self, video_id: UUID) -> Video:
        assert video_id == self.video.id
        return cast(Video, self.video)


class FakeStorage:
    def __init__(self, media_path: Path) -> None:
        self.media_path = media_path
        self.requests: list[tuple[str, int]] = []

    async def verified_path(self, storage_key: str, expected_size: int) -> Path:
        self.requests.append((storage_key, expected_size))
        return self.media_path


class FakePipeline:
    transcriber_model_id = "fake-asr-v1"
    embedding_model_id = "fake-embedding-v1"

    def __init__(
        self,
        fragments: tuple[IndexedFragment, ...],
        *,
        error: Exception | None = None,
    ) -> None:
        self.fragments = fragments
        self.error = error
        self.calls: list[tuple[VideoDescriptor, Path]] = []

    async def index(
        self,
        *,
        video: VideoDescriptor,
        media_path: Path,
    ) -> tuple[IndexedFragment, ...]:
        self.calls.append((video, media_path))
        if self.error is not None:
            raise self.error
        return self.fragments


@dataclass(slots=True)
class WorkerHarness:
    worker: TranscriptIndexWorker
    repository: FakeSearchIndexRepository
    storage: FakeStorage
    pipeline: FakePipeline


def build_worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pipeline_error: Exception | None = None,
    mark_failed_error: Exception | None = None,
) -> WorkerHarness:
    video = FakeVideo(id=uuid4())
    claim = FakeClaim(video_id=video.id)
    fragments = (
        IndexedFragment(
            position=0,
            start_seconds=1.0,
            end_seconds=5.0,
            transcript="test transcript",
            embedding=(1.0, 0.0),
            language="en",
        ),
    )
    repository = FakeSearchIndexRepository(claim, mark_failed_error=mark_failed_error)
    video_repository = FakeVideoRepository(video)
    media_path = Path("/tmp/video.mp4")
    storage = FakeStorage(media_path)
    pipeline = FakePipeline(fragments, error=pipeline_error)
    monkeypatch.setattr(
        worker_module,
        "SearchIndexRepository",
        lambda _session: repository,
    )
    monkeypatch.setattr(
        worker_module,
        "VideoRepository",
        lambda _session: video_repository,
    )
    settings = Settings(
        _env_file=None,
        ml_pipeline_version="test-pipeline-v1",
        ml_worker_lease_seconds=3,
    )
    worker = TranscriptIndexWorker(
        session_factory=cast(
            async_sessionmaker[AsyncSession],
            FakeSessionFactory(),
        ),
        storage=cast(LocalStorage, storage),
        pipeline=cast(TranscriptIndexingService, pipeline),
        settings=settings,
        worker_id="test-worker",
    )
    return WorkerHarness(
        worker=worker,
        repository=repository,
        storage=storage,
        pipeline=pipeline,
    )


def install_observable_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
    worker: TranscriptIndexWorker,
) -> tuple[asyncio.Event, asyncio.Event, list[asyncio.Event]]:
    started = asyncio.Event()
    stopped = asyncio.Event()
    stop_events: list[asyncio.Event] = []

    async def heartbeat(video_id: UUID, stop: asyncio.Event) -> None:
        del video_id
        stop_events.append(stop)
        started.set()
        await stop.wait()
        stopped.set()

    monkeypatch.setattr(worker, "_heartbeat", heartbeat)
    return started, stopped, stop_events


@pytest.mark.asyncio
async def test_successful_job_completes_and_stops_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_worker(monkeypatch)
    started, stopped, stop_events = install_observable_heartbeat(monkeypatch, harness.worker)

    processed = await harness.worker.run_once()

    assert processed is True
    assert started.is_set()
    assert stopped.is_set()
    assert len(stop_events) == 1
    assert stop_events[0].is_set()
    assert harness.storage.requests == [("video.mp4", 123)]
    assert len(harness.pipeline.calls) == 1
    assert harness.repository.failure_calls == []
    assert harness.repository.complete_calls == [
        CompleteCall(
            video_id=harness.repository.claim.video_id,
            worker_id="test-worker",
            fragments=harness.pipeline.fragments,
            pipeline_version="test-pipeline-v1",
            asr_model_version="fake-asr-v1",
            embedding_model_version="fake-embedding-v1",
        )
    ]


@pytest.mark.asyncio
async def test_pipeline_error_marks_job_failed_and_stops_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_worker(monkeypatch, pipeline_error=RuntimeError("pipeline failed"))
    _, stopped, stop_events = install_observable_heartbeat(monkeypatch, harness.worker)

    processed = await harness.worker.run_once()

    assert processed is True
    assert harness.repository.complete_calls == []
    assert harness.repository.failure_calls == [
        FailureCall(
            video_id=harness.repository.claim.video_id,
            worker_id="test-worker",
            error_code="INDEXING_FAILED",
            error_message="Indexing failed; inspect the ML worker logs and retry",
        )
    ]
    assert stopped.is_set()
    assert stop_events[0].is_set()


@pytest.mark.asyncio
async def test_failure_record_error_does_not_escape_run_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_worker(
        monkeypatch,
        pipeline_error=RuntimeError("pipeline failed"),
        mark_failed_error=RuntimeError("database unavailable"),
    )
    _, stopped, stop_events = install_observable_heartbeat(monkeypatch, harness.worker)

    processed = await harness.worker.run_once()

    assert processed is True
    assert len(harness.repository.failure_calls) == 1
    assert stopped.is_set()
    assert stop_events[0].is_set()


@pytest.mark.asyncio
async def test_stop_signal_cancels_and_requeues_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_worker(monkeypatch)
    _, stopped, stop_events = install_observable_heartbeat(monkeypatch, harness.worker)
    indexing_started = asyncio.Event()
    never_complete = asyncio.Event()

    async def blocking_index(_video: Video) -> tuple[IndexedFragment, ...]:
        indexing_started.set()
        await never_complete.wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(harness.worker, "_index_video", blocking_index)
    stop = asyncio.Event()
    task = asyncio.create_task(harness.worker.run_forever(stop))
    await indexing_started.wait()
    stop.set()
    await asyncio.wait_for(task, timeout=1)

    assert harness.repository.requeue_calls == [(harness.repository.claim.video_id, "test-worker")]
    assert stopped.is_set()
    assert stop_events[0].is_set()


@pytest.mark.asyncio
async def test_stop_during_video_load_requeues_before_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_worker(monkeypatch)
    loading_started = asyncio.Event()
    never_complete = asyncio.Event()

    class BlockingVideoRepository:
        async def get(self, _video_id: UUID) -> Video:
            loading_started.set()
            await never_complete.wait()
            raise AssertionError("unreachable")

    monkeypatch.setattr(
        worker_module,
        "VideoRepository",
        lambda _session: BlockingVideoRepository(),
    )
    stop = asyncio.Event()
    task = asyncio.create_task(harness.worker.run_forever(stop))
    await loading_started.wait()
    stop.set()
    await asyncio.wait_for(task, timeout=1)

    assert harness.repository.requeue_calls == [(harness.repository.claim.video_id, "test-worker")]
