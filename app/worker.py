"""Single-concurrency transcript indexing worker for local Compose deployments."""

from __future__ import annotations

import asyncio
import logging
import signal
import socket
from contextlib import suppress
from datetime import timedelta
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.models.video import Video
from app.db.session import create_engine, create_session_factory
from app.repositories.search import SearchIndexLeaseLostError, SearchIndexRepository
from app.repositories.videos import VideoRepository
from app.services.media.storage import LocalStorage
from app.services.ml.adapters import FasterWhisperTranscriber, OnnxE5Embedder
from app.services.ml.chunking import TranscriptChunker
from app.services.ml.contracts import IndexedFragment
from app.services.ml.indexing import TranscriptIndexingService
from app.services.search.base import VideoDescriptor

logger = logging.getLogger(__name__)


class TranscriptIndexWorker:
    """Claim, heartbeat, process, and atomically complete search-index jobs."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        storage: LocalStorage,
        pipeline: TranscriptIndexingService,
        settings: Settings,
        worker_id: str,
    ) -> None:
        self._session_factory = session_factory
        self._storage = storage
        self._pipeline = pipeline
        self._settings = settings
        self._worker_id = worker_id
        self._lease_duration = timedelta(seconds=settings.ml_worker_lease_seconds)

    async def run_forever(self, stop: asyncio.Event) -> None:
        """Poll without busy-waiting until the process receives a stop signal."""
        logger.info(
            "ml_worker_started",
            extra={"event": "ml_worker_started", "worker_id": self._worker_id},
        )
        while not stop.is_set():
            job = asyncio.create_task(self.run_once())
            stopping = asyncio.create_task(stop.wait())
            done, _ = await asyncio.wait(
                (job, stopping),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stopping in done:
                job.cancel()
                with suppress(asyncio.CancelledError):
                    await job
                return
            stopping.cancel()
            with suppress(asyncio.CancelledError):
                await stopping
            processed = job.result()
            if processed:
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), self._settings.ml_worker_poll_seconds)

    async def run_once(self) -> bool:
        """Process at most one claimed job and report whether work was available."""
        async with self._session_factory() as session:
            async with session.begin():
                claimed = await SearchIndexRepository(session).claim_next(
                    worker_id=self._worker_id,
                    lease_duration=self._lease_duration,
                )
        if claimed is None:
            return False

        video_id = claimed.video_id
        heartbeat_stop: asyncio.Event | None = None
        heartbeat: asyncio.Task[None] | None = None
        try:
            if claimed.attempts > self._settings.ml_worker_max_attempts:
                await self._mark_failed(
                    video_id,
                    code="INDEXING_ATTEMPTS_EXHAUSTED",
                    message="Indexing retry limit was exceeded",
                )
                return True

            async with self._session_factory() as session:
                video = await VideoRepository(session).get(video_id)
            if video is None:
                return True

            heartbeat_stop = asyncio.Event()
            heartbeat = asyncio.create_task(self._heartbeat(video_id, heartbeat_stop))
            try:
                fragments = await self._index_video(video)
                async with self._session_factory() as session:
                    async with session.begin():
                        await SearchIndexRepository(session).complete(
                            video_id,
                            worker_id=self._worker_id,
                            fragments=fragments,
                            pipeline_version=self._settings.ml_pipeline_version,
                            asr_model_version=self._pipeline.transcriber_model_id,
                            embedding_model_version=self._pipeline.embedding_model_id,
                        )
                logger.info(
                    "video_search_index_ready",
                    extra={
                        "event": "video_search_index_ready",
                        "video_id": str(video_id),
                        "fragment_count": len(fragments),
                    },
                )
            except SearchIndexLeaseLostError:
                logger.warning(
                    "video_search_index_lease_lost",
                    extra={
                        "event": "video_search_index_lease_lost",
                        "video_id": str(video_id),
                    },
                )
            except Exception:
                logger.exception(
                    "video_search_index_failed",
                    extra={"event": "video_search_index_failed", "video_id": str(video_id)},
                )
                await self._mark_failed(
                    video_id,
                    code="INDEXING_FAILED",
                    message="Indexing failed; inspect the ML worker logs and retry",
                )
        except asyncio.CancelledError:
            await asyncio.shield(self._requeue(video_id))
            raise
        finally:
            if heartbeat_stop is not None and heartbeat is not None:
                heartbeat_stop.set()
                await heartbeat
        return True

    async def _index_video(self, video: Video) -> tuple[IndexedFragment, ...]:
        media_path = await self._storage.verified_path(video.storage_key, video.size_bytes)
        return await self._pipeline.index(
            video=VideoDescriptor(
                id=video.id,
                duration_seconds=video.duration_seconds,
                storage_key=video.storage_key,
            ),
            media_path=media_path,
        )

    async def _heartbeat(self, video_id: UUID, stop: asyncio.Event) -> None:
        interval = max(1.0, self._settings.ml_worker_lease_seconds / 3)
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), interval)
                return
            except TimeoutError:
                pass
            try:
                async with self._session_factory() as session:
                    async with session.begin():
                        renewed = await SearchIndexRepository(session).renew_lease(
                            video_id,
                            worker_id=self._worker_id,
                            lease_duration=self._lease_duration,
                        )
                if not renewed:
                    logger.warning(
                        "video_search_index_heartbeat_rejected",
                        extra={
                            "event": "video_search_index_heartbeat_rejected",
                            "video_id": str(video_id),
                        },
                    )
                    return
            except Exception:
                logger.exception(
                    "video_search_index_heartbeat_failed",
                    extra={
                        "event": "video_search_index_heartbeat_failed",
                        "video_id": str(video_id),
                    },
                )

    async def _mark_failed(self, video_id: UUID, *, code: str, message: str) -> None:
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    await SearchIndexRepository(session).mark_failed(
                        video_id,
                        worker_id=self._worker_id,
                        error_code=code,
                        error_message=message,
                    )
        except SearchIndexLeaseLostError:
            logger.warning(
                "video_search_index_failure_not_recorded",
                extra={
                    "event": "video_search_index_failure_not_recorded",
                    "video_id": str(video_id),
                },
            )
        except Exception:
            logger.exception(
                "video_search_index_failure_record_failed",
                extra={
                    "event": "video_search_index_failure_record_failed",
                    "video_id": str(video_id),
                },
            )

    async def _requeue(self, video_id: UUID) -> None:
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    await SearchIndexRepository(session).requeue(
                        video_id,
                        worker_id=self._worker_id,
                    )
            logger.info(
                "video_search_index_requeued",
                extra={
                    "event": "video_search_index_requeued",
                    "video_id": str(video_id),
                    "worker_id": self._worker_id,
                },
            )
        except SearchIndexLeaseLostError:
            logger.warning(
                "video_search_index_requeue_lease_lost",
                extra={
                    "event": "video_search_index_requeue_lease_lost",
                    "video_id": str(video_id),
                    "worker_id": self._worker_id,
                },
            )
        except Exception:
            logger.exception(
                "video_search_index_requeue_failed",
                extra={
                    "event": "video_search_index_requeue_failed",
                    "video_id": str(video_id),
                    "worker_id": self._worker_id,
                },
            )


async def _run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ml_model_cache_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    storage = LocalStorage(
        settings.upload_dir,
        max_size_bytes=settings.max_upload_size_bytes,
        chunk_size_bytes=settings.upload_chunk_size_bytes,
        temp_max_age_seconds=settings.temp_file_max_age_seconds,
    )
    await storage.initialize()
    transcriber = FasterWhisperTranscriber(
        model_name=settings.whisper_model,
        revision=settings.whisper_model_revision,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        download_root=settings.ml_model_cache_dir / "whisper",
        language=settings.whisper_language,
    )
    embedder = OnnxE5Embedder(
        model_name=settings.embedding_model,
        revision=settings.embedding_model_revision,
        device=settings.embedding_device,
        dimension=384,
        cache_folder=settings.ml_model_cache_dir / "huggingface",
        batch_size=settings.embedding_batch_size,
    )
    pipeline = TranscriptIndexingService(
        transcriber=transcriber,
        embedder=embedder,
        chunker=TranscriptChunker(),
    )
    worker = TranscriptIndexWorker(
        session_factory=session_factory,
        storage=storage,
        pipeline=pipeline,
        settings=settings,
        worker_id=f"{socket.gethostname()}-{uuid4()}",
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for selected_signal in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(selected_signal, stop.set)
    try:
        await worker.run_forever(stop)
    finally:
        await engine.dispose()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
