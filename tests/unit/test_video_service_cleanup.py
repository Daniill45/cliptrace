from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from starlette.datastructures import Headers, UploadFile

from app.core.errors import RequestValidationDomainError, UnsupportedMediaTypeError
from app.services.media.probe import MediaMetadata
from app.services.media.storage import LocalStorage
from app.services.videos import VideoService


class FailingTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> None:
        raise RuntimeError("simulated commit failure")


class FailingSession:
    def begin(self) -> FailingTransaction:
        return FailingTransaction()


class AcceptingRepository:
    async def create(self, video: object) -> object:
        return video


class FixedMediaProbe:
    async def probe(self, path: Path) -> MediaMetadata:
        assert path.exists()
        return MediaMetadata(
            duration_seconds=12.0,
            width=160,
            height=90,
            fps=10.0,
            video_codec="h264",
            video_profile="High",
            pixel_format="yuv420p",
            audio_codec="aac",
            container_names=("mp4",),
        )


@pytest.mark.asyncio
async def test_commit_failure_after_rename_removes_final_file(tmp_path: Path) -> None:
    storage = LocalStorage(
        tmp_path,
        max_size_bytes=1024,
        chunk_size_bytes=4,
        temp_max_age_seconds=60,
    )
    await storage.initialize()
    upload = UploadFile(
        BytesIO(b"video bytes"),
        filename="demo.mp4",
        headers=Headers({"content-type": "video/mp4"}),
    )
    service = VideoService(
        session=FailingSession(),  # type: ignore[arg-type]
        repository=AcceptingRepository(),  # type: ignore[arg-type]
        storage=storage,
        media_probe=FixedMediaProbe(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="simulated commit failure"):
        await service.create(upload=upload, title=None)

    assert not list(tmp_path.glob("*.part"))
    assert not list(tmp_path.glob("*.mp4"))
    assert upload.file.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "content_type", "title", "expected_error"),
    [
        (None, "video/mp4", None, RequestValidationDomainError),
        ("demo.mp4", "video/mp4", "", RequestValidationDomainError),
        ("demo.mp4", "video/webm", None, UnsupportedMediaTypeError),
        ("demo.mov", "video/mp4", None, UnsupportedMediaTypeError),
    ],
)
async def test_early_upload_validation_always_closes_upload(
    tmp_path: Path,
    filename: str | None,
    content_type: str,
    title: str | None,
    expected_error: type[Exception],
) -> None:
    storage = LocalStorage(
        tmp_path,
        max_size_bytes=1024,
        chunk_size_bytes=4,
        temp_max_age_seconds=60,
    )
    upload = UploadFile(
        BytesIO(b"video bytes"),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )
    service = VideoService(
        session=FailingSession(),  # type: ignore[arg-type]
        repository=AcceptingRepository(),  # type: ignore[arg-type]
        storage=storage,
        media_probe=FixedMediaProbe(),  # type: ignore[arg-type]
    )

    with pytest.raises(expected_error):
        await service.create(upload=upload, title=title)

    assert upload.file.closed
