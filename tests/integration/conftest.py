from __future__ import annotations

import os
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from app.core.config import Settings
from app.db.session import create_engine
from app.main import create_app
from tests.db_guard import require_test_database_url, truncate_videos


def _make_video(path: Path, *, duration: float, audio: bool = True) -> bytes:
    command = [
        os.environ.get("FFMPEG_BINARY", "ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=blue:s=160x90:r=10:d={duration}",
    ]
    if audio:
        command.extend(["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=8000:d={duration}"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-threads",
            "1",
        ]
    )
    if audio:
        command.extend(["-c:a", "aac", "-b:a", "32k", "-shortest"])
    command.extend(["-movflags", "+faststart", str(path)])
    subprocess.run(command, check=True, capture_output=True)
    return path.read_bytes()


@pytest.fixture(scope="session")
def media_fixtures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    root = tmp_path_factory.mktemp("media")
    valid = _make_video(root / "valid.mp4", duration=12.0)
    threshold = _make_video(root / "threshold.mp4", duration=10.0)
    short = _make_video(root / "short.mp4", duration=9.5)
    no_audio = _make_video(root / "no-audio.mp4", duration=12.0, audio=False)
    return {
        "valid": valid,
        "threshold": threshold,
        "short": short,
        "no_audio": no_audio,
        # Keep only an incomplete MP4 box header so ffprobe cannot classify the
        # payload as a valid audio-only/video-less MP4 on some versions.
        "truncated": valid[:8],
    }


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    database_url = require_test_database_url(os.environ.get("TEST_DATABASE_URL"))
    return Settings(
        _env_file=None,
        app_env="test",
        database_url=database_url,
        upload_dir=tmp_path / "uploads",
        cors_origins=["http://localhost:3000"],
    )


@pytest_asyncio.fixture
async def clean_database(settings: Settings) -> AsyncIterator[None]:
    engine = create_engine(settings.database_url)
    initialized = False
    try:
        async with engine.begin() as connection:
            await truncate_videos(connection)
        initialized = True
        yield
    finally:
        try:
            if initialized:
                async with engine.begin() as connection:
                    await truncate_videos(connection)
        finally:
            await engine.dispose()


@pytest_asyncio.fixture
async def client(
    settings: Settings,
    clean_database: None,
) -> AsyncIterator[httpx.AsyncClient]:
    del clean_database
    application = create_app(settings)
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api_client:
            yield api_client


@pytest.fixture
def upload_valid(media_fixtures: dict[str, bytes]):
    def upload_payload(
        *, filename: str = "sample.mp4", content_type: str = "video/mp4"
    ) -> dict[str, tuple[str, bytes, str]]:
        return {"file": (filename, media_fixtures["valid"], content_type)}

    return upload_payload
