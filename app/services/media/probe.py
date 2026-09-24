"""Bounded ffprobe adapter and strict MP4 metadata normalization."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from app.core.errors import (
    InvalidVideoError,
    UnsupportedMediaTypeError,
    VideoAudioTrackRequiredError,
    VideoStreamRequiredError,
    VideoTooLongError,
    VideoTooShortError,
)

MAX_PROBE_OUTPUT_BYTES = 1_048_576
MAX_PROBE_ERROR_BYTES = 65_536


@dataclass(frozen=True)
class MediaMetadata:
    duration_seconds: float
    width: int
    height: int
    fps: float | None
    video_codec: str
    video_profile: str | None
    pixel_format: str
    audio_codec: str
    container_names: tuple[str, ...]


class MediaProbe:
    """Run ffprobe safely and enforce ClipTrace's accepted media contract."""

    def __init__(
        self,
        *,
        binary: str,
        timeout_seconds: float,
        max_duration_seconds: float,
        min_duration_seconds: float,
    ) -> None:
        self._binary = binary
        self._timeout = timeout_seconds
        self._max_duration = max_duration_seconds
        self._min_duration = min_duration_seconds

    async def probe(self, path: Path) -> MediaMetadata:
        """Probe one internal path using a fixed argv and bounded output."""
        argv = (
            self._binary,
            "-v",
            "error",
            "-show_entries",
            "format=format_name,duration:"
            "stream=index,codec_type,codec_name,profile,pix_fmt,width,height,"
            "avg_frame_rate,r_frame_rate:stream_disposition=attached_pic",
            "-of",
            "json",
            str(path),
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError) as exc:
            raise InvalidVideoError("Video inspection could not be started") from exc
        try:
            stdout_task = asyncio.create_task(_read_limited(process.stdout, MAX_PROBE_OUTPUT_BYTES))
            stderr_task = asyncio.create_task(_read_limited(process.stderr, MAX_PROBE_ERROR_BYTES))
            try:
                stdout, _stderr = await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task), timeout=self._timeout
                )
                return_code = await asyncio.wait_for(process.wait(), timeout=1.0)
            except (TimeoutError, ProbeOutputTooLargeError) as exc:
                stdout_task.cancel()
                stderr_task.cancel()
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                await _terminate_process(process)
                raise InvalidVideoError("Video inspection failed or timed out") from exc
        finally:
            if process.returncode is None:
                await _terminate_process(process)
        if return_code != 0:
            raise InvalidVideoError
        try:
            payload = json.loads(stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidVideoError from exc
        return parse_probe_payload(
            payload,
            max_duration_seconds=self._max_duration,
            min_duration_seconds=self._min_duration,
        )

    async def check_available(self) -> None:
        """Verify that the configured binary can execute a short version command."""
        try:
            process = await asyncio.create_subprocess_exec(
                self._binary,
                "-version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            return_code = await asyncio.wait_for(process.wait(), timeout=min(self._timeout, 5.0))
        except (OSError, TimeoutError) as exc:
            if "process" in locals() and process.returncode is None:
                await _terminate_process(process)
            raise RuntimeError("ffprobe is unavailable") from exc
        if return_code != 0:
            raise RuntimeError("ffprobe version check failed")


class ProbeOutputTooLargeError(RuntimeError):
    """Internal signal that subprocess output exceeded its safe bound."""


async def _read_limited(
    stream: asyncio.StreamReader | None,
    limit: int,
) -> bytes:
    if stream is None:
        return b""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await stream.read(16_384)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise ProbeOutputTooLargeError
        chunks.append(chunk)


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        await process.wait()
        return
    try:
        process.terminate()
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.wait()


def parse_probe_payload(
    payload: object,
    *,
    max_duration_seconds: float,
    min_duration_seconds: float,
) -> MediaMetadata:
    """Normalize ffprobe JSON and reject media outside the MVP contract."""
    if not isinstance(payload, dict):
        raise InvalidVideoError
    format_data = payload.get("format")
    streams = payload.get("streams")
    if not isinstance(format_data, dict) or not isinstance(streams, list):
        raise InvalidVideoError
    raw_names = format_data.get("format_name")
    if not isinstance(raw_names, str):
        raise UnsupportedMediaTypeError
    container_names = tuple(name.strip().lower() for name in raw_names.split(",") if name.strip())
    if "mp4" not in container_names:
        raise UnsupportedMediaTypeError("The media container is not MP4")
    duration = _positive_float(format_data.get("duration"), allow_zero=False)
    if duration is None:
        raise InvalidVideoError("Video duration is unavailable")
    if duration > max_duration_seconds:
        raise VideoTooLongError
    if duration < min_duration_seconds:
        raise VideoTooShortError

    video_streams = [
        item
        for item in streams
        if isinstance(item, dict)
        and item.get("codec_type") == "video"
        and not _is_attached_picture(item)
    ]
    if not video_streams:
        raise VideoStreamRequiredError
    supported_video = next(
        (
            item
            for item in video_streams
            if item.get("codec_name") == "h264" and item.get("pix_fmt") == "yuv420p"
        ),
        None,
    )
    if supported_video is None:
        raise UnsupportedMediaTypeError("Video must use H.264 with yuv420p pixel format")

    audio_streams = [
        item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise VideoAudioTrackRequiredError
    supported_audio = next(
        (item for item in audio_streams if item.get("codec_name") == "aac"), None
    )
    if supported_audio is None:
        raise UnsupportedMediaTypeError("Audio must use the AAC codec")

    width = _positive_int(supported_video.get("width"))
    height = _positive_int(supported_video.get("height"))
    if width is None or height is None:
        raise InvalidVideoError("Video dimensions are unavailable")
    fps = _parse_fps(supported_video.get("avg_frame_rate") or supported_video.get("r_frame_rate"))
    profile_value = supported_video.get("profile")
    profile = profile_value if isinstance(profile_value, str) else None
    return MediaMetadata(
        duration_seconds=duration,
        width=width,
        height=height,
        fps=fps,
        video_codec="h264",
        video_profile=profile,
        pixel_format="yuv420p",
        audio_codec="aac",
        container_names=container_names,
    )


def _is_attached_picture(stream: dict[str, object]) -> bool:
    disposition = stream.get("disposition")
    return isinstance(disposition, dict) and disposition.get("attached_pic") in {1, "1"}


def _positive_float(value: object, *, allow_zero: bool) -> float | None:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or (not allow_zero and number == 0):
        return None
    return number


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, str | int):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _parse_fps(value: object) -> float | None:
    if not isinstance(value, str) or value in {"0/0", "N/A", ""}:
        return None
    try:
        number = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) and number > 0 else None
