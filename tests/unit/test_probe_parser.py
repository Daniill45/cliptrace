from copy import deepcopy

import pytest

from app.core.errors import (
    InvalidVideoError,
    UnsupportedMediaTypeError,
    VideoAudioTrackRequiredError,
    VideoStreamRequiredError,
    VideoTooLongError,
    VideoTooShortError,
)
from app.services.media.probe import parse_probe_payload


@pytest.fixture
def valid_payload() -> dict[str, object]:
    return {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "31.250"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "profile": "High",
                "pix_fmt": "yuv420p",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "disposition": {"attached_pic": 0},
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }


def parse(payload: object):
    return parse_probe_payload(payload, max_duration_seconds=100, min_duration_seconds=11)


def test_parse_valid_probe_payload(valid_payload: dict[str, object]) -> None:
    metadata = parse(valid_payload)
    assert metadata.duration_seconds == 31.25
    assert metadata.video_codec == "h264"
    assert metadata.pixel_format == "yuv420p"
    assert metadata.audio_codec == "aac"
    assert metadata.fps == pytest.approx(29.97002997)
    assert "mp4" in metadata.container_names


def test_selects_first_supported_main_stream(valid_payload: dict[str, object]) -> None:
    payload = deepcopy(valid_payload)
    streams = payload["streams"]
    assert isinstance(streams, list)
    streams.insert(
        0,
        {
            "codec_type": "video",
            "codec_name": "mjpeg",
            "pix_fmt": "yuvj420p",
            "disposition": {"attached_pic": 1},
        },
    )
    assert parse(payload).video_codec == "h264"


@pytest.mark.parametrize("duration", ["N/A", "NaN", "Infinity", "-1", "0", None])
def test_rejects_invalid_duration(valid_payload: dict[str, object], duration: object) -> None:
    payload = deepcopy(valid_payload)
    assert isinstance(payload["format"], dict)
    payload["format"]["duration"] = duration
    with pytest.raises(InvalidVideoError):
        parse(payload)


def test_rejects_too_short_and_too_long(valid_payload: dict[str, object]) -> None:
    payload = deepcopy(valid_payload)
    assert isinstance(payload["format"], dict)
    payload["format"]["duration"] = "10.999"
    with pytest.raises(VideoTooShortError):
        parse(payload)
    payload["format"]["duration"] = "100.1"
    with pytest.raises(VideoTooLongError):
        parse(payload)


def test_rejects_missing_video(valid_payload: dict[str, object]) -> None:
    payload = deepcopy(valid_payload)
    assert isinstance(payload["streams"], list)
    payload["streams"] = [payload["streams"][1]]
    with pytest.raises(VideoStreamRequiredError):
        parse(payload)


def test_rejects_missing_audio(valid_payload: dict[str, object]) -> None:
    payload = deepcopy(valid_payload)
    assert isinstance(payload["streams"], list)
    payload["streams"] = [payload["streams"][0]]
    with pytest.raises(VideoAudioTrackRequiredError):
        parse(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("codec_name", "hevc"), ("pix_fmt", "yuv420p10le")],
)
def test_rejects_unsupported_video_codec(
    valid_payload: dict[str, object], field: str, value: str
) -> None:
    payload = deepcopy(valid_payload)
    assert isinstance(payload["streams"], list)
    assert isinstance(payload["streams"][0], dict)
    payload["streams"][0][field] = value
    with pytest.raises(UnsupportedMediaTypeError):
        parse(payload)


def test_container_alias_must_contain_exact_mp4(valid_payload: dict[str, object]) -> None:
    payload = deepcopy(valid_payload)
    assert isinstance(payload["format"], dict)
    payload["format"]["format_name"] = "notmp4container,mov"
    with pytest.raises(UnsupportedMediaTypeError):
        parse(payload)
