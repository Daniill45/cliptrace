import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_default_mock_invariants() -> None:
    settings = Settings(_env_file=None)
    assert settings.mock_result_start_seconds == 5.0
    assert settings.mock_result_duration_seconds == 5.0
    assert settings.min_fragment_duration_seconds == 5.0
    assert settings.min_video_duration_seconds == 1.0


@pytest.mark.parametrize(
    "override",
    [
        {"upload_chunk_size_bytes": 2, "max_upload_size_bytes": 1},
        {"min_video_duration_seconds": 20, "max_video_duration_seconds": 10},
        {"mock_result_start_seconds": -1},
        {"mock_result_duration_seconds": 0},
        {"min_fragment_duration_seconds": 0},
        {"mock_result_duration_seconds": 0.5, "min_fragment_duration_seconds": 1},
        {
            "mock_result_start_seconds": 8,
            "mock_result_duration_seconds": 3,
            "min_fragment_duration_seconds": 1,
            "max_video_duration_seconds": 10,
        },
        {"cors_origins": ["http://localhost:3000/path"]},
        {"app_env": "production", "cors_origins": ["*"]},
    ],
)
def test_invalid_settings_fail_fast(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **override)
