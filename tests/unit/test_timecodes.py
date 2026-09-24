import math

import pytest

from app.utils.timecodes import format_timecode


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00.000"),
        (10, "00:00:10.000"),
        (61.234, "00:01:01.234"),
        (3661.001, "01:01:01.001"),
        (0.9995, "00:00:01.000"),
        (59.9995, "00:01:00.000"),
    ],
)
def test_format_timecode(seconds: float, expected: str) -> None:
    assert format_timecode(seconds) == expected


@pytest.mark.parametrize("seconds", [-0.001, math.nan, math.inf, -math.inf])
def test_format_timecode_rejects_invalid(seconds: float) -> None:
    with pytest.raises(ValueError):
        format_timecode(seconds)
