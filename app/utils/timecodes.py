"""Human-readable timecode formatting."""

import math
from decimal import ROUND_HALF_UP, Decimal


def format_timecode(seconds: float) -> str:
    """Format finite non-negative seconds as HH:MM:SS.mmm."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be finite and non-negative")
    milliseconds = int(
        (Decimal(str(seconds)) * Decimal(1000)).quantize(Decimal("1"), ROUND_HALF_UP)
    )
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"
