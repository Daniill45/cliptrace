from pathlib import Path

import pytest

from app.core.errors import InvalidVideoError
from app.services.media.probe import MediaProbe


@pytest.mark.asyncio
async def test_media_probe_timeout_terminates_process(tmp_path: Path) -> None:
    executable = tmp_path / "slow-probe"
    executable.write_text("#!/bin/sh\nexec sleep 30\n", encoding="utf-8")
    executable.chmod(0o700)
    media = tmp_path / "input.mp4"
    media.write_bytes(b"not relevant")
    probe = MediaProbe(
        binary=str(executable),
        timeout_seconds=0.05,
        max_duration_seconds=100,
        min_duration_seconds=11,
    )
    with pytest.raises(InvalidVideoError):
        await probe.probe(media)
