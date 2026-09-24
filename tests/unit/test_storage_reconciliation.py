from __future__ import annotations

import os
import time
from pathlib import Path
from uuid import uuid4

import pytest

from app.services.media.storage import LocalStorage


@pytest.mark.asyncio
async def test_startup_reconciliation_repairs_and_cleans_only_safe_patterns(
    tmp_path: Path,
) -> None:
    storage = LocalStorage(
        tmp_path,
        max_size_bytes=1024,
        chunk_size_bytes=128,
        temp_max_age_seconds=10,
    )
    await storage.initialize()
    referenced_key = f"{uuid4()}.mp4"
    tombstone = tmp_path / f"{referenced_key}.deleting"
    tombstone.write_bytes(b"restore me")

    orphan = tmp_path / f"{uuid4()}.mp4"
    orphan.write_bytes(b"orphan")
    stale_part = tmp_path / f"{uuid4()}.part"
    stale_part.write_bytes(b"partial")
    unrelated = tmp_path / "do-not-touch.txt"
    unrelated.write_bytes(b"safe")
    old = time.time() - 20
    os.utime(orphan, (old, old))
    os.utime(stale_part, (old, old))

    await storage.reconcile({referenced_key})

    assert (tmp_path / referenced_key).read_bytes() == b"restore me"
    assert not tombstone.exists()
    assert not orphan.exists()
    assert not stale_part.exists()
    assert unrelated.read_bytes() == b"safe"


@pytest.mark.asyncio
async def test_reconciliation_removes_unreferenced_tombstone(tmp_path: Path) -> None:
    storage = LocalStorage(
        tmp_path,
        max_size_bytes=1024,
        chunk_size_bytes=128,
        temp_max_age_seconds=10,
    )
    await storage.initialize()
    tombstone = tmp_path / f"{uuid4()}.mp4.deleting"
    tombstone.write_bytes(b"delete me")
    await storage.reconcile(set())
    assert not tombstone.exists()
