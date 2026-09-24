"""Local filesystem storage with strict key and lifecycle handling."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

import anyio
from starlette.datastructures import UploadFile

from app.core.errors import InvalidVideoError, UploadTooLargeError, VideoFileMissingError

FINAL_KEY_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.mp4$")
PART_RE = re.compile(r"^[0-9a-f-]{36}\.part$")


@dataclass(frozen=True)
class StagedUpload:
    """A complete but not yet validated upload inside the storage root."""

    path: Path
    size_bytes: int
    sha256: str


class LocalStorage:
    """Own physical video files while exposing only opaque storage keys."""

    def __init__(
        self,
        root: Path,
        *,
        max_size_bytes: int,
        chunk_size_bytes: int,
        temp_max_age_seconds: int,
    ) -> None:
        self._root = root
        self._max_size = max_size_bytes
        self._chunk_size = chunk_size_bytes
        self._temp_max_age = temp_max_age_seconds

    @property
    def root(self) -> Path:
        """Return the configured root for internal startup coordination only."""
        return self._root

    async def initialize(self) -> None:
        """Create the storage directory and resolve it once."""
        await anyio.to_thread.run_sync(lambda: self._root.mkdir(parents=True, exist_ok=True))
        self._root = self._root.resolve(strict=True)

    def _path_for_key(self, storage_key: str) -> Path:
        if not FINAL_KEY_RE.fullmatch(storage_key):
            raise VideoFileMissingError
        path = self._root / storage_key
        if path.parent != self._root:
            raise VideoFileMissingError
        return path

    async def stage_upload(self, upload: UploadFile) -> StagedUpload:
        """Stream an upload into a random temporary file while hashing and limiting it."""
        temp_path = self._root / f"{uuid4()}.part"
        digest = hashlib.sha256()
        size = 0
        try:
            async with await anyio.open_file(temp_path, "xb") as output:
                while True:
                    chunk = await upload.read(self._chunk_size)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self._max_size:
                        raise UploadTooLargeError
                    digest.update(chunk)
                    await output.write(chunk)
                await output.flush()
            if size == 0:
                raise InvalidVideoError("The uploaded file is empty")
            return StagedUpload(path=temp_path, size_bytes=size, sha256=digest.hexdigest())
        except BaseException:
            await self._unlink_if_exists(temp_path)
            raise

    async def finalize(self, staged: StagedUpload, video_id: UUID) -> str:
        """Atomically move a validated temporary file to its UUID key."""
        storage_key = f"{video_id}.mp4"
        final_path = self._path_for_key(storage_key)
        await anyio.to_thread.run_sync(staged.path.rename, final_path)
        return storage_key

    async def discard_staged(self, staged: StagedUpload) -> None:
        """Remove a staged upload after validation failure."""
        await self._unlink_if_exists(staged.path)

    async def remove_final(self, storage_key: str) -> None:
        """Remove one final file without following symlinks."""
        await self._unlink_if_exists(self._path_for_key(storage_key))

    async def verify(self, storage_key: str, expected_size: int) -> int:
        """Verify regular-file identity and return the current size."""
        path = self._path_for_key(storage_key)

        def check() -> int:
            try:
                info = path.lstat()
            except FileNotFoundError as exc:
                raise VideoFileMissingError from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise VideoFileMissingError
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise VideoFileMissingError from exc
            if resolved.parent != self._root or info.st_size != expected_size:
                raise VideoFileMissingError
            return info.st_size

        return await anyio.to_thread.run_sync(check)

    async def open_verified(self, storage_key: str, expected_size: int) -> BinaryIO:
        """Open and fstat a file descriptor without following a final symlink."""
        path = self._path_for_key(storage_key)

        def open_file() -> BinaryIO:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(path, flags)
            except OSError as exc:
                raise VideoFileMissingError from exc
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
                    raise VideoFileMissingError
                return os.fdopen(descriptor, "rb", closefd=True)
            except BaseException:
                os.close(descriptor)
                raise

        return await anyio.to_thread.run_sync(open_file)

    async def stream(
        self,
        file_handle: BinaryIO,
        *,
        start: int,
        length: int,
    ) -> AsyncIterator[bytes]:
        """Yield exactly a selected byte interval and always close the descriptor."""
        remaining = length
        try:
            await anyio.to_thread.run_sync(file_handle.seek, start)
            while remaining:
                chunk = await anyio.to_thread.run_sync(
                    file_handle.read, min(self._chunk_size, remaining)
                )
                if not chunk:
                    raise VideoFileMissingError
                remaining -= len(chunk)
                yield chunk
        finally:
            await anyio.to_thread.run_sync(file_handle.close)

    async def move_to_tombstone(self, storage_key: str) -> str | None:
        """Atomically hide a final path before its database row is committed away."""
        path = self._path_for_key(storage_key)
        tombstone = self._root / f"{storage_key}.deleting"

        def move() -> str | None:
            try:
                info = path.lstat()
            except FileNotFoundError:
                return None
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                return None
            try:
                tombstone.lstat()
            except FileNotFoundError:
                pass
            else:
                raise OSError("A delete tombstone already exists")
            path.rename(tombstone)
            return tombstone.name

        return await anyio.to_thread.run_sync(move)

    async def restore_tombstone(self, storage_key: str, tombstone_key: str) -> None:
        """Best-effort rollback of a pre-delete rename."""
        path = self._path_for_key(storage_key)
        tombstone = self._root / tombstone_key

        def restore() -> None:
            if tombstone.parent != self._root:
                return
            try:
                tombstone.lstat()
            except FileNotFoundError:
                return
            try:
                path.lstat()
            except FileNotFoundError:
                tombstone.rename(path)

        await anyio.to_thread.run_sync(restore)

    async def remove_tombstone(self, tombstone_key: str) -> None:
        """Unlink a tombstone without resolving it."""
        path = self._root / tombstone_key
        if path.parent != self._root or not tombstone_key.endswith(".mp4.deleting"):
            return
        await self._unlink_if_exists(path)

    async def readiness_probe(self) -> None:
        """Verify create/write/remove access without touching business files."""
        path = self._root / f"{uuid4()}.probe"
        try:
            async with await anyio.open_file(path, "xb") as output:
                await output.write(b"ok")
                await output.flush()
        finally:
            await self._unlink_if_exists(path)

    async def reconcile(self, referenced_keys: set[str]) -> None:
        """Clean stale safe-pattern files and repair interrupted deletions."""
        await anyio.to_thread.run_sync(self._reconcile_sync, referenced_keys, time.time())

    def _reconcile_sync(self, referenced_keys: set[str], now: float) -> None:
        for candidate in self._root.iterdir():
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                continue
            name = candidate.name
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                continue
            if PART_RE.fullmatch(name):
                if now - info.st_mtime > self._temp_max_age:
                    candidate.unlink(missing_ok=True)
                continue
            if name.endswith(".deleting"):
                original_key = name.removesuffix(".deleting")
                if not FINAL_KEY_RE.fullmatch(original_key):
                    continue
                original = self._root / original_key
                if original_key in referenced_keys and not original.exists():
                    candidate.rename(original)
                else:
                    candidate.unlink(missing_ok=True)
                continue
            if FINAL_KEY_RE.fullmatch(name) and name not in referenced_keys:
                if now - info.st_mtime > self._temp_max_age:
                    candidate.unlink(missing_ok=True)

    async def _unlink_if_exists(self, path: Path) -> None:
        def unlink() -> None:
            try:
                path.unlink()
            except FileNotFoundError:
                return

        await anyio.to_thread.run_sync(unlink)
