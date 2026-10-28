"""Fail-closed protection for destructive integration-test database cleanup."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import AsyncConnection

EXPECTED_TEST_DATABASE = "cliptrace_test"


def require_test_database_url(value: str | None) -> str:
    """Accept only an explicit DSN targeting ClipTrace's isolated test database."""
    if not value:
        raise RuntimeError("TEST_DATABASE_URL is required for integration tests")
    try:
        database_name = make_url(value).database
    except (ArgumentError, TypeError, ValueError) as exc:
        raise RuntimeError("TEST_DATABASE_URL must be a valid database URL") from exc
    if database_name != EXPECTED_TEST_DATABASE:
        raise RuntimeError(
            f"Refusing integration tests outside {EXPECTED_TEST_DATABASE!r}; "
            f"configured database is {database_name!r}"
        )
    return value


async def truncate_videos(connection: AsyncConnection) -> None:
    """Verify the connected database itself before deleting any test rows."""
    database_name = await connection.scalar(text("SELECT current_database()"))
    if database_name != EXPECTED_TEST_DATABASE:
        raise RuntimeError(
            f"Refusing TRUNCATE outside {EXPECTED_TEST_DATABASE!r}; "
            f"connected database is {database_name!r}"
        )
    await connection.execute(
        text("TRUNCATE TABLE video_fragments, video_search_indexes, videos RESTART IDENTITY")
    )
