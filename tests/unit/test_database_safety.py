from __future__ import annotations

from collections.abc import Sequence

import pytest
from sqlalchemy.sql.elements import TextClause

from tests.db_guard import EXPECTED_TEST_DATABASE, require_test_database_url, truncate_videos


class FakeConnection:
    def __init__(self, database_name: str) -> None:
        self.database_name = database_name
        self.scalar_statements: list[str] = []
        self.execute_statements: list[str] = []

    async def scalar(self, statement: TextClause) -> str:
        self.scalar_statements.append(statement.text)
        return self.database_name

    async def execute(self, statement: TextClause) -> None:
        self.execute_statements.append(statement.text)


@pytest.mark.parametrize(
    "database_url",
    [
        None,
        "",
        "postgresql+asyncpg://cliptrace:secret@db:5432/cliptrace",
        "postgresql+asyncpg://cliptrace:secret@db:5432/postgres",
        "postgresql+asyncpg://cliptrace:secret@db:5432",
    ],
)
def test_test_database_url_guard_fails_closed_without_leaking_dsn(
    database_url: str | None,
) -> None:
    with pytest.raises(RuntimeError) as error:
        require_test_database_url(database_url)
    assert "secret" not in str(error.value)
    assert "postgresql" not in str(error.value)


def test_test_database_url_guard_accepts_only_expected_database() -> None:
    database_url = f"postgresql+asyncpg://cliptrace:secret@test-db:5432/{EXPECTED_TEST_DATABASE}"
    assert require_test_database_url(database_url) == database_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("database_name", "expected_statements"),
    [
        ("cliptrace", []),
        (
            EXPECTED_TEST_DATABASE,
            ["TRUNCATE TABLE video_fragments, video_search_indexes, videos RESTART IDENTITY"],
        ),
    ],
)
async def test_runtime_database_guard_checks_connection_before_truncate(
    database_name: str,
    expected_statements: Sequence[str],
) -> None:
    connection = FakeConnection(database_name)
    if database_name == EXPECTED_TEST_DATABASE:
        await truncate_videos(connection)  # type: ignore[arg-type]
    else:
        with pytest.raises(RuntimeError, match="Refusing TRUNCATE"):
            await truncate_videos(connection)  # type: ignore[arg-type]
    assert connection.scalar_statements == ["SELECT current_database()"]
    assert connection.execute_statements == expected_statements
