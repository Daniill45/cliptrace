from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import cast
from uuid import UUID

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from alembic.config import Config
from alembic.script import ScriptDirectory
from app.core.config import Settings
from app.db.session import create_engine
from app.main import create_app
from app.repositories.search import FragmentInput, SearchIndexRepository
from app.services.ml.contracts import Embedding
from tests.db_guard import EXPECTED_TEST_DATABASE


async def _upload(
    client: httpx.AsyncClient,
    content: bytes,
    *,
    filename: str = "sample.mp4",
    title: str | None = " Demo interview ",
) -> httpx.Response:
    data = {} if title is None else {"title": title}
    return await client.post(
        "/api/v1/videos",
        data=data,
        files={"file": (filename, content, "video/mp4")},
    )


@pytest.mark.asyncio
async def test_full_video_lifecycle_and_range_contract(
    client: httpx.AsyncClient,
    media_fixtures: dict[str, bytes],
) -> None:
    source = media_fixtures["valid"]
    created = await _upload(client, source)
    assert created.status_code == 201, created.text
    assert created.headers["x-request-id"]
    body = created.json()
    video_id = UUID(body["id"])
    assert body["status"] == "ready"
    assert body["title"] == "Demo interview"
    assert body["content_type"] == "video/mp4"
    assert body["size_bytes"] == len(source)
    assert body["media"]["duration_seconds"] == pytest.approx(12.0, abs=0.05)
    assert body["media"]["video_codec"] == "h264"
    assert body["media"]["audio_codec"] == "aac"
    assert "storage_key" not in created.text
    assert created.headers["location"] == f"/api/v1/videos/{video_id}"

    fetched = await client.get(f"/api/v1/videos/{video_id}")
    assert fetched.status_code == 200
    assert fetched.json() == body

    listed = await client.get("/api/v1/videos?limit=20&offset=0")
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert listed.json()["items"][0]["id"] == str(video_id)

    searched = await client.post(
        f"/api/v1/videos/{video_id}/search",
        json={"query": "  moment about failure  ", "limit": 5},
    )
    assert searched.status_code == 200, searched.text
    search = searched.json()
    assert search["query"] == "moment about failure"
    assert search["mode"] == "mock"
    assert search["engine_version"] == "mock-fixed-v3"
    assert len(search["results"]) == 1
    hit = search["results"][0]
    assert hit["start_seconds"] == 5.0
    assert hit["end_seconds"] == 10.0
    assert hit["start_timecode"] == "00:00:05.000"
    assert hit["end_timecode"] == "00:00:10.000"
    assert hit["score"] is None

    full = await client.get(hit["content_url"])
    assert full.status_code == 200
    assert full.content == source
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-length"] == str(len(source))
    assert full.headers["content-type"] == "video/mp4"

    for range_value, expected in [
        ("bytes=100-199", source[100:200]),
        ("bytes=0-0", source[:1]),
        ("bytes=100-", source[100:]),
        ("bytes=-500", source[-500:]),
    ]:
        ranged = await client.get(hit["content_url"], headers={"Range": range_value})
        assert ranged.status_code == 206
        assert ranged.content == expected
        assert int(ranged.headers["content-length"]) == len(expected)
        assert ranged.headers["content-type"] == "video/mp4"
        assert ranged.headers["content-range"].startswith("bytes ")

    unreachable = await client.get(hit["content_url"], headers={"Range": f"bytes={len(source)}-"})
    assert unreachable.status_code == 416
    assert unreachable.headers["content-range"] == f"bytes */{len(source)}"
    assert unreachable.json()["error"]["code"] == "RANGE_NOT_SATISFIABLE"
    assert unreachable.json()["error"]["request_id"] == unreachable.headers["x-request-id"]

    multiple = await client.get(hit["content_url"], headers={"Range": "bytes=0-1,4-5"})
    assert multiple.status_code == 416
    assert multiple.headers["content-range"] == f"bytes */{len(source)}"

    head = await client.head(hit["content_url"])
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-length"] == str(len(source))
    head_range = await client.head(hit["content_url"], headers={"Range": "bytes=100-199"})
    assert head_range.status_code == 206
    assert head_range.content == b""
    assert head_range.headers["content-range"] == f"bytes 100-199/{len(source)}"
    assert head_range.headers["content-length"] == "100"

    deleted = await client.delete(f"/api/v1/videos/{video_id}")
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert (await client.get(f"/api/v1/videos/{video_id}")).status_code == 404


@pytest.mark.asyncio
async def test_data_survives_new_application_lifespan(
    settings: Settings,
    clean_database: None,
    media_fixtures: dict[str, bytes],
) -> None:
    del clean_database
    first_app = create_app(settings)
    async with first_app.router.lifespan_context(first_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=first_app), base_url="http://test"
        ) as first:
            created = await _upload(first, media_fixtures["valid"])
            assert created.status_code == 201
            video_id = created.json()["id"]

    second_app = create_app(settings)
    async with second_app.router.lifespan_context(second_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=second_app), base_url="http://test"
        ) as second:
            assert (await second.get(f"/api/v1/videos/{video_id}")).status_code == 200
            content = await second.get(f"/api/v1/videos/{video_id}/content")
            assert content.content == media_fixtures["valid"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fixture_name", "expected_code"),
    [
        ("short", "VIDEO_TOO_SHORT"),
        ("no_audio", "VIDEO_AUDIO_TRACK_REQUIRED"),
        ("truncated", "INVALID_VIDEO"),
    ],
)
async def test_rejects_invalid_media_without_artifacts(
    client: httpx.AsyncClient,
    settings: Settings,
    media_fixtures: dict[str, bytes],
    fixture_name: str,
    expected_code: str,
) -> None:
    response = await _upload(client, media_fixtures[fixture_name])
    assert response.status_code in {415, 422}
    assert response.json()["error"]["code"] == expected_code
    listing = await client.get("/api/v1/videos")
    assert listing.json()["total"] == 0
    assert not list(settings.upload_dir.glob("*.part"))
    assert not list(settings.upload_dir.glob("*.mp4"))


@pytest.mark.asyncio
async def test_threshold_duration_is_accepted(
    client: httpx.AsyncClient, media_fixtures: dict[str, bytes]
) -> None:
    response = await _upload(client, media_fixtures["threshold"])
    assert response.status_code == 201, response.text
    assert response.json()["media"]["duration_seconds"] == pytest.approx(1.0, abs=0.1)
    video_id = response.json()["id"]
    searched = await client.post(
        f"/api/v1/videos/{video_id}/search",
        json={"query": "short mock fragment"},
    )
    assert searched.status_code == 200, searched.text
    result = searched.json()["results"][0]
    assert result["start_seconds"] == 0.0
    assert result["end_seconds"] == pytest.approx(1.0, abs=0.1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "filename", "expected_code"),
    [
        (b"", "empty.mp4", "INVALID_VIDEO"),
        (b"plain text", "fake.mp4", "INVALID_VIDEO"),
        (b"plain text", "fake.txt", "UNSUPPORTED_MEDIA_TYPE"),
    ],
)
async def test_rejects_empty_fake_and_wrong_extension(
    client: httpx.AsyncClient,
    content: bytes,
    filename: str,
    expected_code: str,
) -> None:
    response = await _upload(client, content, filename=filename)
    assert response.json()["error"]["code"] == expected_code
    assert (await client.get("/api/v1/videos")).json()["total"] == 0


@pytest.mark.asyncio
async def test_path_traversal_and_duplicate_display_names_are_safe(
    client: httpx.AsyncClient,
    settings: Settings,
    media_fixtures: dict[str, bytes],
) -> None:
    first = await _upload(client, media_fixtures["valid"], filename="../../видео.mp4")
    second = await _upload(client, media_fixtures["valid"], filename="../../видео.mp4")
    assert first.status_code == second.status_code == 201
    assert first.json()["original_filename"] == "видео.mp4"
    assert first.json()["id"] != second.json()["id"]
    files = list(settings.upload_dir.glob("*.mp4"))
    assert len(files) == 2
    expected_names = {f"{first.json()['id']}.mp4", f"{second.json()['id']}.mp4"}
    assert all(path.name in expected_names for path in files)


@pytest.mark.asyncio
async def test_validation_not_found_and_cors_use_stable_errors(
    client: httpx.AsyncClient,
) -> None:
    invalid_uuid = await client.get("/api/v1/videos/not-a-uuid")
    assert invalid_uuid.status_code == 422
    assert invalid_uuid.json()["error"]["code"] == "REQUEST_VALIDATION_ERROR"
    assert invalid_uuid.headers["x-request-id"] == invalid_uuid.json()["error"]["request_id"]

    unknown = await client.post(
        "/api/v1/videos/00000000-0000-0000-0000-000000000001/search",
        json={"query": "valid"},
    )
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "VIDEO_NOT_FOUND"

    for query in ["   ", "x" * 501]:
        invalid_query = await client.post(
            "/api/v1/videos/00000000-0000-0000-0000-000000000001/search",
            json={"query": query},
        )
        assert invalid_query.status_code == 422
        assert invalid_query.json()["error"]["code"] == "REQUEST_VALIDATION_ERROR"

    allowed = await client.options(
        "/api/v1/videos",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    denied = await client.options(
        "/api/v1/videos",
        headers={
            "Origin": "https://unconfigured.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in denied.headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path", "status_code", "code", "message"),
    [
        ("GET", "/api/v1/not-a-route", 404, "ROUTE_NOT_FOUND", "Route was not found"),
        ("POST", "/health/live", 405, "METHOD_NOT_ALLOWED", "Method is not allowed"),
    ],
)
async def test_framework_http_errors_use_stable_envelope(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    status_code: int,
    code: str,
    message: str,
) -> None:
    response = await client.request(method, path)

    assert response.status_code == status_code
    assert response.json() == {
        "error": {
            "code": code,
            "message": message,
            "details": {},
            "request_id": response.headers["x-request-id"],
        }
    }
    if status_code == 405:
        assert "GET" in response.headers["allow"].split(", ")


@pytest.mark.asyncio
async def test_missing_truncated_and_symlinked_file_return_gone(
    client: httpx.AsyncClient,
    settings: Settings,
    media_fixtures: dict[str, bytes],
) -> None:
    for mutation in ("missing", "truncated", "symlink"):
        created = await _upload(client, media_fixtures["valid"])
        assert created.status_code == 201
        video_id = created.json()["id"]
        path = settings.upload_dir / f"{video_id}.mp4"
        if mutation == "missing":
            path.unlink()
        elif mutation == "truncated":
            path.write_bytes(b"short")
        else:
            backup = settings.upload_dir / f"{video_id}.backup"
            path.rename(backup)
            path.symlink_to(backup)

        content = await client.get(f"/api/v1/videos/{video_id}/content")
        search = await client.post(
            f"/api/v1/videos/{video_id}/search", json={"query": "valid query"}
        )
        assert content.status_code == search.status_code == 410
        assert content.json()["error"]["code"] == "VIDEO_FILE_MISSING"
        assert str(settings.upload_dir) not in content.text
        assert str(settings.upload_dir) not in search.text
        assert (await client.delete(f"/api/v1/videos/{video_id}")).status_code == 204


@pytest.mark.asyncio
async def test_oversized_upload_is_rejected_without_row_or_file(
    settings: Settings,
    clean_database: None,
    media_fixtures: dict[str, bytes],
) -> None:
    del clean_database
    limited = settings.model_copy(
        update={"max_upload_size_bytes": 1024, "upload_chunk_size_bytes": 512}
    )
    application = create_app(limited)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application, raise_app_exceptions=False),
            base_url="http://test",
        ) as local_client:
            response = await _upload(local_client, media_fixtures["valid"])
            assert response.status_code == 413
            assert response.json()["error"]["code"] == "UPLOAD_TOO_LARGE"
            assert response.headers["x-request-id"] == response.json()["error"]["request_id"]
            assert (await local_client.get("/api/v1/videos")).json()["total"] == 0
    assert not list(limited.upload_dir.glob("*.part"))
    assert not list(limited.upload_dir.glob("*.mp4"))


@pytest.mark.asyncio
async def test_health_endpoints(client: httpx.AsyncClient) -> None:
    live = await client.get("/health/live")
    ready = await client.get("/health/ready")
    assert live.status_code == 200 and live.json() == {"status": "ok"}
    assert ready.status_code == 200 and ready.json() == {"status": "ready"}


@pytest.mark.asyncio
async def test_frontend_assets_do_not_shadow_api_or_docs(client: httpx.AsyncClient) -> None:
    frontend = await client.get("/")
    assert frontend.status_code == 200
    assert frontend.headers["content-type"].startswith("text/html")
    assert "ClipTrace" in frontend.text
    assert "ML-поиск" in frontend.text and "по смыслу речи" in frontend.text

    stylesheet = await client.get("/assets/styles.css")
    script = await client.get("/assets/app.js")
    assert stylesheet.status_code == script.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert "javascript" in script.headers["content-type"]
    assert 'const API = "/api/v1"' in script.text

    assert (await client.get("/docs")).status_code == 200
    assert (await client.get("/health/live")).status_code == 200
    assert (await client.get("/api/v1/videos")).status_code == 200


@pytest.mark.asyncio
async def test_upload_rejects_multiple_file_fields(
    client: httpx.AsyncClient,
    media_fixtures: dict[str, bytes],
) -> None:
    response = await client.post(
        "/api/v1/videos",
        files=[
            ("file", ("one.mp4", media_fixtures["valid"], "video/mp4")),
            ("file", ("two.mp4", media_fixtures["valid"], "video/mp4")),
        ],
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_ERROR"
    assert (await client.get("/api/v1/videos")).json()["total"] == 0


@pytest.mark.asyncio
async def test_openapi_describes_single_binary_upload(client: httpx.AsyncClient) -> None:
    response = await client.get("/openapi.json")
    schema = response.json()
    upload_schema = schema["paths"]["/api/v1/videos"]["post"]["requestBody"]
    multipart = upload_schema["content"]["multipart/form-data"]["schema"]
    assert multipart["required"] == ["file"]
    assert multipart["properties"]["file"] == {"type": "string", "format": "binary"}

    content_path = schema["paths"]["/api/v1/videos/{video_id}/content"]
    get_responses = content_path["get"]["responses"]
    head_responses = content_path["head"]["responses"]
    for responses in (get_responses, head_responses):
        assert {"200", "206", "416", "422"} <= responses.keys()
        assert "Content-Range" in responses["206"]["headers"]
        assert "Content-Range" in responses["416"]["headers"]
        assert responses["416"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ErrorResponse"
        }
        assert responses["422"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ErrorResponse"
        }
    for status_code in ("200", "206"):
        assert set(get_responses[status_code]["content"]) == {"video/mp4"}
        assert get_responses[status_code]["content"]["video/mp4"]["schema"] == {
            "type": "string",
            "format": "binary",
        }
        assert "content" not in head_responses[status_code]
    for operation in (content_path["get"], content_path["head"]):
        range_parameters = [
            item
            for item in operation["parameters"]
            if item["in"] == "header" and item["name"].lower() == "range"
        ]
        assert len(range_parameters) == 1


@pytest.mark.asyncio
async def test_integration_database_is_isolated_and_migrated(
    settings: Settings,
    clean_database: None,
) -> None:
    del clean_database
    expected_heads = set(ScriptDirectory.from_config(Config("alembic.ini")).get_heads())
    engine = create_engine(settings.database_url)
    try:
        async with engine.connect() as connection:
            database_name = await connection.scalar(text("SELECT current_database()"))
            applied_revisions = set(
                (
                    await connection.execute(text("SELECT version_num FROM alembic_version"))
                ).scalars()
            )
    finally:
        await engine.dispose()

    assert database_name == EXPECTED_TEST_DATABASE
    assert applied_revisions == expected_heads


class _FakeSemanticEmbedder:
    model_id = "fake-e5-integration-v1"
    dimension = 384

    async def embed_query(self, text: str) -> Embedding:
        assert text == "первый сюжет"
        return (1.0, *([0.0] * 383))

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Embedding]:
        raise AssertionError(f"online API must not index documents: {texts!r}")


@pytest.mark.asyncio
async def test_semantic_index_lifecycle_and_exact_pgvector_search(
    settings: Settings,
    clean_database: None,
    media_fixtures: dict[str, bytes],
) -> None:
    del clean_database
    semantic_settings = settings.model_copy(update={"search_engine": "semantic"})
    application = create_app(semantic_settings)
    application.state.query_embedder = _FakeSemanticEmbedder()

    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await _upload(client, media_fixtures["valid"])
            assert created.status_code == 201, created.text
            video_id = UUID(created.json()["id"])

            pending = await client.get(f"/api/v1/videos/{video_id}/search-index")
            assert pending.status_code == 200
            assert pending.json()["status"] == "pending"
            assert pending.json()["fragment_count"] == 0

            blocked = await client.post(
                f"/api/v1/videos/{video_id}/search",
                json={"query": "первый сюжет", "limit": 2},
            )
            assert blocked.status_code == 409
            assert blocked.json()["error"]["code"] == "SEARCH_INDEX_NOT_READY"

            factory = cast(async_sessionmaker[AsyncSession], application.state.session_factory)
            async with factory() as session:
                async with session.begin():
                    repository = SearchIndexRepository(session)
                    claimed = await repository.claim_next(
                        worker_id="integration-worker",
                        lease_duration=timedelta(minutes=1),
                    )
                    assert claimed is not None and claimed.video_id == video_id
            async with factory() as session:
                async with session.begin():
                    await SearchIndexRepository(session).requeue(
                        video_id,
                        worker_id="integration-worker",
                    )
            requeued = await client.get(f"/api/v1/videos/{video_id}/search-index")
            assert requeued.json()["status"] == "pending"
            async with factory() as session:
                async with session.begin():
                    claimed = await SearchIndexRepository(session).claim_next(
                        worker_id="integration-worker",
                        lease_duration=timedelta(minutes=1),
                    )
                    assert claimed is not None and claimed.video_id == video_id
            first_embedding = (1.0, *([0.0] * 383))
            second_embedding = (0.0, 1.0, *([0.0] * 382))
            async with factory() as session:
                async with session.begin():
                    await SearchIndexRepository(session).complete(
                        video_id,
                        worker_id="integration-worker",
                        fragments=(
                            FragmentInput(0, 2.0, 4.0, "первый сюжет", first_embedding, "ru"),
                            FragmentInput(1, 7.0, 9.0, "второй сюжет", second_embedding, "ru"),
                        ),
                        pipeline_version="integration-v1",
                        asr_model_version="fake-asr-v1",
                        embedding_model_version="fake-e5-v1",
                    )

            ready = await client.get(f"/api/v1/videos/{video_id}/search-index")
            assert ready.status_code == 200
            assert ready.json()["status"] == "ready"
            assert ready.json()["fragment_count"] == 2
            assert ready.json()["indexed_at"] is not None

            searched = await client.post(
                f"/api/v1/videos/{video_id}/search",
                json={"query": "первый сюжет", "limit": 2},
            )
            assert searched.status_code == 200, searched.text
            payload = searched.json()
            assert payload["mode"] == "semantic-transcript"
            assert [item["preview_text"] for item in payload["results"]] == [
                "первый сюжет",
                "второй сюжет",
            ]
            assert payload["results"][0]["score"] == pytest.approx(1.0)
            assert payload["results"][0]["start_seconds"] == 1.0
            assert payload["results"][0]["end_seconds"] == 5.0

            reindexed = await client.post(f"/api/v1/videos/{video_id}/search-index/reindex")
            assert reindexed.status_code == 202
            assert reindexed.json()["status"] == "pending"
            assert reindexed.json()["fragment_count"] == 2
            blocked_again = await client.post(
                f"/api/v1/videos/{video_id}/search",
                json={"query": "первый сюжет"},
            )
            assert blocked_again.status_code == 409

            assert (await client.delete(f"/api/v1/videos/{video_id}")).status_code == 204
            assert (await client.get(f"/api/v1/videos/{video_id}/search-index")).status_code == 404
