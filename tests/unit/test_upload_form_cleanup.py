from __future__ import annotations

from io import BytesIO

import pytest
from starlette.datastructures import FormData, Headers, UploadFile
from starlette.requests import Request

from app.api.dependencies import parse_upload_form
from app.core.errors import RequestValidationDomainError


class TrackingFormData(FormData):
    def __init__(self, items: list[tuple[str, str | UploadFile]]) -> None:
        super().__init__(items)
        self.close_called = False

    async def close(self) -> None:
        self.close_called = True
        await super().close()


def make_upload(filename: str) -> UploadFile:
    return UploadFile(
        BytesIO(b"video"),
        filename=filename,
        headers=Headers({"content-type": "video/mp4"}),
    )


def request_with_form(form: FormData) -> Request:
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    request._form = form
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["missing_file", "duplicate_file", "duplicate_title", "file_as_title"],
)
async def test_invalid_upload_form_closes_form_and_all_uploads(case: str) -> None:
    primary = make_upload("primary.mp4")
    secondary = make_upload("secondary.mp4")
    items: list[tuple[str, str | UploadFile]]
    if case == "missing_file":
        items = [("title", "demo")]
    elif case == "duplicate_file":
        items = [("file", primary), ("file", secondary)]
    elif case == "duplicate_title":
        items = [("file", primary), ("title", "one"), ("title", "two")]
    else:
        items = [("file", primary), ("title", secondary)]
    form = TrackingFormData(items)
    dependency = parse_upload_form(request_with_form(form))

    with pytest.raises(RequestValidationDomainError):
        await anext(dependency)

    assert form.close_called
    for _, item in form.multi_items():
        if isinstance(item, UploadFile):
            assert item.file.closed


@pytest.mark.asyncio
async def test_valid_upload_form_transfers_open_file_ownership_to_service() -> None:
    upload = make_upload("demo.mp4")
    form = TrackingFormData([("file", upload), ("title", "Demo")])
    dependency = parse_upload_form(request_with_form(form))

    parsed = await anext(dependency)

    assert parsed.file is upload
    assert parsed.title == "Demo"
    assert not form.close_called
    assert not upload.file.closed
    await dependency.aclose()
    assert form.close_called
    assert upload.file.closed
