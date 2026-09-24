import pytest

from app.services.media.range_requests import InvalidByteRange, parse_byte_range


@pytest.mark.parametrize(
    ("header", "size", "start", "end"),
    [
        ("bytes=0-0", 1000, 0, 0),
        ("bytes=0-99", 1000, 0, 99),
        ("bytes=100-", 1000, 100, 999),
        ("bytes=-500", 1000, 500, 999),
        ("bytes=100-999", 200, 100, 199),
        ("bytes=-500", 200, 0, 199),
    ],
)
def test_parse_byte_range(header: str, size: int, start: int, end: int) -> None:
    result = parse_byte_range(header, size)
    assert (result.start, result.end, result.length) == (start, end, end - start + 1)


@pytest.mark.parametrize(
    "header",
    [
        "bytes=1000-",
        "bytes=10-9",
        "",
        "items=0-1",
        "bytes=--1",
        "bytes=-0",
        "bytes=0-1,4-5",
        " bytes=0-1",
        "bytes=0 -1",
    ],
)
def test_parse_byte_range_rejects_invalid(header: str) -> None:
    with pytest.raises(InvalidByteRange):
        parse_byte_range(header, 1000)


def test_parse_byte_range_rejects_long_header() -> None:
    with pytest.raises(InvalidByteRange):
        parse_byte_range("bytes=" + "1" * 300 + "-", 1000)
