"""Pure RFC 7233 single byte-range parser."""

from dataclasses import dataclass

MAX_RANGE_HEADER_LENGTH = 256


class InvalidByteRange(ValueError):
    """Raised when a Range value cannot select bytes from the resource."""


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def parse_byte_range(value: str, file_size: int) -> ByteRange:
    """Parse one strict bytes range, clipping an explicit end at EOF."""
    if file_size <= 0 or not value or len(value) > MAX_RANGE_HEADER_LENGTH:
        raise InvalidByteRange
    if value.strip() != value or not value.startswith("bytes="):
        raise InvalidByteRange
    spec = value[6:]
    if not spec or "," in spec or " " in spec or "\t" in spec or spec.count("-") != 1:
        raise InvalidByteRange
    start_text, end_text = spec.split("-", 1)
    if not start_text:
        if not end_text.isascii() or not end_text.isdigit():
            raise InvalidByteRange
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise InvalidByteRange
        length = min(suffix_length, file_size)
        return ByteRange(start=file_size - length, end=file_size - 1)
    if not start_text.isascii() or not start_text.isdigit():
        raise InvalidByteRange
    start = int(start_text)
    if start >= file_size:
        raise InvalidByteRange
    if not end_text:
        return ByteRange(start=start, end=file_size - 1)
    if not end_text.isascii() or not end_text.isdigit():
        raise InvalidByteRange
    end = int(end_text)
    if start > end:
        raise InvalidByteRange
    return ByteRange(start=start, end=min(end, file_size - 1))
