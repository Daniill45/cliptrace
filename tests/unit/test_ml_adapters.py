from __future__ import annotations

import math
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from faster_whisper.audio import decode_audio

from app.services.ml import adapters
from app.services.ml.adapters import FasterWhisperTranscriber, OnnxE5Embedder


class FakeWhisperModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def transcribe(
        self,
        audio: str,
        **kwargs: object,
    ) -> tuple[list[SimpleNamespace], SimpleNamespace]:
        self.calls.append((audio, kwargs))
        return (
            [
                SimpleNamespace(
                    start=0.0,
                    end=2.0,
                    text="fallback",
                    words=[
                        SimpleNamespace(start=0.0, end=0.8, word=" first "),
                        SimpleNamespace(start=0.8, end=1.6, word="second"),
                    ],
                ),
                SimpleNamespace(start=3.0, end=4.0, text=" final ", words=None),
            ],
            SimpleNamespace(language="ru", language_probability=0.97),
        )


def test_pinned_pyav_decodes_audio_with_faster_whisper(tmp_path: Path) -> None:
    audio_path = tmp_path / "silence.wav"
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 1_600)

    decoded = decode_audio(str(audio_path), sampling_rate=16_000)

    assert decoded.shape == (1_600,)


@pytest.mark.asyncio
async def test_faster_whisper_adapter_prefers_word_timestamps(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "video.mp4"
    media_path.touch()
    adapter = FasterWhisperTranscriber(language="ru", revision="test-revision")
    model = FakeWhisperModel()
    adapter._model = model

    transcription = await adapter.transcribe(media_path)

    assert [
        (segment.start_seconds, segment.end_seconds, segment.text)
        for segment in transcription.segments
    ] == [
        (0.0, 0.8, "first"),
        (0.8, 1.6, "second"),
        (3.0, 4.0, "final"),
    ]
    assert transcription.language == "ru"
    assert transcription.language_probability == 0.97
    assert adapter.model_id.endswith("@test-revision:cpu-int8:language-ru")
    assert model.calls == [
        (
            str(media_path),
            {
                "beam_size": 5,
                "vad_filter": True,
                "word_timestamps": True,
                "language": "ru",
            },
        )
    ]


@dataclass
class FakeEncoding:
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class FakeTokenizer:
    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []

    def encode_batch(self, inputs: Sequence[str]) -> list[FakeEncoding]:
        self.batches.append(tuple(inputs))
        return [FakeEncoding([1, 2, 999], [1, 1, 0], [0, 0, 0]) for _ in inputs]


@dataclass
class FakeOnnxInput:
    name: str


class FakeOnnxSession:
    def get_inputs(self) -> Sequence[FakeOnnxInput]:
        return (FakeOnnxInput("input_ids"), FakeOnnxInput("attention_mask"))

    def run(
        self,
        output_names: None,
        input_feed: dict[str, object],
    ) -> Sequence[object]:
        del output_names
        batch = input_feed["input_ids"]
        assert isinstance(batch, list)
        return [[[[float(token_id), 1.0] for token_id in row] for row in batch]]


class FakeNumpy:
    int64 = object()

    def array(self, value: object, *, dtype: object) -> object:
        assert dtype is self.int64
        return value


@pytest.mark.asyncio
async def test_onnx_e5_adapter_prefixes_pools_and_normalizes_without_model_download() -> None:
    tokenizer = FakeTokenizer()
    adapter = OnnxE5Embedder(dimension=2)
    adapter._runtime = adapters._E5Runtime(
        tokenizer=tokenizer,
        session=FakeOnnxSession(),
        numpy=FakeNumpy(),
    )

    query = await adapter.embed_query("  красная   машина ")
    documents = await adapter.embed_documents(("первый фрагмент", "second fragment"))

    assert tokenizer.batches == [
        ("query: красная машина",),
        ("passage: первый фрагмент", "passage: second fragment"),
    ]
    assert query == pytest.approx((1.5 / math.sqrt(3.25), 1.0 / math.sqrt(3.25)))
    assert documents[0] == pytest.approx(query)
    assert documents[1] == pytest.approx(query)
    assert math.sqrt(sum(value * value for value in query)) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_onnx_e5_adapter_rejects_blank_text_without_loading_runtime() -> None:
    adapter = OnnxE5Embedder(dimension=2)

    with pytest.raises(ValueError, match="blank"):
        await adapter.embed_query("   ")
    with pytest.raises(ValueError, match="blank"):
        await adapter.embed_documents(("valid", "   "))

    assert adapter._runtime is None
