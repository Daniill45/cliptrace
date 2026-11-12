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
from app.services.ml.adapters import (
    FasterWhisperTranscriber,
    OnnxCrossEncoderReranker,
    OnnxE5Embedder,
)


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


class FakeRerankerTokenizer:
    def __init__(self) -> None:
        self.batches: list[tuple[tuple[str, str], ...]] = []
        self.truncation_lengths: list[int] = []
        self.padding_calls: list[tuple[str, int, str]] = []
        self._next_token_id = 1

    def token_to_id(self, token: str) -> int | None:
        return 1 if token == "<pad>" else None

    def enable_truncation(self, max_length: int) -> None:
        self.truncation_lengths.append(max_length)

    def enable_padding(
        self,
        *,
        direction: str,
        pad_id: int,
        pad_token: str,
    ) -> None:
        self.padding_calls.append((direction, pad_id, pad_token))

    def encode_batch(
        self,
        inputs: Sequence[str | tuple[str, str]],
    ) -> list[FakeEncoding]:
        pairs: list[tuple[str, str]] = []
        encodings: list[FakeEncoding] = []
        for value in inputs:
            assert isinstance(value, tuple)
            pairs.append(value)
            token_id = self._next_token_id
            self._next_token_id += 1
            encodings.append(
                FakeEncoding(
                    ids=[token_id, 999, 0],
                    attention_mask=[1, 1, 0],
                    type_ids=[0, 1, 0],
                )
            )
        self.batches.append(tuple(pairs))
        return encodings


@dataclass
class FakeArray:
    value: object

    def tolist(self) -> object:
        return self.value


class FakeRerankerSession:
    def __init__(
        self,
        input_names: Sequence[str] = (
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ),
    ) -> None:
        self._input_names = tuple(input_names)
        self.input_feeds: list[dict[str, object]] = []

    def get_inputs(self) -> Sequence[FakeOnnxInput]:
        return tuple(FakeOnnxInput(name) for name in self._input_names)

    def run(
        self,
        output_names: None,
        input_feed: dict[str, object],
    ) -> Sequence[object]:
        assert output_names is None
        self.input_feeds.append(input_feed)
        input_ids = input_feed["input_ids"]
        assert isinstance(input_ids, list)
        return (FakeArray([[float(row[0])] for row in input_ids]),)


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


@pytest.mark.asyncio
async def test_onnx_reranker_tokenizes_pairs_in_batches_and_extracts_logits() -> None:
    tokenizer = FakeRerankerTokenizer()
    session = FakeRerankerSession()
    adapter = OnnxCrossEncoderReranker(batch_size=2)
    adapter._runtime = adapters._CrossEncoderRuntime(
        tokenizer=tokenizer,
        session=session,
        numpy=FakeNumpy(),
    )

    scores = await adapter.score(
        "  найти   Docker ",
        (" первый   фрагмент ", "second fragment", " третий фрагмент "),
    )

    assert scores == (1.0, 2.0, 3.0)
    assert tokenizer.batches == [
        (
            ("найти Docker", "первый фрагмент"),
            ("найти Docker", "second fragment"),
        ),
        (("найти Docker", "третий фрагмент"),),
    ]
    assert session.input_feeds == [
        {
            "input_ids": [[1, 999, 0], [2, 999, 0]],
            "attention_mask": [[1, 1, 0], [1, 1, 0]],
            "token_type_ids": [[0, 1, 0], [0, 1, 0]],
        },
        {
            "input_ids": [[3, 999, 0]],
            "attention_mask": [[1, 1, 0]],
            "token_type_ids": [[0, 1, 0]],
        },
    ]


@pytest.mark.asyncio
async def test_onnx_reranker_returns_empty_without_loading_runtime() -> None:
    adapter = OnnxCrossEncoderReranker()

    scores = await adapter.score("valid query", ())

    assert scores == ()
    assert adapter._runtime is None


@pytest.mark.asyncio
async def test_onnx_reranker_rejects_blank_inputs_without_loading_runtime() -> None:
    adapter = OnnxCrossEncoderReranker()

    with pytest.raises(ValueError, match="blank"):
        await adapter.score("   ", ("valid document",))
    with pytest.raises(ValueError, match="blank"):
        await adapter.score("valid query", ("valid document", "   "))

    assert adapter._runtime is None


@pytest.mark.parametrize(
    ("machine", "expected_file"),
    [
        ("arm64", "model_qint8_arm64.onnx"),
        ("x86_64", "model.onnx"),
        ("riscv64", "model.onnx"),
    ],
)
def test_onnx_reranker_auto_selects_artifact_for_cpu_architecture(
    monkeypatch: pytest.MonkeyPatch,
    machine: str,
    expected_file: str,
) -> None:
    monkeypatch.setattr(adapters.platform, "machine", lambda: machine)

    adapter = OnnxCrossEncoderReranker(onnx_file="auto", revision="test-revision")

    assert adapter.model_id.endswith(f"@test-revision:{expected_file}")
    assert adapter._runtime is None


def test_onnx_reranker_loads_auto_artifact_from_mocked_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(adapters.platform, "machine", lambda: "arm64")
    snapshot_root = tmp_path / "snapshot"
    cache_folder = tmp_path / "cache"
    tokenizer = FakeRerankerTokenizer()
    session = FakeRerankerSession(("input_ids", "attention_mask"))
    snapshot_calls: list[dict[str, object]] = []
    tokenizer_paths: list[str] = []
    session_calls: list[tuple[str, tuple[str, ...]]] = []

    def snapshot_download(**kwargs: object) -> str:
        snapshot_calls.append(kwargs)
        return str(snapshot_root)

    class TokenizerFactory:
        @staticmethod
        def from_file(path: str) -> FakeRerankerTokenizer:
            tokenizer_paths.append(path)
            return tokenizer

    def create_session(path: str, *, providers: Sequence[str]) -> FakeRerankerSession:
        session_calls.append((path, tuple(providers)))
        return session

    modules: dict[str, object] = {
        "huggingface_hub": SimpleNamespace(snapshot_download=snapshot_download),
        "tokenizers": SimpleNamespace(Tokenizer=TokenizerFactory),
        "onnxruntime": SimpleNamespace(InferenceSession=create_session),
        "numpy": FakeNumpy(),
    }
    monkeypatch.setattr(adapters, "import_module", modules.__getitem__)
    adapter = OnnxCrossEncoderReranker(
        revision="test-revision",
        onnx_file="auto",
        max_length=256,
        cache_folder=cache_folder,
    )

    first_runtime = adapter._get_runtime()
    second_runtime = adapter._get_runtime()

    assert first_runtime is second_runtime
    assert snapshot_calls == [
        {
            "repo_id": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "allow_patterns": (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "sentencepiece.bpe.model",
                "onnx/model_qint8_arm64.onnx",
            ),
            "revision": "test-revision",
            "cache_dir": str(cache_folder),
        }
    ]
    assert tokenizer_paths == [str(snapshot_root / "tokenizer.json")]
    assert tokenizer.truncation_lengths == [256]
    assert tokenizer.padding_calls == [("right", 1, "<pad>")]
    assert session_calls == [
        (
            str(snapshot_root / "onnx" / "model_qint8_arm64.onnx"),
            ("CPUExecutionProvider",),
        )
    ]
