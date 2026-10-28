"""Lazy CPU adapters for optional local ML dependencies."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Protocol, cast

import anyio

from app.services.ml.contracts import Embedding, Transcription, TranscriptSegment


class _WhisperModel(Protocol):
    def transcribe(
        self,
        audio: str,
        **kwargs: object,
    ) -> tuple[Iterable[object], object]: ...


class _TokenEncoding(Protocol):
    @property
    def ids(self) -> list[int]: ...

    @property
    def attention_mask(self) -> list[int]: ...

    @property
    def type_ids(self) -> list[int]: ...


class _Tokenizer(Protocol):
    def token_to_id(self, token: str) -> int | None: ...

    def enable_truncation(self, max_length: int) -> None: ...

    def enable_padding(
        self,
        *,
        direction: str,
        pad_id: int,
        pad_token: str,
    ) -> None: ...

    def encode_batch(self, inputs: Sequence[str]) -> list[_TokenEncoding]: ...


class _OnnxInput(Protocol):
    @property
    def name(self) -> str: ...


class _OnnxSession(Protocol):
    def get_inputs(self) -> Sequence[_OnnxInput]: ...

    def run(
        self,
        output_names: None,
        input_feed: dict[str, object],
    ) -> Sequence[object]: ...


class _NumpyModule(Protocol):
    int64: object

    def array(self, value: object, *, dtype: object) -> object: ...


@dataclass(frozen=True, slots=True)
class _E5Runtime:
    tokenizer: _Tokenizer
    session: _OnnxSession
    numpy: _NumpyModule


class FasterWhisperTranscriber:
    """Run faster-whisper on CPU without importing or loading it at app startup."""

    def __init__(
        self,
        *,
        model_name: str = "small",
        revision: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: Path | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        language: str | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Whisper model name must not be blank")
        if cpu_threads < 0 or num_workers <= 0 or beam_size <= 0:
            raise ValueError("Whisper worker and beam settings must be positive")
        self._model_name = model_name
        normalized_revision = revision.strip() if revision is not None else ""
        self._revision = normalized_revision or None
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._num_workers = num_workers
        self._download_root = download_root
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        normalized_language = language.strip() if language is not None else ""
        self._language = normalized_language or None
        self._model: _WhisperModel | None = None
        self._model_lock = Lock()

    @property
    def model_id(self) -> str:
        language = self._language or "auto"
        revision = self._revision or "main"
        return (
            f"faster-whisper/{self._model_name}@{revision}:"
            f"{self._device}-{self._compute_type}"
            f":language-{language}"
        )

    async def transcribe(self, media_path: Path) -> Transcription:
        """Transcribe in a worker thread because model inference is synchronous."""
        return await anyio.to_thread.run_sync(
            self._transcribe_sync,
            media_path,
            abandon_on_cancel=True,
        )

    def _transcribe_sync(self, media_path: Path) -> Transcription:
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        model = self._get_model()
        transcribe_options: dict[str, object] = {
            "beam_size": self._beam_size,
            "vad_filter": self._vad_filter,
            "word_timestamps": True,
        }
        if self._language is not None:
            transcribe_options["language"] = self._language
        raw_segments, raw_info = model.transcribe(str(media_path), **transcribe_options)
        segments: list[TranscriptSegment] = []
        for raw_segment in raw_segments:
            words: object = getattr(raw_segment, "words", None)
            word_segments: list[TranscriptSegment] = []
            if isinstance(words, Iterable) and not isinstance(words, str | bytes):
                word_segments = [
                    segment
                    for word in cast(Iterable[object], words)
                    if (segment := _read_transcript_segment(word, "word")) is not None
                ]
            if word_segments:
                segments.extend(word_segments)
                continue
            segment = _read_transcript_segment(raw_segment, "text")
            if segment is not None:
                segments.append(segment)

        language = _read_optional_string(raw_info, "language")
        language_probability = _read_optional_float(raw_info, "language_probability")
        return Transcription(
            segments=tuple(segments),
            language=language,
            language_probability=language_probability,
        )

    def _get_model(self) -> _WhisperModel:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                try:
                    module = import_module("faster_whisper")
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "faster-whisper is required to build the transcript index"
                    ) from exc
                model_factory = cast(Callable[..., object], module.WhisperModel)
                kwargs: dict[str, object] = {
                    "device": self._device,
                    "compute_type": self._compute_type,
                    "cpu_threads": self._cpu_threads,
                    "num_workers": self._num_workers,
                }
                if self._download_root is not None:
                    kwargs["download_root"] = str(self._download_root)
                if self._revision is not None:
                    kwargs["revision"] = self._revision
                self._model = cast(_WhisperModel, model_factory(self._model_name, **kwargs))
        if self._model is None:
            raise RuntimeError("Failed to initialize faster-whisper")
        return self._model


class OnnxE5Embedder:
    """Encode multilingual E5 text with a small CPU-only ONNX runtime."""

    _ALLOW_PATTERNS = (
        "onnx/model.onnx",
        "onnx/tokenizer.json",
        "onnx/tokenizer_config.json",
        "onnx/special_tokens_map.json",
        "onnx/sentencepiece.bpe.model",
    )

    def __init__(
        self,
        *,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "cpu",
        revision: str | None = None,
        dimension: int = 384,
        max_length: int = 512,
        cache_folder: Path | None = None,
        batch_size: int = 16,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Embedding model name must not be blank")
        if device != "cpu":
            raise ValueError("The ONNX E5 adapter currently supports only the CPU device")
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        if max_length <= 0:
            raise ValueError("Tokenizer maximum length must be positive")
        if batch_size <= 0:
            raise ValueError("Embedding batch size must be positive")
        self._model_name = model_name
        self._revision = revision
        self._dimension = dimension
        self._max_length = max_length
        self._cache_folder = cache_folder
        self._batch_size = batch_size
        self._runtime: _E5Runtime | None = None
        self._runtime_lock = Lock()

    @property
    def model_id(self) -> str:
        revision = self._revision or "main"
        return f"onnx/{self._model_name}@{revision}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_query(self, text: str) -> Embedding:
        normalized = _require_text(text)
        embeddings = await self._encode((f"query: {normalized}",))
        return embeddings[0]

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Embedding]:
        if not texts:
            return ()
        prefixed = tuple(f"passage: {_require_text(text)}" for text in texts)
        return await self._encode(prefixed)

    async def _encode(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        return await anyio.to_thread.run_sync(
            self._encode_sync,
            texts,
            abandon_on_cancel=True,
        )

    def _encode_sync(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        embeddings: list[Embedding] = []
        for start in range(0, len(texts), self._batch_size):
            embeddings.extend(self._encode_batch_sync(texts[start : start + self._batch_size]))
        return tuple(embeddings)

    def _encode_batch_sync(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        runtime = self._get_runtime()
        encodings = runtime.tokenizer.encode_batch(texts)
        if len(encodings) != len(texts):
            raise ValueError("Tokenizer returned an unexpected batch size")
        input_names = {input_value.name for input_value in runtime.session.get_inputs()}
        unknown_inputs = input_names - {"input_ids", "attention_mask", "token_type_ids"}
        if unknown_inputs:
            names = ", ".join(sorted(unknown_inputs))
            raise RuntimeError(f"Unsupported ONNX model inputs: {names}")
        input_feed: dict[str, object] = {
            "input_ids": runtime.numpy.array(
                [encoding.ids for encoding in encodings],
                dtype=runtime.numpy.int64,
            ),
            "attention_mask": runtime.numpy.array(
                [encoding.attention_mask for encoding in encodings],
                dtype=runtime.numpy.int64,
            ),
        }
        if "token_type_ids" in input_names:
            input_feed["token_type_ids"] = runtime.numpy.array(
                [encoding.type_ids for encoding in encodings],
                dtype=runtime.numpy.int64,
            )
        outputs = runtime.session.run(None, input_feed)
        if not outputs:
            raise ValueError("ONNX model returned no outputs")
        return _mean_pool_and_normalize(
            outputs[0],
            masks=[encoding.attention_mask for encoding in encodings],
            expected_dimension=self._dimension,
        )

    def _get_runtime(self) -> _E5Runtime:
        if self._runtime is not None:
            return self._runtime
        with self._runtime_lock:
            if self._runtime is None:
                try:
                    hub_module = import_module("huggingface_hub")
                    tokenizers_module = import_module("tokenizers")
                    onnx_module = import_module("onnxruntime")
                    numpy_module = import_module("numpy")
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "huggingface-hub, tokenizers, onnxruntime, and numpy are required for E5"
                    ) from exc
                snapshot_download = cast(
                    Callable[..., str],
                    hub_module.snapshot_download,
                )
                snapshot_kwargs: dict[str, object] = {
                    "repo_id": self._model_name,
                    "allow_patterns": self._ALLOW_PATTERNS,
                }
                if self._revision is not None:
                    snapshot_kwargs["revision"] = self._revision
                if self._cache_folder is not None:
                    snapshot_kwargs["cache_dir"] = str(self._cache_folder)
                snapshot_root = Path(snapshot_download(**snapshot_kwargs))

                from_file_value: object = tokenizers_module.Tokenizer.from_file
                from_file = cast(Callable[[str], object], from_file_value)
                tokenizer = cast(
                    _Tokenizer,
                    from_file(str(snapshot_root / "onnx" / "tokenizer.json")),
                )
                tokenizer.enable_truncation(self._max_length)
                pad_token, pad_id = _resolve_pad_token(tokenizer)
                tokenizer.enable_padding(
                    direction="right",
                    pad_id=pad_id,
                    pad_token=pad_token,
                )

                session_factory = cast(
                    Callable[..., object],
                    onnx_module.InferenceSession,
                )
                session = cast(
                    _OnnxSession,
                    session_factory(
                        str(snapshot_root / "onnx" / "model.onnx"),
                        providers=["CPUExecutionProvider"],
                    ),
                )
                self._runtime = _E5Runtime(
                    tokenizer=tokenizer,
                    session=session,
                    numpy=cast(_NumpyModule, numpy_module),
                )
        if self._runtime is None:
            raise RuntimeError("Failed to initialize E5 runtime")
        return self._runtime


def _read_transcript_segment(value: object, text_attribute: str) -> TranscriptSegment | None:
    start = _read_optional_float(value, "start")
    end = _read_optional_float(value, "end")
    text = _read_optional_string(value, text_attribute)
    if start is None or end is None or text is None or start >= end:
        return None
    return TranscriptSegment(start_seconds=start, end_seconds=end, text=text)


def _read_optional_float(value: object, attribute: str) -> float | None:
    raw: object = getattr(value, attribute, None)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    converted = float(raw)
    return converted if math.isfinite(converted) else None


def _read_optional_string(value: object, attribute: str) -> str | None:
    raw: object = getattr(value, attribute, None)
    if not isinstance(raw, str):
        return None
    normalized = " ".join(raw.split())
    return normalized or None


def _require_text(text: str) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("Text to embed must not be blank")
    return normalized


def _to_plain_sequence(value: object) -> Sequence[object]:
    tolist: object = getattr(value, "tolist", None)
    if callable(tolist):
        value = cast(Callable[[], object], tolist)()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError("Embedding model returned a non-sequence value")
    return cast(Sequence[object], value)


def _resolve_pad_token(tokenizer: _Tokenizer) -> tuple[str, int]:
    for token in ("<pad>", "[PAD]"):
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            return token, token_id
    raise ValueError("Tokenizer does not define a supported padding token")


def _mean_pool_and_normalize(
    raw_hidden_state: object,
    *,
    masks: Sequence[Sequence[int]],
    expected_dimension: int,
) -> tuple[Embedding, ...]:
    batches = _to_plain_sequence(raw_hidden_state)
    if len(batches) != len(masks):
        raise ValueError("ONNX output batch size does not match the tokenizer output")

    embeddings: list[Embedding] = []
    for raw_tokens, mask in zip(batches, masks, strict=True):
        tokens = _to_plain_sequence(raw_tokens)
        if len(tokens) != len(mask):
            raise ValueError("ONNX sequence length does not match the attention mask")
        pooled = [0.0] * expected_dimension
        active_tokens = 0
        for raw_token, is_active in zip(tokens, mask, strict=True):
            if not is_active:
                continue
            token = _to_plain_sequence(raw_token)
            if len(token) != expected_dimension:
                raise ValueError("ONNX hidden state has an unexpected dimension")
            for index, raw_value in enumerate(token):
                pooled[index] += _require_finite_number(raw_value)
            active_tokens += 1
        if active_tokens == 0:
            raise ValueError("Tokenizer produced an empty attention mask")
        pooled = [value / active_tokens for value in pooled]
        norm = math.sqrt(sum(value * value for value in pooled))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Cannot normalize a zero or non-finite embedding")
        embeddings.append(tuple(value / norm for value in pooled))
    return tuple(embeddings)


def _require_finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Embedding contains a non-numeric value")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("Embedding contains a non-finite value")
    return converted
