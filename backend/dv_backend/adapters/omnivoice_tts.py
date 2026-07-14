from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..errors import AppError
from ..models import ErrorInfo
from ..omnivoice_chunk_synthesis import record_direct_segment_result, synthesize_short_or_chunked
from ..omnivoice_chunking import chunking_required
from .omnivoice_infer import _strip_surrogates
from .tts import parse_tts_voice_string, prepare_spoken_text_for_tts

OMNIVOICE_INSTRUCT_PREFIX = "instruct:"
OMNIVOICE_DEFAULT_CHUNK_THRESHOLD_SEC = 30.0
OMNIVOICE_DEFAULT_CHUNK_DURATION_SEC = 15.0


def _aggregate_response_perf(perf_items: list[dict[str, Any]]) -> dict[str, Any]:
    worker_sizes = [int(item.get("worker_batch_size") or 0) for item in perf_items if item.get("worker_batch_size")]
    flush_reasons = [str(item.get("flush_reason") or "") for item in perf_items if item.get("flush_reason")]
    queue_waits = [float(item.get("queue_wait_ms") or 0.0) for item in perf_items]
    model_ms = [float(item.get("model_synthesis_ms") or 0.0) for item in perf_items]
    encode_ms = [float(item.get("encode_ms") or 0.0) for item in perf_items]
    from collections import Counter

    return {
        "worker_batch_size_histogram": dict(Counter(worker_sizes)),
        "flush_reason_histogram": dict(Counter(flush_reasons)),
        "queue_wait_ms_mean": round(sum(queue_waits) / len(queue_waits), 2) if queue_waits else 0.0,
        "model_synthesis_ms_mean": round(sum(model_ms) / len(model_ms), 2) if model_ms else 0.0,
        "encode_ms_mean": round(sum(encode_ms) / len(encode_ms), 2) if encode_ms else 0.0,
    }


class OmniVoiceTtsAdapter:
    """Adapter backed by a long-lived OmniVoice worker subprocess."""

    def __init__(
        self,
        *,
        model: str,
        device: str = "cuda:0",
        num_step: int = 32,
        speed: float = 1.0,
        language_id: str | None = None,
        audio_chunk_threshold: float = OMNIVOICE_DEFAULT_CHUNK_THRESHOLD_SEC,
        audio_chunk_duration: float = OMNIVOICE_DEFAULT_CHUNK_DURATION_SEC,
        data_dir: Path | None = None,
        runner: object | None = None,
        settings: dict[str, Any] | None = None,
        _client: object | None = None,
    ) -> None:
        from ..omnivoice_env import OMNIVOICE_DEFAULT_MODEL

        self.model = (model or OMNIVOICE_DEFAULT_MODEL).strip() or OMNIVOICE_DEFAULT_MODEL
        self.device = (device or "cuda:0").strip() or "cuda:0"
        self.num_step = max(4, min(64, int(num_step)))
        self.speed = max(0.5, min(1.5, float(speed)))
        self.language_id = (language_id or "").strip() or None
        self.audio_chunk_threshold = max(4.0, min(60.0, float(audio_chunk_threshold)))
        self.audio_chunk_duration = max(4.0, min(30.0, float(audio_chunk_duration)))
        self._data_dir = Path(data_dir) if data_dir is not None else None
        self._runner = runner
        self._settings = dict(settings or {})
        self._client = None
        self._injected_client = _client
        self._last_batch_diagnostics: dict[str, Any] = {
            "mode": "none",
            "submitted_block_size": 0,
            "max_worker_batch_size": 0,
            "direct_blocks": 0,
            "chunked_segments": 0,
        }
        self._last_batch_perf: dict[str, Any] = {}

    def _include_worker_perf(self) -> bool:
        value = self._settings.get("omnivoice_tts_include_perf", False)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    @property
    def last_batch_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_batch_diagnostics)

    @property
    def last_batch_perf(self) -> dict[str, Any]:
        return dict(self._last_batch_perf)

    def _resolve_data_dir(self) -> Path:
        if self._data_dir is not None:
            return self._data_dir
        from ..config import AppConfig

        try:
            return AppConfig.from_env().data_dir
        except Exception:
            return Path.cwd() / "data"

    def _ensure_runtime(self) -> None:
        if self._client is not None:
            return
        if self._injected_client is not None:
            self._client = self._injected_client
            return
        from .omnivoice_client import acquire_client

        self._client = acquire_client(
            data_dir=self._resolve_data_dir(),
            model=self.model,
            device=self.device,
            num_step=self.num_step,
            speed=self.speed,
            language_id=self.language_id,
            audio_chunk_threshold=self.audio_chunk_threshold,
            audio_chunk_duration=self.audio_chunk_duration,
        )
        self._client.register_with_runner(self._runner)

    def _voice_kwargs(self, voice: str, ref_text: str | None) -> dict[str, str | None]:
        ref_audio, _, voice_design = parse_tts_voice_string(voice)
        return {
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "instruct": voice_design,
        }

    def _synthesize_single(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        ref_text: str | None,
        anchor_text: str | None = None,
    ) -> None:
        self._ensure_runtime()
        kwargs = self._voice_kwargs(voice, ref_text)
        client = self._client
        assert client is not None
        response = client.synthesize(
            text=text,
            output_path=output_path,
            ref_audio=kwargs["ref_audio"],
            ref_text=kwargs["ref_text"],
            anchor_text=anchor_text,
            instruct=kwargs["instruct"],
            include_perf=self._include_worker_perf(),
        )
        if not response.get("ok", False):
            raise AppError(
                502,
                ErrorInfo(
                    code=response.get("code") or "OMNIVOICE_TTS_FAILED",
                    message=response.get("message") or "OmniVoice could not generate narration.",
                    action="Check OmniVoice settings and reference audio.",
                    detail=response.get("detail"),
                    retryable=bool(response.get("retryable", True)),
                ),
            )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_TTS_FAILED",
                    message="OmniVoice produced an empty audio file.",
                    action="Try another reference clip or switch to auto voice mode.",
                    retryable=True,
                ),
            )

    def _validate_worker_response(self, response: dict[str, Any], output_path: Path) -> None:
        if not response.get("ok", False):
            raise AppError(
                502,
                ErrorInfo(
                    code=response.get("code") or "OMNIVOICE_TTS_FAILED",
                    message=response.get("message") or "OmniVoice could not generate narration.",
                    action="Check OmniVoice settings and reference audio.",
                    detail=response.get("detail"),
                    retryable=bool(response.get("retryable", True)),
                ),
            )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_TTS_FAILED",
                    message="OmniVoice produced an empty audio file.",
                    action="Try another reference clip or switch to auto voice mode.",
                    retryable=True,
                ),
            )

    def _configured_batch_size(self) -> int:
        try:
            configured = int(self._settings.get("tts_micro_batch_size") or 4)
        except (TypeError, ValueError):
            configured = 4
        configured = max(1, configured)
        client = self._client or self._injected_client
        client_max = getattr(client, "max_batch", None)
        if client_max is not None:
            try:
                configured = min(configured, max(1, int(client_max)))
            except (TypeError, ValueError):
                pass
        return configured

    def _synthesize_direct_block(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {
                "submitted_block_size": 0,
                "configured_batch_size": self._configured_batch_size(),
                "explicit_batch_sizes": [],
                "max_worker_batch_size": 0,
                "flush_reasons": [],
            }
        self._ensure_runtime()
        client = self._client
        assert client is not None
        vendor_dir = self._default_vendor_dir()
        synth_many = getattr(client, "synthesize_many", None)
        configured_batch_size = self._configured_batch_size()

        prepared: list[dict[str, Any]] = []
        for item in items:
            text = _strip_surrogates(str(item.get("text") or "")).strip()
            output_path = Path(item["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            voice_kwargs = self._voice_kwargs(str(item.get("voice") or ""), item.get("ref_text"))
            prepared.append(
                {
                    "text": text,
                    "output_path": output_path,
                    "ref_audio": voice_kwargs["ref_audio"],
                    "ref_text": voice_kwargs["ref_text"],
                    "anchor_text": item.get("anchor_text"),
                    "instruct": voice_kwargs["instruct"],
                    "item": item,
                }
            )

        responses: list[dict[str, Any]] = []
        explicit_batch_sizes: list[int] = []
        if callable(synth_many):
            for start in range(0, len(prepared), configured_batch_size):
                chunk = prepared[start : start + configured_batch_size]
                worker_requests = [
                    {
                        "text": entry["text"],
                        "output_path": entry["output_path"],
                        "ref_audio": entry["ref_audio"],
                        "ref_text": entry["ref_text"],
                        "anchor_text": entry["anchor_text"],
                        "instruct": entry["instruct"],
                        "include_perf": self._include_worker_perf(),
                    }
                    for entry in chunk
                ]
                explicit_batch_sizes.append(len(chunk))
                chunk_responses = synth_many(worker_requests)
                if len(chunk_responses) != len(chunk):
                    raise AppError(
                        502,
                        ErrorInfo(
                            code="OMNIVOICE_TTS_FAILED",
                            message="OmniVoice batch returned an unexpected number of responses.",
                            retryable=True,
                        ),
                    )
                responses.extend(chunk_responses)
        else:
            for entry in prepared:
                explicit_batch_sizes.append(1)
                responses.append(
                    client.synthesize(
                        text=entry["text"],
                        output_path=entry["output_path"],
                        ref_audio=entry["ref_audio"],
                        ref_text=entry["ref_text"],
                        anchor_text=entry["anchor_text"],
                        instruct=entry["instruct"],
                        include_perf=self._include_worker_perf(),
                    )
                )

        if len(responses) != len(prepared):
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_TTS_FAILED",
                    message="OmniVoice batch returned an unexpected number of responses.",
                    retryable=True,
                ),
            )

        for entry, response in zip(prepared, responses):
            output_path = Path(entry["output_path"])
            self._validate_worker_response(response, output_path)
            item = entry["item"]
            record_direct_segment_result(
                text=entry["text"],
                output_path=output_path,
                settings=self._settings,
                segment=item.get("segment"),
                language=str(item.get("language") or "vi"),
                transcribe_fn=item.get("transcribe_fn"),
                vendor_dir=vendor_dir,
            )
        worker_batch_sizes = [
            int((response.get("perf") or {}).get("worker_batch_size") or 0)
            for response in responses
            if response.get("ok")
        ]
        flush_reasons = [
            str((response.get("perf") or {}).get("flush_reason") or "")
            for response in responses
            if response.get("ok")
        ]
        perf_items = [response.get("perf") or {} for response in responses if response.get("ok")]
        self._last_batch_perf = _aggregate_response_perf(perf_items)
        max_worker_batch = max(worker_batch_sizes) if worker_batch_sizes else 0
        if max_worker_batch > configured_batch_size:
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_TTS_FAILED",
                    message=(
                        "OmniVoice worker batch exceeded configured batch size "
                        f"({max_worker_batch} > {configured_batch_size})."
                    ),
                    retryable=True,
                ),
            )
        return {
            "submitted_block_size": len(prepared),
            "configured_batch_size": configured_batch_size,
            "explicit_batch_sizes": explicit_batch_sizes,
            "max_worker_batch_size": max_worker_batch,
            "flush_reasons": [reason for reason in flush_reasons if reason],
        }

    def synthesize_batch(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        direct_blocks = 0
        chunked_segments = 0
        max_submitted = 0
        max_worker_batch = 0
        flush_reasons: list[str] = []
        explicit_batch_sizes: list[int] = []
        configured_batch_size = self._configured_batch_size()
        index = 0
        while index < len(items):
            item = items[index]
            text = _strip_surrogates(str(item.get("text") or "")).strip()
            if not text:
                raise AppError(
                    422,
                    ErrorInfo(
                        code="EMPTY_TTS_TEXT",
                        message="Cannot synthesize empty narration text.",
                        action="Verify translation output for this segment.",
                    ),
                )
            if chunking_required(text, self._settings):
                chunked_segments += 1
                self.synthesize(
                    text,
                    Path(item["output_path"]),
                    voice=str(item.get("voice") or ""),
                    ref_text=item.get("ref_text"),
                    anchor_text=item.get("anchor_text"),
                    clone=bool(item.get("clone")),
                    clone_mode=item.get("clone_mode"),
                    segment=item.get("segment"),
                    language=str(item.get("language") or "vi"),
                    transcribe_fn=item.get("transcribe_fn"),
                )
                index += 1
                continue

            block_end = index
            while block_end < len(items):
                block_text = _strip_surrogates(str(items[block_end].get("text") or "")).strip()
                if not block_text or chunking_required(block_text, self._settings):
                    break
                block_end += 1
            block_stats = self._synthesize_direct_block(items[index:block_end])
            direct_blocks += 1
            max_submitted = max(max_submitted, int(block_stats.get("submitted_block_size") or 0))
            max_worker_batch = max(max_worker_batch, int(block_stats.get("max_worker_batch_size") or 0))
            flush_reasons.extend(list(block_stats.get("flush_reasons") or []))
            explicit_batch_sizes.extend(list(block_stats.get("explicit_batch_sizes") or []))
            configured_batch_size = int(block_stats.get("configured_batch_size") or configured_batch_size)
            index = block_end

        if chunked_segments and not direct_blocks:
            mode = "omnivoice_chunked_only"
        elif max_submitted <= 1:
            mode = "omnivoice_single_direct"
        elif max_worker_batch > 1:
            mode = "omnivoice_queued_batch"
        elif max_submitted > 1:
            mode = "omnivoice_queued_submit_only"
        else:
            mode = "omnivoice_direct"
        self._last_batch_diagnostics = {
            "mode": mode,
            "submitted_block_size": max_submitted,
            "configured_batch_size": configured_batch_size,
            "explicit_batch_sizes": explicit_batch_sizes,
            "max_worker_batch_size": max_worker_batch,
            "direct_blocks": direct_blocks,
            "direct_block_count": direct_blocks,
            "chunked_segments": chunked_segments,
            "flush_reasons": flush_reasons,
        }

    def close(self) -> None:
        return None

    def _default_vendor_dir(self) -> Path | None:
        env = os.environ.get("DV_VENDOR_DIR", "").strip()
        if env:
            return Path(env)
        here = Path(__file__).resolve()
        for candidate in (here.parents[2] / "vendor", here.parents[3] / "vendor"):
            if candidate.is_dir():
                return candidate
        return None

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        ref_text: str | None = None,
        anchor_text: str | None = None,
        clone: bool = False,
        clone_mode: str | None = None,
        segment: dict | None = None,
        language: str = "vi",
        transcribe_fn=None,
        vendor_dir: Path | None = None,
    ) -> None:
        _ = clone, clone_mode, ref_text
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned = _strip_surrogates(text).strip()
        if not cleaned:
            raise AppError(
                422,
                ErrorInfo(
                    code="EMPTY_TTS_TEXT",
                    message="Cannot synthesize empty narration text.",
                    action="Verify translation output for this segment.",
                ),
            )
        try:
            def _synthesize_chunk(chunk_text: str, chunk_path: Path) -> None:
                self._synthesize_single(
                    chunk_text,
                    chunk_path,
                    voice=voice,
                    ref_text=None,
                    anchor_text=anchor_text,
                )

            vendor_dir = self._default_vendor_dir()
            # Prepare once, then route short → single-shot or long → external chunking.
            prepared = prepare_spoken_text_for_tts(cleaned)
            synthesize_short_or_chunked(
                text=prepared,
                output_path=output_path,
                synthesize_fn=_synthesize_chunk,
                settings=self._settings,
                segment=segment,
                language=language,
                transcribe_fn=transcribe_fn,
                vendor_dir=vendor_dir,
            )
        except AppError:
            raise
        except Exception as cause:
            raise AppError(
                502,
                ErrorInfo(
                    code="OMNIVOICE_TTS_FAILED",
                    message="OmniVoice could not generate narration.",
                    action="Ensure the OmniVoice virtualenv is installed and the GPU is available.",
                    detail=str(cause),
                    retryable=True,
                ),
            ) from cause
