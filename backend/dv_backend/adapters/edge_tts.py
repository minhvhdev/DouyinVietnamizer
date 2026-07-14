import asyncio
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from ..errors import AppError
from ..models import ErrorInfo
from .tts import split_tts_text

DEFAULT_EDGE_TTS_VOICE = "vi-VN-HoaiMyNeural"

EDGE_TTS_VI_VOICES = (
    {"id": "vi-VN-HoaiMyNeural", "name": "Hoài My (Nữ)", "gender": "Female"},
    {"id": "vi-VN-NamMinhNeural", "name": "Nam Minh (Nam)", "gender": "Male"},
)

EDGE_TTS_TH_VOICES = (
    {"id": "th-TH-PremwadeeNeural", "name": "Premwadee (Nữ)", "gender": "Female"},
    {"id": "th-TH-NiwatNeural", "name": "Niwat (Nam)", "gender": "Male"},
)

EDGE_TTS_FALLBACK_VOICES: dict[str, tuple[dict, ...]] = {
    "vi": EDGE_TTS_VI_VOICES,
    "th": EDGE_TTS_TH_VOICES,
}

_voice_cache: dict[str, list[dict]] = {}


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("edge_tts adapter cannot run inside an active asyncio loop")


async def _fetch_edge_voices(locale_prefix: str = "vi") -> list[dict]:
    import edge_tts

    voices = await edge_tts.list_voices()
    results: list[dict] = []
    for voice in voices:
        locale = str(voice.get("Locale") or "")
        if not locale.lower().startswith(locale_prefix.lower()):
            continue
        short_name = str(voice.get("ShortName") or "").strip()
        if not short_name:
            continue
        friendly = str(voice.get("FriendlyName") or short_name).strip()
        results.append(
            {
                "id": short_name,
                "name": friendly,
                "gender": str(voice.get("Gender") or ""),
                "locale": locale,
            }
        )
    results.sort(key=lambda item: (item["locale"], item["name"]))
    return results


def _fallback_edge_voices(locale_prefix: str) -> list[dict]:
    prefix = (locale_prefix or "vi").strip().lower()
    if prefix.startswith("th"):
        return list(EDGE_TTS_TH_VOICES)
    return list(EDGE_TTS_VI_VOICES)


def list_edge_tts_voices(*, locale_prefix: str = "vi", refresh: bool = False) -> list[dict]:
    global _voice_cache
    cache_key = (locale_prefix or "vi").strip().lower()
    if cache_key in _voice_cache and not refresh:
        return list(_voice_cache[cache_key])
    try:
        voices = _run_async(_fetch_edge_voices(locale_prefix))
    except Exception:
        voices = _fallback_edge_voices(locale_prefix)
    if voices:
        _voice_cache[cache_key] = voices
    return list(voices or _fallback_edge_voices(locale_prefix))


def _mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AppError(
            503,
            ErrorInfo(
                code="FFMPEG_UNAVAILABLE",
                message="ffmpeg is required to convert Edge TTS output to WAV.",
                action="Install ffmpeg or add it under vendor/.",
            ),
        )
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(mp3_path),
            "-ac",
            "1",
            "-ar",
            "24000",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not wav_path.is_file() or wav_path.stat().st_size == 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise AppError(
            502,
            ErrorInfo(
                code="EDGE_TTS_CONVERT_FAILED",
                message="Could not convert Edge TTS audio to WAV.",
                action="Verify ffmpeg is installed and working.",
                detail=detail or None,
            ),
        )


async def _synthesize_edge_chunk(
    text: str,
    voice: str,
    mp3_path: Path,
    *,
    max_attempts: int = 4,
) -> None:
    import edge_tts

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(mp3_path))
            if mp3_path.is_file() and mp3_path.stat().st_size > 0:
                return
            last_exc = RuntimeError("Edge TTS returned an empty MP3 file.")
        except Exception as exc:
            last_exc = exc
        if attempt + 1 < max_attempts:
            await asyncio.sleep(0.75 * (2**attempt))
    if last_exc is not None:
        raise last_exc


def _concat_wavs(parts: list[Path], output_path: Path) -> None:
    import wave

    if not parts:
        raise AppError(
            422,
            ErrorInfo(
                code="EMPTY_TTS_TEXT",
                message="Cannot synthesize empty narration text.",
                action="Verify translation output for this segment.",
            ),
        )
    if len(parts) == 1:
        output_path.write_bytes(parts[0].read_bytes())
        return

    params = None
    frames: list[bytes] = []
    for part in parts:
        with wave.open(str(part), "rb") as wav:
            current = (
                wav.getnchannels(),
                wav.getsampwidth(),
                wav.getframerate(),
                wav.getcomptype(),
                wav.getcompname(),
            )
            if params is None:
                params = current
            elif current != params:
                raise AppError(
                    502,
                    ErrorInfo(
                        code="EDGE_TTS_MERGE_FAILED",
                        message="Edge TTS chunk formats did not match.",
                        action="Try a shorter preview sentence.",
                    ),
                )
            frames.append(wav.readframes(wav.getnframes()))

    assert params is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as merged:
        merged.setnchannels(params[0])
        merged.setsampwidth(params[1])
        merged.setframerate(params[2])
        merged.setcomptype(params[3], params[4])
        for chunk in frames:
            merged.writeframes(chunk)


class EdgeTtsAdapter:
    def __init__(self, *, voice: str = DEFAULT_EDGE_TTS_VOICE) -> None:
        self.voice = voice or DEFAULT_EDGE_TTS_VOICE

    def close(self) -> None:
        return None

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str | None = None,
        ref_text: str | None = None,
        anchor_text: str | None = None,
        clone: bool = False,
        clone_mode: str | None = None,
        **kwargs,
    ) -> None:
        del ref_text, anchor_text, clone, clone_mode, kwargs
        resolved_voice = (voice or self.voice or DEFAULT_EDGE_TTS_VOICE).strip()
        chunks = split_tts_text(text)
        if not chunks:
            raise AppError(
                422,
                ErrorInfo(
                    code="EMPTY_TTS_TEXT",
                    message="Cannot synthesize empty narration text.",
                    action="Verify translation output for this segment.",
                ),
            )

        wav_parts: list[Path] = []
        try:
            for index, chunk in enumerate(chunks):
                if index > 0:
                    time.sleep(0.15)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
                    mp3_path = Path(tmp_mp3.name)
                wav_part = output_path.with_name(f"{output_path.stem}.part{index:03d}.wav")
                try:
                    _run_async(_synthesize_edge_chunk(chunk, resolved_voice, mp3_path))
                    _mp3_to_wav(mp3_path, wav_part)
                    wav_parts.append(wav_part)
                except AppError:
                    raise
                except Exception as exc:
                    raise AppError(
                        502,
                        ErrorInfo(
                            code="EDGE_TTS_SYNTHESIZE_FAILED",
                            message="Edge TTS could not synthesize narration.",
                            action="Check your internet connection and voice selection, then retry.",
                            detail=str(exc),
                            retryable=True,
                        ),
                    ) from exc
                finally:
                    mp3_path.unlink(missing_ok=True)

            _concat_wavs(wav_parts, output_path)
        finally:
            for part in wav_parts:
                part.unlink(missing_ok=True)
