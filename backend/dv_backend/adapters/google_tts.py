import base64
import json
import tempfile
import urllib.error
import urllib.request
import wave
from pathlib import Path

from ..errors import AppError
from ..models import ErrorInfo
from .edge_tts import _concat_wavs
from .tts import split_tts_text

GOOGLE_CLOUD_TTS_API = "https://texttospeech.googleapis.com/v1/text:synthesize"
DEFAULT_GOOGLE_TTS_VOICE = "vi-VN-Standard-A"
DEFAULT_GOOGLE_TTS_SPEAKING_RATE = 1.0

GOOGLE_TTS_VI_VOICES = (
    {"id": "vi-VN-Standard-A", "name": "Standard A — Nữ", "gender": "Female", "tier": "Standard"},
    {"id": "vi-VN-Standard-B", "name": "Standard B — Nam", "gender": "Male", "tier": "Standard"},
    {"id": "vi-VN-Standard-C", "name": "Standard C — Nữ", "gender": "Female", "tier": "Standard"},
    {"id": "vi-VN-Standard-D", "name": "Standard D — Nam", "gender": "Male", "tier": "Standard"},
)

GOOGLE_TTS_TH_VOICES = (
    {"id": "th-TH-Standard-A", "name": "Standard A — Nữ", "gender": "Female", "tier": "Standard"},
    {"id": "th-TH-Neural2-C", "name": "Neural2 C — Nữ", "gender": "Female", "tier": "Neural2"},
    {"id": "th-TH-Neural2-D", "name": "Neural2 D — Nam", "gender": "Male", "tier": "Neural2"},
)

GOOGLE_TTS_VOICES = GOOGLE_TTS_VI_VOICES + GOOGLE_TTS_TH_VOICES

GOOGLE_TTS_VOICE_IDS = frozenset(voice["id"] for voice in GOOGLE_TTS_VOICES)


def list_google_tts_voices(*, locale: str | None = None) -> list[dict]:
    if not locale:
        return list(GOOGLE_TTS_VOICES)
    prefix = str(locale).strip().lower()
    return [
        voice
        for voice in GOOGLE_TTS_VOICES
        if str(voice["id"]).lower().startswith(prefix)
    ]


def _language_code_for_voice(voice: str) -> str:
    if "-" in voice:
        parts = voice.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
    return "vi-VN"


def _write_pcm_wav(pcm: bytes, output_path: Path, *, sample_rate: int = 24000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def _cloud_tts_request(api_key: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        GOOGLE_CLOUD_TTS_API,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as cause:
        detail = cause.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Google Cloud TTS HTTP {cause.code}: {detail}") from cause


def _synthesize_cloud_chunk(
    text: str,
    voice: str,
    wav_path: Path,
    *,
    api_key: str,
    speaking_rate: float,
) -> None:
    response = _cloud_tts_request(
        api_key,
        {
            "input": {"text": text},
            "voice": {
                "languageCode": _language_code_for_voice(voice),
                "name": voice,
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": 24000,
                "speakingRate": speaking_rate,
            },
        },
    )
    audio_b64 = response.get("audioContent")
    if not audio_b64:
        raise ValueError("Google Cloud TTS returned no audioContent.")
    pcm = base64.b64decode(audio_b64)
    if not pcm:
        raise ValueError("Google Cloud TTS returned empty audio.")
    _write_pcm_wav(pcm, wav_path)


def _classify_cloud_tts_failure(error: Exception) -> tuple[str, str, str]:
    message = str(error)
    if "HTTP 401" in message or "HTTP 403" in message or "API key not valid" in message:
        return (
            "GOOGLE_TTS_KEY_INVALID",
            "The Google Cloud Text-to-Speech API key was rejected.",
            "Create an API key in Google Cloud Console with Text-to-Speech API enabled.",
        )
    if "HTTP 429" in message or "RESOURCE_EXHAUSTED" in message:
        return (
            "GOOGLE_TTS_QUOTA_EXCEEDED",
            "Google Cloud TTS quota or rate limit was exceeded.",
            "Check quota in Google Cloud Console or wait before retrying.",
        )
    if "HTTP 400" in message:
        return (
            "GOOGLE_TTS_BAD_REQUEST",
            "Google Cloud TTS rejected the synthesis request.",
            "Verify the selected voice and API key permissions.",
        )
    return (
        "GOOGLE_TTS_SYNTHESIZE_FAILED",
        "Google Cloud TTS could not synthesize narration.",
        "Check your API key, internet connection, and voice selection.",
    )


class GoogleTtsAdapter:
    def __init__(
        self,
        *,
        api_key: str,
        voice: str = DEFAULT_GOOGLE_TTS_VOICE,
        speaking_rate: float = DEFAULT_GOOGLE_TTS_SPEAKING_RATE,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.voice = voice or DEFAULT_GOOGLE_TTS_VOICE
        self.speaking_rate = speaking_rate

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
        if not self.api_key:
            raise AppError(
                400,
                ErrorInfo(
                    code="MISSING_GOOGLE_TTS_API_KEY",
                    message="No Google Cloud Text-to-Speech API key is configured.",
                    action="Add a Google Cloud API key in Settings → Lồng tiếng → Google TTS.",
                ),
            )

        resolved_voice = (voice or self.voice or DEFAULT_GOOGLE_TTS_VOICE).strip()
        if resolved_voice not in GOOGLE_TTS_VOICE_IDS:
            resolved_voice = DEFAULT_GOOGLE_TTS_VOICE

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
                wav_part = output_path.with_name(f"{output_path.stem}.part{index:03d}.wav")
                try:
                    _synthesize_cloud_chunk(
                        chunk,
                        resolved_voice,
                        wav_part,
                        api_key=self.api_key,
                        speaking_rate=self.speaking_rate,
                    )
                    wav_parts.append(wav_part)
                except AppError:
                    raise
                except Exception as exc:
                    code, message, action = _classify_cloud_tts_failure(exc)
                    raise AppError(
                        502,
                        ErrorInfo(
                            code=code,
                            message=message,
                            action=action,
                            detail=str(exc),
                            retryable=code in {"GOOGLE_TTS_QUOTA_EXCEEDED", "GOOGLE_TTS_SYNTHESIZE_FAILED"},
                        ),
                    ) from exc

            _concat_wavs(wav_parts, output_path)
        finally:
            for part in wav_parts:
                part.unlink(missing_ok=True)
