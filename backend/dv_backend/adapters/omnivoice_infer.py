"""Direct OmniVoice inference in the isolated virtualenv."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from ..errors import AppError
from ..models import ErrorInfo
from ..omnivoice_env import OMNIVOICE_DEFAULT_MODEL, build_omnivoice_subprocess_env, resolve_omnivoice_python
from .tts import resolve_omnivoice_device

OMNIVOICE_DEFAULT_CHUNK_THRESHOLD_SEC = 30.0
OMNIVOICE_DEFAULT_CHUNK_DURATION_SEC = 15.0
OMNIVOICE_DEFAULT_NUM_STEP = 32
OMNIVOICE_DEFAULT_GUIDANCE_SCALE = 2.0
OMNIVOICE_SAMPLE_RATE = 24_000


def _strip_surrogates(text: str) -> str:
    return "".join(
        character
        for character in str(text or "")
        if not (0xD800 <= ord(character) <= 0xDFFF)
    )


def resolve_omnivoice_clone_ref_text(anchor_text: str | None) -> str | None:
    """Return user-provided ref_text without truncation or ASR."""
    cleaned = _strip_surrogates(str(anchor_text or "")).strip()
    return cleaned or None


def resolve_omnivoice_language(language_id: str | None) -> str | None:
    value = (language_id or "").strip()
    if not value or value.lower() in {"auto", "none"}:
        return None
    if value.lower() in {"vi", "vietnamese", "viet nam", "vietnam"}:
        return "vi"
    if value.lower() in {"th", "thai", "thailand"}:
        return "th"
    return value


def resolve_omnivoice_speed(speed: float) -> float | None:
    """Official demo only passes speed when it differs from 1.0."""
    try:
        value = float(speed)
    except (TypeError, ValueError):
        return None
    if abs(value - 1.0) < 1e-6:
        return None
    return max(0.5, min(1.5, value))


def build_official_generation_config(
    *,
    num_step: int,
    audio_chunk_threshold: float,
    audio_chunk_duration: float,
) -> dict[str, Any]:
    """Mirror ``OmniVoiceGenerationConfig`` defaults from the official demo."""
    return {
        "num_step": max(4, min(64, int(num_step))),
        "guidance_scale": OMNIVOICE_DEFAULT_GUIDANCE_SCALE,
        "denoise": True,
        "preprocess_prompt": True,
        "postprocess_output": True,
        "audio_chunk_threshold": max(4.0, min(60.0, float(audio_chunk_threshold))),
        "audio_chunk_duration": max(4.0, min(30.0, float(audio_chunk_duration))),
    }


def plan_official_omnivoice_call(
    *,
    text: str,
    speed: float,
    num_step: int,
    language_id: str | None,
    ref_audio: str | None,
    anchor_text: str | None,
    instruct: str | None,
    audio_chunk_threshold: float,
    audio_chunk_duration: float,
) -> dict[str, Any]:
    """Plan a generate() call aligned with omnivoice/omnivoice/cli/demo.py."""
    text_clean = _strip_surrogates(text).strip()
    if not text_clean:
        raise ValueError("Cannot synthesize empty narration text.")

    plan: dict[str, Any] = {
        "text": text_clean,
        "generation_config": build_official_generation_config(
            num_step=num_step,
            audio_chunk_threshold=audio_chunk_threshold,
            audio_chunk_duration=audio_chunk_duration,
        ),
    }
    language = resolve_omnivoice_language(language_id)
    if language:
        plan["language"] = language

    speed_value = resolve_omnivoice_speed(speed)
    if speed_value is not None:
        plan["speed"] = speed_value

    if ref_audio:
        ref_text = resolve_omnivoice_clone_ref_text(anchor_text)
        if not ref_text:
            raise ValueError(
                "OmniVoice voice clone requires ref_text that matches the reference audio."
            )
        plan["ref_audio"] = ref_audio
        plan["ref_text"] = ref_text
    elif instruct and instruct.strip():
        plan["instruct"] = instruct.strip()

    return plan


_OFFICIAL_GENERATE_SCRIPT = """
import json
from pathlib import Path

import soundfile as sf
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

args = json.loads(__PAYLOAD__)
dtype = torch.float16 if str(args.get("device", "")).startswith("cuda") else torch.float32
model = OmniVoice.from_pretrained(args["model"], device_map=args["device"], dtype=dtype)
plan = dict(args["plan"])
generation_config = OmniVoiceGenerationConfig(**dict(plan.pop("generation_config")))
generate_kwargs = {"generation_config": generation_config, **plan}
ref_audio = generate_kwargs.pop("ref_audio", None)
ref_text = generate_kwargs.pop("ref_text", None)
if ref_audio:
    generate_kwargs["voice_clone_prompt"] = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        preprocess_prompt=generation_config.preprocess_prompt,
    )
audio = model.generate(**generate_kwargs)
if not audio:
    raise RuntimeError("OmniVoice returned no audio.")
out = Path(args["output_path"])
out.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(out), audio[0], model.sampling_rate)
print(json.dumps({"ok": True, "output_path": str(out)}))
"""


def synthesize_omnivoice_in_subprocess(
    *,
    text: str,
    output_path: Path,
    ref_audio: str | None,
    ref_text: str | None,
    instruct: str | None,
    model: str = OMNIVOICE_DEFAULT_MODEL,
    device: str = "cuda:0",
    num_step: int = OMNIVOICE_DEFAULT_NUM_STEP,
    speed: float = 1.0,
    language_id: str | None = None,
    audio_chunk_threshold: float = OMNIVOICE_DEFAULT_CHUNK_THRESHOLD_SEC,
    audio_chunk_duration: float = OMNIVOICE_DEFAULT_CHUNK_DURATION_SEC,
    anchor_text: str | None = None,
) -> None:
    _ = ref_text
    resolved_device = resolve_omnivoice_device(device)
    try:
        plan = plan_official_omnivoice_call(
            text=text,
            speed=speed,
            num_step=num_step,
            language_id=language_id,
            ref_audio=ref_audio,
            anchor_text=anchor_text,
            instruct=instruct,
            audio_chunk_threshold=audio_chunk_threshold,
            audio_chunk_duration=audio_chunk_duration,
        )
    except ValueError as exc:
        raise AppError(
            422,
            ErrorInfo(
                code="OMNIVOICE_BAD_REQUEST",
                message=str(exc),
                action="Provide valid clone ref_text and narration text.",
            ),
        ) from exc

    payload = {
        "output_path": str(output_path),
        "model": (model or OMNIVOICE_DEFAULT_MODEL).strip() or OMNIVOICE_DEFAULT_MODEL,
        "device": resolved_device,
        "plan": plan,
    }
    script = _OFFICIAL_GENERATE_SCRIPT.replace("__PAYLOAD__", json.dumps(json.dumps(payload)))

    python = resolve_omnivoice_python()
    env = build_omnivoice_subprocess_env()
    try:
        completed = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise AppError(
            502,
            ErrorInfo(
                code="OMNIVOICE_TIMEOUT",
                message="OmniVoice synthesis timed out.",
                action="Retry with shorter text or release VRAM, then try again.",
                retryable=True,
            ),
        ) from exc
    except OSError as exc:
        raise AppError(
            502,
            ErrorInfo(
                code="OMNIVOICE_TTS_FAILED",
                message="OmniVoice subprocess could not be started.",
                action="Run 'python scripts/setup_omnivoice.py' to install the isolated virtualenv.",
                detail=str(exc),
                retryable=True,
            ),
        ) from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:] or None
        raise AppError(
            502,
            ErrorInfo(
                code="OMNIVOICE_TTS_FAILED",
                message="OmniVoice could not generate narration.",
                action="Check OmniVoice model, GPU availability, and reference audio settings.",
                detail=detail,
                retryable=True,
            ),
        )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise AppError(
            502,
            ErrorInfo(
                code="OMNIVOICE_TTS_FAILED",
                message="OmniVoice produced an empty audio file.",
                action="Try another reference clip or shorter test text.",
                retryable=True,
            ),
        )
