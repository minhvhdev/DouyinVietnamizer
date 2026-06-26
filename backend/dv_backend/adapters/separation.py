import logging
import shutil
import sys
from pathlib import Path

import soundfile as sf
import torch

from ..errors import AppError
from ..models import ErrorInfo

logger = logging.getLogger(__name__)

MIX_MODE_DUCK = "duck"
MIX_MODE_SEPARATE = "separate"
SUPPORTED_MIX_MODES = {MIX_MODE_DUCK, MIX_MODE_SEPARATE}
DEFAULT_DEMUCS_MODEL = "htdemucs_ft"
FALLBACK_DEMUCS_MODEL = "htdemucs"


def _save_stem_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    from demucs.audio import prevent_clip

    tensor = prevent_clip(wav, mode="rescale").detach().cpu()
    if tensor.dim() == 2:
        tensor = tensor.transpose(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), tensor.numpy(), sample_rate, subtype="PCM_16")


def separate_vocals(
    input_wav: Path,
    *,
    vocals_out: Path,
    bgm_out: Path,
    ffmpeg_path: Path,
    device: str = "cuda",
    job_id: str,
    runner,
) -> None:
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    from demucs.pretrained import get_model

    if not input_wav.is_file():
        raise AppError(
            400,
            ErrorInfo(
                code="MISSING_AUDIO_FILE",
                message="Source audio for vocal separation is missing.",
                action="Resume the extract_audio step.",
            ),
        )

    if runner and runner.is_cancelled(job_id):
        raise AppError(
            400,
            ErrorInfo(
                code="JOB_CANCELLED",
                message="The job was cancelled by the user.",
                action="Create a new job to start over.",
            ),
        )

    work_dir = input_wav.parent / "demucs_stems"
    if work_dir.is_dir():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    demucs_device = (
        "cuda"
        if str(device).lower().startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    stem_name = "vocals"
    raw_vocals = work_dir / "vocals_raw.wav"
    raw_bgm = work_dir / "no_vocals_raw.wav"

    logger.info(
        "Running Demucs vocal separation on %s (device=%s)",
        input_wav,
        demucs_device,
    )
    try:
        try:
            model = get_model(DEFAULT_DEMUCS_MODEL)
        except Exception:
            logger.warning(
                "Demucs model %s unavailable, falling back to %s",
                DEFAULT_DEMUCS_MODEL,
                FALLBACK_DEMUCS_MODEL,
            )
            model = get_model(FALLBACK_DEMUCS_MODEL)
        model.to(demucs_device)
        model.eval()

        if stem_name not in model.sources:
            raise RuntimeError(
                f"Demucs model does not expose '{stem_name}' stem: {model.sources}"
            )

        wav = AudioFile(input_wav).read(
            streams=0,
            samplerate=model.samplerate,
            channels=model.audio_channels,
        )
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(
            model,
            wav[None],
            device=demucs_device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
        sources = sources * ref.std() + ref.mean()

        vocals = sources[model.sources.index(stem_name)]
        background = torch.zeros_like(sources[0])
        for index, name in enumerate(model.sources):
            if name != stem_name:
                background += sources[index]

        _save_stem_wav(raw_vocals, vocals, model.samplerate)
        _save_stem_wav(raw_bgm, background, model.samplerate)
    except AppError:
        raise
    except Exception as error:
        raise AppError(
            502,
            ErrorInfo(
                code="VOCAL_SEPARATION_FAILED",
                message="Demucs could not separate vocals from background audio.",
                action="Install demucs in the backend environment or switch mix mode to duck.",
                detail=str(error),
                retryable=True,
            ),
        ) from error
    finally:
        if demucs_device == "cuda":
            torch.cuda.empty_cache()

    vocals_out.parent.mkdir(parents=True, exist_ok=True)
    _normalize_to_48k_stereo(ffmpeg_path, raw_vocals, vocals_out, job_id, runner)
    _normalize_to_48k_stereo(ffmpeg_path, raw_bgm, bgm_out, job_id, runner)

    shutil.rmtree(work_dir, ignore_errors=True)
    logger.info("Vocal separation complete: %s, %s", vocals_out, bgm_out)


def _normalize_to_48k_stereo(
    ffmpeg_path: Path,
    source_path: Path,
    output_path: Path,
    job_id: str,
    runner,
) -> None:
    from ..pipeline import run_subprocess_with_cancel

    cmd = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(source_path),
        "-ar",
        "48000",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    run_subprocess_with_cancel(cmd, job_id, runner)
