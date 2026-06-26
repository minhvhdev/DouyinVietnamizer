from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dv_backend.adapters.diarization.service import run_diarization_with_fallback
from dv_backend.diarization_models import DiarizationOptions, DiarizationResult, DiarizationTimeline
from dv_backend.errors import AppError
from dv_backend.models import ErrorInfo


def test_run_diarization_falls_back_when_pyannote_not_bootstrapped(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    options = DiarizationOptions(device="cpu", model_cache_dir=str(tmp_path / "pyannote"))
    fallback_result = DiarizationResult(
        regular=DiarizationTimeline(backend="funasr_campp", model="m", device="cpu", turns=[]),
        exclusive=DiarizationTimeline(backend="funasr_campp", model="m", device="cpu", turns=[]),
    )

    pyannote_error = AppError(
        503,
        ErrorInfo(
            code="PYANNOTE_MODEL_NOT_BOOTSTRAPPED",
            message="missing",
            action="bootstrap",
            retryable=True,
        ),
    )

    with patch(
        "dv_backend.adapters.diarization.service.PyannoteCommunity1Backend.diarize",
        side_effect=pyannote_error,
    ), patch(
        "dv_backend.adapters.diarization.service.FunASRCampPlusBackend.diarize",
        return_value=fallback_result,
    ) as funasr_diarize, patch(
        "dv_backend.adapters.diarization.service.gpu_lease",
        return_value=MagicMock(__enter__=lambda self: None, __exit__=lambda *args: None),
    ):
        result, backend_used, used_fallback, reason = run_diarization_with_fallback(
            audio,
            options,
            primary_backend="pyannote_community_1",
            fallback_backend="funasr_campp",
            job_id="job-1",
        )

    assert backend_used == "funasr_campp"
    assert used_fallback == "funasr_campp"
    assert reason == "PYANNOTE_MODEL_NOT_BOOTSTRAPPED"
    assert result is fallback_result
    funasr_diarize.assert_called_once()
