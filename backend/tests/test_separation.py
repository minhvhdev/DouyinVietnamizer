from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from dv_backend.adapters.separation import separate_vocals
from dv_backend.errors import AppError


@patch("dv_backend.adapters.separation._normalize_to_48k_stereo")
@patch("demucs.apply.apply_model")
@patch("demucs.pretrained.get_model")
@patch("demucs.audio.AudioFile")
def test_separate_vocals_writes_stems(
    mock_audio_file,
    mock_get_model,
    mock_apply_model,
    mock_normalize,
    tmp_path: Path,
) -> None:
    input_wav = tmp_path / "original_48k.wav"
    input_wav.write_bytes(b"wav")
    vocals_out = tmp_path / "vocals.wav"
    bgm_out = tmp_path / "bgm.wav"
    ffmpeg_path = tmp_path / "ffmpeg.exe"

    mock_model = MagicMock()
    mock_model.samplerate = 44100
    mock_model.audio_channels = 2
    mock_model.sources = ["drums", "bass", "other", "vocals"]
    mock_get_model.return_value = mock_model

    mock_audio_file.return_value.read.return_value = torch.randn(2, 44100)
    mock_apply_model.return_value = [torch.randn(4, 2, 44100)]

    runner = MagicMock()
    runner.is_cancelled.return_value = False

    separate_vocals(
        input_wav,
        vocals_out=vocals_out,
        bgm_out=bgm_out,
        ffmpeg_path=ffmpeg_path,
        device="cuda",
        job_id="job-1",
        runner=runner,
    )

    mock_get_model.assert_called_once_with("htdemucs")
    mock_apply_model.assert_called_once()
    assert mock_normalize.call_count == 2


def test_separate_vocals_missing_input(tmp_path: Path) -> None:
    with pytest.raises(AppError) as exc:
        separate_vocals(
            tmp_path / "missing.wav",
            vocals_out=tmp_path / "vocals.wav",
            bgm_out=tmp_path / "bgm.wav",
            ffmpeg_path=tmp_path / "ffmpeg.exe",
            job_id="job-1",
            runner=MagicMock(),
        )

    assert exc.value.info.code == "MISSING_AUDIO_FILE"
