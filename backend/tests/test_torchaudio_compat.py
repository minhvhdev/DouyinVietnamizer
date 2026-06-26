from dv_backend.torchaudio_compat import apply_torchaudio_compat


def test_apply_torchaudio_compat_restores_metadata_api() -> None:
    apply_torchaudio_compat()
    import torchaudio

    assert hasattr(torchaudio, "AudioMetaData")
    assert hasattr(torchaudio, "info")
    assert hasattr(torchaudio, "list_audio_backends")
    assert "soundfile" in torchaudio.list_audio_backends()
