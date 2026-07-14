from pathlib import Path
import sys
import array
import wave

from unittest.mock import MagicMock



import pytest



from dv_backend.adapters.omnivoice_tts import OmniVoiceTtsAdapter

from dv_backend.adapters.tts import (

    SUPPORTED_TTS_BACKENDS,

    create_tts_adapter,

    estimate_omnivoice_duration_sec,

    resolve_omnivoice_device,

    resolve_tts_voice,

    split_omnivoice_tts_text,

    tts_backend_from_settings,

)

from dv_backend import omnivoice_env





def test_omnivoice_in_supported_backends() -> None:

    assert "omnivoice" in SUPPORTED_TTS_BACKENDS





def test_create_tts_adapter_selects_omnivoice() -> None:

    adapter = create_tts_adapter(

        {

            "tts_backend": "omnivoice",

            "omnivoice_model": "k2-fsa/OmniVoice",

            "omnivoice_device": "cuda:0",

            "omnivoice_num_steps": 16,

            "omnivoice_language_id": "vi",

        }

    )

    assert type(adapter).__name__ == "OmniVoiceTtsAdapter"

    assert adapter.num_step == 16

    assert adapter.speed == 1.0

    assert adapter.language_id == "vi"





def test_resolve_tts_voice_omnivoice_instruct() -> None:

    voice = resolve_tts_voice(

        {

            "tts_backend": "omnivoice",

            "omnivoice_instruct": "female, low pitch",

        }

    )

    assert voice == "instruct:female, low pitch"





def test_resolve_tts_voice_omnivoice_ref_audio() -> None:

    voice = resolve_tts_voice(

        {

            "tts_backend": "omnivoice",

            "omnivoice_ref_audio": "/tmp/ref.wav",

        }

    )

    assert voice == "/tmp/ref.wav"





def test_tts_backend_from_settings_defaults_to_omnivoice() -> None:

    assert tts_backend_from_settings({}) == "omnivoice"

    assert tts_backend_from_settings({"tts_backend": "omnivoice"}) == "omnivoice"





def test_resolve_omnivoice_device_falls_back_to_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:

    class _Cuda:

        @staticmethod

        def is_available() -> bool:

            return False



    class _Torch:

        cuda = _Cuda()



    import sys



    monkeypatch.setitem(sys.modules, "torch", _Torch())

    assert resolve_omnivoice_device("cuda:0") == "cpu"





def test_omnivoice_venv_root_default(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.delenv("DV_OMNIVOICE_VENV", raising=False)

    root = omnivoice_env.omnivoice_venv_root()

    assert root.name == "omnivoice"


def test_resolve_omnivoice_python_prefers_venv_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DV_OMNIVOICE_PYTHON", raising=False)
    resolved = omnivoice_env.resolve_omnivoice_python()
    venv_root = omnivoice_env.omnivoice_venv_root()
    expected = venv_root / ("Scripts" if sys.platform == "win32" else "bin") / (
        "python.exe" if sys.platform == "win32" else "python3"
    )
    assert resolved == expected.resolve()





def test_split_omnivoice_tts_text_prefers_word_boundaries() -> None:

    sentence = " ".join(["tu"] * 80)

    chunks = split_omnivoice_tts_text(sentence, max_chars=120)

    assert len(chunks) >= 2

    assert all(len(chunk) <= 120 for chunk in chunks)

    assert all(not chunk.startswith("tu") or " " in chunk or len(chunk.split()) == 1 for chunk in chunks)





def test_split_omnivoice_tts_text_splits_on_vietnamese_punctuation() -> None:

    text = "Xin chao, ban khoe khong? Toi rat vui duoc gap ban."

    chunks = split_omnivoice_tts_text(text, max_chars=20)

    assert len(chunks) >= 2

    joined = " ".join(chunks)

    assert "Xin chao" in joined

    assert "gap ban" in joined





def test_estimate_omnivoice_duration_scales_with_text_length() -> None:

    short = estimate_omnivoice_duration_sec("Xin chao.")

    long = estimate_omnivoice_duration_sec("Tôi nghĩ con trai tôi có lẽ là gay. Tôi phát hiện ra chuyện đó thế nào ư?")

    assert long > short

    assert 5.0 <= long <= 22.0





def test_resolve_omnivoice_clone_ref_text_preserves_full_transcript() -> None:

    from dv_backend.adapters.omnivoice_infer import resolve_omnivoice_clone_ref_text



    long_text = "xin chào " * 40

    assert resolve_omnivoice_clone_ref_text(f"  {long_text}  ") == long_text.strip()

    assert resolve_omnivoice_clone_ref_text("  xin chào  ") == "xin chào"





def test_plan_official_omnivoice_call_matches_demo_defaults() -> None:

    from dv_backend.adapters.omnivoice_infer import plan_official_omnivoice_call



    plan = plan_official_omnivoice_call(

        text="Xin chào bạn.",

        speed=1.0,

        num_step=32,

        language_id="",

        ref_audio="/tmp/ref.wav",

        anchor_text="xin chào",

        instruct=None,

        audio_chunk_threshold=30.0,

        audio_chunk_duration=15.0,

    )

    assert plan["text"] == "Xin chào bạn."

    assert plan["ref_audio"] == "/tmp/ref.wav"

    assert plan["ref_text"] == "xin chào"

    assert "language" not in plan

    assert "speed" not in plan

    config = plan["generation_config"]

    assert config["num_step"] == 32

    assert config["guidance_scale"] == 2.0

    assert config["denoise"] is True

    assert config["preprocess_prompt"] is True

    assert config["postprocess_output"] is True





def test_plan_official_omnivoice_call_requires_ref_text_for_clone() -> None:

    from dv_backend.adapters.omnivoice_infer import plan_official_omnivoice_call



    with pytest.raises(ValueError, match="ref_text"):

        plan_official_omnivoice_call(

            text="Xin chào bạn.",

            speed=1.0,

            num_step=32,

            language_id="",

            ref_audio="/tmp/ref.wav",

            anchor_text=None,

            instruct=None,

            audio_chunk_threshold=30.0,

            audio_chunk_duration=15.0,

        )





def _write_tone_wav(path: Path, *, duration_sec: float = 0.2, rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(rate * duration_sec)
    samples = array.array("h", [8000 if (index // 100) % 2 == 0 else -8000 for index in range(frames)])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def test_omnivoice_adapter_uses_injected_client(tmp_path: Path) -> None:

    output = tmp_path / "out.wav"

    client = MagicMock()

    def _write_and_ok(**kwargs) -> dict:
        out = Path(kwargs["output_path"])
        _write_tone_wav(out, duration_sec=0.2)
        return {"ok": True, "output_path": str(out)}

    client.synthesize.side_effect = _write_and_ok

    adapter = OmniVoiceTtsAdapter(model="k2-fsa/OmniVoice", _client=client, settings={"omnivoice_fidelity_check_enabled": False})

    adapter.synthesize("Xin chao", output, voice="instruct:female, low pitch")



    client.synthesize.assert_called_once()

    kwargs = client.synthesize.call_args.kwargs

    assert kwargs["instruct"] == "female, low pitch"

    assert kwargs["ref_audio"] is None

    assert kwargs.get("anchor_text") is None


def test_omnivoice_adapter_external_chunks_by_default(tmp_path: Path) -> None:
    output = tmp_path / "out.wav"
    client = MagicMock()

    def _write_and_ok(**kwargs) -> dict:
        out = Path(kwargs["output_path"])
        _write_tone_wav(out, duration_sec=0.2)
        return {"ok": True, "output_path": str(out)}

    client.synthesize.side_effect = _write_and_ok
    adapter = OmniVoiceTtsAdapter(
        model="k2-fsa/OmniVoice",
        _client=client,
        settings={"omnivoice_fidelity_check_enabled": False},
    )
    long_text = " ".join(["Đây là một câu tiếng Việt đủ dài để vượt ngưỡng chunk ký tự cũ." for _ in range(12)])

    adapter.synthesize(long_text, output, voice="instruct:female, low pitch")

    assert client.synthesize.call_count >= 2
    for call in client.synthesize.call_args_list:
        assert len(call.kwargs["text"]) <= 220


def test_omnivoice_adapter_respects_external_chunking_disabled(tmp_path: Path) -> None:
    output = tmp_path / "out.wav"
    client = MagicMock()

    def _write_and_ok(**kwargs) -> dict:
        out = Path(kwargs["output_path"])
        _write_tone_wav(out, duration_sec=0.2)
        return {"ok": True, "output_path": str(out)}

    client.synthesize.side_effect = _write_and_ok
    adapter = OmniVoiceTtsAdapter(
        model="k2-fsa/OmniVoice",
        _client=client,
        settings={
            "omnivoice_fidelity_check_enabled": False,
            "omnivoice_external_chunking_enabled": False,
        },
    )
    long_text = " ".join(["Đây là một câu tiếng Việt đủ dài để vượt ngưỡng chunk ký tự cũ." for _ in range(12)])

    adapter.synthesize(long_text, output, voice="instruct:female, low pitch")

    client.synthesize.assert_called_once()
    assert client.synthesize.call_args.kwargs["text"] == long_text


