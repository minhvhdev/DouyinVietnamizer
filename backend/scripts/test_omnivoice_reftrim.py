"""Test ref_text + trim approach for OmniVoice clone TTS."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPECTED = (
    "Tôi nghĩ con trai tôi có lẽ là gay. "
    "Tôi phát hiện ra chuyện đó thế nào ư? "
    "Hôm đó, chúng tôi đang ngồi ăn cơm cùng nhau."
)
REF = Path(
    r"C:\Users\Admin\AppData\Local\DouyinVietnamizer\cloned_voices_omnivoice"
    r"\4a8b0c1f-2acf-49b4-8e99-ff3731b8f4e8.wav"
)
VENDOR = ROOT.parent / "vendor"
PY = ROOT / "venvs" / "omnivoice" / "Scripts" / "python.exe"


def main() -> None:
    from dv_backend.adapters.asr import transcribe_audio
    from dv_backend.adapters.omnivoice_infer import prepare_omnivoice_ref_audio
    from dv_backend.adapters.tts import estimate_omnivoice_duration_sec

    ref_text = REF.with_suffix(".txt").read_text(encoding="utf-8").strip()[:160]
    ref_audio = prepare_omnivoice_ref_audio(str(REF), max_sec=6.0)
    ref_dur = estimate_omnivoice_duration_sec(ref_text, buffer=1.35)
    target_dur = estimate_omnivoice_duration_sec(EXPECTED, buffer=1.35)
    out = Path(r"C:\Users\Admin\AppData\Local\Temp\omni_reftrim.wav")
    payload = {
        "text": EXPECTED,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "duration": ref_dur + target_dur,
        "out": str(out),
    }
    payload_path = Path(r"C:\Users\Admin\AppData\Local\Temp\omni_reftrim_payload.json")
    script_path = Path(r"C:\Users\Admin\AppData\Local\Temp\omni_reftrim_script.py")
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    script_path.write_text(
        """
import json
import soundfile as sf
import torch
from omnivoice import OmniVoice

args = json.loads(open(r"C:/Users/Admin/AppData/Local/Temp/omni_reftrim_payload.json", encoding="utf-8").read())
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
audio = model.generate(
    text=args["text"],
    ref_audio=args["ref_audio"],
    ref_text=args["ref_text"],
    language="vi",
    num_step=32,
    speed=1.0,
    duration=args["duration"],
    postprocess_output=True,
    audio_chunk_threshold=30.0,
)[0]
sf.write(args["out"], audio, 24000)
""",
        encoding="utf-8",
    )
    subprocess.run([str(PY), str(script_path)], check=True)

    with wave.open(str(out), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        params = wav_file.getparams()
        data = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
    cut = int(ref_dur * sample_rate)
    trimmed = Path(r"C:\Users\Admin\AppData\Local\Temp\omni_reftrim_cut.wav")
    with wave.open(str(trimmed), "wb") as wav_file:
        wav_file.setparams(params)
        wav_file.writeframes(data[cut:].tobytes())

    for label, path in [("raw", out), ("trim", trimmed)]:
        transcript = "".join(
            str(item["text"])
            for item in transcribe_audio(path, vendor_dir=VENDOR, language="Vietnamese", device="cuda:0")
        )
        with wave.open(str(path), "rb") as wav_file:
            duration = wav_file.getnframes() / float(wav_file.getframerate())
        print(label, f"dur={duration:.1f}", transcript)


if __name__ == "__main__":
    main()
