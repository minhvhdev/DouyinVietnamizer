"""Test trim by ref audio duration."""
from __future__ import annotations

import json
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
    from dv_backend.adapters.tts import estimate_omnivoice_duration_sec, split_omnivoice_sentences

    ref_audio = prepare_omnivoice_ref_audio(str(REF), max_sec=6.0)
    with wave.open(ref_audio, "rb") as wav_file:
        ref_sec = wav_file.getnframes() / float(wav_file.getframerate())

    ref_text = REF.with_suffix(".txt").read_text(encoding="utf-8").strip()
    ref_text = ref_text[: max(40, min(100, int(ref_sec * 12)))]
    target_dur = estimate_omnivoice_duration_sec(EXPECTED, buffer=1.35)
    duration = ref_sec + target_dur
    out = Path(r"C:\Users\Admin\AppData\Local\Temp\omni_reftrim3.wav")
    payload = {
        "text": EXPECTED,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "duration": duration,
        "out": str(out),
    }
    Path(r"C:/Users/Admin/AppData/Local/Temp/omni_reftrim3_payload.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    subprocess.run(
        [
            str(PY),
            "-c",
            """
import json
import soundfile as sf
import torch
from omnivoice import OmniVoice
args = json.loads(open(r"C:/Users/Admin/AppData/Local/Temp/omni_reftrim3_payload.json", encoding="utf-8").read())
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
audio = model.generate(
    text=args["text"], ref_audio=args["ref_audio"], ref_text=args["ref_text"],
    language="vi", num_step=32, speed=1.0, duration=args["duration"],
    postprocess_output=True, audio_chunk_threshold=30.0,
)[0]
sf.write(args["out"], audio, 24000)
""",
        ],
        check=True,
    )

    with wave.open(str(out), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        params = wav_file.getparams()
        data = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
    cut = int(ref_sec * sample_rate)
    trimmed = Path(r"C:\Users\Admin\AppData\Local\Temp\omni_reftrim3_cut.wav")
    with wave.open(str(trimmed), "wb") as wav_file:
        wav_file.setparams(params)
        wav_file.writeframes(data[cut:].tobytes())

    def score(got: str) -> tuple[float, bool]:
        import re

        words = re.findall(r"[a-zà-ỹ0-9]+", EXPECTED.lower(), flags=re.UNICODE)
        recall = sum(1 for word in words if word in got.lower()) / len(words)
        ordered = True
        pos = 0
        got_l = got.lower()
        for sentence in split_omnivoice_sentences(EXPECTED):
            sw = re.findall(r"[a-zà-ỹ0-9]+", sentence.lower(), flags=re.UNICODE)
            if len(sw) < 2:
                continue
            idx = got_l.find(sw[0], pos)
            if idx < 0:
                ordered = False
                break
            pos = idx
        return recall, ordered

    for label, path in [("raw", out), ("trim", trimmed)]:
        transcript = "".join(
            str(item["text"])
            for item in transcribe_audio(path, vendor_dir=VENDOR, language="Vietnamese", device="cuda:0")
        )
        with wave.open(str(path), "rb") as wav_file:
            duration = wav_file.getnframes() / float(wav_file.getframerate())
        recall, ordered = score(transcript)
        print(label, f"dur={duration:.1f}", f"recall={recall:.3f}", f"ordered={ordered}")
        print(transcript)


if __name__ == "__main__":
    main()
