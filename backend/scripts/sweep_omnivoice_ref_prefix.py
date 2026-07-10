"""Sweep ref_text prefix lengths for OmniVoice clone quality."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import wave
from pathlib import Path

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


def score(got: str) -> tuple[float, bool, bool]:
    words = re.findall(r"[a-zà-ỹ0-9]+", EXPECTED.lower(), flags=re.UNICODE)
    recall = sum(1 for word in words if word in got.lower()) / len(words)
    got_l = got.lower()
    first_ok = "nghĩ" in got_l and got_l.find("nghĩ") < got_l.find("phát") if "phát" in got_l else "nghĩ" in got_l
    ordered = all(word in got_l for word in ["nghĩ", "phát", "hôm"])
    return recall, first_ok, ordered


def main() -> None:
    from dv_backend.adapters.asr import transcribe_audio
    from dv_backend.adapters.omnivoice_infer import prepare_omnivoice_ref_audio
    from dv_backend.adapters.tts import estimate_omnivoice_duration_sec

    sidecar = REF.with_suffix(".txt").read_text(encoding="utf-8").strip()
    ref_audio = prepare_omnivoice_ref_audio(str(REF), max_sec=6.0)
    target_dur = estimate_omnivoice_duration_sec(EXPECTED, buffer=1.4)

    for prefix_len in (0, 25, 40, 60, 80):
        ref_text = "" if prefix_len == 0 else sidecar[:prefix_len]
        ref_dur = 0.0 if not ref_text else estimate_omnivoice_duration_sec(ref_text, buffer=1.2)
        out = Path(rf"C:\Users\Admin\AppData\Local\Temp\omni_prefix_{prefix_len}.wav")
        payload = {
            "text": EXPECTED,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "duration": (ref_dur + target_dur) if ref_text else target_dur,
            "out": str(out),
            "use_ref": bool(ref_text),
        }
        Path(r"C:/Users/Admin/AppData/Local/Temp/omni_prefix_payload.json").write_text(
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
args = json.loads(open(r"C:/Users/Admin/AppData/Local/Temp/omni_prefix_payload.json", encoding="utf-8").read())
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
kw = dict(text=args["text"], ref_audio=args["ref_audio"], language="vi", num_step=32, speed=1.0,
          postprocess_output=True, audio_chunk_threshold=30.0, duration=args["duration"])
if args["use_ref"]:
    kw["ref_text"] = args["ref_text"]
else:
    kw["ref_text"] = ""
audio = model.generate(**kw)[0]
sf.write(args["out"], audio, 24000)
""",
            ],
            check=True,
            capture_output=True,
        )
        transcript = "".join(
            str(item["text"])
            for item in transcribe_audio(out, vendor_dir=VENDOR, language="Vietnamese", device="cuda:0")
        )
        with wave.open(str(out), "rb") as wav_file:
            duration = wav_file.getnframes() / float(wav_file.getframerate())
        recall, first_ok, ordered = score(transcript)
        print(
            f"prefix={prefix_len} dur={duration:.1f} recall={recall:.3f} "
            f"first_ok={first_ok} ordered={ordered}"
        )
        print(" ", transcript[:160])


if __name__ == "__main__":
    main()
