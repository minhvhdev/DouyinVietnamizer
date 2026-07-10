"""ASR verification for OmniVoice clone TTS."""
from __future__ import annotations

import argparse
import re
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_DEFAULT = (
    "Tôi nghĩ con trai tôi có lẽ là gay. "
    "Tôi phát hiện ra chuyện đó thế nào ư? "
    "Hôm đó, chúng tôi đang ngồi ăn cơm cùng nhau."
)


def norm(text: str) -> str:
    cleaned = text.lower()
    cleaned = cleaned.replace("gây", "gay").replace("gầy", "gay")
    cleaned = re.sub(r"[^\w]", "", cleaned, flags=re.UNICODE)
    return cleaned


def sentence_order_ok(expected: str, got: str) -> bool:
    from dv_backend.adapters.tts import split_omnivoice_sentences

    got_norm = norm(got)
    position = 0
    for sentence in split_omnivoice_sentences(expected):
        sentence_norm = norm(sentence)
        if not sentence_norm:
            continue
        index = got_norm.find(sentence_norm, position)
        if index < 0:
            words = re.findall(r"[a-zà-ỹ0-9]+", sentence.lower(), flags=re.UNICODE)
            if len(words) < 2:
                return False
            word_positions = [got_norm.find(norm(word), position) for word in words if norm(word)]
            if any(item < 0 for item in word_positions):
                return False
            if word_positions != sorted(word_positions):
                return False
            position = max(word_positions) + len(words[-1])
            continue
        position = index + len(sentence_norm)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=EXPECTED_DEFAULT)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--vendor-dir", default=str(ROOT.parent / "vendor"))
    parser.add_argument("--output", default=str(Path("C:/Users/Admin/AppData/Local/Temp/omni_verify_final.wav")))
    args = parser.parse_args()

    from dv_backend.adapters.asr import transcribe_audio
    from dv_backend.adapters.omnivoice_tts import OmniVoiceTtsAdapter

    output = Path(args.output)
    adapter = OmniVoiceTtsAdapter(num_step=32, speed=1.0, language_id="vi")
    adapter.synthesize(args.text.strip(), output, voice=args.ref_audio)

    with wave.open(str(output), "rb") as wav_file:
        duration = wav_file.getnframes() / float(wav_file.getframerate())

    segments = transcribe_audio(
        output,
        vendor_dir=Path(args.vendor_dir),
        language="Vietnamese",
        device="cuda:0",
    )
    transcript = " ".join(str(item.get("text") or "").strip() for item in segments).strip()
    got_l = transcript.lower()
    expected_words = re.findall(r"[a-zà-ỹ0-9]+", args.text.lower(), flags=re.UNICODE)
    got_for_words = got_l.replace("gây", "gay").replace("gầy", "gay")
    recall = (
        sum(1 for word in expected_words if word in got_for_words) / len(expected_words)
        if expected_words
        else 0.0
    )
    ordered = sentence_order_ok(args.text, transcript)
    first_ok = "nghĩ" in got_l and (
        got_l.find("nghĩ") < got_l.find("phát") if "phát" in got_l else True
    )
    start_pos = got_l.find("nghĩ")
    clean_start = start_pos >= 0 and start_pos < 12
    ok = recall >= 0.95 and ordered and first_ok and clean_start and 5.0 <= duration <= 14.0

    print(
        f"duration={duration:.2f}s recall={recall:.3f} ordered={ordered} "
        f"first_ok={first_ok} clean_start={clean_start} ok={ok}"
    )
    print(f"expected={args.text}")
    print(f"got={transcript}")
    print(f"file={output}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
