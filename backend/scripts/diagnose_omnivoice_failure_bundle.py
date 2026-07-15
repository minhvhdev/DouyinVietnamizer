"""Replay an OmniVoice clone failure bundle with controlled experiments.

Requires a bundle from Phase 2 capture:
  <dir>/<request-id>_clone_failure/{manifest.json, ref_audio.*, ref_text.txt, target_text.txt, ...}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bundle(bundle: Path) -> dict:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing manifest.json in {bundle}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ref_candidates = list(bundle.glob("ref_audio.*"))
    if not ref_candidates:
        raise SystemExit("Missing ref_audio.* in bundle")
    ref_audio = ref_candidates[0]
    expected = str(manifest.get("ref_audio_sha256") or "")
    actual = _sha256(ref_audio)
    if expected and actual != expected and actual[: len(expected)] != expected and expected not in actual:
        # Accept short or full hashes from capture.
        if not (expected.startswith(actual[:12]) or actual.startswith(expected[:12])):
            raise SystemExit(f"ref_audio hash mismatch: expected={expected} actual={actual}")
    ref_text_path = bundle / "ref_text.txt"
    target_text_path = bundle / "target_text.txt"
    if not ref_text_path.is_file() or not target_text_path.is_file():
        raise SystemExit("Missing ref_text.txt or target_text.txt")
    return {
        "manifest": manifest,
        "ref_audio": ref_audio,
        "ref_text": ref_text_path.read_text(encoding="utf-8"),
        "target_text": target_text_path.read_text(encoding="utf-8"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    from dv_backend.audio_probe import probe_wav_path, short_hash
    from dv_backend.omnivoice_env import resolve_omnivoice_python

    bundle = Path(args.bundle)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verified = _verify_bundle(bundle)
    payload = {
        "output_dir": str(out_dir),
        "ref_audio": str(verified["ref_audio"]),
        "ref_text": verified["ref_text"],
        "target_text": verified["target_text"],
        "runs": max(1, int(args.runs)),
        "device": "cuda:0",
        "model": "k2-fsa/OmniVoice",
    }
    payload_path = out_dir / "replay_payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    script = r'''
import json, unicodedata
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

args = json.loads(Path(r''' + json.dumps(str(payload_path)) + r''').read_text(encoding="utf-8"))
out = Path(args["output_dir"])
dtype = torch.float16 if str(args.get("device","")).startswith("cuda") else torch.float32
model = OmniVoice.from_pretrained(args["model"], device_map=args["device"], dtype=dtype)
sr_expected = 24000

raw_audio, sr = sf.read(args["ref_audio"], always_2d=True)
def to_mono(x):
    return x.mean(axis=1) if x.ndim == 2 and x.shape[1] > 1 else x.reshape(-1)

def write_case(name, audio, rate, text, n):
    path = out / f"{name}_run{n}.wav"
    # save temporary ref
    ref_path = out / f"_ref_{name}.wav"
    sf.write(str(ref_path), audio, rate)
    prompt = model.create_voice_clone_prompt(ref_audio=str(ref_path), ref_text=text, preprocess_prompt=True)
    cfg = OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0, denoise=True, preprocess_prompt=True, postprocess_output=True, audio_chunk_threshold=30.0, audio_chunk_duration=15.0)
    samples = model.generate(text=args["target_text"], language="vi", generation_config=cfg, voice_clone_prompt=prompt)[0]
    sf.write(str(path), samples, model.sampling_rate)
    return str(path)

mono = to_mono(np.asarray(raw_audio, dtype=np.float64))
# silence trim experimental
abs_vals = np.abs(mono)
thr = 0.01
idx = np.where(abs_vals >= thr)[0]
if len(idx):
    trimmed = mono[idx[0]: idx[-1]+1]
else:
    trimmed = mono
norm_text = " ".join(unicodedata.normalize("NFC", args["ref_text"]).replace("\ufeff","").split())

results = []
for n in range(1, int(args["runs"])+1):
    results.append(("exact", write_case("exact", mono, sr, args["ref_text"], n)))
    # decoded-canonical: re-encode float mono wav (no loudness normalize)
    results.append(("decoded_canonical", write_case("decoded_canonical", mono.astype(np.float32), sr, args["ref_text"], n)))
    results.append(("silence_trimmed", write_case("silence_trimmed", trimmed.astype(np.float32), sr, args["ref_text"], n)))
    results.append(("text_normalized", write_case("text_normalized", mono, sr, norm_text, n)))
    results.append(("canonical_text_normalized", write_case("canonical_text_normalized", mono.astype(np.float32), sr, norm_text, n)))
print(json.dumps({"ok": True, "results": results}, ensure_ascii=False))
'''
    python = resolve_omnivoice_python()
    completed = subprocess.run(
        [str(python), "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
    )
    (out_dir / "replay_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (out_dir / "replay_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)

    rows = []
    for path in sorted(out_dir.glob("*_run*.wav")):
        probe = probe_wav_path(path)
        case = path.name.rsplit("_run", 1)[0]
        rows.append(
            {
                "case": case,
                "path": str(path),
                "duration_sec": probe.get("duration_sec"),
                "peak_abs": probe.get("peak_abs"),
                "rms": probe.get("rms"),
                "speech_detected": probe.get("speech_detected"),
            }
        )
    summary: dict[str, dict] = {}
    for row in rows:
        bucket = summary.setdefault(row["case"], {"runs": 0, "silent_runs": 0, "durations": [], "rms": []})
        bucket["runs"] += 1
        if not row["speech_detected"]:
            bucket["silent_runs"] += 1
        bucket["durations"].append(row["duration_sec"])
        bucket["rms"].append(row["rms"])
    report = {
        "bundle": str(bundle),
        "ref_text_hash": short_hash(verified["ref_text"]),
        "target_text_hash": short_hash(verified["target_text"]),
        "rows": rows,
        "summary": summary,
    }
    report_path = out_dir / "replay_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
