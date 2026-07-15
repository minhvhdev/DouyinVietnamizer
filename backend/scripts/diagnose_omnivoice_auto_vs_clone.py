"""Diagnose OmniVoice auto vs clone silence using production call shape.

Example:
  set DV_OMNIVOICE_DIAGNOSTICS=1
  python scripts/diagnose_omnivoice_auto_vs_clone.py ^
    --text "Xin chào, đây là bản nghe thử." ^
    --ref-audio C:\\path\\voice.wav ^
    --ref-text "transcript khớp audio mẫu" ^
    --output-dir %TEMP%\\omnivoice-diagnose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _probe_row(stage: str, probe: dict) -> dict:
    return {
        "stage": stage,
        "duration_sec": probe.get("duration_sec"),
        "peak_abs": probe.get("peak_abs"),
        "rms": probe.get("rms"),
        "speech_detected": probe.get("speech_detected"),
        "suspect": probe.get("suspect"),
        "error": probe.get("error"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare OmniVoice auto vs clone for one sentence.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language-id", default="vi")
    parser.add_argument("--num-step", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    args = parser.parse_args()

    from dv_backend.adapters.omnivoice_infer import (
        OMNIVOICE_SAMPLE_RATE,
        plan_official_omnivoice_call,
        resolve_omnivoice_device,
    )
    from dv_backend.audio_probe import file_content_hash, probe_wav_path, probe_waveform, short_hash
    from dv_backend.omnivoice_env import resolve_omnivoice_python

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_audio = str(Path(args.ref_audio).resolve())
    ref_text = str(args.ref_text).strip()
    text = str(args.text).strip()
    if not Path(ref_audio).is_file():
        raise SystemExit(f"ref-audio not found: {ref_audio}")
    if not ref_text:
        raise SystemExit("ref-text is required for clone")
    if not text:
        raise SystemExit("text is required")

    # Use isolated OmniVoice python via an inline script so we match production deps.
    payload = {
        "text": text,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "language_id": args.language_id,
        "num_step": int(args.num_step),
        "device": resolve_omnivoice_device(args.device),
        "model": args.model,
        "output_dir": str(out_dir),
        "auto_plan": plan_official_omnivoice_call(
            text=text,
            speed=1.0,
            num_step=int(args.num_step),
            language_id=args.language_id,
            ref_audio=None,
            anchor_text=None,
            instruct=None,
            audio_chunk_threshold=30.0,
            audio_chunk_duration=15.0,
        ),
        "clone_plan": plan_official_omnivoice_call(
            text=text,
            speed=1.0,
            num_step=int(args.num_step),
            language_id=args.language_id,
            ref_audio=ref_audio,
            anchor_text=ref_text,
            instruct=None,
            audio_chunk_threshold=30.0,
            audio_chunk_duration=15.0,
        ),
        "sample_rate": OMNIVOICE_SAMPLE_RATE,
    }
    payload_path = out_dir / "payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    script = r"""
import json
from pathlib import Path
import soundfile as sf
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

args = json.loads(Path(r""" + json.dumps(str(payload_path)) + r""").read_text(encoding="utf-8"))
out = Path(args["output_dir"])
dtype = torch.float16 if str(args.get("device", "")).startswith("cuda") else torch.float32
model = OmniVoice.from_pretrained(args["model"], device_map=args["device"], dtype=dtype)
sr = int(args["sample_rate"])

def run(label, plan, *, postprocess: bool, clone_prompt=None):
    local = dict(plan)
    cfg = dict(local.pop("generation_config"))
    cfg["postprocess_output"] = bool(postprocess)
    generation_config = OmniVoiceGenerationConfig(**cfg)
    kwargs = {"generation_config": generation_config, **local}
    kwargs.pop("ref_audio", None)
    kwargs.pop("ref_text", None)
    if clone_prompt is not None:
        kwargs["voice_clone_prompt"] = clone_prompt
    samples = model.generate(**kwargs)[0]
    path = out / f"{label}.wav"
    sf.write(str(path), samples, sr)
    return path

# auto raw/post
run("auto_raw", args["auto_plan"], postprocess=False)
run("auto_post", args["auto_plan"], postprocess=True)

# clone prompt once, reuse for cache-hit check
clone_ref_audio = args["clone_plan"]["ref_audio"]
clone_ref_text = args["clone_plan"]["ref_text"]
prompt = model.create_voice_clone_prompt(
    ref_audio=clone_ref_audio,
    ref_text=clone_ref_text,
    preprocess_prompt=True,
)
run("clone_raw", args["clone_plan"], postprocess=False, clone_prompt=prompt)
run("clone_post", args["clone_plan"], postprocess=True, clone_prompt=prompt)
# second clone post uses same prompt object => cache-hit equivalent
run("clone_cache_hit_post", args["clone_plan"], postprocess=True, clone_prompt=prompt)
print(json.dumps({"ok": True}))
"""
    import subprocess

    python = resolve_omnivoice_python()
    completed = subprocess.run(
        [str(python), "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)

    stages = [
        "auto_raw",
        "auto_post",
        "clone_raw",
        "clone_post",
        "clone_cache_hit_post",
    ]
    rows = []
    for stage in stages:
        path = out_dir / f"{stage}.wav"
        rows.append({"run": stage, **_probe_row(stage, probe_wav_path(path)), "path": str(path)})

    report = {
        "text_length": len(text),
        "text_hash": short_hash(text),
        "ref_audio_hash": file_content_hash(ref_audio),
        "ref_text_hash": short_hash(ref_text),
        "ref_text_length": len(ref_text),
        "ref_probe": probe_wav_path(ref_audio),
        "rows": rows,
        "first_silent_clone_stage": next(
            (row["run"] for row in rows if row["run"].startswith("clone") and not row.get("speech_detected")),
            None,
        ),
        "stderr_tail": (completed.stderr or "")[-2000:],
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
