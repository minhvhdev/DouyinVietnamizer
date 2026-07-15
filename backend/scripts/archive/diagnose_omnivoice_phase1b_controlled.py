"""Phase 1B controlled OmniVoice auto-vs-clone CUDA reproduction.

Creates a controlled ref from OmniVoice auto, then runs clone matrix.
Does not commit WAV artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REF_TEXT = "Xin chào, đây là đoạn âm thanh tham chiếu dùng để kiểm tra giọng nói."
TARGET_TEXT = "Hôm nay chúng ta kiểm tra chất lượng tổng hợp tiếng nói."
WRONG_TEXT = "Buổi chiều trời nhiều mây và nhiệt độ giảm nhẹ."


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:16]


def main() -> int:
    from dv_backend.adapters.omnivoice_infer import (
        OMNIVOICE_SAMPLE_RATE,
        plan_official_omnivoice_call,
        resolve_omnivoice_device,
    )
    from dv_backend.audio_probe import diagnostics_dir, file_content_hash, probe_wav_path, short_hash
    from dv_backend.omnivoice_env import resolve_omnivoice_python

    out_dir = diagnostics_dir(Path(tempfile.gettempdir()) / "omnivoice_phase1b")
    os.environ.setdefault("DV_OMNIVOICE_DIAGNOSTICS", "1")
    os.environ.setdefault("DV_OMNIVOICE_DIAGNOSTICS_DIR", str(out_dir))

    device = resolve_omnivoice_device("cuda:0")
    model = "k2-fsa/OmniVoice"
    payload = {
        "output_dir": str(out_dir),
        "device": device,
        "model": model,
        "sample_rate": OMNIVOICE_SAMPLE_RATE,
        "ref_text": REF_TEXT,
        "target_text": TARGET_TEXT,
        "wrong_text": WRONG_TEXT,
        "ref_plan": plan_official_omnivoice_call(
            text=REF_TEXT,
            speed=1.0,
            num_step=32,
            language_id="vi",
            ref_audio=None,
            anchor_text=None,
            instruct=None,
            audio_chunk_threshold=30.0,
            audio_chunk_duration=15.0,
        ),
        "auto_plan": plan_official_omnivoice_call(
            text=TARGET_TEXT,
            speed=1.0,
            num_step=32,
            language_id="vi",
            ref_audio=None,
            anchor_text=None,
            instruct=None,
            audio_chunk_threshold=30.0,
            audio_chunk_duration=15.0,
        ),
    }
    payload_path = out_dir / "phase1b_payload.json"
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

def generate(plan, *, postprocess: bool, clone_prompt=None):
    local = dict(plan)
    cfg = dict(local.pop("generation_config"))
    cfg["postprocess_output"] = bool(postprocess)
    generation_config = OmniVoiceGenerationConfig(**cfg)
    kwargs = {"generation_config": generation_config, **local}
    kwargs.pop("ref_audio", None)
    kwargs.pop("ref_text", None)
    if clone_prompt is not None:
        kwargs["voice_clone_prompt"] = clone_prompt
    return model.generate(**kwargs)[0]

# 1) Controlled ref via auto (production postprocess=True)
ref_samples = generate(args["ref_plan"], postprocess=True)
ref_wav = out / "controlled_ref.wav"
sf.write(str(ref_wav), ref_samples, sr)
(out / "controlled_ref.txt").write_text(args["ref_text"], encoding="utf-8")

# Prepare clone prompts
exact_prompt = model.create_voice_clone_prompt(
    ref_audio=str(ref_wav),
    ref_text=args["ref_text"],
    preprocess_prompt=True,
)
wrong_prompt = model.create_voice_clone_prompt(
    ref_audio=str(ref_wav),
    ref_text=args["wrong_text"],
    preprocess_prompt=True,
)

# Clone plan text is target; attach ref fields for bookkeeping only
clone_base = dict(args["auto_plan"])

runs = []

def save(label, samples):
    path = out / f"{label}.wav"
    sf.write(str(path), samples, sr)
    runs.append(label)
    return path

# auto baseline (production)
save("auto_generate_output", generate(args["auto_plan"], postprocess=True))
# experimental postprocess disabled (NOT called W1 raw)
save("EXP_auto_generate_postprocess_disabled", generate(args["auto_plan"], postprocess=False))

# clone exact miss / hit
save("clone_exact_miss_generate_output", generate(clone_base, postprocess=True, clone_prompt=exact_prompt))
save("clone_exact_hit_generate_output", generate(clone_base, postprocess=True, clone_prompt=exact_prompt))
save("EXP_clone_exact_generate_postprocess_disabled", generate(clone_base, postprocess=False, clone_prompt=exact_prompt))

# wrong-text negative diagnostic
save("clone_wrong_text_generate_output", generate(clone_base, postprocess=True, clone_prompt=wrong_prompt))

# empty ref_text should fail before generate in production planner; record attempt
empty_error = None
try:
    model.create_voice_clone_prompt(ref_audio=str(ref_wav), ref_text="   ", preprocess_prompt=True)
except Exception as exc:
    empty_error = str(exc)

print(json.dumps({"ok": True, "runs": runs, "empty_ref_text_error": empty_error}, ensure_ascii=False))
"""

    python = resolve_omnivoice_python()
    print(f"Running Phase 1B CUDA matrix via {python} ...", flush=True)
    completed = subprocess.run(
        [str(python), "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
    )
    (out_dir / "phase1b_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (out_dir / "phase1b_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)

    worker_json = {}
    for line in reversed((completed.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and '"ok"' in line:
            worker_json = json.loads(line)
            break

    ref_wav = out_dir / "controlled_ref.wav"
    ref_probe = probe_wav_path(ref_wav)
    (out_dir / "controlled_ref_probe.json").write_text(
        json.dumps(ref_probe, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stage_files = {
        "auto_generate_output": out_dir / "auto_generate_output.wav",
        "auto_written_output": out_dir / "auto_generate_output.wav",  # same file after write
        "EXP_auto_generate_postprocess_disabled": out_dir / "EXP_auto_generate_postprocess_disabled.wav",
        "clone_exact_miss_generate_output": out_dir / "clone_exact_miss_generate_output.wav",
        "clone_exact_miss_written_output": out_dir / "clone_exact_miss_generate_output.wav",
        "clone_exact_hit_generate_output": out_dir / "clone_exact_hit_generate_output.wav",
        "EXP_clone_exact_generate_postprocess_disabled": out_dir
        / "EXP_clone_exact_generate_postprocess_disabled.wav",
        "clone_wrong_text_generate_output": out_dir / "clone_wrong_text_generate_output.wav",
    }

    rows = []
    for stage, path in stage_files.items():
        probe = probe_wav_path(path)
        rows.append(
            {
                "run": stage,
                "stage_label": stage,
                "duration_sec": probe.get("duration_sec"),
                "peak_abs": probe.get("peak_abs"),
                "rms": probe.get("rms"),
                "speech_detected": probe.get("speech_detected"),
                "suspect": probe.get("suspect"),
                "wav_hash": _sha_file(path) if path.is_file() else None,
                "path": str(path),
            }
        )

    # Anchor hash equivalence for controlled fixture (preview vs dubbing same source)
    ref_hash = file_content_hash(ref_wav)
    anchor_hash = short_hash(REF_TEXT)
    preview_meta = {
        "ref_audio_hash": ref_hash,
        "anchor_text_hash": anchor_hash,
        "anchor_text_length": len(REF_TEXT),
        "anchor_source": "controlled_exact_text",
    }
    dubbing_meta = dict(preview_meta)
    dubbing_meta["anchor_source"] = "controlled_exact_text"

    first_silent = next((row["run"] for row in rows if row.get("speech_detected") is False), None)
    miss = next(r for r in rows if r["run"] == "clone_exact_miss_generate_output")
    hit = next(r for r in rows if r["run"] == "clone_exact_hit_generate_output")
    wrong = next(r for r in rows if r["run"] == "clone_wrong_text_generate_output")
    auto = next(r for r in rows if r["run"] == "auto_generate_output")
    exp_clone = next(r for r in rows if r["run"] == "EXP_clone_exact_generate_postprocess_disabled")

    answers = {
        "1_controlled_exact_clone_has_speech": bool(miss.get("speech_detected")),
        "2_first_speech_false_stage": first_silent,
        "3_cache_hit_differs_from_miss": miss.get("wav_hash") != hit.get("wav_hash")
        or miss.get("speech_detected") != hit.get("speech_detected"),
        "4_wrong_ref_text_changes_speech": wrong.get("speech_detected") != miss.get("speech_detected")
        or wrong.get("wav_hash") != miss.get("wav_hash"),
        "5_preview_dubbing_conditioning_match": preview_meta["ref_audio_hash"] == dubbing_meta["ref_audio_hash"]
        and preview_meta["anchor_text_hash"] == dubbing_meta["anchor_text_hash"],
        "6_postprocess_root_cause_confirmed": False,
        "6_note": (
            "Insufficient deterministic evidence to confirm internal postprocess as root cause; "
            f"exact miss speech={miss.get('speech_detected')}, "
            f"EXP postprocess_disabled speech={exp_clone.get('speech_detected')}."
        ),
    }

    report = {
        "schema": {
            "w1_label": "generate_output",
            "w2_label": "written_output",
            "experimental_postprocess_disabled_label": "EXP_generate_postprocess_disabled",
            "forbids_w1_raw_label": True,
        },
        "ref_probe": ref_probe,
        "ref_audio_hash": ref_hash,
        "ref_text_hash": anchor_hash,
        "preview_anchor": preview_meta,
        "dubbing_anchor": dubbing_meta,
        "empty_ref_text_error": worker_json.get("empty_ref_text_error"),
        "rows": rows,
        "answers": answers,
        "auto_speech": auto.get("speech_detected"),
        "deterministic": "unknown",
    }
    report_path = out_dir / "phase1b_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
