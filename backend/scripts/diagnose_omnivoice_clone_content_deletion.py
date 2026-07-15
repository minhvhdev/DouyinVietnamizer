"""Phase 2R: diagnose OmniVoice clone content deletion (mất chữ).

Uses production planner + official create_voice_clone_prompt / generate().
Does not use legacy prepend/ref-trim.

Example:
  uv run python scripts/diagnose_omnivoice_clone_content_deletion.py \\
    --voice capcut_female_vn \\
    --text "xin chào? bạn là ai? Tôi là Minh?  rất vui được làm quen với bạn" \\
    --output-dir %TEMP%\\omnivoice_content_deletion
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_TEXT = "xin chào? bạn là ai? Tôi là Minh?  rất vui được làm quen với bạn"
CRITICAL = "Tôi là Minh"
CONTROLLED_REF_TEXT = "Xin chào, đây là đoạn âm thanh tham chiếu dùng để kiểm tra giọng nói."


def _resolve_voice(name: str) -> tuple[str, str]:
    db = Path(os.environ.get("LOCALAPPDATA", "")) / "DouyinVietnamizer" / "app.db"
    if not db.is_file():
        raise SystemExit(f"app.db not found: {db}")
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT id, name, wav_filename, transcript FROM cloned_voices WHERE lower(name)=lower(?)",
        (name,),
    ).fetchone()
    con.close()
    if row is None:
        raise SystemExit(f"voice not found: {name}")
    wav = Path(os.environ.get("LOCALAPPDATA", "")) / "DouyinVietnamizer" / "cloned_voices_omnivoice" / str(
        row["wav_filename"]
    )
    if not wav.is_file():
        raise SystemExit(f"voice wav missing: {wav}")
    transcript = str(row["transcript"] or "").strip()
    if not transcript:
        raise SystemExit(f"voice transcript empty: {name}")
    return str(wav), transcript


def _summarize_case(rows: list[dict]) -> dict:
    missing_critical = sum(1 for row in rows if not row.get("critical_ok"))
    missing_any = sum(1 for row in rows if row.get("missing_any_clause"))
    coverages = [float(row.get("ordered_token_coverage") or 0.0) for row in rows]
    return {
        "runs": len(rows),
        "missing_critical": missing_critical,
        "missing_any_clause": missing_any,
        "ordered_coverage_mean": round(sum(coverages) / max(1, len(coverages)), 4),
        "ordered_coverage_min": round(min(coverages) if coverages else 0.0, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="capcut_female_vn")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--ref-audio", default="")
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--clause-runs", type=int, default=3)
    parser.add_argument("--language-id", default="vi")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--skip-asr", action="store_true")
    args = parser.parse_args()

    from dv_backend.adapters.omnivoice_infer import (
        OMNIVOICE_SAMPLE_RATE,
        plan_official_omnivoice_call,
        resolve_omnivoice_device,
    )
    from dv_backend.audio_probe import probe_wav_path, short_hash
    from dv_backend.omnivoice_content_fidelity import (
        describe_target_text_for_generate,
        evaluate_content_fidelity,
        normalize_content_compare_text,
        plan_clone_semantic_chunks,
        split_content_clauses,
    )
    from dv_backend.omnivoice_env import resolve_omnivoice_python
    from dv_backend.omnivoice_wav_concat import concat_omnivoice_chunks
    from dv_backend.tts_fidelity import transcribe_wav_to_text

    if args.ref_audio and args.ref_text:
        ref_audio, ref_text = str(Path(args.ref_audio).resolve()), str(args.ref_text).strip()
    else:
        ref_audio, ref_text = _resolve_voice(args.voice)

    out_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.gettempdir()) / "omnivoice_content_deletion"
    out_dir.mkdir(parents=True, exist_ok=True)
    vendor_dir = Path(os.environ.get("DV_VENDOR_DIR", ROOT.parent / "vendor"))
    if not vendor_dir.is_dir():
        vendor_dir = ROOT.parent / "vendor"

    text = str(args.text)
    target_meta_auto = describe_target_text_for_generate(text, mode="auto")
    target_meta_clone = describe_target_text_for_generate(text, mode="clone")
    if target_meta_auto["target_text_sha256"] != target_meta_clone["target_text_sha256"]:
        raise SystemExit("auto/clone target hash mismatch before planning")

    auto_plan = plan_official_omnivoice_call(
        text=text,
        speed=1.0,
        num_step=32,
        language_id=args.language_id,
        ref_audio=None,
        anchor_text=None,
        instruct=None,
        audio_chunk_threshold=30.0,
        audio_chunk_duration=15.0,
    )
    clone_plan = plan_official_omnivoice_call(
        text=text,
        speed=1.0,
        num_step=32,
        language_id=args.language_id,
        ref_audio=ref_audio,
        anchor_text=ref_text,
        instruct=None,
        audio_chunk_threshold=30.0,
        audio_chunk_duration=15.0,
    )
    if auto_plan["text"] != clone_plan["text"]:
        raise SystemExit(
            f"planner text diverged: auto={short_hash(auto_plan['text'])} clone={short_hash(clone_plan['text'])}"
        )

    cases: list[dict] = [
        {"id": "A1_auto_original", "mode": "auto", "text": text, "runs": args.runs},
        {"id": "A2_clone_original", "mode": "clone", "text": text, "runs": args.runs, "ref": "capcut"},
        {"id": "A3_controlled_clone", "mode": "clone", "text": text, "runs": args.runs, "ref": "controlled"},
        {
            "id": "B1_no_double_space",
            "mode": "clone",
            "text": "xin chào? bạn là ai? Tôi là Minh? rất vui được làm quen với bạn",
            "runs": args.runs,
            "ref": "capcut",
        },
        {
            "id": "B2_periods",
            "mode": "clone",
            "text": "xin chào. bạn là ai. Tôi là Minh. rất vui được làm quen với bạn.",
            "runs": args.runs,
            "ref": "capcut",
        },
        {
            "id": "B3_commas",
            "mode": "clone",
            "text": "xin chào, bạn là ai, Tôi là Minh, rất vui được làm quen với bạn",
            "runs": args.runs,
            "ref": "capcut",
        },
        {
            "id": "B4_no_punctuation",
            "mode": "clone",
            "text": "xin chào bạn là ai Tôi là Minh rất vui được làm quen với bạn",
            "runs": args.runs,
            "ref": "capcut",
        },
    ]
    for index, clause in enumerate(split_content_clauses(text), start=1):
        cases.append(
            {
                "id": f"C{index}_isolated",
                "mode": "clone",
                "text": clause,
                "runs": args.clause_runs,
                "ref": "capcut",
                "expected_override": clause,
                "critical_override": [CRITICAL] if CRITICAL.lower() in clause.lower() else [],
            }
        )
    cases.append(
        {
            "id": "D1_balanced",
            "mode": "clone_chunks",
            "chunks": plan_clone_semantic_chunks(text, strategy="d1"),
            "runs": args.runs,
            "ref": "capcut",
        }
    )
    cases.append(
        {
            "id": "D2_balanced",
            "mode": "clone_chunks",
            "chunks": plan_clone_semantic_chunks(text, strategy="d2"),
            "runs": args.runs,
            "ref": "capcut",
        }
    )

    payload = {
        "output_dir": str(out_dir),
        "device": resolve_omnivoice_device(args.device),
        "model": args.model,
        "sample_rate": OMNIVOICE_SAMPLE_RATE,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "controlled_ref_text": CONTROLLED_REF_TEXT,
        "cases": cases,
        "auto_plan_template": auto_plan,
        "clone_plan_template": clone_plan,
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

def make_plan(text, *, clone=False, ref_audio=None, ref_text=None):
    from copy import deepcopy
    if clone:
        plan = deepcopy(args["clone_plan_template"])
        plan["text"] = text
        if ref_audio is not None:
            plan["ref_audio"] = ref_audio
        if ref_text is not None:
            plan["ref_text"] = ref_text
        return plan
    plan = deepcopy(args["auto_plan_template"])
    plan["text"] = text
    return plan

def generate(path, plan, clone_prompt=None):
    local = dict(plan)
    cfg = OmniVoiceGenerationConfig(**dict(local.pop("generation_config")))
    kwargs = {"generation_config": cfg, **local}
    kwargs.pop("ref_audio", None)
    kwargs.pop("ref_text", None)
    if clone_prompt is not None:
        kwargs["voice_clone_prompt"] = clone_prompt
    samples = model.generate(**kwargs)[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr)
    return {
        "path": str(path),
        "target_text": str(plan.get("text") or ""),
        "target_len": len(str(plan.get("text") or "")),
    }

# controlled ref once
controlled_ref = out / "controlled_ref.wav"
ref_plan = make_plan(args["controlled_ref_text"], clone=False)
generate(controlled_ref, ref_plan, clone_prompt=None)

capcut_prompt = model.create_voice_clone_prompt(
    ref_audio=args["ref_audio"],
    ref_text=args["ref_text"],
    preprocess_prompt=True,
)
controlled_prompt = model.create_voice_clone_prompt(
    ref_audio=str(controlled_ref),
    ref_text=args["controlled_ref_text"],
    preprocess_prompt=True,
)

manifest = {"controlled_ref": str(controlled_ref), "results": []}
for case in args["cases"]:
    case_id = case["id"]
    runs = int(case.get("runs") or 1)
    mode = case["mode"]
    for run_i in range(runs):
        run_dir = out / case_id / f"run_{run_i}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if mode == "auto":
            plan = make_plan(case["text"], clone=False)
            meta = generate(run_dir / "out.wav", plan, clone_prompt=None)
            meta.update({"case_id": case_id, "run": run_i, "mode": mode, "chunks": None})
            manifest["results"].append(meta)
            continue
        if mode == "clone":
            prompt = controlled_prompt if case.get("ref") == "controlled" else capcut_prompt
            ref_a = str(controlled_ref) if case.get("ref") == "controlled" else args["ref_audio"]
            ref_t = args["controlled_ref_text"] if case.get("ref") == "controlled" else args["ref_text"]
            plan = make_plan(case["text"], clone=True, ref_audio=ref_a, ref_text=ref_t)
            meta = generate(run_dir / "out.wav", plan, clone_prompt=prompt)
            meta.update({"case_id": case_id, "run": run_i, "mode": mode, "chunks": None})
            manifest["results"].append(meta)
            continue
        if mode == "clone_chunks":
            chunks = list(case["chunks"])
            chunk_paths = []
            for ci, chunk_text in enumerate(chunks):
                plan = make_plan(chunk_text, clone=True, ref_audio=args["ref_audio"], ref_text=args["ref_text"])
                cpath = run_dir / f"chunk_{ci}.wav"
                generate(cpath, plan, clone_prompt=capcut_prompt)
                chunk_paths.append(str(cpath))
            manifest["results"].append({
                "case_id": case_id,
                "run": run_i,
                "mode": mode,
                "path": None,
                "chunks": chunk_paths,
                "target_text": " ".join(chunks),
                "target_len": len(" ".join(chunks)),
            })
            continue
        raise RuntimeError(f"unknown mode {mode}")

(out / "synth_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"ok": True, "results": len(manifest["results"])}))
"""
    python = resolve_omnivoice_python()
    print(f"Running CUDA matrix via {python} ...")
    completed = subprocess.run(
        [str(python), "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
    )
    (out_dir / "synth_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (out_dir / "synth_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)

    synth = json.loads((out_dir / "synth_manifest.json").read_text(encoding="utf-8"))
    case_lookup = {case["id"]: case for case in cases}
    evaluated: dict[str, list[dict]] = {}

    for item in synth["results"]:
        case = case_lookup[item["case_id"]]
        expected = str(case.get("expected_override") or case.get("text") or text)
        critical = list(case.get("critical_override") if "critical_override" in case else [CRITICAL])
        out_path: Path
        if item.get("chunks"):
            out_path = out_dir / item["case_id"] / f"run_{item['run']}" / "concat.wav"
            chunk_paths = [Path(p) for p in item["chunks"]]
            pauses = [120] * max(0, len(chunk_paths) - 1)
            concat_omnivoice_chunks(chunk_paths, pause_ms_list=pauses, output_path=out_path)
            expected = text
            critical = [CRITICAL]
        else:
            out_path = Path(item["path"])

        probe = probe_wav_path(out_path)
        heard = ""
        fidelity = {}
        if not args.skip_asr:
            heard = transcribe_wav_to_text(out_path, vendor_dir=vendor_dir, language="Vietnamese")
            fidelity = evaluate_content_fidelity(
                expected_text=expected,
                recognized_text=heard,
                critical_phrases=critical,
            )
        critical_key = normalize_content_compare_text(CRITICAL) if critical else ""
        row = {
            "case_id": item["case_id"],
            "run": item["run"],
            "path": str(out_path),
            "speech_detected": probe.get("speech_detected"),
            "duration_sec": probe.get("duration_sec"),
            "heard": heard,
            "critical_ok": bool(fidelity.get("critical_phrases", {}).get(critical_key, True)) if critical else True,
            "missing_any_clause": bool(fidelity.get("missing_any_clause")),
            "ordered_token_coverage": fidelity.get("ordered_token_coverage"),
            "ordered_clause_ok": fidelity.get("ordered_clause_ok"),
            "missing_clauses": fidelity.get("missing_clauses"),
            "target_text_len": item.get("target_len"),
            "pre_generate_target_sha256": short_hash(str(item.get("target_text") or "")),
        }
        evaluated.setdefault(item["case_id"], []).append(row)

    summary = {case_id: _summarize_case(rows) for case_id, rows in evaluated.items()}
    report = {
        "voice": args.voice,
        "ref_audio": ref_audio,
        "ref_text_len": len(ref_text),
        "text": text,
        "target_meta_auto": target_meta_auto,
        "target_meta_clone": target_meta_clone,
        "planner_text_equal": auto_plan["text"] == clone_plan["text"],
        "planner_text": auto_plan["text"],
        "contains_toi_la_minh_before_generate": CRITICAL in auto_plan["text"],
        "summary": summary,
        "rows": evaluated,
        "stderr_tail": (completed.stderr or "")[-3000:],
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "report": str(report_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
