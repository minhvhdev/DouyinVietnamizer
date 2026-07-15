#!/usr/bin/env python3
"""Audit translation candidate semantic safeguards for a job."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dv_backend.checkpoints import load_checkpoint  # noqa: E402
from dv_backend.semantic_safeguards import evaluate_semantic_safeguards  # noqa: E402


def _load_segments(data_dir: Path, job_id: str) -> list[dict]:
    for step in ("translate", "duration_repair", "tts"):
        cp = load_checkpoint(data_dir, job_id, step)
        if cp and cp.get("segments"):
            return list(cp["segments"])
    return []


def audit_segments(segments: list[dict]) -> dict:
    stats: Counter[str] = Counter()
    rows: list[dict] = []
    for segment in segments:
        source = str(segment.get("text") or "")
        selected_index = int(segment.get("selected_candidate_index") or 0)
        candidates = segment.get("translation_candidates") or []
        natural = next((c for c in candidates if c.get("style") == "natural"), candidates[0] if candidates else {})
        selected = candidates[selected_index] if 0 <= selected_index < len(candidates) else {}
        result = evaluate_semantic_safeguards(
            str(selected.get("text") or segment.get("translation") or ""),
            source_text=source,
            reference_text=str(natural.get("text") or segment.get("translation") or ""),
        )
        for penalty in result.get("penalties") or []:
            stats[penalty] += 1
        rows.append(
            {
                "index": segment.get("index"),
                "source_text": source,
                "natural_candidate": natural.get("text"),
                "selected_candidate": selected.get("text") or segment.get("translation"),
                "penalties": result.get("penalties"),
                "semantic_score": result.get("semantic_score"),
                "critical_violation": result.get("critical_violation"),
                "selected_reason": segment.get("selected_candidate_reason"),
            }
        )
    return {"stats": dict(stats), "segments": rows}


def export_html(path: Path, payload: dict) -> None:
    rows = "".join(
        f"<tr><td>{r['index']}</td><td>{html.escape(str(r.get('source_text') or '')[:60])}</td>"
        f"<td>{html.escape(str(r.get('natural_candidate') or '')[:60])}</td>"
        f"<td>{html.escape(str(r.get('selected_candidate') or '')[:60])}</td>"
        f"<td>{r.get('penalties')}</td><td>{r.get('selected_reason')}</td></tr>"
        for r in payload.get("segments") or []
    )
    path.write_text(
        f"<html><body><h1>Semantic Audit</h1><pre>{json.dumps(payload.get('stats'), indent=2)}</pre>"
        f"<table border='1'><tr><th>Idx</th><th>Source</th><th>Natural</th><th>Selected</th><th>Penalties</th><th>Reason</th></tr>{rows}</table></body></html>",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit translation semantics")
    parser.add_argument("job_id")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".douyin-vietnamizer")
    parser.add_argument("--export-html", action="store_true")
    args = parser.parse_args()

    segments = _load_segments(args.data_dir, args.job_id)
    if not segments:
        print("No segments found", file=sys.stderr)
        return 1
    payload = audit_segments(segments)
    if args.export_html:
        out = args.data_dir / "jobs" / args.job_id / "artifacts" / "semantic_audit.html"
        export_html(out, payload)
        print(f"Wrote {out}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
