#!/usr/bin/env python3
"""Voice duration profile convergence report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dv_backend.database import Database  # noqa: E402
from dv_backend.config import AppConfig  # noqa: E402
from dv_backend.voice_profile_policy import profile_convergence_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate voice duration profile convergence")
    parser.add_argument("--voice", default=None, help="Voice ID hint (optional)")
    parser.add_argument("--export-html", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    config = AppConfig.from_env() if args.data_dir is None else AppConfig(args.data_dir)
    database = Database(config.database_path)
    database.migrate()
    rows = database.connection.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: json.loads(r["value"]) for r in rows}

    report = profile_convergence_report(config.data_dir, settings)
    if args.voice and args.voice not in str(report.get("voice_id")):
        print(f"Note: active voice_id={report.get('voice_id')}", file=sys.stderr)

    out_json = config.data_dir / "artifacts" / "voice_profile_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.export_html:
        out_html = config.data_dir / "artifacts" / "voice_profile_report.html"
        out_html.write_text(
            f"<html><body><h1>Voice Profile Report</h1><pre>{json.dumps(report, ensure_ascii=False, indent=2)}</pre></body></html>",
            encoding="utf-8",
        )
        print(f"Wrote {out_html}")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
