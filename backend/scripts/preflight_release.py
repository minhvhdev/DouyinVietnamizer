#!/usr/bin/env python3
"""Production release preflight checks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("DV_VENDOR_DIR", str(ROOT.parent / "vendor"))


def _status(level: str, name: str, detail: str = "") -> dict:
    return {"level": level, "check": name, "detail": detail}


def check_python() -> dict:
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 10:
        return _status("PASS", "python_version", f"{major}.{minor}")
    return _status("FAIL", "python_version", f"{major}.{minor}")


def check_ffmpeg_subtitles(ffmpeg_path: str) -> dict:
    try:
        proc = subprocess.run([ffmpeg_path, "-filters"], capture_output=True, text=True, check=False)
        text = proc.stdout + proc.stderr
        if "subtitles" not in text and " ass " not in text:
            return _status("FAIL", "ffmpeg_subtitles_filter", "subtitles/ass filter not listed")
    except OSError as exc:
        return _status("FAIL", "ffmpeg_executable", str(exc))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ass_path = tmp_path / "test.ass"
        frame_path = tmp_path / "frame.png"
        ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 640
PlayResY: 360
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,32,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,3,2,0,2,20,20,20,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Preflight box test
"""
        ass_path.write_text(ass_content, encoding="utf-8-sig")
        vf = f"subtitles='{ass_path.as_posix()}'"
        try:
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=orange:s=640x360:d=2",
                    "-vf",
                    vf,
                    "-frames:v",
                    "1",
                    str(frame_path),
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            return _status("FAIL", "libass_render", (exc.stderr or exc.stdout or b"").decode(errors="replace")[:400])

        if not frame_path.is_file():
            return _status("FAIL", "libass_render", "no output frame")

        try:
            from PIL import Image

            image = Image.open(frame_path).convert("RGB")
            dark = 0
            total = 0
            w, h = image.size
            for y in range(int(h * 0.7), int(h * 0.95)):
                for x in range(int(w * 0.1), int(w * 0.9)):
                    r, g, b = image.getpixel((x, y))
                    total += 1
                    if r < 80 and g < 80 and b < 80:
                        dark += 1
            if dark / max(total, 1) < 0.01:
                return _status("FAIL", "libass_render", "ASS box not visible in rendered frame")
        except ImportError:
            return _status("WARNING", "libass_render", "PIL unavailable; render succeeded but not verified")

    return _status("PASS", "libass_render", ffmpeg_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Production release preflight")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    from dv_backend.adapters.subtitles import build_ass_content, ffmpeg_subtitles_filter, subtitles_filter_available
    from dv_backend.config import AppConfig
    from dv_backend.vendor import VendorResolver

    config = AppConfig.from_env() if args.data_dir is None else AppConfig(args.data_dir)
    config.ensure_directories()
    results: list[dict] = []

    results.append(check_python())

    resolver = VendorResolver(config.vendor_dir)
    ffmpeg_path = shutil.which("ffmpeg") or str(resolver.resolve("ffmpeg"))
    results.append(_status("PASS" if Path(ffmpeg_path).exists() else "FAIL", "ffmpeg_path", ffmpeg_path))

    if subtitles_filter_available(ffmpeg_path):
        results.append(_status("PASS", "subtitles_filter_listed", ffmpeg_path))
    else:
        results.append(_status("FAIL", "subtitles_filter_listed", ffmpeg_path))
    results.append(check_ffmpeg_subtitles(ffmpeg_path))

    try:
        import torch

        accel = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        results.append(_status("PASS" if accel != "cpu" else "WARNING", "accelerator", accel))
    except ImportError:
        results.append(_status("WARNING", "pytorch", "not installed"))

    free = shutil.disk_usage(config.data_dir).free
    results.append(_status("PASS" if free > 500_000_000 else "WARNING", "disk_free_bytes", str(free)))

    try:
        conn = sqlite3.connect(config.database_path)
        conn.execute("CREATE TABLE IF NOT EXISTS _preflight (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO _preflight DEFAULT VALUES")
        conn.commit()
        conn.close()
        results.append(_status("PASS", "sqlite_write", str(config.database_path)))
    except OSError as exc:
        results.append(_status("FAIL", "sqlite_write", str(exc)))

    cp_test = config.data_dir / "artifacts" / "_preflight.tmp"
    try:
        cp_test.write_text("ok", encoding="utf-8")
        cp_test.unlink()
        results.append(_status("PASS", "data_dir_writable", str(config.data_dir)))
    except OSError as exc:
        results.append(_status("FAIL", "data_dir_writable", str(exc)))

    blocking = [r for r in results if r["level"] == "FAIL"]
    warnings = [r for r in results if r["level"] == "WARNING"]
    report = {"results": results, "blocking_count": len(blocking), "warning_count": len(warnings)}
    out = config.data_dir / "artifacts" / "preflight_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if blocking:
        return 2
    if warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
