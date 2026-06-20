import json
from pathlib import Path

import pytest

from dv_backend.vendor import VendorManifest, VendorResolver


def write_manifest(path: Path, executable: str = "ffmpeg/ffmpeg.exe") -> Path:
    path.write_text(json.dumps({
        "schema_version": 1,
        "tools": [{
            "id": "ffmpeg",
            "display_name": "FFmpeg",
            "executable": executable,
            "dev_command": "ffmpeg",
            "version_args": ["-version"],
            "version_contains": "ffmpeg",
            "required": True,
            "capability": "media"
        }]
    }), encoding="utf-8")
    return path


def test_manifest_parses_valid_tool(tmp_path: Path) -> None:
    manifest = VendorManifest.load(write_manifest(tmp_path / "manifest.json"))
    assert manifest.tools[0].id == "ffmpeg"
    assert manifest.tools[0].required is True


def test_manifest_accepts_windows_utf8_bom(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "manifest.json")
    path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8-sig")

    assert VendorManifest.load(path).tools[0].id == "ffmpeg"


def test_manifest_rejects_unsafe_executable_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        VendorManifest.load(write_manifest(tmp_path / "manifest.json", "../ffmpeg.exe"))


def test_resolver_can_disable_path_fallback(tmp_path: Path, monkeypatch) -> None:
    tool = VendorManifest.load(write_manifest(tmp_path / "manifest.json")).tools[0]
    monkeypatch.setattr("shutil.which", lambda _name: "C:/tools/ffmpeg.exe")

    resolved = VendorResolver(tmp_path, allow_path_tools=False).resolve(tool)

    assert resolved.path is None
    assert resolved.source == "missing"


def test_development_path_fallback_is_default(tmp_path: Path, monkeypatch) -> None:
    tool = VendorManifest.load(write_manifest(tmp_path / "manifest.json")).tools[0]
    monkeypatch.setattr("shutil.which", lambda _name: "C:/tools/ffmpeg.exe")

    resolved = VendorResolver(tmp_path).resolve(tool)

    assert resolved.path == Path("C:/tools/ffmpeg.exe")
    assert resolved.source == "path"
