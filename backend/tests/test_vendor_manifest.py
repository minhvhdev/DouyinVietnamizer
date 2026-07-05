import json
from pathlib import Path

from dv_backend.vendor import VendorManifest, VendorResolver


def _write_manifest(path: Path, executable: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tools": [
                    {
                        "id": "ffmpeg",
                        "display_name": "FFmpeg",
                        "executable": executable,
                        "dev_command": "ffmpeg",
                        "version_args": ["-version"],
                        "version_contains": "ffmpeg",
                        "required": True,
                        "capability": "media",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_repo_manifest_resolves_against_dev_vendor_dir(tmp_path: Path) -> None:
    repo_manifest = Path(__file__).resolve().parents[2] / "vendor" / "manifest.json"
    vendor_dir = tmp_path / "vendor"
    ffmpeg_dir = vendor_dir / "ffmpeg"
    ffmpeg_dir.mkdir(parents=True)
    ffmpeg_path = ffmpeg_dir / "ffmpeg.exe"
    ffmpeg_path.write_text("binary", encoding="utf-8")

    manifest = VendorManifest.load(repo_manifest)
    resolved = VendorResolver(vendor_dir, allow_path_tools=False).resolve(manifest.tools[0])
    assert resolved.path == ffmpeg_path


def test_repo_manifest_resolves_against_portable_tools_dir(tmp_path: Path) -> None:
    repo_manifest = Path(__file__).resolve().parents[2] / "vendor" / "manifest.json"
    tools_root = tmp_path / "portable-runtime" / "tools"
    ffmpeg_dir = tools_root / "ffmpeg"
    ffmpeg_dir.mkdir(parents=True)
    ffmpeg_path = ffmpeg_dir / "ffmpeg.exe"
    ffmpeg_path.write_text("binary", encoding="utf-8")

    manifest = VendorManifest.load(repo_manifest)
    resolved = VendorResolver(tools_root, allow_path_tools=False).resolve(manifest.tools[0])
    assert resolved.path == ffmpeg_path


def test_legacy_tools_prefix_still_resolves(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools" / "ffmpeg"
    tools_dir.mkdir(parents=True)
    ffmpeg_path = tools_dir / "ffmpeg.exe"
    ffmpeg_path.write_text("binary", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, "tools/ffmpeg/ffmpeg.exe")

    manifest = VendorManifest.load(manifest_path)
    resolved = VendorResolver(tools_dir.parent, allow_path_tools=False).resolve(manifest.tools[0])
    assert resolved.path == ffmpeg_path
