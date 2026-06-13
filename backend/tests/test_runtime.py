import json
from pathlib import Path
import sys

from dv_backend.config import AppConfig
from dv_backend.database import Database
from dv_backend.runtime import RuntimeSmokeTestService


def write_manifest(path: Path, required: bool = True) -> Path:
    path.write_text(json.dumps({
        "schema_version": 1,
        "tools": [{
            "id": "python",
            "display_name": "Python fixture",
            "executable": "missing/python.exe",
            "dev_command": Path(sys.executable).name,
            "version_args": ["-c", "print('fixture-ready')"],
            "version_contains": "fixture-ready",
            "required": required,
            "capability": "fixture"
        }]
    }), encoding="utf-8")
    return path


def service(tmp_path: Path, required: bool = True, allow_path: bool = True):
    database = Database(tmp_path / "app.db")
    database.migrate()
    return RuntimeSmokeTestService(
        AppConfig(tmp_path),
        database,
        write_manifest(tmp_path / "manifest.json", required),
        tmp_path / "vendor",
        allow_path_tools=allow_path,
    )


def test_smoke_test_ready_and_persisted(tmp_path: Path) -> None:
    report = service(tmp_path).run()
    assert report.status == "warning"
    assert report.checks[-1].source == "path"
    assert service(tmp_path).latest().status == "warning"


def test_missing_required_tool_blocks_runtime(tmp_path: Path) -> None:
    report = service(tmp_path, allow_path=False).run()
    assert report.status == "blocked"


def test_missing_optional_tool_only_warns(tmp_path: Path) -> None:
    report = service(tmp_path, required=False, allow_path=False).run()
    assert report.status == "warning"

