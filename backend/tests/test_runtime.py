import json
from pathlib import Path
import sys
from unittest.mock import patch

from dv_backend.config import AppConfig
from dv_backend.database import Database
from dv_backend.runtime import RuntimeCheck, RuntimeSmokeTestService

READY_VIENEU = RuntimeCheck(
    id="vieneu", display_name="VieNeu-TTS", status="ready", required=True,
    message="ok", action="none",
)
READY_ESPEAK = RuntimeCheck(
    id="espeak", display_name="eSpeak NG", status="ready", required=True,
    message="ok", action="none",
)


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
    with (
        patch("dv_backend.adapters.asr.cuda_available", return_value=True),
        patch("dv_backend.runtime.RuntimeSmokeTestService._check_vieneu", return_value=READY_VIENEU),
        patch("dv_backend.runtime.RuntimeSmokeTestService._check_espeak", return_value=READY_ESPEAK),
    ):
        report = service(tmp_path).run()
    assert report.status == "ready"
    assert report.checks[-1].source == "path"
    with (
        patch("dv_backend.adapters.asr.cuda_available", return_value=True),
        patch("dv_backend.runtime.RuntimeSmokeTestService._check_vieneu", return_value=READY_VIENEU),
        patch("dv_backend.runtime.RuntimeSmokeTestService._check_espeak", return_value=READY_ESPEAK),
    ):
        assert service(tmp_path).latest().status == "ready"


def test_missing_required_tool_blocks_runtime(tmp_path: Path) -> None:
    report = service(tmp_path, allow_path=False).run()
    assert report.status == "blocked"


def test_missing_optional_tool_only_warns(tmp_path: Path) -> None:
    with (
        patch("dv_backend.adapters.asr.cuda_available", return_value=True),
        patch("dv_backend.runtime.RuntimeSmokeTestService._check_vieneu", return_value=READY_VIENEU),
        patch("dv_backend.runtime.RuntimeSmokeTestService._check_espeak", return_value=READY_ESPEAK),
    ):
        report = service(tmp_path, required=False, allow_path=False).run()
    assert report.status == "warning"
