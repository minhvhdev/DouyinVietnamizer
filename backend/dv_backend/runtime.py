from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from .config import AppConfig
from .database import Database
from .pyannote_vendor import huggingface_token, pyannote_bootstrap_action, pyannote_model_dir, validate_pyannote_model_dir
from .settings import SettingsService
from .tool_probe import probe_executable
from .vendor import VendorManifest, VendorResolver


class RuntimeCheck(BaseModel):
    id: str
    display_name: str
    status: str
    required: bool
    message: str
    action: str
    detail: str | None = None
    resolved_path: str | None = None
    source: str | None = None
    version: str | None = None
    duration_ms: int = 0


class RuntimeReport(BaseModel):
    status: str
    checked_at: str
    checks: list[RuntimeCheck]


class RuntimeSmokeTestService:
    def __init__(
        self,
        config: AppConfig,
        database: Database,
        manifest_path: Path,
        vendor_dir: Path,
        allow_path_tools: bool = True,
    ) -> None:
        self.config = config
        self.database = database
        self.manifest_path = manifest_path
        self.vendor_dir = vendor_dir
        self.allow_path_tools = allow_path_tools

    def latest(self) -> RuntimeReport | None:
        row = self.database.connection.execute(
            "SELECT report_json FROM runtime_reports ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return RuntimeReport.model_validate_json(row["report_json"]) if row else None

    def run(self) -> RuntimeReport:
        checks = [
            self._check_storage(),
            self._check_sqlite(),
            self._check_qwen3_asr(),
            self._check_omnivoice(),
        ]
        try:
            manifest = VendorManifest.load(self.manifest_path)
            checks.append(RuntimeCheck(
                id="manifest", display_name="Vendor manifest", status="ready", required=True,
                message="Vendor manifest is valid.", action="No action required.",
            ))
            resolver = VendorResolver(self.vendor_dir, self.allow_path_tools)
            for tool in manifest.tools:
                resolved = resolver.resolve(tool)
                if resolved.path is None:
                    checks.append(RuntimeCheck(
                        id=tool.id, display_name=tool.display_name,
                        status="blocked" if tool.required else "warning", required=tool.required,
                        message=f"{tool.display_name} was not found.",
                        action="Install the tool under vendor/ or add it to PATH, then rerun the smoke test.",
                        source="missing",
                    ))
                    continue
                probe = probe_executable(tool, resolved.path)
                status = probe.status
                if status == "blocked" and not tool.required:
                    status = "warning"
                checks.append(RuntimeCheck(
                    id=tool.id, display_name=tool.display_name, status=status, required=tool.required,
                    message=probe.message, action=probe.action,
                    detail=probe.detail, resolved_path=str(resolved.path), source=resolved.source,
                    version=probe.version, duration_ms=probe.duration_ms,
                ))
        except Exception as error:
            checks.append(RuntimeCheck(
                id="manifest", display_name="Vendor manifest", status="blocked", required=True,
                message="Vendor manifest could not be loaded.",
                action="Restore a valid vendor/manifest.json file.", detail=str(error),
            ))
        report = RuntimeReport(
            status=self._aggregate(checks),
            checked_at=datetime.now(timezone.utc).isoformat(),
            checks=checks,
        )
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO runtime_reports (status, report_json, created_at) VALUES (?, ?, ?)",
                (report.status, report.model_dump_json(), report.checked_at),
            )
        return report

    def _check_storage(self) -> RuntimeCheck:
        try:
            self.config.ensure_directories()
            path = self.config.data_dir / f".smoke-{uuid4().hex}"
            path.write_text("ok", encoding="ascii")
            path.unlink()
            return RuntimeCheck(id="storage", display_name="Local storage", status="ready", required=True, message="Local storage is writable.", action="No action required.")
        except OSError as error:
            return RuntimeCheck(id="storage", display_name="Local storage", status="blocked", required=True, message="Local storage is not writable.", action="Choose a writable data directory.", detail=str(error))

    def _check_sqlite(self) -> RuntimeCheck:
        try:
            self.database.connection.execute("SELECT 1").fetchone()
            return RuntimeCheck(id="sqlite", display_name="SQLite", status="ready", required=True, message="SQLite is available.", action="No action required.")
        except Exception as error:
            return RuntimeCheck(id="sqlite", display_name="SQLite", status="blocked", required=True, message="SQLite check failed.", action="Check application data permissions.", detail=str(error))

    def _check_qwen3_asr(self) -> RuntimeCheck:
        try:
            from .adapters.asr import cuda_available, DEFAULT_ASR_MODEL

            if not cuda_available():
                return RuntimeCheck(
                    id="qwen3_asr",
                    display_name="Qwen3-ASR GPU",
                    status="blocked",
                    required=True,
                    message="CUDA GPU is not available for Qwen3-ASR.",
                    action="Install NVIDIA drivers and a CUDA-enabled PyTorch build, then rerun the smoke test.",
                )
            return RuntimeCheck(
                id="qwen3_asr",
                display_name="Qwen3-ASR GPU",
                status="ready",
                required=True,
                message=f"Qwen3-ASR is configured for GPU inference ({DEFAULT_ASR_MODEL}).",
                action="No action required.",
            )
        except Exception as error:
            return RuntimeCheck(
                id="qwen3_asr",
                display_name="Qwen3-ASR GPU",
                status="blocked",
                required=True,
                message="Qwen3-ASR dependencies are not available.",
                action="Install qwen-asr, torch, and soundfile in the backend environment.",
                detail=str(error),
            )

    def _check_omnivoice(self) -> RuntimeCheck:
        from .omnivoice_env import is_omnivoice_available, omnivoice_venv_root
        from .adapters.omnivoice_client import acquire_client, release_all_clients

        if not is_omnivoice_available():
            return RuntimeCheck(
                id="omnivoice",
                display_name="OmniVoice",
                status="blocked",
                required=True,
                message="OmniVoice is not installed in the isolated virtualenv.",
                action="Run 'python scripts/setup_omnivoice.py' in the backend folder.",
                resolved_path=str(omnivoice_venv_root()),
            )
        try:
            client = acquire_client(
                data_dir=self.config.data_dir,
                model="k2-fsa/OmniVoice",
                device="cpu",
                num_steps=8,
            )
            client._ensure_alive()
        except Exception as exc:  # noqa: BLE001
            return RuntimeCheck(
                id="omnivoice",
                display_name="OmniVoice",
                status="blocked",
                required=True,
                message="OmniVoice worker could not be started.",
                action="Re-run 'python scripts/setup_omnivoice.py' and verify the worker script.",
                resolved_path=str(omnivoice_venv_root()),
                detail=str(exc),
            )
        finally:
            release_all_clients()
        return RuntimeCheck(
            id="omnivoice",
            display_name="OmniVoice",
            status="ready",
            required=True,
            message="OmniVoice isolated environment and worker are operational.",
            action="No action required.",
            resolved_path=str(omnivoice_venv_root()),
        )


    def _check_espeak(self) -> RuntimeCheck:
        from .hardware import detect_espeak

        if detect_espeak():
            return RuntimeCheck(
                id="espeak",
                display_name="eSpeak NG",
                status="ready",
                required=False,
                message="eSpeak NG phonemizer is available.",
                action="No action required.",
            )
        return RuntimeCheck(
            id="espeak",
            display_name="eSpeak NG",
            status="warning",
            required=False,
            message="eSpeak NG was not found.",
            action="No action required.",
        )

    @staticmethod
    def _aggregate(checks: list[RuntimeCheck]) -> str:
        if any(check.required and check.status == "blocked" for check in checks):
            return "blocked"
        if any(check.status == "warning" for check in checks):
            return "warning"
        return "ready"


def default_runtime_service(config: AppConfig, database: Database) -> RuntimeSmokeTestService:
    project_root = Path(__file__).resolve().parents[2]
    vendor_dir = Path(os.environ.get("DV_VENDOR_DIR", project_root / "vendor"))
    manifest_path = Path(os.environ.get("DV_VENDOR_MANIFEST", vendor_dir / "manifest.json"))
    return RuntimeSmokeTestService(
        config, database, manifest_path, vendor_dir,
        allow_path_tools=os.environ.get("DV_ALLOW_PATH_TOOLS", "1") == "1",
    )

