from __future__ import annotations

import gc
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from uuid import uuid4

from pydantic import BaseModel, Field

from .config import AppConfig
from .database import Database
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
    gpu: RuntimeGpuStatus | None = None


class RuntimeGpuStatus(BaseModel):
    cuda_supported: bool
    device_name: str | None = None
    total_vram_mb: float | None = None
    used_vram_mb: float | None = None
    free_vram_mb: float | None = None
    torch_allocated_mb: float | None = None
    torch_reserved_mb: float | None = None
    torch_peak_mb: float | None = None
    active_omnivoice_clients: int = 0
    resident_models: list[str] = Field(default_factory=list)
    helper_processes: list[str] = Field(default_factory=list)


class ReleaseVramResult(BaseModel):
    status: str
    released: list[str] = Field(default_factory=list)
    terminated_processes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    gpu: RuntimeGpuStatus


def _attach_gpu_status(report: RuntimeReport) -> RuntimeReport:
    return report.model_copy(update={"gpu": collect_runtime_gpu_status()})


def _truncate_process_detail(text: str, *, limit: int = 180) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def _list_gpu_helper_processes() -> list[str]:
    try:
        if os.name == "nt":
            script = (
                "$targets = Get-CimInstance Win32_Process | Where-Object { "
                "$_.Name -in @('llama-tts-server.exe') -or "
                "((($_.CommandLine -as [string]) -ne '') -and ($_.CommandLine -match 'dv_backend\\.adapters\\.omnivoice_worker')) "
                "}; "
                "foreach ($p in $targets) { "
                "$cmd = ($p.CommandLine -as [string]); "
                "Write-Output ('{0} ({1}) {2}' -f $p.Name, $p.ProcessId, $cmd) "
                "}"
            )
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            return [
                _truncate_process_detail(line)
                for line in completed.stdout.splitlines()
                if line.strip()
            ]
        completed = subprocess.run(
            ["ps", "-ax", "-o", "pid=", "-o", "command="],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        matches: list[str] = []
        for line in completed.stdout.splitlines():
            lowered = line.lower()
            if (
                "llama-tts-server" in lowered
                or "dv_backend.adapters.omnivoice_worker" in line
            ):
                matches.append(_truncate_process_detail(line))
        return matches
    except Exception:
        return []


def _terminate_gpu_helper_processes() -> list[str]:
    processes = _list_gpu_helper_processes()
    try:
        if os.name == "nt":
            script = (
                "$targets = Get-CimInstance Win32_Process | Where-Object { "
                "$_.Name -in @('llama-tts-server.exe') -or "
                "((($_.CommandLine -as [string]) -ne '') -and ($_.CommandLine -match 'dv_backend\\.adapters\\.omnivoice_worker')) "
                "}; "
                "foreach ($p in $targets) { "
                "Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue "
                "}"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
        else:
            completed = subprocess.run(
                ["ps", "-ax", "-o", "pid=", "-o", "command="],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            for line in completed.stdout.splitlines():
                lowered = line.lower()
                if (
                    "llama-tts-server" not in lowered
                    and "dv_backend.adapters.omnivoice_worker" not in line
                ):
                    continue
                pid_text = line.strip().split(maxsplit=1)[0]
                try:
                    os.kill(int(pid_text), signal.SIGKILL)
                except Exception:
                    pass
    except Exception:
        pass
    return processes


def _clear_torch_cuda_state() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats()
            return
        if sys.platform == "darwin" and torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def collect_runtime_gpu_status() -> RuntimeGpuStatus:
    from .adapters.omnivoice_client import client_debug_snapshot
    from .gpu_manager import global_gpu_manager

    client_snapshot = client_debug_snapshot()
    manager_snapshot = global_gpu_manager().snapshot()
    helper_processes = _list_gpu_helper_processes()

    status = RuntimeGpuStatus(
        cuda_supported=False,
        active_omnivoice_clients=int(client_snapshot.get("count") or 0),
        resident_models=[
            f"{item['family']} ({item['device']}): {item['model']}"
            for item in manager_snapshot.get("resident_models", [])
        ],
        helper_processes=helper_processes,
    )
    try:
        import torch

        if not torch.cuda.is_available():
            if sys.platform == "darwin" and torch.backends.mps.is_available():
                status.cuda_supported = True
                status.device_name = "Apple MPS"
            return status
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        device_index = torch.cuda.current_device()
        total_mb = round(float(total_bytes) / (1024 * 1024), 2)
        free_mb = round(float(free_bytes) / (1024 * 1024), 2)
        status.cuda_supported = True
        status.device_name = torch.cuda.get_device_name(device_index)
        status.total_vram_mb = total_mb
        status.free_vram_mb = free_mb
        status.used_vram_mb = round(total_mb - free_mb, 2)
        status.torch_allocated_mb = round(float(torch.cuda.memory_allocated(device_index)) / (1024 * 1024), 2)
        status.torch_reserved_mb = round(float(torch.cuda.memory_reserved(device_index)) / (1024 * 1024), 2)
        status.torch_peak_mb = round(float(torch.cuda.max_memory_allocated(device_index)) / (1024 * 1024), 2)
    except Exception:
        pass
    return status


def release_vram_resources(*, runner=None) -> ReleaseVramResult:
    released: list[str] = []
    terminated_processes: list[str] = []
    errors: list[str] = []

    if runner is not None and hasattr(runner, "kill_managed_processes"):
        try:
            killed = runner.kill_managed_processes()
            if killed:
                released.append("managed_job_processes")
                terminated_processes.extend(killed)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"managed_job_processes: {exc}")

    try:
        from .adapters.omnivoice_client import release_all_clients as release_omnivoice_clients

        release_omnivoice_clients()
        released.append("omnivoice_clients")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"omnivoice_clients: {exc}")

    try:
        from .adapters.asr import reset_model_cache

        reset_model_cache()
        released.append("qwen3_asr_cache")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"qwen3_asr_cache: {exc}")

    try:
        from .gpu_manager import global_gpu_manager

        previous = global_gpu_manager().reset()
        if previous.get("resident_models"):
            released.append("gpu_manager_state")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"gpu_manager_state: {exc}")

    try:
        from .gpu_lease import clear_gpu_lease_state

        cleared = clear_gpu_lease_state(reason="release_vram")
        if cleared:
            released.append("gpu_lease_state")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"gpu_lease_state: {exc}")

    try:
        helpers = _terminate_gpu_helper_processes()
        if helpers:
            released.append("gpu_helper_processes")
            terminated_processes.extend(helpers)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"gpu_helper_processes: {exc}")

    try:
        _clear_torch_cuda_state()
        released.append("torch_cuda_cache")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"torch_cuda_cache: {exc}")

    return ReleaseVramResult(
        status="ok" if not errors else ("warning" if released or terminated_processes else "error"),
        released=released,
        terminated_processes=terminated_processes,
        errors=errors,
        gpu=collect_runtime_gpu_status(),
    )


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
        return _attach_gpu_status(RuntimeReport.model_validate_json(row["report_json"])) if row else None

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
        report = _attach_gpu_status(report)
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
            import importlib.util

            from .adapters.asr import DEFAULT_ASR_MODEL
            from .hardware import accelerator_available, resolve_inference_device

            if importlib.util.find_spec("qwen_asr") is None:
                return RuntimeCheck(
                    id="qwen3_asr",
                    display_name="Qwen3-ASR GPU",
                    status="blocked",
                    required=True,
                    message="Qwen3-ASR Python package is missing in the backend runtime.",
                    action="Install backend dependencies (qwen-asr and related packages), then rerun smoke test.",
                )
            if not accelerator_available():
                return RuntimeCheck(
                    id="qwen3_asr",
                    display_name="Qwen3-ASR GPU",
                    status="blocked",
                    required=True,
                    message="No GPU accelerator is available for Qwen3-ASR (CUDA or Apple MPS).",
                    action=(
                        "Use an NVIDIA GPU with CUDA on Windows/Linux, or run on Apple Silicon "
                        "with PyTorch MPS enabled, then rerun the smoke test."
                    ),
                )
            resolved_device = resolve_inference_device("cuda:0")
            backend_label = (
                "CUDA"
                if resolved_device.startswith("cuda")
                else "Apple MPS"
                if resolved_device == "mps"
                else resolved_device
            )
            return RuntimeCheck(
                id="qwen3_asr",
                display_name="Qwen3-ASR GPU",
                status="ready",
                required=True,
                message=(
                    f"Qwen3-ASR is configured for GPU inference ({DEFAULT_ASR_MODEL}) "
                    f"on {backend_label} ({resolved_device})."
                ),
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
        from .omnivoice_env import is_omnivoice_available, omnivoice_venv_root, resolve_omnivoice_python

        if not is_omnivoice_available():
            return RuntimeCheck(
                id="omnivoice",
                display_name="OmniVoice",
                status="warning",
                required=False,
                message="OmniVoice is not installed in the isolated virtualenv.",
                action="Run 'python scripts/setup_omnivoice.py' to enable the OmniVoice TTS backend.",
                resolved_path=str(omnivoice_venv_root()),
            )
        return RuntimeCheck(
            id="omnivoice",
            display_name="OmniVoice",
            status="ready",
            required=False,
            message="OmniVoice package is available for the OmniVoice TTS backend.",
            action="No action required.",
            resolved_path=str(resolve_omnivoice_python()),
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

