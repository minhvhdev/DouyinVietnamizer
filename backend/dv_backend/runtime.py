from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from .config import AppConfig
from .database import Database
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
        allow_path_tools: bool = False,
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
        checks = [self._check_storage(), self._check_sqlite()]
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
                        action="Install the complete bundled runtime and rerun the smoke test.",
                        source="missing",
                    ))
                    continue
                probe = probe_executable(tool, resolved.path)
                status = probe.status
                if status == "blocked" and not tool.required:
                    status = "warning"
                if status == "ready" and resolved.source == "path":
                    status = "warning"
                checks.append(RuntimeCheck(
                    id=tool.id, display_name=tool.display_name, status=status, required=tool.required,
                    message=probe.message, action=(
                        "Bundle this tool before creating a customer build."
                        if resolved.source == "path" and probe.status == "ready" else probe.action
                    ),
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

    @staticmethod
    def _aggregate(checks: list[RuntimeCheck]) -> str:
        if any(check.required and check.status == "blocked" for check in checks):
            return "blocked"
        if any(check.status == "warning" or check.source == "path" for check in checks):
            return "warning"
        return "ready"


def default_runtime_service(config: AppConfig, database: Database) -> RuntimeSmokeTestService:
    project_root = Path(__file__).resolve().parents[2]
    vendor_dir = Path(os.environ.get("DV_VENDOR_DIR", project_root / "vendor"))
    manifest_path = Path(os.environ.get("DV_VENDOR_MANIFEST", vendor_dir / "manifest.json"))
    return RuntimeSmokeTestService(
        config, database, manifest_path, vendor_dir,
        allow_path_tools=os.environ.get("DV_ALLOW_PATH_TOOLS") == "1",
    )

