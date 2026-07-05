from pathlib import Path
import subprocess
import time

from pydantic import BaseModel

from .vendor import VendorTool


class ProbeResult(BaseModel):
    status: str
    message: str
    action: str
    detail: str | None = None
    version: str | None = None
    duration_ms: int


def probe_executable(
    tool: VendorTool,
    executable: Path,
    timeout_seconds: float = 5,
) -> ProbeResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(executable), *tool.version_args],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()[:16_384]
        duration_ms = round((time.perf_counter() - started) * 1000)
        if completed.returncode not in tool.success_exit_codes:
            return ProbeResult(
                status="blocked",
                message=f"{tool.display_name} exited with code {completed.returncode}.",
                action=(
                    "Replace the executable with a supported release."
                    if tool.success_exit_codes == [0]
                    else f"Verify the probe arguments and expected exit codes: {tool.success_exit_codes}."
                ),
                detail=output,
                duration_ms=duration_ms,
            )
        if tool.version_contains and tool.version_contains.lower() not in output.lower():
            return ProbeResult(
                status="blocked",
                message=f"{tool.display_name} returned unrecognized version output.",
                action="Verify that the executable matches the vendor manifest.",
                detail=output,
                duration_ms=duration_ms,
            )
        return ProbeResult(
            status="ready",
            message=f"{tool.display_name} is available.",
            action="No action required.",
            version=output.splitlines()[0] if output else "available",
            duration_ms=duration_ms,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(
            status="blocked",
            message=f"{tool.display_name} version probe timed out.",
            action="Replace the executable or check antivirus interference.",
            duration_ms=round((time.perf_counter() - started) * 1000),
        )
    except OSError as error:
        return ProbeResult(
            status="blocked",
            message=f"{tool.display_name} could not start.",
            action="Install the tool under vendor/ or add it to PATH, then retry.",
            detail=str(error),
            duration_ms=round((time.perf_counter() - started) * 1000),
        )

