from pathlib import Path
import sys

from dv_backend.tool_probe import probe_executable
from dv_backend.vendor import VendorTool


def tool(args: list[str], contains: str = "probe-ok") -> VendorTool:
    return VendorTool(
        id="fake",
        display_name="Fake tool",
        executable="fake.exe",
        dev_command="fake",
        version_args=args,
        version_contains=contains,
        required=True,
        capability="test",
    )


def test_probe_recognizes_version_output() -> None:
    result = probe_executable(tool(["-c", "print('probe-ok 1.2.3')"]), Path(sys.executable))
    assert result.status == "ready"
    assert "probe-ok 1.2.3" in result.version


def test_probe_reports_non_zero_exit() -> None:
    result = probe_executable(tool(["-c", "raise SystemExit(7)"]), Path(sys.executable))
    assert result.status == "blocked"
    assert "code 7" in result.message


def test_probe_reports_timeout() -> None:
    result = probe_executable(
        tool(["-c", "import time; time.sleep(2)"]),
        Path(sys.executable),
        timeout_seconds=0.05,
    )
    assert result.status == "blocked"
    assert "timed out" in result.message.lower()

