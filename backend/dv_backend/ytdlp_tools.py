from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .errors import AppError
from .models import ErrorInfo

YTDLP_WINDOWS_RELEASE = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
YTDLP_MACOS_RELEASE = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos"

# Prefer Firefox first, then try other browsers.
COOKIE_BROWSER_FALLBACK_ORDER: tuple[str, ...] = ("firefox", "chrome", "edge", "brave")


def yt_dlp_cookie_args_for_browser(browser: str) -> list[str]:
    return ["--cookies-from-browser", browser]


def yt_dlp_cookie_args_for_file(cookies_file: str | Path) -> list[str]:
    return ["--cookies", str(cookies_file)]


def format_browsers_attempted(browsers: list[str]) -> str:
    if not browsers:
        return "firefox → chrome → edge → brave"
    return " → ".join(browsers)


def _tail(text: str, limit: int = 2400) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"...\n{cleaned[-limit:]}"


def classify_yt_dlp_failure(
    *,
    operation: str,
    stderr: str,
    stdout: str,
    browsers_attempted: list[str],
    source_label: str,
    cookie_browser: str | None = None,
) -> ErrorInfo:
    detail = _tail(stderr or stdout)
    lowered = detail.lower()
    browser_hint = format_browsers_attempted(browsers_attempted)
    if cookie_browser:
        browser_hint = f"{cookie_browser} (đã thử: {browser_hint})"

    if any(token in lowered for token in ("sign in", "login required", "cookies", "cookie", "authentication")):
        auth_hint = (
            f"Đặt file cookies.txt (Netscape) trong Cài đặt → Tải video, hoặc đăng nhập "
            f"{source_label} trên Firefox/Chrome/Edge/Brave. "
            f"Đã thử: {format_browsers_attempted(browsers_attempted)}."
        )
        return ErrorInfo(
            code="YTDLP_AUTH_REQUIRED",
            message=f"Không thể {operation} — video yêu cầu đăng nhập hoặc cookie hợp lệ ({source_label}).",
            action=auth_hint,
            detail=detail,
            retryable=True,
        )

    if any(token in lowered for token in ("unsupported url", "no video formats", "video unavailable")):
        return ErrorInfo(
            code="YTDLP_UNSUPPORTED_URL",
            message=f"Liên kết không hợp lệ hoặc video không còn khả dụng ({source_label}).",
            action="Kiểm tra lại URL video công khai từ Douyin hoặc Bilibili.",
            detail=detail,
            retryable=False,
        )

    if any(
        token in lowered
        for token in (
            "unable to extract",
            "extractor",
            "no suitable extractor",
            "signature",
            "fresh cookies",
            "confirm you are on the latest version",
        )
    ):
        return ErrorInfo(
            code="YTDLP_EXTRACTOR_OUTDATED",
            message=f"Không thể {operation} — nền tảng có thể đã đổi cơ chế hoặc yt-dlp đã cũ.",
            action=(
                "Douyin đang yêu cầu chữ ký web (không chỉ cookie). "
                "Cập nhật yt-dlp trong Cài đặt, thử cookies.txt mới hơn; "
                "nếu vẫn lỗi thì cần extractor mới từ yt-dlp hoặc tải file video thủ công rồi import."
            ),
            detail=detail,
            retryable=True,
        )

    if "geo" in lowered or "not available in your country" in lowered:
        return ErrorInfo(
            code="YTDLP_GEO_BLOCKED",
            message=f"Video bị chặn theo khu vực ({source_label}).",
            action="Thử VPN hoặc đăng nhập tài khoản hợp lệ trên Chrome trước khi tạo job.",
            detail=detail,
            retryable=True,
        )

    return ErrorInfo(
        code="YTDLP_COMMAND_FAILED",
        message=f"yt-dlp không thể {operation} ({source_label}).",
        action=(
            f"Đã thử cookie trình duyệt theo thứ tự {browser_hint}. "
            "Xem chi tiết lỗi bên dưới hoặc cập nhật yt-dlp."
        ),
        detail=detail,
        retryable=True,
    )


def yt_dlp_version(yt_dlp_path: Path) -> str:
    result = subprocess.run(
        [str(yt_dlp_path), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )
    if result.returncode != 0:
        raise AppError(
            500,
            ErrorInfo(
                code="YTDLP_VERSION_FAILED",
                message="Không đọc được phiên bản yt-dlp.",
                action="Kiểm tra yt-dlp trong vendor/ hoặc PATH.",
                detail=_tail(result.stderr or result.stdout),
            ),
        )
    return (result.stdout or result.stderr or "").strip()


def _release_download_url() -> tuple[str, str]:
    if sys.platform == "darwin":
        return YTDLP_MACOS_RELEASE, "yt-dlp_macos"
    if os.name == "nt":
        return YTDLP_WINDOWS_RELEASE, "yt-dlp.exe"
    return YTDLP_WINDOWS_RELEASE, "yt-dlp"


def update_yt_dlp_binary(yt_dlp_path: Path) -> dict[str, str]:
    previous_version = yt_dlp_version(yt_dlp_path)

    try:
        self_update = subprocess.run(
            [str(yt_dlp_path), "-U"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        if self_update.returncode == 0:
            new_version = yt_dlp_version(yt_dlp_path)
            return {
                "status": "updated",
                "method": "self_update",
                "previous_version": previous_version,
                "version": new_version,
                "detail": _tail(self_update.stdout or self_update.stderr, 800),
            }
    except (subprocess.TimeoutExpired, OSError):
        pass

    release_url, release_name = _release_download_url()
    temp_path: Path | None = None
    try:
        with urllib.request.urlopen(release_url, timeout=120) as response:
            payload = response.read()
        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / f"dv-ytdlp-{os.getpid()}-{release_name}"
        temp_path.write_bytes(payload)
        if sys.platform != "win32":
            temp_path.chmod(temp_path.stat().st_mode | 0o111)
        backup_path = yt_dlp_path.with_suffix(yt_dlp_path.suffix + ".bak")
        if yt_dlp_path.exists():
            shutil.copy2(yt_dlp_path, backup_path)
        shutil.copy2(temp_path, yt_dlp_path)
        if sys.platform != "win32":
            yt_dlp_path.chmod(yt_dlp_path.stat().st_mode | 0o111)
    except urllib.error.URLError as exc:
        raise AppError(
            502,
            ErrorInfo(
                code="YTDLP_UPDATE_DOWNLOAD_FAILED",
                message="Không tải được bản yt-dlp mới từ GitHub.",
                action="Kiểm tra kết nối mạng và thử lại.",
                detail=str(exc),
            ),
        ) from exc
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    new_version = yt_dlp_version(yt_dlp_path)
    return {
        "status": "updated",
        "method": "binary_replace",
        "previous_version": previous_version,
        "version": new_version,
        "platform": platform.system(),
    }
