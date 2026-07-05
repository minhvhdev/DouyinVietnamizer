from unittest.mock import patch

import pytest

from dv_backend.config import AppConfig
from dv_backend.pipeline import run_yt_dlp_with_browser_fallback
from dv_backend.ytdlp_tools import (
    COOKIE_BROWSER_FALLBACK_ORDER,
    classify_yt_dlp_failure,
    format_browsers_attempted,
)


def test_cookie_browser_fallback_starts_with_chrome() -> None:
    assert COOKIE_BROWSER_FALLBACK_ORDER[0] == "chrome"


def test_classify_yt_dlp_auth_error() -> None:
    info = classify_yt_dlp_failure(
        operation="tải video",
        stderr="ERROR: Sign in to confirm your age",
        stdout="",
        browsers_attempted=["chrome", "edge"],
        source_label="Douyin",
    )
    assert info.code == "YTDLP_AUTH_REQUIRED"
    assert "chrome" in info.action.lower()


def test_classify_yt_dlp_extractor_outdated() -> None:
    info = classify_yt_dlp_failure(
        operation="phân tích liên kết",
        stderr="ERROR: Unable to extract video data",
        stdout="",
        browsers_attempted=list(COOKIE_BROWSER_FALLBACK_ORDER),
        source_label="Bilibili",
    )
    assert info.code == "YTDLP_EXTRACTOR_OUTDATED"
    assert "yt-dlp" in info.action


def test_format_browsers_attempted() -> None:
    assert format_browsers_attempted(["chrome", "edge"]) == "chrome → edge"


@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_run_yt_dlp_with_browser_fallback_tries_next_browser(mock_run, tmp_path) -> None:
    mock_run.side_effect = [
        __import__("subprocess").CalledProcessError(1, "yt-dlp", stderr="chrome failed"),
        __import__("subprocess").CompletedProcess(args=[], returncode=0, stdout="ok", stderr=""),
    ]
    result, browser, tried = run_yt_dlp_with_browser_fallback(
        tmp_path / "yt-dlp.exe",
        ["--dump-single-json", "https://example.com"],
        "job-1",
        None,
    )
    assert result.stdout == "ok"
    assert browser == "edge"
    assert tried == ["chrome", "edge"]
    assert mock_run.call_count == 2


@patch("dv_backend.pipeline.run_subprocess_with_cancel")
def test_run_yt_dlp_with_browser_fallback_raises_after_all_browsers(mock_run, tmp_path) -> None:
    mock_run.side_effect = __import__("subprocess").CalledProcessError(
        1, "yt-dlp", stderr="failed"
    )
    with pytest.raises(__import__("subprocess").CalledProcessError) as exc_info:
        run_yt_dlp_with_browser_fallback(
            tmp_path / "yt-dlp.exe",
            ["--dump-single-json", "https://example.com"],
            "job-1",
            None,
        )
    assert exc_info.value.browsers_tried == list(COOKIE_BROWSER_FALLBACK_ORDER)
    assert mock_run.call_count == len(COOKIE_BROWSER_FALLBACK_ORDER)
