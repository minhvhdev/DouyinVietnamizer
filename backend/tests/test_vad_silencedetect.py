"""Tests for FFmpeg silencedetect VAD parser."""

from __future__ import annotations

from dv_backend.adapters.vad_silencedetect import (
    detect_speech_regions_silencedetect,
    parse_silencedetect_stderr,
    silencedetect_filter,
)


def test_parse_silencedetect_stderr_inverts_silence_to_speech() -> None:
    stderr = (
        "  Duration: 00:00:10.00, start: 0.000000, bitrate: 256 kb/s\n"
        "[silencedetect @ 0x1] silence_start: 2.0\n"
        "[silencedetect @ 0x1] silence_end: 4.5 | silence_duration: 2.5\n"
        "[silencedetect @ 0x1] silence_start: 7.0\n"
        "[silencedetect @ 0x1] silence_end: 9.0 | silence_duration: 2.0\n"
    )
    regions = parse_silencedetect_stderr(stderr, 10.0)
    assert regions == [
        {"start": 0.0, "end": 2.0},
        {"start": 4.5, "end": 7.0},
        {"start": 9.0, "end": 10.0},
    ]


def test_silencedetect_filter_uses_configured_thresholds() -> None:
    assert silencedetect_filter(-35, 0.75) == "silencedetect=n=-35dB:d=0.75"


def test_detect_speech_regions_silencedetect_wrapper() -> None:
    stderr = "[silencedetect @ 0x1] silence_start: 1.0\n[silencedetect @ 0x1] silence_end: 2.0\n"
    regions = detect_speech_regions_silencedetect("ignored.wav", total_duration=3.0, stderr=stderr)
    assert regions == [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]
