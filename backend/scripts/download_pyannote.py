"""Deprecated Pyannote downloader kept as a compatibility no-op."""

from __future__ import annotations


def main() -> int:
    print(
        "Pyannote download skipped: speaker diarization was removed; "
        "single-voice VoxCPM2 does not need it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
