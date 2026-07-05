---
id: clearpath-product
type: product-summary
status: draft
canonical: true
---
# Product Summary

## Product
Douyin Vietnamizer

## Target Users
Windows users and maintainers who want a local workflow for turning
Douyin or Bilibili videos into Vietnamese-dubbed outputs.

## Core Outcome
Download a source video, transcribe Chinese audio, translate to
Vietnamese, synthesize Vietnamese speech, repair timing, and produce a
final `dubbed.mp4` plus JSON/HTML QC reports.

## Non-Goals
- Speaker diarization or per-speaker voice assignment.
- A cloud-hosted dubbing service.
- Non-portable desktop packaging by default.

## Constraints
- Node.js 20+, `pnpm`, Python 3.12, and `uv`.
- NVIDIA GPU for the local accelerated pipeline.
- FFmpeg and `yt-dlp` on `PATH` or bundled under `vendor/`.
- Portable runtime and model assets for desktop distribution.

## Success Criteria
- Resolve and download a supported source video URL.
- Complete ASR, translation, TTS, timing repair, and render stages.
- Produce `dubbed.mp4` and machine-readable QC artifacts.
- Run through the Tauri desktop shell and portable runtime layout.
