# Douyin Vietnamizer

Windows workflow for producing Vietnamese dubbed Douyin videos.

## Current status

The GPU pipeline is implemented end to end:

1. Resolve and download a Douyin video.
2. Extract audio, detect speech, and transcribe Chinese with Qwen3-ASR on CUDA.
3. Translate to Vietnamese with Google Translate or Gemini.
4. Synthesize Vietnamese speech with VieNeu-TTS v3 Turbo (48 kHz) on GPU.
5. Repair timing, mix audio, render `dubbed.mp4`, and produce JSON/HTML QC reports.

## Quick start

Requirements: Node.js 20+, pnpm, **Python 3.12** (not 3.13), [uv](https://docs.astral.sh/uv/), NVIDIA GPU (RTX 50-series needs **PyTorch cu128**), FFmpeg and yt-dlp on `PATH` (or under `vendor/`).

```powershell
pnpm run setup   # pnpm install + uv sync (installs vieneu 3.x [gpu], torch, torchaudio)
pnpm run dev     # backend + UI, opens browser automatically
```

Backend dependencies (including VieNeu-TTS) are installed into `backend/.venv` via `uv sync`. Use `cd backend && uv python pin 3.12` if uv picks the wrong Python version.

Press `Ctrl+C` to stop both processes.

Run tests:

```powershell
pnpm test
```

Development state defaults to `%LOCALAPPDATA%\DouyinVietnamizer`. Set `DV_DATA_DIR` to override it.

## Vendor tools

`vendor/manifest.json` declares FFmpeg and yt-dlp.

Tools are resolved from `vendor/` first, then from `%PATH%`. Set `DV_ALLOW_PATH_TOOLS=0` to require files under `vendor/` only.

Missing tools or Qwen3 models? Use the setup wizard in the UI (Môi trường) to download them into `vendor/`.

## Privacy and limitations

- Douyin URLs and downloaded media are processed locally.
- Google Translate or Gemini receives transcript text when selected for translation.
- VieNeu-TTS runs fully offline after models are downloaded.
- Browser cookies are optional and are only passed to yt-dlp when selected.
- Douyin may change its site or require authentication, which can break a URL.
