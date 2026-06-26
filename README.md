# Douyin Vietnamizer

Windows workflow for producing Vietnamese dubbed Douyin videos.

## Current status

The GPU pipeline is implemented end to end:

1. Resolve and download a Douyin or Bilibili video.
2. Extract audio, detect speech, and transcribe Chinese with Qwen3-ASR on CUDA.
3. Translate to Vietnamese with Google Translate or Gemini.
4. Synthesize Vietnamese speech with OmniVoice.
5. Repair timing, mix audio, render `dubbed.mp4`, and produce JSON/HTML QC reports.

Speaker diarization/per-speaker voice assignment has been removed; all segments use the single OmniVoice configuration.

## Quick start

Requirements: Node.js 20+, pnpm, **Python 3.12** (not 3.13), [uv](https://docs.astral.sh/uv/), NVIDIA GPU (RTX 50-series needs **PyTorch cu128**), FFmpeg and yt-dlp on `PATH` (or under `vendor/`).

```powershell
pnpm run setup   # pnpm install + uv sync (backend deps)
pnpm run dev     # backend + UI, opens browser automatically
```

Backend dependencies are installed into `backend/.venv` via `uv sync`. Use `cd backend && uv python pin 3.12` if uv picks the wrong Python version. Run `python scripts/setup_omnivoice.py` in `backend` to prepare OmniVoice.

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

- Douyin and Bilibili URLs and downloaded media are processed locally.
- Google Translate or Gemini receives transcript text when selected for translation.
- OmniVoice runs through the isolated `backend/.venv-omnivoice` environment after setup.
- Browser cookies are optional and are only passed to yt-dlp when selected.
- Douyin and Bilibili may change their sites or require authentication, which can break a URL.
