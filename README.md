# Douyin Vietnamizer

Windows workflow for producing Vietnamese dubbed Douyin videos.

## Current status

The GPU pipeline is implemented end to end:

1. Resolve and download a Douyin or Bilibili video.
2. Extract audio, detect speech, and transcribe Chinese with Qwen3-ASR on CUDA.
3. Translate to Vietnamese with Google Translate or Gemini.
4. Synthesize Vietnamese speech with VoxCPM2.
5. Repair timing, mix audio, render `dubbed.mp4`, and produce JSON/HTML QC reports.

Speaker diarization/per-speaker voice assignment has been removed; all segments use the single VoxCPM2 configuration.

## Quick start

Requirements: Node.js 20+, pnpm, **Python 3.12** (not 3.13), [uv](https://docs.astral.sh/uv/), NVIDIA GPU (RTX 50-series needs **PyTorch cu128**), FFmpeg and yt-dlp on `PATH` (or under `vendor/`).

```powershell
pnpm run setup   # pnpm install + uv sync (backend deps)
pnpm run dev     # backend + UI, opens browser automatically
```

Backend dependencies are installed into `backend/.venv` via `uv sync`. Use `cd backend && uv python pin 3.12` if uv picks the wrong Python version. Run `python scripts/setup_voxcpm.py` in `backend` to prepare VoxCPM2.

Press `Ctrl+C` to stop both processes.

Run tests:

```powershell
pnpm test
```

Development state defaults to `%LOCALAPPDATA%\DouyinVietnamizer`. Set `DV_DATA_DIR` to override it.

## Tauri desktop app

`pnpm tauri:dev` opens the app in a Tauri window. Rust spawns the Python backend from `vendor/portable-runtime` on `127.0.0.1:8765`. The dev app does not run first-time setup; prepare `vendor/portable-runtime` once, then frontend and backend source changes reload without rebuilding the runtime.

Portable runtime layout:

```text
vendor/portable-runtime/
├── .venv/ or python/
├── backend/
├── tools/
│   ├── ffmpeg/
│   └── yt-dlp/
├── models/
│   ├── qwen3-asr/
│   └── voxcpm2/
└── manifest.json
```

Hot-reload during development:
- Edit `frontend/src/renderer/**` — Vite HMR refreshes the window.
- Edit `backend/dv_backend/**` — uvicorn reloads because `DV_RELOAD=1`.
- Edit `src-tauri/src/**` — Cargo rebuilds the affected crate, window refreshes.

`pnpm tauri:build` produces a Windows app bundle (NSIS) that includes the prepared portable runtime. For the folder-style portable release, copy the built `DouyinVietnamizer.exe` together with its sibling `portable-runtime/` directory. Target machines must be Windows x64 with compatible NVIDIA/CUDA drivers. Existing `pnpm dev` and `pnpm test` workflows remain available for non-Tauri work.

## Vendor tools

`vendor/manifest.json` declares FFmpeg and yt-dlp.

Tools are resolved from `vendor/` first, then from `%PATH%`. Set `DV_ALLOW_PATH_TOOLS=0` to require files under `vendor/` only.

Missing tools or Qwen3 models? Use the setup wizard in the UI (Môi trường) to download them into `vendor/`.

## Privacy and limitations

- Douyin and Bilibili URLs and downloaded media are processed locally.
- Google Translate or Gemini receives transcript text when selected for translation.
- VoxCPM2 runs through the isolated `backend/.venv-voxcpm` environment after setup.
- Browser cookies are optional and are only passed to yt-dlp when selected.
- Douyin and Bilibili may change their sites or require authentication, which can break a URL.
