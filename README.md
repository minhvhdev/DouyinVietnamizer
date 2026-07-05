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
│   ├── yt-dlp/
│   └── voxcpm2/          # voxcpm2-cli.exe + llama-tts-server.exe + ggml DLLs
├── models/
│   ├── qwen3-asr/
│   └── voxcpm2/          # VoxCPM2-BaseLM-Q8_0.gguf + Acoustic F16
└── manifest.json
```

Hot-reload during development:
- Edit `frontend/src/renderer/**` — Vite HMR refreshes the window.
- Edit `backend/dv_backend/**` — uvicorn reloads because `DV_RELOAD=1`.
- Edit `src-tauri/src/**` — Cargo rebuilds the affected crate, window refreshes.

`pnpm tauri:build` produces a Windows app bundle (NSIS) that includes the prepared portable runtime. For the folder-style portable release, run:

```powershell
pnpm tauri:build:portable
```

This refreshes `vendor/portable-runtime` (GGUF weights + `voxcpm2-cli` under `tools/voxcpm2/`), mirrors it into `dist-portable/DouyinVietnamizer-0.1.0-portable/`, builds the Tauri binary, and syncs backend code. To refresh only the runtime folder (for `tauri:dev`), use `pnpm tauri:build:portable:runtime`.

Prerequisites before the first portable build:

1. `vendor/portable-runtime/` with embedded Python `.venv` (existing dev bootstrap).
2. `vendor/voxcpm2/voxcpm2-cli.exe` (+ CUDA DLLs) built from llama.cpp-omni.
3. Network access for the first GGUF download (~3.3 GB into `portable-runtime/models/voxcpm2/`).

Copy `douyin-vietnamizer.exe` together with its sibling `portable-runtime/` directory to another Windows x64 machine with compatible NVIDIA/CUDA drivers.

## macOS portable (Apple Silicon)

Pre-built portable zips are available on the [Releases page](../../releases) as `DouyinVietnamizer-0.1.0-portable-macos.zip` (arm64 only; macOS 12+).

To use:

1. Download the zip from the latest release.
2. Unzip anywhere (e.g. `~/Applications/`).
3. Open the resulting folder and double-click `DouyinVietnamizer.app`.
4. **First launch only:** macOS Gatekeeper will block the unsigned app. Right-click `DouyinVietnamizer.app` → **Open** → confirm. Subsequent launches double-click normally.

The `.app` looks for a sibling `portable-runtime/` folder with the bundled Python interpreter, models, and tools. Keep the folder intact.

### Building from source (maintainers)

Build host requirements:
- Apple Silicon Mac (M1/M2/M3/M4), macOS 12+
- Internet connection (first run downloads Python runtime, pip packages, tools, and models)
- Free disk space: at least 20 GB recommended

```bash
# from repository root
pnpm run tauri:build:mac:m4
```

What this command does automatically:
- Homebrew (if missing)
- Node.js + pnpm
- rustup + Rust target `aarch64-apple-darwin`
- uv
- python-build-standalone runtime, Python packages, tools, and models

Step-by-step flow on a fresh Mac:
1. Install/prepare Xcode Command Line Tools.
2. Install missing package/build tools listed above.
3. Build frontend assets.
4. Build Tauri app for `aarch64-apple-darwin`.
5. Build portable runtime at `dist-portable/macos-staging/portable-runtime`.
6. Assemble final bundle and zip.

If Xcode Command Line Tools are missing, the script will trigger `xcode-select --install` and stop. Complete installation, then run the same build command again.

Build outputs:
- App bundle: `dist-portable/DouyinVietnamizer-0.1.0-portable/DouyinVietnamizer.app`
- Runtime folder: `dist-portable/DouyinVietnamizer-0.1.0-portable/portable-runtime`
- Release zip: `dist-portable/DouyinVietnamizer-0.1.0-portable-macos.zip`

Run the built app locally:
1. Unzip `DouyinVietnamizer-0.1.0-portable-macos.zip`.
2. Keep `DouyinVietnamizer.app` and `portable-runtime/` in the same folder.
3. First launch: right-click `DouyinVietnamizer.app` -> **Open** -> confirm.

Equivalent manual steps (if you want full control):

```bash
pnpm install
pnpm tauri build --target aarch64-apple-darwin
bash scripts/build-portable-runtime-mac.sh
bash scripts/build-portable-mac.sh
```

Common issues:
- `xcode-select: error`: install Command Line Tools, then rerun.
- `pnpm` not found: rerun once; bootstrap installs it automatically.
- Download/model step is slow: first run may download ~11 GB; later runs reuse `dist-portable/macos-staging/`.
- Gatekeeper blocks app: use right-click -> Open for first launch.

## Vendor tools

`vendor/manifest.json` declares FFmpeg and yt-dlp.

Tools are resolved from `vendor/` first, then from `%PATH%`. Set `DV_ALLOW_PATH_TOOLS=0` to require files under `vendor/` only.

Missing tools or Qwen3 models? Use the setup wizard in the UI (Môi trường) to download them into `vendor/`.

## Privacy and limitations

- Douyin and Bilibili URLs and downloaded media are processed locally.
- Google Translate or Gemini receives transcript text when selected for translation.
- VoxCPM2 runs through native `voxcpm2-cli` + GGUF weights bundled under `portable-runtime/tools/voxcpm2` and `portable-runtime/models/voxcpm2`.
- Browser cookies are optional and are only passed to yt-dlp when selected.
- Douyin and Bilibili may change their sites or require authentication, which can break a URL.
