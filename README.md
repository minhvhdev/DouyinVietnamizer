# Douyin Vietnamizer

Desktop workflow for producing Vietnamese dubbed Douyin/Bilibili videos on **Windows (NVIDIA CUDA)** and **macOS Apple Silicon (MPS)**.

## Pipeline

1. Resolve and download a Douyin or Bilibili video.
2. Extract audio, detect speech, and transcribe Chinese with Qwen3-ASR.
3. Translate to Vietnamese with Gemini or an OpenAPI-compatible LLM.
4. Synthesize Vietnamese speech with OmniVoice (voice clone).
5. Repair timing, mix audio, render `dubbed.mp4`, and produce JSON/HTML QC reports.

Speaker diarization/per-speaker voice assignment has been removed; all segments use the single OmniVoice configuration.

## Supported platforms

| Platform | Status | Acceleration |
|----------|--------|--------------|
| Windows 10/11 x64 | Primary dev target | NVIDIA CUDA (PyTorch cu128) |
| macOS 12+ Apple Silicon (`arm64`) | Supported for dev/build | MPS for OmniVoice; ASR may use MPS when available |
| Intel Mac | Not supported | — |

## Requirements

Shared:

- Node.js 20+
- [pnpm](https://pnpm.io/) 10+
- **Python 3.12** (not 3.13)
- [uv](https://docs.astral.sh/uv/)
- [Rust](https://rustup.rs/) (for Tauri)
- FFmpeg and yt-dlp on `PATH`, or under `vendor/` (download via the in-app **Môi trường** wizard)

Windows additionally:

- NVIDIA GPU with a recent driver
- RTX 50-series GPUs need **PyTorch cu128** (already pinned in `backend/pyproject.toml`)
- [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (for `pnpm tauri:build`)
- [WebView2 Runtime](https://developer.microsoft.com/microsoft-edge/webview2/) (usually preinstalled on Windows 11)

macOS Apple Silicon additionally:

- Xcode Command Line Tools: `xcode-select --install`
- Homebrew recommended for FFmpeg: `brew install ffmpeg yt-dlp`
- See [docs/BUILD_MACOS.md](docs/BUILD_MACOS.md) for MPS smoke-test and troubleshooting

## Repository layout

```text
backend/
├── .venv/                 # main backend env (`uv sync`)
├── venvs/omnivoice/       # isolated OmniVoice worker env
├── dv_backend/
├── models/                # qwen3-asr, omnivoice weights
└── scripts/
    ├── eval/              # benchmarks, A/B experiments, QC dashboards
    └── archive/           # one-off diagnostic / P0 scripts
vendor/
├── manifest.json
├── ffmpeg/
└── yt-dlp/
src-tauri/                 # Tauri desktop shell
frontend/                  # Vite UI
```

Application state defaults to `%LOCALAPPDATA%\DouyinVietnamizer` on Windows. On macOS, set `DV_DATA_DIR` to a writable folder (for example `~/Library/Application Support/DouyinVietnamizer`) until native macOS data-dir defaults land.

## Development

Stop any running Douyin Vietnamizer instance before starting a new dev session.

### Windows

```powershell
git clone https://github.com/minhvhdev/DouyinVietnamizer.git
cd DouyinVietnamizer

pnpm run setup
python backend\scripts\setup_omnivoice.py

# optional: pin Python 3.12 for uv
cd backend
uv python pin 3.12
cd ..

pnpm tauri:dev
```

Alternative without the Tauri shell (backend + Vite UI only):

```powershell
pnpm run dev
```

Verify OmniVoice:

```powershell
backend\venvs\omnivoice\Scripts\python.exe -m dv_backend.adapters.omnivoice_worker --health-check
```

### macOS Apple Silicon

```bash
git clone https://github.com/minhvhdev/DouyinVietnamizer.git
cd DouyinVietnamizer

pnpm run setup
python3 backend/scripts/setup_omnivoice.py

export DV_DATA_DIR="$HOME/Library/Application Support/DouyinVietnamizer"

pnpm tauri:dev
```

Alternative without the Tauri shell:

```bash
pnpm run dev
```

Verify OmniVoice MPS:

```bash
backend/venvs/omnivoice/bin/python3 \
  -m dv_backend.adapters.omnivoice_worker --health-check
```

Before claiming a Mac is supported, run the strict MPS smoke test documented in [docs/BUILD_MACOS.md](docs/BUILD_MACOS.md).

### Hot reload

- `frontend/src/renderer/**` — Vite HMR
- `backend/dv_backend/**` — uvicorn reload when `DV_RELOAD=1` (dev profile)
- `src-tauri/src/**` — Cargo rebuilds the shell

Press `Ctrl+C` (Windows) or `Ctrl+C` / `Cmd+.` (macOS) to stop dev processes.

### Tests

```bash
pnpm test
```

Backend only:

```bash
cd backend && uv run pytest -v
```

## Build

`pnpm tauri:build` packages the Tauri desktop shell. The Python backend, `vendor/` tools, `backend/models/`, and `backend/venvs/omnivoice/` must still be present in the expected layout when you run the built app. In practice, build and run from a prepared repository checkout.

### Windows

```powershell
pnpm run setup
python backend\scripts\setup_omnivoice.py
pnpm tauri:build
```

Installer artifacts:

```text
src-tauri/target/release/bundle/nsis/
src-tauri/target/release/bundle/msi/
```

Run the installer, then launch the app from a machine that also has the repository layout available (or copy `backend/`, `vendor/`, and downloaded models next to the install location as your deployment process requires).

### macOS Apple Silicon

```bash
pnpm run setup
python3 backend/scripts/setup_omnivoice.py
pnpm tauri:build
```

Bundle artifacts:

```text
src-tauri/target/release/bundle/macos/*.app
src-tauri/target/release/bundle/dmg/*.dmg
```

OmniVoice on Mac uses **MPS/float16** for the main model and keeps the Higgs audio tokenizer on **CPU/float32**. Do not set `PYTORCH_ENABLE_MPS_FALLBACK=1` for acceptance runs.

## Vendor tools

`vendor/manifest.json` declares FFmpeg and yt-dlp.

Tools are resolved from `vendor/` first, then from `PATH`. Set `DV_ALLOW_PATH_TOOLS=0` to require files under `vendor/` only.

Missing tools or Qwen3 models? Use the setup wizard in the UI (**Môi trường**) to download them into `vendor/` and `backend/models/`.

## Privacy and limitations

- Douyin and Bilibili URLs and downloaded media are processed locally.
- Gemini or an OpenAPI-compatible LLM receives transcript text for translation.
- OmniVoice runs locally with cloned reference audio and user-provided `ref_text`.
- Browser cookies are optional and are only passed to yt-dlp when selected.
- Douyin and Bilibili may change their sites or require authentication, which can break a URL.

## Further reading

- [docs/BUILD_MACOS.md](docs/BUILD_MACOS.md) — Apple Silicon setup, OmniVoice MPS smoke test, troubleshooting
