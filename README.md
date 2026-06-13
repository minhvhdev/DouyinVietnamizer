# Douyin Vietnamizer Portable Edition

Windows-first desktop application for producing Vietnamese dubbed Douyin videos.

## Current foundation

The application includes the Electron/React jobs dashboard, local FastAPI backend, SQLite job state, twelve declared checkpoint steps, settings, logs, actionable API errors, and persisted vendor runtime smoke tests. Media processing arrives incrementally in later phases.

Customer builds will bundle Python, FFmpeg, yt-dlp, whisper.cpp, and Piper. They will not require Docker, ROCm, CUDA, WSL2, Redis, or a manual Python installation.

## Development

Requirements: Node.js 20+ and Python 3.11+.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/dev.ps1
```

Run verification:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test.ps1
```

Development state defaults to `%LOCALAPPDATA%\DouyinVietnamizer`. Set `DV_DATA_DIR` to override it.

## Vendor runtime

`vendor/manifest.json` declares FFmpeg, yt-dlp, whisper.cpp CPU, optional whisper.cpp Vulkan, and optional Piper. Customer and packaged execution only accept executables bundled under `vendor/`.

Development may explicitly allow tools installed on `%PATH%` by setting `DV_ALLOW_PATH_TOOLS=1`; `scripts/dev.ps1` sets this flag for the development backend. PATH-resolved tools always produce a runtime warning so they cannot be mistaken for a complete customer build.

The Runtime panel shows storage, SQLite, manifest, and executable probe results. A missing required CPU tool blocks new-job creation. Optional Vulkan or Piper failures only produce warnings.

