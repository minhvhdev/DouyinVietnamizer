# Douyin Vietnamizer Portable Edition

Windows-first desktop application for producing Vietnamese dubbed Douyin videos.

## Phase 1

Phase 1 includes the Electron/React jobs dashboard, local FastAPI backend, SQLite job state, twelve declared checkpoint steps, settings, logs, and actionable API errors. Media processing arrives incrementally in later phases.

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

