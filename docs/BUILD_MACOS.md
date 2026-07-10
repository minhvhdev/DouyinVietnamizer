# macOS development (Apple Silicon)

Hướng dẫn chạy và phát triển DouyinVietnamizer trên **Mac M1/M2/M3/M4** (arm64, macOS 12+).

> **Lưu ý:** Bản portable đã được loại bỏ. Dùng workflow dev (`pnpm run dev` / `pnpm tauri:dev`) từ repo checkout.

## Yêu cầu

- macOS 12+, Apple Silicon (`uname -m` → `arm64`)
- Xcode Command Line Tools: `xcode-select --install`
- Homebrew (khuyến nghị): [https://brew.sh](https://brew.sh)
- Node 20+, pnpm, [uv](https://docs.astral.sh/uv/), Rust (cho Tauri)

## Quick start

```bash
git clone https://github.com/minhvhdev/DouyinVietnamizer.git
cd DouyinVietnamizer
pnpm run setup
pnpm tauri:dev                     # hoặc: pnpm run dev
```

## Layout dev

```text
backend/.venv/          # Python deps (uv sync)
backend/models/         # qwen3-asr, omnivoice weights
vendor/manifest.json
vendor/ffmpeg/
vendor/yt-dlp/
```

Tauri desktop spawn backend qua `uv run python -m dv_backend.main` từ `backend/`, với `DV_VENDOR_DIR=vendor/` và `DV_MODELS_DIR=backend/models/`.

## Không commit

- `backend/models/`
- `backend/data/`
- `.cache/`
- file WAV test tạm

## Troubleshooting

| Vấn đề | Cách xử lý |
|--------|------------|
| `backend/.venv` thiếu | `pnpm run setup` |
| OmniVoice chưa sẵn sàng | Kiểm tra tab Môi trường / cài model OmniVoice |
| FFmpeg/yt-dlp missing | Bootstrap trong UI tab Môi trường hoặc đặt vào `vendor/` |
| Port 8765 bận | `pnpm dlx kill-port 8765` |
