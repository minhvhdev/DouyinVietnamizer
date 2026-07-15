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
python3 backend/scripts/setup_omnivoice.py
pnpm tauri:dev                     # hoặc: pnpm run dev
```

## OmniVoice trên Apple Silicon (MPS)

OmniVoice 0.2.x chạy model chính trên MPS bằng `float16` và giữ Higgs audio
tokenizer trên CPU. Không bật `PYTORCH_ENABLE_MPS_FALLBACK=1`, vì biến này có
thể âm thầm chuyển operator chưa hỗ trợ sang CPU.

Kiểm tra môi trường:

```bash
backend/venvs/omnivoice/bin/python3 \
  -m dv_backend.adapters.omnivoice_worker --health-check
```

Smoke test thực, bắt buộc trước khi xác nhận một máy Mac được hỗ trợ:

```bash
cd backend
venvs/omnivoice/bin/python3 scripts/smoke_omnivoice_mps.py \
  --ref-audio /path/to/reference.wav \
  --ref-text "Toàn bộ lời nói khớp với reference.wav" \
  --output-dir omnivoice_mps_smoke
```

Kết quả hợp lệ phải báo `device: "mps"`, `model_dtype: "float16"`,
audio tokenizer CPU/float32, và tạo được cả `cold.wav` lẫn `warm.wav` ở 24 kHz
với kiểm tra finite/non-silent/clipping đạt. Nghe thủ công cả hai WAV để xác
nhận không muffled/noisy/clipped. App không tự fallback toàn bộ model sang CPU
trên Apple Silicon; chỉ dùng CPU nếu người vận hành bật
`DV_OMNIVOICE_ALLOW_CPU_FALLBACK=1` rõ ràng. Operator fallback là policy riêng
và không được bật trong acceptance run.

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
