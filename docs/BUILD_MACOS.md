# Build DouyinVietnamizer on macOS (Apple Silicon)

Hướng dẫn đầy đủ để build app portable trên **Mac M1/M2/M3/M4** (arm64, macOS 12+).

Tài liệu này được viết để **Cursor Agent** (hoặc developer) có thể tự chạy từng bước trên máy Mac mới mà không cần đoán thêm.

---

## Tóm tắt nhanh

| Bước | Lệnh | Thời gian (ước tính) |
|------|------|----------------------|
| 1. Clone repo | `git clone …` | 1 phút |
| 2. Xcode CLT | `xcode-select --install` | 5–15 phút (một lần) |
| 3. Build VoxCPM2 CLI | `bash scripts/build-voxcpm2-mac.sh` | 5–20 phút |
| 4. Build portable app | `pnpm run tauri:build:mac:m4` | 30–90 phút (lần đầu) |

**Dung lượng đĩa khuyến nghị:** ≥ 25 GB trống (lần đầu tải ~11 GB models + build cache).

**Không commit lên git:** `backend/models/`, `backend/data/`, `backend/out*.wav`, `dist-portable/`, `.cache/llama.cpp-omni/`.

---

## Playbook cho Cursor Agent

Khi user yêu cầu *"build full trên macOS"*, agent thực hiện **theo thứ tự** sau. Sau mỗi bước, chạy lệnh verify; nếu fail thì dừng và sửa trước khi sang bước tiếp.

### Bước 0 — Kiểm tra máy

```bash
uname -s    # phải là Darwin
uname -m    # phải là arm64
sw_vers     # macOS 12+
```

Nếu `uname -m` ≠ `arm64` → **dừng**: bản portable này chỉ hỗ trợ Apple Silicon.

### Bước 1 — Lấy source code

```bash
# Lần đầu
git clone https://github.com/minhvhdev/DouyinVietnamizer.git
cd DouyinVietnamizer

# Đã có repo
git pull origin main
```

### Bước 2 — Xcode Command Line Tools

```bash
xcode-select -p || xcode-select --install
```

Nếu vừa chạy `--install`, **chờ user cài xong** trong GUI rồi chạy lại `xcode-select -p` cho đến khi có đường dẫn (ví dụ `/Library/Developer/CommandLineTools`).

### Bước 3 — Homebrew + cmake (cho VoxCPM2)

Script build portable tự cài Homebrew/Node/pnpm/rust/uv, nhưng **build VoxCPM2 cần cmake** trước:

```bash
if ! command -v brew >/dev/null 2>&1; then
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi
brew install cmake git
```

### Bước 4 — Build `vendor/voxcpm2/voxcpm2-cli` (bắt buộc)

Đây là bước **không tự động** trong `tauri:build:mac:m4`. Phải có binary trước khi build portable.

```bash
bash scripts/build-voxcpm2-mac.sh
```

Verify:

```bash
test -x vendor/voxcpm2/voxcpm2-cli
vendor/voxcpm2/voxcpm2-cli --help
# Tuỳ chọn nhưng khuyến nghị:
test -x vendor/voxcpm2/llama-tts-server && vendor/voxcpm2/llama-tts-server --help
```

Script clone/update [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) vào `.cache/llama.cpp-omni`, build với **Metal** (tự bật trên Mac), rồi copy vào `vendor/voxcpm2/`.

Tuỳ chọn — chỉ định thư mục build khác:

```bash
DV_LLAMA_CPP_OMNI_DIR=~/src/llama.cpp-omni bash scripts/build-voxcpm2-mac.sh
```

### Bước 5 — Build portable app end-to-end

```bash
pnpm run tauri:build:mac:m4
```

Lệnh này (qua `scripts/build-portable-runtime-mac.sh --full-build`) tự:

1. Cài Homebrew, Node.js, pnpm, rustup, uv (nếu thiếu)
2. `pnpm install --frozen-lockfile`
3. Build frontend
4. `pnpm tauri build --target aarch64-apple-darwin --no-sign`
5. Tải Python embedded 3.12, pip packages (PyTorch MPS/CPU, Silero VAD, Edge TTS, Demucs, …)
6. Tải ffmpeg, yt-dlp
7. Bundle `vendor/voxcpm2/` vào runtime
8. Tải models Qwen3-ASR + VoxCPM2 GGUF (~3.3 GB cho TTS)
9. Đóng gói `.app` + zip

Verify output:

```bash
test -d dist-portable/DouyinVietnamizer-0.1.0-portable/DouyinVietnamizer.app
test -f dist-portable/DouyinVietnamizer-0.1.0-portable-macos.zip
ls -lh dist-portable/DouyinVietnamizer-0.1.0-portable-macos.zip
```

### Bước 6 — Chạy app đã build

```bash
cd dist-portable
unzip -o DouyinVietnamizer-0.1.0-portable-macos.zip
open DouyinVietnamizer-0.1.0-portable/DouyinVietnamizer.app
```

**Lần mở đầu:** Gatekeeper chặn app chưa ký → chuột phải `DouyinVietnamizer.app` → **Open** → xác nhận.

Giữ `DouyinVietnamizer.app` và thư mục `portable-runtime/` **cùng một folder**.

---

## Dev mode trên macOS (`tauri:dev`)

Dev app đọc runtime từ `vendor/portable-runtime/` (không phải `dist-portable/macos-staging/`).

Sau khi build runtime (có thể chỉ build runtime, không cần full zip):

```bash
# Cần vendor/voxcpm2/voxcpm2-cli trước
bash scripts/build-portable-runtime-mac.sh

# Trỏ dev runtime
ln -sfn "$(pwd)/dist-portable/macos-staging/portable-runtime" vendor/portable-runtime

# Hoặc dùng biến môi trường
export DV_PORTABLE_RUNTIME_DIR="$(pwd)/dist-portable/macos-staging/portable-runtime"

pnpm install
pnpm tauri:dev
```

Backend dev riêng (không qua Tauri):

```bash
pnpm run setup
cd backend && uv python pin 3.12
uv run python scripts/setup_voxcpm.py   # tải GGUF (~3.3 GB)
pnpm run dev:backend
```

Data mặc định: `~/Library/Application Support/DouyinVietnamizer` (hoặc set `DV_DATA_DIR`).

---

## Build thủ công từng phần

### A. Chỉ build VoxCPM2 CLI

```bash
git clone https://github.com/tc-mb/llama.cpp-omni.git
cd llama.cpp-omni
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target voxcpm2-cli --target llama-tts-server -j"$(sysctl -n hw.ncpu)"
mkdir -p /path/to/DouyinVietnamizer/vendor/voxcpm2
cp build/bin/voxcpm2-cli build/bin/llama-tts-server /path/to/DouyinVietnamizer/vendor/voxcpm2/
chmod +x /path/to/DouyinVietnamizer/vendor/voxcpm2/*
```

Hoặc dùng script có sẵn: `bash scripts/build-voxcpm2-mac.sh`.

### B. Chỉ build runtime (không build Tauri)

```bash
bash scripts/build-portable-runtime-mac.sh
# Output: dist-portable/macos-staging/portable-runtime/
```

### C. Chỉ đóng gói (đã có Tauri .app + runtime)

```bash
pnpm tauri build --target aarch64-apple-darwin --no-sign
bash scripts/build-portable-runtime-mac.sh
bash scripts/build-portable-mac.sh
```

---

## Cấu trúc output

```text
dist-portable/
├── macos-staging/
│   └── portable-runtime/          # cache runtime (tái sử dụng lần build sau)
│       ├── python/
│       ├── .venv/
│       ├── backend/
│       ├── tools/
│       │   ├── ffmpeg/
│       │   ├── yt-dlp/
│       │   └── voxcpm2/         # copy từ vendor/voxcpm2/
│       ├── models/
│       │   ├── qwen3-asr/
│       │   └── voxcpm2/         # GGUF ~3.3 GB
│       └── manifest.json
└── DouyinVietnamizer-0.1.0-portable/
    ├── DouyinVietnamizer.app
    └── portable-runtime/
└── DouyinVietnamizer-0.1.0-portable-macos.zip
```

`vendor/voxcpm2/` (input, commit **không** bắt buộc — build locally):

```text
vendor/voxcpm2/
├── voxcpm2-cli          # bắt buộc
└── llama-tts-server     # khuyến nghị (TTS nhanh hơn qua HTTP server)
```

---

## Tải GGUF weights (nếu test TTS riêng)

```bash
cd backend
uv sync --group dev
uv run python scripts/setup_voxcpm.py
```

Test nhanh:

```bash
vendor/voxcpm2/voxcpm2-cli \
  -t "Xin chào, đây là thử nghiệm VoxCPM2." \
  -o /tmp/dv-tts-test.wav \
  backend/models/voxcpm2/VoxCPM2-BaseLM-Q8_0.gguf \
  backend/models/voxcpm2/VoxCPM2-Acoustic-F16.gguf
afplay /tmp/dv-tts-test.wav
```

---

## Biến môi trường hữu ích

| Biến | Mục đích |
|------|----------|
| `DV_PORTABLE_RUNTIME_DIR` | Trỏ runtime khác cho `tauri:dev` |
| `DV_DATA_DIR` | Thư mục data job/settings |
| `DV_VOXCPM_CLI` | Override đường dẫn `voxcpm2-cli` |
| `DV_VOXCPM_TTS_SERVER` | Override `llama-tts-server` |
| `DV_LLAMA_CPP_OMNI_DIR` | Thư mục clone llama.cpp-omni khi build VoxCPM2 |
| `DV_BUILD_JOBS` | Số job parallel khi compile C++ |

---

## Xử lý lỗi thường gặp

### `xcode-select: error` / thiếu compiler

```bash
xcode-select --install
# Chờ cài xong, rồi chạy lại build
```

### `vendor/voxcpm2/voxcpm2-cli not found`

Chạy trước:

```bash
brew install cmake
bash scripts/build-voxcpm2-mac.sh
```

### `target llama-tts-server not found`

Chỉ build `voxcpm2-cli` vẫn được — app fallback sang CLI mode (chậm hơn). Script `build-voxcpm2-mac.sh` tự phát hiện target có tồn tại hay không.

### `pnpm` / `node` not found sau bootstrap

Mở terminal mới hoặc:

```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
```

Rồi chạy lại `pnpm run tauri:build:mac:m4`.

### Build Tauri lỗi Rust

```bash
rustup target add aarch64-apple-darwin
rustup default stable
```

### Tải model Hugging Face chậm / timeout

Lần sau script bỏ qua file đã có trong `dist-portable/macos-staging/portable-runtime/models/`. Có thể set `HF_TOKEN` nếu rate-limit.

### VAD / TTS mới không chạy sau khi pull code mới

Script Mac luôn chạy lại `uv pip install` và `uv pip install -e backend` khi build runtime. Nếu venv cũ vẫn lỗi, xóa staging rồi build lại:

```bash
rm -rf dist-portable/macos-staging/portable-runtime/.venv
pnpm run tauri:build:mac:m4
```

### Gatekeeper chặn app

Chuột phải → **Open** (lần đầu). Không dùng `xattr -cr` trừ khi bạn hiểu rủi ro bảo mật.

### ASR/TTS chậm trên Mac

Mac dùng **Metal/MPS hoặc CPU**, không có CUDA. Đây là hành vi bình thường so với Windows + NVIDIA.

---

## Chạy tests

```bash
pnpm install
pnpm run setup
pnpm test
```

Một số test GPU/CUDA chỉ áp dụng Windows — pytest trên Mac có thể skip một phần.

---

## Không muốn build — dùng release có sẵn

Tải `DouyinVietnamizer-0.1.0-portable-macos.zip` từ [GitHub Releases](https://github.com/minhvhdev/DouyinVietnamizer/releases), giải nén, mở app như mục **Bước 6**.

---

## Checklist hoàn tất (agent tự verify)

- [ ] `uname -m` = `arm64`
- [ ] `xcode-select -p` thành công
- [ ] `vendor/voxcpm2/voxcpm2-cli` tồn tại và `--help` chạy được
- [ ] `dist-portable/.../DouyinVietnamizer.app` tồn tại
- [ ] `dist-portable/DouyinVietnamizer-0.1.0-portable-macos.zip` tồn tại
- [ ] App mở được (sau Gatekeeper lần đầu)

---

## Tham khảo

- [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) — engine C++ cho VoxCPM2
- [VoxCPM2 GGUF weights](https://huggingface.co/DennisHuang648/VoxCPM2-GGUF)
- Script repo: `scripts/build-voxcpm2-mac.sh`, `scripts/build-portable-runtime-mac.sh`, `scripts/build-portable-mac.sh`
