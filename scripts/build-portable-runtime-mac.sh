#!/usr/bin/env bash
# Builds a macOS arm64 portable runtime into dist-portable/macos-staging/portable-runtime/.
# With --full-build, this script also bootstraps local build dependencies and builds the app zip end-to-end.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="$REPO_ROOT/dist-portable/macos-staging"
RUNTIME="$STAGING/portable-runtime"
PY_VERSION="3.12.13"
PY_TAG="20260623"
PBS_BASE="https://github.com/astral-sh/python-build-standalone/releases/download"
PBS_URL="$PBS_BASE/$PY_TAG/cpython-$PY_VERSION+$PY_TAG-aarch64-apple-darwin-install_only.tar.gz"
FULL_BUILD="${1:-}"
PNPM_VERSION="10.12.1"

ensure_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

install_homebrew_if_missing() {
  if ensure_cmd brew; then
    return 0
  fi
  echo ">>> Homebrew not found. Installing Homebrew..."
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [ -x /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
}

bootstrap_mac_build_tools() {
  local arch
  arch="$(uname -m)"
  if [ "$arch" != "arm64" ]; then
    echo "ERROR: This build is only supported on Apple Silicon (arm64). Detected: $arch" >&2
    exit 1
  fi

  if ! xcode-select -p >/dev/null 2>&1; then
    echo ">>> Xcode Command Line Tools are missing. Triggering installer..."
    xcode-select --install || true
    echo "ERROR: Complete the Xcode Command Line Tools installation, then re-run this command." >&2
    exit 1
  fi

  install_homebrew_if_missing
  if [ -x /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi

  if ! ensure_cmd node; then
    echo ">>> Installing Node.js via Homebrew..."
    brew install node
  fi

  if ! ensure_cmd pnpm; then
    echo ">>> Installing pnpm..."
    corepack enable || true
    corepack prepare "pnpm@${PNPM_VERSION}" --activate || true
    if ! ensure_cmd pnpm; then
      brew install pnpm
    fi
  fi

  if ! ensure_cmd rustup; then
    echo ">>> Installing rustup..."
    if ! ensure_cmd rustup-init; then
      brew install rustup-init
    fi
    rustup-init -y --default-toolchain stable
  fi
  export PATH="$HOME/.cargo/bin:$PATH"

  if ! ensure_cmd uv; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  rustup target add aarch64-apple-darwin
}

if [ "$FULL_BUILD" = "--full-build" ]; then
  echo ">>> Bootstrapping local macOS build environment..."
  bootstrap_mac_build_tools

  cd "$REPO_ROOT"
  echo ">>> Installing JS dependencies..."
  pnpm install --frozen-lockfile
  echo ">>> Building frontend..."
  pnpm --filter frontend build
  echo ">>> Building Tauri app (aarch64-apple-darwin, unsigned local .app)..."
  pnpm tauri build --target aarch64-apple-darwin --no-sign
fi

mkdir -p "$STAGING"

if [ ! -x "$RUNTIME/python/bin/python3" ]; then
  echo ">>> Downloading python-build-standalone CPython $PY_VERSION (arm64 macOS)..."
  mkdir -p "$RUNTIME"
  curl -fL "$PBS_URL" -o "$STAGING/pbs.tar.gz"
  tar -xzf "$STAGING/pbs.tar.gz" -C "$RUNTIME"
  rm "$STAGING/pbs.tar.gz"
fi
PY="$RUNTIME/python/bin/python3"
echo ">>> Python: $($PY --version) at $PY"

# macOS requirements omit the Windows-only pytorch-cu128 extra index.
MAC_REQUIREMENTS="$STAGING/requirements-mac.txt"
cat > "$MAC_REQUIREMENTS" <<'EOF'
deep-translator>=1.11,<2
fastapi>=0.115,<1
uvicorn>=0.34,<1
qwen-asr>=0.0.6
funasr==1.3.10
librosa==0.11.0
numba==0.65.1
llvmlite==0.47.0
torch>=2.7
torchaudio>=2.7
soundfile>=0.12
numpy>=2
huggingface-hub>=0.26
transformers>=4.57
onnxruntime>=1.20
demucs>=4.0
pyannote-audio>=4.0,<5
hf-xet>=1.5
edge-tts>=7,<8
silero-vad>=5.1,<6
EOF

if [ ! -x "$RUNTIME/.venv/bin/python" ]; then
  echo ">>> Creating venv..."
  "$PY" -m venv "$RUNTIME/.venv"
fi
export VIRTUAL_ENV="$RUNTIME/.venv"
export UV_DEFAULT_INDEX="https://pypi.org/simple"
echo ">>> Syncing portable Python venv (macOS CPU/MPS torch)..."
"$RUNTIME/.venv/bin/python" -m pip install --upgrade pip
uv pip install --python "$RUNTIME/.venv/bin/python" -r "$MAC_REQUIREMENTS"
echo ">>> Venv ready: $RUNTIME/.venv"

# Sync backend sources.
echo ">>> Syncing backend sources..."
mkdir -p "$RUNTIME/backend"
rsync -a --delete "$REPO_ROOT/backend/dv_backend/" "$RUNTIME/backend/dv_backend/"
rsync -a "$REPO_ROOT/backend/scripts/" "$RUNTIME/backend/scripts/"
cp "$REPO_ROOT/backend/pyproject.toml" "$RUNTIME/backend/pyproject.toml"
if [ -f "$REPO_ROOT/backend/uv.lock" ]; then
  cp "$REPO_ROOT/backend/uv.lock" "$RUNTIME/backend/uv.lock"
fi

echo ">>> Installing backend package into portable venv..."
uv pip install --python "$RUNTIME/.venv/bin/python" -e "$RUNTIME/backend"

# Tools.
TOOLS="$RUNTIME/tools"
mkdir -p "$TOOLS/ffmpeg" "$TOOLS/yt-dlp" "$TOOLS/voxcpm2"
if [ ! -x "$TOOLS/ffmpeg/ffmpeg" ]; then
  echo ">>> Downloading ffmpeg (macOS arm64 static)..."
  curl -fL "https://www.osxexperts.net/ffmpeg7arm.zip" -o "$STAGING/ffmpeg.zip"
  unzip -o "$STAGING/ffmpeg.zip" -d "$TOOLS/ffmpeg/"
  rm "$STAGING/ffmpeg.zip"
  chmod +x "$TOOLS/ffmpeg/ffmpeg" "$TOOLS/ffmpeg/ffprobe" 2>/dev/null || true
fi
if [ ! -x "$TOOLS/yt-dlp/yt-dlp_macos" ]; then
  echo ">>> Downloading yt-dlp (macOS binary)..."
  curl -fL "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos" -o "$TOOLS/yt-dlp/yt-dlp_macos"
  chmod +x "$TOOLS/yt-dlp/yt-dlp_macos"
  # Make it discoverable as `yt-dlp` (no .exe on Mac).
  ln -sf yt-dlp_macos "$TOOLS/yt-dlp/yt-dlp"
fi
if [ -x "$REPO_ROOT/vendor/voxcpm2/voxcpm2-cli" ]; then
  echo ">>> Bundling voxcpm2-cli into portable tools..."
  rsync -a "$REPO_ROOT/vendor/voxcpm2/" "$TOOLS/voxcpm2/"
  chmod +x "$TOOLS/voxcpm2/voxcpm2-cli" 2>/dev/null || true
elif [ ! -x "$TOOLS/voxcpm2/voxcpm2-cli" ]; then
  echo "ERROR: vendor/voxcpm2/voxcpm2-cli not found." >&2
  echo "Build/copy the macOS arm64 VoxCPM2 CLI into vendor/voxcpm2/, then rerun this script." >&2
  exit 1
fi

# Models via huggingface_hub.
download_model() {
  local repo="$1" dest="$2"
  if [ -d "$dest" ] && [ -f "$dest/config.json" ]; then
    echo ">>> Model already present: $dest"
    return
  fi
  echo ">>> Downloading $repo -> $dest"
  mkdir -p "$dest"
  "$RUNTIME/.venv/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$repo', local_dir='$dest', local_dir_use_symlinks=False, allow_patterns=['*.json','*.txt','*.safetensors','*.pth','*.py','tokenizer*','vocab*','merges*','special_tokens*','preprocessor*','chat_template*','*.md','generation*'])"
}

download_model "Qwen/Qwen3-ASR-1.7B" "$RUNTIME/models/qwen3-asr/Qwen3-ASR-1.7B"
download_model "Qwen/Qwen3-ForcedAligner-0.6B" "$RUNTIME/models/qwen3-asr/Qwen3-ForcedAligner-0.6B"
download_model_gguf() {
  local repo="$1" dest="$2"
  shift 2
  local files=("$@")
  mkdir -p "$dest"
  for file in "${files[@]}"; do
    if [ -f "$dest/$file" ]; then
      echo ">>> GGUF already present: $dest/$file"
      continue
    fi
    echo ">>> Downloading $repo/$file -> $dest"
    "$RUNTIME/.venv/bin/python" -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$repo', filename='$file', local_dir='$dest')"
  done
}

download_model_gguf "DennisHuang648/VoxCPM2-GGUF" "$RUNTIME/models/voxcpm2" \
  "VoxCPM2-BaseLM-Q8_0.gguf" "VoxCPM2-Acoustic-F16.gguf"

# Manifest (portable format with macOS binary names).
cat > "$RUNTIME/manifest.json" <<'EOF'
{
  "schema_version": 1,
  "tools": [
    {
      "id": "ffmpeg",
      "display_name": "FFmpeg",
      "executable": "tools/ffmpeg/ffmpeg",
      "dev_command": "ffmpeg",
      "version_args": ["-version"],
      "version_contains": "ffmpeg",
      "required": true,
      "capability": "media"
    },
    {
      "id": "voxcpm2-cli",
      "display_name": "VoxCPM2 CLI",
      "executable": "tools/voxcpm2/voxcpm2-cli",
      "dev_command": "voxcpm2-cli",
      "version_args": ["--help"],
      "version_contains": "",
      "success_exit_codes": [0, 1],
      "required": true,
      "capability": "tts"
    },
    {
      "id": "yt_dlp",
      "display_name": "yt-dlp",
      "executable": "tools/yt-dlp/yt-dlp_macos",
      "dev_command": "yt-dlp",
      "version_args": ["--version"],
      "version_contains": "",
      "required": true,
      "capability": "download"
    }
  ]
}
EOF

echo ">>> Runtime built at: $RUNTIME"
du -sh "$RUNTIME"

if [ "$FULL_BUILD" = "--full-build" ]; then
  echo ">>> Assembling final portable zip..."
  bash "$REPO_ROOT/scripts/build-portable-mac.sh"
fi
