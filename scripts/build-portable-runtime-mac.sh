#!/usr/bin/env bash
# Builds a macOS arm64 portable runtime into dist-portable/macos-staging/portable-runtime/.
# Idempotent: re-runs are safe; existing files are reused.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="$REPO_ROOT/dist-portable/macos-staging"
RUNTIME="$STAGING/portable-runtime"
PY_VERSION="3.12.13"
PY_TAG="20260623"
PBS_BASE="https://github.com/astral-sh/python-build-standalone/releases/download"
PBS_URL="$PBS_BASE/$PY_TAG/cpython-$PY_VERSION+$PY_TAG-aarch64-apple-darwin-install_only.tar.gz"

mkdir -p "$STAGING"

if [ ! -x "$RUNTIME/python/bin/python3" ]; then
  echo ">>> Downloading python-build-standalone CPython $PY_VERSION (arm64 macOS)..."
  mkdir -p "$RUNTIME"
  curl -fL "$PBS_URL" -o "$STAGING/pbs.tar.gz"
  tar -xzf "$STAGING/pbs.tar.gz" -C "$RUNTIME" --strip-components=1
  rm "$STAGING/pbs.tar.gz"
fi
PY="$RUNTIME/python/bin/python3"
echo ">>> Python: $($PY --version) at $PY"

# Build a separate macOS pyproject that omits the pytorch-cu128 extra index.
MAC_PYPROJECT="$STAGING/pyproject.mac.toml"
cat > "$MAC_PYPROJECT" <<'EOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "douyin-vietnamizer-backend"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "deep-translator>=1.11,<2",
  "fastapi>=0.115,<1",
  "uvicorn>=0.34,<1",
  "qwen-asr>=0.0.6",
  "torch>=2.7",
  "torchaudio>=2.7",
  "soundfile>=0.12",
  "numpy>=2",
  "huggingface-hub>=0.26",
  "transformers>=4.57",
  "onnxruntime>=1.20",
  "demucs>=4.0",
  "funasr>=1.3.3",
  "pyannote-audio>=4.0,<5",
  "hf-xet>=1.5",
]

[tool.hatch.build.targets.wheel]
packages = ["dv_backend"]
EOF

if [ ! -x "$RUNTIME/.venv/bin/python" ]; then
  echo ">>> Creating venv with uv..."
  "$PY" -m venv "$RUNTIME/.venv"
  export VIRTUAL_ENV="$RUNTIME/.venv"
  export UV_DEFAULT_INDEX="https://pypi.org/simple"
  uv pip install --upgrade pip
  uv pip install -r "$MAC_PYPROJECT"
  uv pip install huggingface_hub
fi
echo ">>> Venv ready: $RUNTIME/.venv"

# Sync backend sources.
echo ">>> Syncing backend sources..."
rsync -a --delete "$REPO_ROOT/backend/dv_backend/" "$RUNTIME/backend/dv_backend/"
rsync -a "$REPO_ROOT/backend/scripts/" "$RUNTIME/backend/scripts/"
cp "$REPO_ROOT/backend/pyproject.toml" "$RUNTIME/backend/pyproject.toml"

# Tools.
TOOLS="$RUNTIME/tools"
mkdir -p "$TOOLS/ffmpeg" "$TOOLS/yt-dlp"
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
download_model "OpenBMB/VoxCPM2" "$RUNTIME/models/voxcpm2/VoxCPM2"

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
