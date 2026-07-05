#!/usr/bin/env bash
# Build voxcpm2-cli (+ llama-tts-server) from llama.cpp-omni for Apple Silicon
# and install binaries into vendor/voxcpm2/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor/voxcpm2"
BUILD_ROOT="${DV_LLAMA_CPP_OMNI_DIR:-$REPO_ROOT/.cache/llama.cpp-omni}"
JOBS="${DV_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "ERROR: This script is macOS-only." >&2
  exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
  echo "ERROR: Apple Silicon (arm64) required. Detected: $(uname -m)" >&2
  exit 1
fi
if ! xcode-select -p >/dev/null 2>&1; then
  echo "ERROR: Xcode Command Line Tools are missing. Run: xcode-select --install" >&2
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "ERROR: cmake not found. Install with: brew install cmake" >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found. Install with: brew install git" >&2
  exit 1
fi

mkdir -p "$VENDOR_DIR"

if [ ! -d "$BUILD_ROOT/.git" ]; then
  echo ">>> Cloning llama.cpp-omni into $BUILD_ROOT"
  mkdir -p "$(dirname "$BUILD_ROOT")"
  git clone https://github.com/tc-mb/llama.cpp-omni.git "$BUILD_ROOT"
else
  echo ">>> Updating llama.cpp-omni in $BUILD_ROOT"
  git -C "$BUILD_ROOT" pull --ff-only
fi

echo ">>> Configuring llama.cpp-omni (Metal auto-detected on macOS)..."
cmake -B "$BUILD_ROOT/build" -DCMAKE_BUILD_TYPE=Release -S "$BUILD_ROOT"

TARGETS=(voxcpm2-cli)
if cmake --build "$BUILD_ROOT/build" --target help 2>/dev/null | grep -q 'llama-tts-server'; then
  TARGETS+=(llama-tts-server)
else
  echo ">>> Note: llama-tts-server target not found; building voxcpm2-cli only."
fi

echo ">>> Building: ${TARGETS[*]}"
cmake --build "$BUILD_ROOT/build" --target "${TARGETS[@]}" -j"$JOBS"

BIN_DIR="$BUILD_ROOT/build/bin"
if [ ! -x "$BIN_DIR/voxcpm2-cli" ]; then
  echo "ERROR: $BIN_DIR/voxcpm2-cli was not produced." >&2
  exit 1
fi

echo ">>> Installing into $VENDOR_DIR"
install -m 755 "$BIN_DIR/voxcpm2-cli" "$VENDOR_DIR/voxcpm2-cli"
if [ -x "$BIN_DIR/llama-tts-server" ]; then
  install -m 755 "$BIN_DIR/llama-tts-server" "$VENDOR_DIR/llama-tts-server"
fi

echo ">>> Verifying binaries"
"$VENDOR_DIR/voxcpm2-cli" --help >/dev/null
ls -la "$VENDOR_DIR"

echo ">>> Done. voxcpm2-cli is ready at $VENDOR_DIR/voxcpm2-cli"
