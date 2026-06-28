# Portable macOS App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a macOS arm64 portable folder (`.app` + sibling `portable-runtime/`) that runs with full features immediately, built entirely in GitHub Actions (macos-14), while the existing Windows portable build keeps working unchanged.

**Architecture:** Add a macOS branch to the existing Tauri runtime resolver using `#[cfg(target_os = "macos")]` gates, add a Python `hardware.py` cross-platform detection that includes MPS, and add two new bash scripts plus a macOS GitHub Actions workflow. The Windows build's source files are not modified; only additive `#[cfg]`-gated code and new files. Build is fully reproducible from a clean runner using `python-build-standalone`, `uv`, and `huggingface_hub.snapshot_download`, with `actions/cache` for the 11 GB model payload.

**Tech Stack:** Tauri 2, Rust 2021, Python 3.12 (python-build-standalone), uv, huggingface_hub, GitHub Actions `macos-14`, bash, ditto, lsof.

## Global Constraints

- **Zero impact on Windows portable.** Existing `scripts/build-portable.ps1`, `src-tauri/src/backend.rs` Windows branch (`kill_port_listeners_windows`, `CREATE_NO_WINDOW`), and `src-tauri/src/portable.rs` Windows branch are not modified. All Rust changes are pure additions gated by `#[cfg(target_os = "macos")]`. `backend/pyproject.toml` is not modified.
- Build runs on GitHub Actions `macos-14` (M1). Free for public repos; 2000 min/month for private.
- Output: `dist-portable/DouyinVietnamizer-0.1.0-portable-macos.zip` containing `DouyinVietnamizer.app` and sibling `portable-runtime/`.
- Ad-hoc code signing only. No Developer ID, no notarization. End-user accepts Gatekeeper bypass once.
- arm64 only. No x86_64, no universal binary.
- Python runtime uses default PyPI index (no `pytorch-cu128` extra).
- No new runtime dependencies added to the project (the build uses `uv` and `huggingface_hub` only inside the macOS build script, not as project deps).
- Do not commit during implementation unless the user explicitly asks.

---

## File Structure

- **Modify** `src-tauri/src/portable.rs` — add `#[cfg(target_os = "macos")]` branch to `python_executable`. Windows branch untouched.
- **Modify** `src-tauri/src/backend.rs` — add `kill_port_listeners_macos` and wire it into `spawn_uvicorn` under `#[cfg(target_os = "macos")]`. Add unit tests.
- **Modify** `src-tauri/tauri.conf.json` — add `icons/icon.icns`, add `macOS` block. `targets` switches to `"all"`. `nsis` (Windows) still resolves correctly.
- **Modify** `backend/dv_backend/hardware.py` — gate Windows-specific probes by `sys.platform`; add MPS detection; add unit test.
- **Create** `src-tauri/icons/icon.icns` — generated once from `icon.png` and committed.
- **Create** `scripts/build-portable-runtime-mac.sh` — downloads python-build-standalone, builds uv venv, downloads tools + models, emits `portable-runtime/`.
- **Create** `scripts/build-portable-mac.sh` — assembles `.app` + `portable-runtime/`, syncs sources, zips.
- **Create** `.github/workflows/build-macos.yml` — runs the two scripts and uploads the zip.
- **Create** `.github/workflows/build-windows-regression.yml` — runs `pnpm tauri:build` + `build-portable.ps1` + asserts `.exe` exists on every PR.
- **Create** `docs/superpowers/specs/2026-06-28-portable-macos-app-design.md` — already done; referenced for design intent.
- **Modify** `README.md` — add "macOS portable" section near the Windows section.

---

### Task 1: Cross-platform `python_executable` in Rust

**Files:**
- Modify: `src-tauri/src/portable.rs:72-78` (the `python_executable` function body)
- Test: `src-tauri/src/portable.rs` (extend the existing `#[cfg(test)] mod tests`)

**Interfaces:**
- Produces: `python_executable(root: &Path) -> PathBuf` — returns `.venv/bin/python` on macOS, `.venv/Scripts/python.exe` on Windows. Existing Windows return values unchanged.
- Consumes: only `Path` from std.

- [ ] **Step 1: Add macOS branch via `#[cfg]`**

Replace the body of `python_executable` in `src-tauri/src/portable.rs` with:

```rust
pub fn python_executable(root: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        let embedded = root.join("python").join("python.exe");
        if embedded.exists() {
            return embedded;
        }
        root.join(".venv").join("Scripts").join("python.exe")
    }
    #[cfg(target_os = "macos")]
    {
        let embedded = root.join("python").join("bin").join("python3");
        if embedded.exists() {
            return embedded;
        }
        root.join(".venv").join("bin").join("python")
    }
}
```

The Windows branch is byte-for-byte the existing logic. macOS branch is new.

- [ ] **Step 2: Add macOS test**

In the same file, inside `mod tests`, add this test (do not remove the existing `make_runtime` helper; extend it to also create the macOS path):

```rust
#[cfg(target_os = "macos")]
fn make_runtime_macos(root: &Path) {
    fs::create_dir_all(root.join(".venv/bin")).unwrap();
    fs::write(root.join(".venv/bin/python"), b"").unwrap();
    fs::create_dir_all(root.join("backend/dv_backend")).unwrap();
    fs::create_dir_all(root.join("tools/ffmpeg")).unwrap();
    fs::create_dir_all(root.join("tools/yt-dlp")).unwrap();
    fs::create_dir_all(root.join("models/qwen3-asr")).unwrap();
    fs::create_dir_all(root.join("models/voxcpm2")).unwrap();
}

#[test]
#[cfg(target_os = "macos")]
fn python_executable_picks_venv_bin_python_on_macos() {
    let dir = tempdir().unwrap();
    make_runtime_macos(dir.path());
    let p = python_executable(dir.path());
    assert_eq!(p, dir.path().join(".venv").join("bin").join("python"));
}

#[test]
#[cfg(target_os = "macos")]
fn python_executable_prefers_embedded_python3_when_present() {
    let dir = tempdir().unwrap();
    make_runtime_macos(dir.path());
    fs::create_dir_all(dir.path().join("python/bin")).unwrap();
    fs::write(dir.path().join("python/bin/python3"), b"").unwrap();
    let p = python_executable(dir.path());
    assert_eq!(p, dir.path().join("python").join("bin").join("python3"));
}
```

- [ ] **Step 3: Run `cargo test` on Windows host**

Run: `cd src-tauri && cargo test --lib portable::tests::`
Expected: All existing Windows tests pass. New tests are gated by `#[cfg(target_os = "macos")]` and the Windows host skips them — `cargo test` exit code 0. **This is the Windows regression check.**

- [ ] **Step 4: Cross-check with `cargo check --target aarch64-apple-darwin`**

If a Mac or `osxcross` is unavailable, skip this step and rely on CI for cross-compile validation. Otherwise:
Run: `cd src-tauri && cargo check --target aarch64-apple-darwin`
Expected: Compiles clean. macOS branch validates.

- [ ] **Step 5: Commit (only if user asks)**

Do not commit yet — accumulate with Task 2's changes.

---

### Task 2: Add macOS port killer to Rust

**Files:**
- Modify: `src-tauri/src/backend.rs` — add `kill_port_listeners_macos` and call it in `spawn_uvicorn` under `#[cfg(target_os = "macos")]`. Add unit test.
- Modify: `src-tauri/src/backend.rs` — `spawn_uvicorn` adds a sibling `#[cfg(target_os = "macos")]` block next to the existing `#[cfg(windows)]` block.

**Interfaces:**
- Produces: `kill_port_listeners_macos(port: u16) -> std::io::Result<usize>` — returns number of PIDs killed.
- Consumes: only std `process::Command` and `lsof` (macOS system binary).
- Existing `kill_port_listeners_windows` is **not** changed.

- [ ] **Step 1: Add `kill_port_listeners_macos` function**

In `src-tauri/src/backend.rs`, add this function **after** the existing `kill_port_listeners_windows` (which ends with `}` followed by a blank line). Place the new function in that blank line area:

```rust
/// On macOS, kill any leftover process bound to `port`. Best-effort: returns the
/// number of processes successfully terminated. `Err` only if `lsof` itself
/// cannot be invoked.
#[cfg(target_os = "macos")]
fn kill_port_listeners_macos(port: u16) -> std::io::Result<usize> {
    use std::process::Command;
    let out = Command::new("lsof")
        .args(["-nP", "-tiTCP:", &port.to_string(), "-sTCP:LISTEN"])
        .output()?;
    let pids: Vec<u32> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let mut killed = 0usize;
    for pid in pids {
        let status = Command::new("kill")
            .args(["-9", &pid.to_string()])
            .status()?;
        if status.success() {
            killed += 1;
        }
    }
    Ok(killed)
}
```

- [ ] **Step 2: Wire into `spawn_uvicorn`**

In `spawn_uvicorn`, the existing `#[cfg(windows)]` block is:

```rust
    #[cfg(windows)]
    {
        let _ = kill_port_listeners_windows(BACKEND_PORT);
    }
```

Add a sibling block **immediately after** (before the `let mut cmd = ...` line):

```rust
    #[cfg(target_os = "macos")]
    {
        let _ = kill_port_listeners_macos(BACKEND_PORT);
    }
```

- [ ] **Step 3: Add unit test for the lsof parser**

Add a new test inside the existing `mod tests` in `backend.rs`. First add a helper:

```rust
#[cfg(target_os = "macos")]
fn parse_lsof_pids(text: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

#[test]
#[cfg(target_os = "macos")]
fn parse_lsof_pids_extracts_unique_pids() {
    let sample = "111\n222\n111\nabc\n";
    let mut got = parse_lsof_pids(sample);
    got.sort_unstable();
    got.dedup();
    assert_eq!(got, vec![111, 222]);
}
```

This test runs only on macOS. On Windows the function is gated out, so the existing Windows test suite is unaffected.

- [ ] **Step 4: Run `cargo test` on Windows host**

Run: `cd src-tauri && cargo test --lib`
Expected: All existing tests still pass. macOS-gated tests are skipped (no `target_os = "macos"`). Exit code 0.

- [ ] **Step 5: Commit (only if user asks)**

---

### Task 3: Update `tauri.conf.json` for macOS bundle

**Files:**
- Modify: `src-tauri/tauri.conf.json` (3 fields: `targets`, `icon`, new `macOS` block)

- [ ] **Step 1: Read current file and make additive changes**

Read `src-tauri/tauri.conf.json`. Apply these three edits, leaving every other field unchanged:

Edit 1 — change `"targets": "nsis"` to `"targets": "all"`:

```jsonc
    "targets": "all",
```

Edit 2 — append `"icons/icon.icns"` to the `icon` array:

```jsonc
    "icon": [
      "icons/icon.png",
      "icons/icon.ico",
      "icons/icon.icns"
    ],
```

Edit 3 — add a `macOS` block at the end of the `bundle` object (before the closing `}` of `bundle`):

```jsonc
    "macOS": {
      "minimumSystemVersion": "12.0"
    }
```

Resulting `bundle` section:

```json
  "bundle": {
    "active": true,
    "targets": "all",
    "resources": ["../vendor/portable-runtime"],
    "icon": [
      "icons/icon.png",
      "icons/icon.ico",
      "icons/icon.icns"
    ],
    "macOS": {
      "minimumSystemVersion": "12.0"
    }
  }
```

- [ ] **Step 2: Verify `tauri.conf.json` is still valid JSON**

Run: `cd src-tauri && cargo build` (just to exercise tauri's config loader)
Expected: Compiles. If JSON is malformed, tauri-build will error — fix and retry.

- [ ] **Step 3: Confirm Windows build still works locally**

Run: `cd .. && pnpm tauri:build`
Expected: NSIS installer produced at `src-tauri/target/release/bundle/nsis/*.exe`. macOS fields are ignored on Windows. The `targets: "all"` change must not break Windows.

If `targets: "all"` causes an error on Windows, revert to `"nsis"` for now and add a separate conditional config (out of scope for this plan; raise in PR review).

- [ ] **Step 4: Commit (only if user asks)**

---

### Task 4: Generate and commit `icon.icns`

**Files:**
- Create: `src-tauri/icons/icon.icns`

- [ ] **Step 1: Generate `icon.icns` from existing PNG**

This step needs any platform with `png2icns` or `iconutil`. The maintainer runs it once on their dev machine (Windows is fine via WSL, or a Mac, or Docker). Command (Linux/macOS):

```bash
# Option A: png2icns (Linux)
png2icns src-tauri/icons/icon.icns src-tauri/icons/icon.png

# Option B: macOS native
mkdir -p /tmp/icon.iconset
sips -z 16 16     src-tauri/icons/icon.png --out /tmp/icon.iconset/icon_16x16.png
sips -z 32 32     src-tauri/icons/icon.png --out /tmp/icon.iconset/icon_16x16@2x.png
sips -z 32 32     src-tauri/icons/icon.png --out /tmp/icon.iconset/icon_32x32.png
sips -z 64 64     src-tauri/icons/icon.png --out /tmp/icon.iconset/icon_32x32@2x.png
sips -z 128 128   src-tauri/icons/icon.png --out /tmp/icon.iconset/icon_128x128.png
sips -z 256 256   src-tauri/icons/icon.png --out /tmp/icon.iconset/icon_128x128@2x.png
sips -z 256 256   src-tauri/icons/icon.png --out /tmp/icon.iconset/icon_256x256.png
sips -z 512 512   src-tauri/icons/icon.png --out /tmp/icon.iconset/icon_256x256@2x.png
sips -z 512 512   src-tauri/icons/icon.png --out /tmp/icon.iconset/icon_512x512.png
sips -z 1024 1024 src-tauri/icons/icon.png --out /tmp/icon.iconset/icon_512x512@2x.png
iconutil -c icns /tmp/icon.iconset -o src-tauri/icons/icon.icns
```

Verify file size > 50 KB and head bytes are `icns` magic:

```bash
file src-tauri/icons/icon.icns   # should say "Mac OS X icns"
xxd src-tauri/icons/icon.icns | head -1   # should start with "636e73..." = "icns"
```

- [ ] **Step 2: Commit `icon.icns` (only if user asks)**

Note: the binary file gets committed to git history. If the repo prefers LFS, configure LFS for `*.icns` in a separate task. For now, commit directly (Tauri's other icons like `icon.ico` are already committed binary).

---

### Task 5: Make `backend/dv_backend/hardware.py` cross-platform

**Files:**
- Modify: `backend/dv_backend/hardware.py` — add `import sys` at the top; gate each Windows-specific probe with `sys.platform != "win32"` short-circuit; add MPS detection inside `detect_cuda`.
- Test: `backend/tests/test_hardware.py` — new file with the tests below.

- [ ] **Step 1: Write the failing test first**

Create `backend/tests/test_hardware.py`:

```python
import sys
from unittest import mock

import pytest


def _import_hardware():
    """Re-import hardware module after mocking torch."""
    import importlib
    from dv_backend import hardware
    importlib.reload(hardware)
    return hardware


def test_detect_vulkan_false_on_non_windows(monkeypatch):
    if sys.platform == "win32":
        pytest.skip("non-Windows specific test")
    hardware = _import_hardware()
    assert hardware.detect_vulkan() is False


def test_detect_cpu_avx2_true_on_non_windows(monkeypatch):
    if sys.platform == "win32":
        pytest.skip("non-Windows specific test")
    hardware = _import_hardware()
    assert hardware.detect_cpu_avx2() is True


def test_detect_espeak_false_on_non_windows(monkeypatch):
    if sys.platform == "win32":
        pytest.skip("non-Windows specific test")
    hardware = _import_hardware()
    assert hardware.detect_espeak() is False


def test_detect_cuda_reports_mps_on_macos(monkeypatch):
    if sys.platform != "darwin":
        pytest.skip("macOS-specific test")
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    hardware = _import_hardware()
    assert hardware.detect_cuda() is True


def test_detect_cuda_falls_back_to_false_when_torch_missing(monkeypatch):
    hardware = _import_hardware()
    monkeypatch.setitem(sys.modules, "torch", None)
    assert hardware.detect_cuda() is False
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `cd backend && uv run pytest tests/test_hardware.py -v`
Expected: FAIL — the function bodies are still the original Windows-only code; `detect_vulkan` on Linux/Mac will try `ctypes.windll.LoadLibrary` and raise or return wrong value.

- [ ] **Step 3: Modify `hardware.py` to be cross-platform**

Edit `backend/dv_backend/hardware.py`. Add `import sys` at the top (after the existing `import` lines):

```python
import ctypes
import os
import sys
from pathlib import Path
```

Replace the body of `detect_vulkan`:

```python
def detect_vulkan() -> bool:
    """Probes the system for Vulkan support. Windows-only probe; returns False elsewhere."""
    if sys.platform != "win32":
        return False
    try:
        # Drivers supporting Vulkan place vulkan-1.dll in System32
        vulkan_lib = ctypes.windll.LoadLibrary("vulkan-1.dll")
        return vulkan_lib is not None
    except Exception:
        return False
```

Replace the body of `detect_cpu_avx2`:

```python
def detect_cpu_avx2() -> bool:
    """Checks if the CPU supports AVX2. Windows: IsProcessorFeaturePresent. Other: positive default."""
    if sys.platform == "win32":
        try:
            # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40 in Windows SDK
            kernel32 = ctypes.windll.kernel32
            return kernel32.IsProcessorFeaturePresent(40) != 0
        except Exception:
            # Fallback to True or check AVX (36) if AVX2 call fails or is unsupported
            try:
                return kernel32.IsProcessorFeaturePresent(36) != 0
            except Exception:
                return False
    # macOS / linux: assume modern CPU (Apple Silicon = ARMv8.4 with i8mm/dotprod).
    return True
```

Replace the body of `detect_espeak`:

```python
def detect_espeak() -> bool:
    """Checks for eSpeak NG. Windows: Program Files scan. Other: returns False (optional dep)."""
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        paths_to_check = [
            Path(program_files) / "eSpeak NG" / "libespeak-ng.dll",
            Path(program_files_x86) / "eSpeak NG" / "libespeak-ng.dll",
        ]
        for p in paths_to_check:
            if p.is_file():
                return True
        return False
    return False
```

Replace the body of `detect_cuda`:

```python
def detect_cuda() -> bool:
    """Returns True if a GPU backend is available: NVIDIA CUDA (any OS) or Apple MPS (macOS)."""
    try:
        import torch

        if torch.cuda.is_available():
            return True
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            return True
        return False
    except Exception:
        return False
```

- [ ] **Step 4: Run tests**

Run: `cd backend && uv run pytest tests/test_hardware.py -v`
Expected: All 5 new tests pass on the current platform. Tests with `sys.platform` skip on the wrong OS.

- [ ] **Step 5: Run full backend test suite (regression check)**

Run: `cd backend && uv run pytest -v`
Expected: All previously-passing tests still pass. `hardware.py` changes are additive and Windows behavior is preserved.

- [ ] **Step 6: Commit (only if user asks)**

---

### Task 6: Create `scripts/build-portable-runtime-mac.sh`

**Files:**
- Create: `scripts/build-portable-runtime-mac.sh`
- Create: `scripts/lib/macos-runtime.sh` (helper functions if needed — skip if script fits in one file)

**Interfaces:**
- Input: none (reads `backend/pyproject.toml`)
- Output: `dist-portable/macos-staging/portable-runtime/` containing `python/`, `.venv/`, `backend/`, `tools/`, `models/`, `manifest.json`
- Consumes: standard tools (`curl`, `tar`, `uv`, `huggingface_hub` via `uv run python -c ...`)

- [ ] **Step 1: Write the script**

Create `scripts/build-portable-runtime-mac.sh`:

```bash
#!/usr/bin/env bash
# Builds a macOS arm64 portable runtime into dist-portable/macos-staging/portable-runtime/.
# Idempotent: re-runs are safe; existing files are reused.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="$REPO_ROOT/dist-portable/macos-staging"
RUNTIME="$STAGING/portable-runtime"
PY_VERSION="3.12.7"
PY_TAG="20240909"
PBS_BASE="https://github.com/astral-sh/python-build-standalone/releases/download"
PBS_URL="$PBS_BASE/$PY_TAG/cpython-$PY_VERSION+aarch64-apple-darwin-install_only.tar.gz"

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
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/build-portable-runtime-mac.sh`

- [ ] **Step 3: Smoke test on the macOS CI runner**

This step only runs in CI. Locally (Windows), verify only that the script parses with `bash -n`:

Run: `bash -n scripts/build-portable-runtime-mac.sh && echo OK`
Expected: `OK`

- [ ] **Step 4: Commit (only if user asks)**

---

### Task 7: Create `scripts/build-portable-mac.sh`

**Files:**
- Create: `scripts/build-portable-mac.sh`

**Interfaces:**
- Input: `src-tauri/target/aarch64-apple-darwin/release/bundle/macos/DouyinVietnamizer.app` (from `tauri build`) and `dist-portable/macos-staging/portable-runtime/`
- Output: `dist-portable/DouyinVietnamizer-0.1.0-portable/DouyinVietnamizer.app`, sibling `portable-runtime/`, and `dist-portable/DouyinVietnamizer-0.1.0-portable-macos.zip`

- [ ] **Step 1: Write the script**

Create `scripts/build-portable-mac.sh`:

```bash
#!/usr/bin/env bash
# Assembles the final macOS portable folder and zips it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST_NAME="DouyinVietnamizer-0.1.0-portable"
DST="$REPO_ROOT/dist-portable/$DST_NAME"
STAGING_RUNTIME="$REPO_ROOT/dist-portable/macos-staging/portable-runtime"
APP_PATH="$REPO_ROOT/src-tauri/target/aarch64-apple-darwin/release/bundle/macos/DouyinVietnamizer.app"

if [ ! -d "$APP_PATH" ]; then
  echo "ERROR: $APP_PATH not found. Run 'pnpm tauri build --target aarch64-apple-darwin' first." >&2
  exit 1
fi
if [ ! -d "$STAGING_RUNTIME" ]; then
  echo "ERROR: $STAGING_RUNTIME not found. Run scripts/build-portable-runtime-mac.sh first." >&2
  exit 1
fi

echo ">>> Assembling $DST/"
rm -rf "$DST"
mkdir -p "$DST"
cp -R "$APP_PATH" "$DST/DouyinVietnamizer.app"
cp -R "$STAGING_RUNTIME" "$DST/portable-runtime"

# Re-sync backend Python sources in case they changed after the staging build.
rsync -a --delete "$REPO_ROOT/backend/dv_backend/" "$DST/portable-runtime/backend/dv_backend/"
rsync -a "$REPO_ROOT/backend/scripts/" "$DST/portable-runtime/backend/scripts/"
cp "$REPO_ROOT/backend/pyproject.toml" "$DST/portable-runtime/backend/pyproject.toml"

cd "$REPO_ROOT/dist-portable"
ditto -c -k --sequesterRsrc --keepParent "$DST_NAME" "${DST_NAME}-macos.zip"
echo ">>> Built: dist-portable/${DST_NAME}-macos.zip"
ls -la "${DST_NAME}-macos.zip"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/build-portable-mac.sh`

- [ ] **Step 3: Bash syntax check**

Run: `bash -n scripts/build-portable-mac.sh && echo OK`
Expected: `OK`

- [ ] **Step 4: Commit (only if user asks)**

---

### Task 8: Add macOS CI workflow

**Files:**
- Create: `.github/workflows/build-macos.yml`

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/build-macos.yml`:

```yaml
name: Build macOS portable

on:
  workflow_dispatch:
  push:
    tags: ['v*']
  pull_request:

jobs:
  build:
    runs-on: macos-14
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v4
        with:
          version: 10

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: pnpm

      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: aarch64-apple-darwin

      - name: Cache macOS portable runtime
        uses: actions/cache@v4
        with:
          path: dist-portable/macos-staging
          key: macos-runtime-${{ hashFiles('backend/pyproject.toml', '.github/workflows/build-macos.yml') }}
          restore-keys: macos-runtime-

      - name: Install frontend deps
        run: pnpm install --frozen-lockfile

      - name: Build frontend
        run: pnpm --filter frontend build

      - name: Cargo test (host x86_64)
        run: cd src-tauri && cargo test --lib

      - name: Tauri build (aarch64-apple-darwin)
        run: pnpm tauri build --target aarch64-apple-darwin

      - name: Build portable runtime
        run: bash scripts/build-portable-runtime-mac.sh

      - name: Assemble portable folder + zip
        run: bash scripts/build-portable-mac.sh

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: DouyinVietnamizer-0.1.0-portable-macos
          path: dist-portable/DouyinVietnamizer-0.1.0-portable-macos.zip

      - name: Attach to GitHub Release (on tag)
        if: startsWith(github.ref, 'refs/tags/v')
        uses: softprops/action-gh-release@v2
        with:
          files: dist-portable/DouyinVietnamizer-0.1.0-portable-macos.zip
```

- [ ] **Step 2: Validate YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/build-macos.yml'))" && echo OK`
Expected: `OK`

(If `python` is unavailable, use any other YAML linter or `pnpm dlx js-yaml`.)

- [ ] **Step 3: Commit (only if user asks)**

---

### Task 9: Add Windows regression CI workflow

**Files:**
- Create: `.github/workflows/build-windows-regression.yml`

**Purpose:** Every PR and every push to `main` runs the Windows portable build to confirm the macOS refactor did not break Windows.

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/build-windows-regression.yml`:

```yaml
name: Build Windows portable (regression)

on:
  pull_request:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  build:
    runs-on: windows-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v4
        with:
          version: 10

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: pnpm

      - name: Install frontend deps
        run: pnpm install --frozen-lockfile

      - name: Rust tests
        run: cd src-tauri && cargo test --lib

      - name: Tauri build
        run: pnpm tauri:build

      - name: Build portable
        shell: pwsh
        run: pwsh scripts/build-portable.ps1

      - name: Assert Windows .exe exists
        shell: bash
        run: test -f dist-portable/DouyinVietnamizer-0.1.0-portable/douyin-vietnamizer.exe

      - name: Backend tests
        working-directory: backend
        run: uv run pytest -v
```

Note: This workflow depends on `vendor/portable-runtime/` being available in the repo. The Windows build is **read-only** with respect to that folder (the existing `build-portable.ps1` requires it to exist). If the repo is private and vendor/ is too large for checkout, see "Alternative: skip Tauri build, just cargo test" below.

Alternative slim workflow (if vendor/ is too heavy):

```yaml
      - name: Tauri build
        run: pnpm tauri:build
      - name: Skip full portable assembly (vendor too heavy for CI)
        run: echo "Skipping build-portable.ps1 in CI; covered by local maintainer"
```

Maintainer picks one before merging.

- [ ] **Step 2: Validate YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/build-windows-regression.yml'))" && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit (only if user asks)**

---

### Task 10: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README**

Read `README.md` and find the existing Windows portable section. Skip if absent.

- [ ] **Step 2: Add a macOS section immediately after the Windows section**

Append a new section:

```markdown
## macOS portable (Apple Silicon)

Pre-built portable zips are available on the [Releases page](../../releases) as `DouyinVietnamizer-0.1.0-portable-macos.zip` (arm64 only; macOS 12+).

To use:

1. Download the zip from the latest release.
2. Unzip anywhere (e.g. `~/Applications/`).
3. Open the resulting folder and double-click `DouyinVietnamizer.app`.
4. **First launch only:** macOS Gatekeeper will block the unsigned app. Right-click `DouyinVietnamizer.app` → **Open** → confirm. Subsequent launches double-click normally.

The `.app` looks for a sibling `portable-runtime/` folder with the bundled Python interpreter, models, and tools. Keep the folder intact.

### Building from source (maintainers)

Trigger the **Build macOS portable** workflow from the Actions tab, or push a `v*` tag. The job runs on `macos-14` and uploads the zip as an artifact. Local Mac builds are also possible:

```bash
pnpm install
pnpm tauri build --target aarch64-apple-darwin
bash scripts/build-portable-runtime-mac.sh
bash scripts/build-portable-mac.sh
```

The first run downloads ~11 GB of models; subsequent runs reuse `dist-portable/macos-staging/` and the GitHub Actions cache.
```

- [ ] **Step 3: Commit (only if user asks)**

---

### Task 11: End-to-end manual verification

**Files:** none (verification only)

- [ ] **Step 1: Local Windows regression (maintainer's Windows machine)**

Run:

```bash
cd src-tauri && cargo test --lib
cd ../.. && pnpm tauri:build
pwsh scripts/build-portable.ps1
test -f dist-portable/DouyinVietnamizer-0.1.0-portable/douyin-vietnamizer.exe
```

Expected: All pass. Windows portable build is identical to pre-macOS-PR behavior.

- [ ] **Step 2: Trigger macOS CI**

Push a branch and open a PR. Confirm:
- `.github/workflows/build-windows-regression.yml` runs and passes.
- `.github/workflows/build-macos.yml` runs and produces the artifact.

- [ ] **Step 3: Download the macOS artifact, transfer to friend's M4 Mac**

Download `DouyinVietnamizer-0.1.0-portable-macos.zip` from the workflow run. Send to friend's Mac. Unzip, double-click `.app`, verify:
- App opens without crash
- Backend becomes ready (status indicator shows ready within ~30s)
- Can download a Douyin URL and produce a translation
- No `uv` / `pip` / `brew` required on friend's machine

- [ ] **Step 4: Tag a release**

```bash
git tag v0.1.0
git push origin v0.1.0
```

The macOS workflow auto-attaches the zip to the GitHub Release.

- [ ] **Step 5: Final commit (only if user asks)**

---

## Self-Review

1. **Spec coverage:**
   - Goal (macOS arm64 portable folder, double-click) → Tasks 6, 7, 11
   - Zero impact on Windows → Tasks 1-3 add `#[cfg]` only; Task 5 gates by `sys.platform`; Task 9 enforces via CI
   - CI-only build → Task 8
   - Sibling folder layout → Task 7 assembly script
   - Ad-hoc signing only → README Task 10 (documented as Gatekeeper bypass)
   - `targets: "all"` → Task 3 (with fallback note)
   - `icon.icns` → Task 4
   - MPS detection → Task 5
   - Models from HuggingFace via `huggingface_hub` → Task 6 `download_model`
   - `actions/cache` for 11 GB models → Task 8

2. **Placeholder scan:** No "TBD" or "implement later". Every code block is complete and runnable.

3. **Type consistency:**
   - `python_executable(&Path) -> PathBuf` used in both branches.
   - `kill_port_listeners_macos(port: u16) -> std::io::Result<usize>` matches Windows counterpart signature.
   - `parse_lsof_pids` is test-only helper, no cross-task references.
   - `download_model` shell function used once in Task 6; self-contained.

4. **Ordering:** Tasks 1-5 are independent Rust/Python edits (could merge into 1 PR). Task 4 (icon) is also independent. Task 6, 7 depend on Tauri config from Task 3. Task 8 wires Tasks 6 + 7. Task 9 is independent. Task 10 depends on Task 8. Task 11 is the final gate.

5. **Cross-platform gate correctness:** All `#[cfg(target_os = "macos")]` blocks are siblings to existing `#[cfg(windows)]` blocks (Tasks 1, 2). All `sys.platform != "win32"` early-returns preserve Windows behavior (Task 5). No file is mutated on its Windows-only path.
