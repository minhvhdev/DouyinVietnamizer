# Portable macOS App — Design Spec

**Date:** 2026-06-28
**Status:** Approved (pending user review)
**Target:** Ship a macOS arm64 portable folder (Apple Silicon M-series) that runs with full features immediately, matching the existing Windows portable model, without any user-facing setup. Built entirely in CI (GitHub Actions macos-14) since the maintainer has no Mac locally.

## Goal

Extend the existing Windows portable model to macOS arm64 (Apple Silicon M1/M2/M3/M4) with the same "unzip → double-click → run" experience:

1. The release artifact is a `.zip` containing `DouyinVietnamizer.app` and a sibling `portable-runtime/` folder. End user unzips, double-clicks `.app`, the app launches the bundled Python backend, and full pipeline features work immediately.
2. The build runs entirely in GitHub Actions (`macos-14` runner, free for public repos, 2000 min/month for private) so no Mac is required locally.
3. The existing Windows portable build (`scripts/build-portable.ps1`, NSIS `.exe`) continues to work **unchanged**. Windows is the regression baseline.
4. Ad-hoc code signing only (no Apple Developer ID, no notarization). End user runs the app with one-time "right-click → Open" bypass on first launch.

## Non-Goals

- Intel Mac (x86_64) support. arm64 only, matching the maintainer's friend's M4.
- Universal/fat binary (arm64 + x86_64). Not needed; doubles disk for no benefit.
- Code signing with Apple Developer ID. Out of scope; ad-hoc only.
- Notarization. Out of scope; end user accepts Gatekeeper bypass.
- Auto-update mechanism.
- Single-file `.app` (app must keep `portable-runtime/` as a sibling folder).
- Any change that alters the Windows portable build's behavior, output paths, or required assets. **Windows is the regression baseline.**

## Architecture

### Artifact topology (macOS)

```
DouyinVietnamizer-0.1.0-portable-macos.zip
└─ DouyinVietnamizer-0.1.0-portable/
   ├─ DouyinVietnamizer.app/           # Tauri-built app bundle (double-click target)
   │  └─ Contents/
   │     ├─ MacOS/douyin-vietnamizer   # aarch64 binary
   │     └─ ...
   └─ portable-runtime/                # discovered via release_runtime_root() beside_exe
      ├─ python/bin/python3            # python-build-standalone CPython 3.12
      ├─ .venv/bin/python              # uv-built venv with all Python deps
      ├─ backend/
      │  ├─ dv_backend/                # synced from repo at build time
      │  ├─ scripts/
      │  └─ pyproject.toml
      ├─ tools/
      │  ├─ ffmpeg/ffmpeg
      │  ├─ yt-dlp/yt-dlp
      │  └─ ...                        # macOS arm64 binaries
      ├─ models/
      │  ├─ qwen3-asr/
      │  │  ├─ Qwen3-ASR-1.7B/         # safetensors
      │  │  └─ Qwen3-ForcedAligner-0.6B/
      │  └─ voxcpm2/
      │     └─ VoxCPM2/                # safetensors + audiovae.pth
      └─ manifest.json                 # vendored tool manifest
```

This mirrors the Windows topology at `dist-portable/DouyinVietnamizer-0.1.0-portable/`. The `.app` plus sibling `portable-runtime/` is the analog of `douyin-vietnamizer.exe` plus sibling `portable-runtime/`.

### Build pipeline (GitHub Actions)

```
.github/workflows/build-macos.yml
   runner: macos-14  (M1, free for public repos)
   trigger: workflow_dispatch + tag v*

   1. Setup pnpm, Node 20, Rust toolchain
   2. Cache:
        ~/.cargo/registry, src-tauri/target, pnpm store,
        python-build-standalone tarball, models archive (~11GB)
   3. pnpm install
   4. pnpm --filter frontend build
   5. pnpm tauri build --target aarch64-apple-darwin
        → produces src-tauri/target/.../release/bundle/macos/DouyinVietnamizer.app
   6. ./scripts/build-portable-runtime-mac.sh
        → produces portable-runtime/ in a staging dir
        (idempotent: reuses cache for python interpreter, venv deps, models)
   7. ./scripts/build-portable-mac.sh
        → assembles final folder, syncs dv_backend sources, zips
   8. Upload .zip as artifact (and attach to release on tag)
```

### Why this design

- **CI-only build**: maintainer has no Mac. `macos-14` GitHub-hosted runner is M1, free for public repos and 2000 min/month for private.
- **No committed binaries**: `vendor/portable-runtime/` is Windows-only and stays that way. macOS runtime is built on-the-fly in CI, keeping the repo lean.
- **Cache by content hash**: `actions/cache` keyed on model dir + pyproject.toml hash. First build ~25 min (downloads 11GB models, builds Python venv), subsequent builds ~8 min.
- **Sibling folder layout** matches Windows exactly. The Tauri runtime discovery logic (`portable.rs::release_runtime_root`) already supports `portable-runtime/` beside the executable — `Contents/MacOS/../portable-runtime/` resolves the same way.

## Components

### 1. Rust cross-platform refactor

**`src-tauri/src/portable.rs::python_executable`** — split by `#[cfg]`:

```rust
pub fn python_executable(root: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        let embedded = root.join("python").join("python.exe");
        if embedded.exists() { return embedded; }
        return root.join(".venv").join("Scripts").join("python.exe");
    }
    #[cfg(target_os = "macos")]
    {
        let embedded = root.join("python").join("bin").join("python3");
        if embedded.exists() { return embedded; }
        return root.join(".venv").join("bin").join("python");
    }
}
```

**Windows branch is byte-for-byte unchanged.**

**`src-tauri/src/backend.rs`** — add a macOS port killer. Windows path (`kill_port_listeners_windows`, `CREATE_NO_WINDOW`) stays gated with `#[cfg(windows)]` exactly as today. New addition:

```rust
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
        let status = Command::new("kill").args(["-9", &pid.to_string()]).status()?;
        if status.success() { killed += 1; }
    }
    Ok(killed)
}
```

`spawn_uvicorn` extends the `#[cfg(windows)]` block to also call the macOS variant under `#[cfg(target_os = "macos")]`. The existing `CREATE_NO_WINDOW` block is unchanged.

**`src-tauri/src/portable.rs::validate_portable_runtime`** — required paths list already uses forward-slash-style joins (e.g. `tools_dir.join("ffmpeg")`); the joiner works on macOS. No code change needed there.

**`src-tauri/tauri.conf.json`** — additive only:

```jsonc
{
  "bundle": {
    "active": true,
    "targets": "all",          // was "nsis"; "all" picks platform-appropriate targets
    "icon": [
      "icons/icon.png",
      "icons/icon.ico",
      "icons/icon.icns"        // NEW
    ],
    "macOS": {                 // NEW
      "minimumSystemVersion": "12.0"
    }
  }
}
```

Tauri merges `macOS` block additively. On Windows builds, Tauri ignores it. `targets: "all"` on Windows still resolves to `nsis` (only available target on Windows). On macOS, it resolves to `app`. (If this proves fragile during implementation, fall back to env-driven `targets` via `tauri.conf.json` per platform — but try `"all"` first.)

### 2. Python cross-platform refactor

**`backend/dv_backend/hardware.py`** — gate Windows-specific probes:

```python
import sys

def detect_vulkan() -> bool:
    if sys.platform != "win32":
        return False
    # ... existing Windows code unchanged ...

def detect_cpu_avx2() -> bool:
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            return kernel32.IsProcessorFeaturePresent(40) != 0
        except Exception:
            try:
                return kernel32.IsProcessorFeaturePresent(36) != 0
            except Exception:
                return False
    # macOS / linux: Apple Silicon M-series has ARMv8.4 with dotprod + i8mm;
    # use a positive default and let runtime handle ISA-specific kernels.
    return True

def detect_espeak() -> bool:
    if sys.platform == "win32":
        # existing Windows Program Files checks unchanged
        ...
    return False  # macOS uses bundled tokenizer; not a hard dep

def detect_cuda() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            return True
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            return True  # expose mps via cuda flag for upstream gpu_lease logic
        return False
    except Exception:
        return False
```

**Windows behavior is preserved**: `detect_vulkan` still probes `vulkan-1.dll` on win32; `detect_cpu_avx2` still calls `IsProcessorFeaturePresent` on win32; `detect_espeak` still scans Program Files on win32; `detect_cuda` still reports `torch.cuda.is_available()` on Windows (MPS branch is gated by `sys.platform == "darwin"`).

Add a new top-level helper for the report:

```python
def get_hardware_report() -> dict:
    cuda = detect_cuda()
    vulkan = detect_vulkan()
    avx2 = detect_cpu_avx2()
    espeak = detect_espeak()
    if cuda:              recommendation = "gpu_cuda"     # or gpu_mps on Mac, surfaced as gpu_cuda
    elif vulkan:          recommendation = "gpu_vulkan"
    elif avx2:            recommendation = "cpu_avx2"
    else:                 recommendation = "cpu_legacy"
    return {...}
```

The report's `recommendation` field continues to be one of the existing 4 values. On macOS with MPS, `cuda` is True → `gpu_cuda` is returned; the label is a misnomer on Mac but downstream consumers (`gpu_lease.py`) only check truthiness on the boolean, not the label. Verified-safe.

**`backend/dv_backend/gpu_lease.py`** — no change required if it only checks `cuda_supported` boolean. If it inspects `recommendation == "gpu_cuda"` and assumes NVIDIA, add a small `is_apple_silicon()` branch in this file to flip device selection to `mps`. Inspect during implementation; defer if no consumers branch on label.

**`backend/pyproject.toml`** — **unchanged**. The pytorch-cu128 index is gated by `explicit = true` and only resolved when the `[[tool.uv.index]]` block is referenced. On macOS, the runtime build script will create a sibling `pyproject.mac.toml` (or pass `UV_INDEX` / `UV_DEFAULT_INDEX` overrides) that omits the CUDA extra index. Repo file untouched.

### 3. Build scripts (new files only)

**`scripts/build-portable-runtime-mac.sh`** (new):

- Inputs: `backend/pyproject.toml`, list of model IDs, list of tool URLs
- Outputs: `dist-portable/macos-staging/portable-runtime/`
- Steps:
  1. Download `python-build-standalone` CPython 3.12.x macOS arm64 from `astral-sh/python-build-standalone` GitHub releases
  2. Extract to `portable-runtime/python/`
  3. Create venv: `portable-runtime/.venv` using the embedded Python
  4. `UV_DEFAULT_INDEX=https://pypi.org/simple uv pip install` the deps from `backend/pyproject.mac.toml` (a copy generated by stripping CUDA sources)
  5. Copy `backend/dv_backend/`, `backend/scripts/`, `backend/pyproject.toml` to `portable-runtime/backend/`
  6. Download ffmpeg (static macOS arm64 from `osxexperts.net/ffmpeg7arm.zip` or `evermeet.cx` mirror) and yt-dlp (PyInstaller `yt-dlp_macos` from `yt-dlp/yt-dlp` GitHub releases) into `portable-runtime/tools/`
  7. Download models from HuggingFace via `huggingface_hub.snapshot_download`:
     - `Qwen/Qwen3-ASR-1.7B` → `models/qwen3-asr/Qwen3-ASR-1.7B/`
     - `Qwen/Qwen3-ForcedAligner-0.6B` → `models/qwen3-asr/Qwen3-ForcedAligner-0.6B/`
     - `OpenBMB/VoxCPM2` → `models/voxcpm2/VoxCPM2/`
  8. Generate `portable-runtime/manifest.json` mirroring `vendor/manifest.json` (portable format with macOS binary names)
- Idempotent: every step checks target path before download/extract

**`scripts/build-portable-mac.sh`** (new):

- Inputs: `src-tauri/target/.../bundle/macos/DouyinVietnamizer.app`, `dist-portable/macos-staging/portable-runtime/`
- Outputs: `dist-portable/DouyinVietnamizer-0.1.0-portable/DouyinVietnamizer.app` + sibling `portable-runtime/`, then `dist-portable/DouyinVietnamizer-0.1.0-portable-macos.zip`
- Steps:
  1. `mkdir -p dist-portable/DouyinVietnamizer-0.1.0-portable/`
  2. Copy `.app` into the folder
  3. Copy `portable-runtime/` next to it
  4. Sync Python sources: `rsync -a --delete backend/dv_backend/ portable-runtime/backend/dv_backend/` (mirrors the robocopy logic in `build-portable.ps1`)
  5. `cd dist-portable && ditto -c -k --sequesterRsrc --keepParent DouyinVietnamizer-0.1.0-portable DouyinVietnamizer-0.1.0-portable-macos.zip`
- `build-portable.ps1` is **not modified** at all

### 4. CI workflow

**`.github/workflows/build-macos.yml`** (new):

```yaml
name: Build macOS portable
on:
  workflow_dispatch:
  push:
    tags: ['v*']
jobs:
  build:
    runs-on: macos-14
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with: { version: 10 }
      - uses: actions/setup-node@v4
        with: { node-version: 20, cache: pnpm }
      - uses: dtolnay/rust-toolchain@stable
        with: { targets: aarch64-apple-darwin }
      - run: pnpm install --frozen-lockfile
      - run: pnpm --filter frontend build
      - run: pnpm tauri build --target aarch64-apple-darwin
      - name: Restore runtime cache
        uses: actions/cache@v4
        with:
          path: dist-portable/macos-staging
          key: macos-runtime-${{ hashFiles('backend/pyproject.toml', '.github/workflows/build-macos.yml') }}
      - run: bash scripts/build-portable-runtime-mac.sh
      - run: bash scripts/build-portable-mac.sh
      - uses: actions/upload-artifact@v4
        with:
          name: DouyinVietnamizer-0.1.0-portable-macos
          path: dist-portable/DouyinVietnamizer-0.1.0-portable-macos.zip
      - if: startsWith(github.ref, 'refs/tags/v')
        uses: softprops/action-gh-release@v2
        with:
          files: dist-portable/DouyinVietnamizer-0.1.0-portable-macos.zip
```

### 5. Windows regression guard

**`.github/workflows/build-windows.yml`** (new, or extend existing CI if one exists) — a minimal Windows job that ensures the portable build still works:

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
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with: { version: 10 }
      - uses: actions/setup-node@v4
        with: { node-version: 20, cache: pnpm }
      - run: pnpm install --frozen-lockfile
      - run: cd src-tauri && cargo test --locked
      - run: pnpm tauri:build
      - run: pwsh scripts/build-portable.ps1
      - run: test -f dist-portable/DouyinVietnamizer-0.1.0-portable/douyin-vietnamizer.exe
```

If a Windows CI workflow already exists, **add the `dist-portable/.../douyin-vietnamizer.exe` existence check to it**; do not create a parallel workflow.

## Data flow

**Runtime data flow is unchanged.** Tauri spawns `python -m dv_backend.main` → uvicorn listens on `:8765` → Tauri polls `/api/health` → frontend calls API. Only the **build** pipeline changes; the **runtime** behavior is platform-agnostic by design (Tauri abstracts the .exe vs .app launcher; the Python backend already speaks HTTP).

## Error handling

- **CI build fail on macOS**: log all steps; upload `dist-portable/macos-staging/` partial output for debugging. Do not silently fail.
- **Tauri icon missing**: the `tauri build` step requires `icon.icns`. The maintainer generates it once from `icons/icon.png` (using `png2icns icons/icon.png icons/icon.icns` on any platform) and commits it to `src-tauri/icons/icon.icns`. No runtime generation in CI.
- **Port conflict on macOS**: `kill_port_listeners_macos` (lsof + kill -9) handles leftover uvicorn. `lsof` is in macOS base system.
- **Python deps fail to resolve**: macOS arm64 wheels exist for torch (MPS build), torchaudio, transformers, demucs, funasr, pyannote-audio, onnxruntime, qwen-asr, scipy. The build script uses default PyPI index (no CUDA extras) to avoid the cu128 mismatch.
- **Model download fail**: `huggingface_hub` retries on transient errors; cache key includes pyproject hash so a stable cache is reused.
- **First-launch Gatekeeper warning**: documented in `README.md` end-user section. Friend does right-click → Open → Open.

## Testing

- **Unit tests (Rust)**: existing `cargo test` suite in `portable.rs` and `backend.rs`. Add one test in `backend.rs` that constructs a fake `PortableRuntime` with macOS-style `.venv/bin/python` path and asserts `python_executable` returns it. Add one test for `kill_port_listeners_macos` parsing.
- **Unit tests (Python)**: existing `pytest` suite in `backend/tests/`. Add one test for `detect_cuda` returning True when `torch.backends.mps.is_available()` is True (mock both `torch.cuda` and `torch.backends.mps`).
- **CI smoke (Windows regression)**: `cargo test` + `tauri build` + `build-portable.ps1` + assert the `.exe` exists. Runs on every PR.
- **CI smoke (macOS)**: `cargo test --target aarch64-apple-darwin` + `tauri build` + `build-portable-mac.sh` + assert the `.app` exists. Runs on tag push.
- **Manual end-to-end**: maintainer downloads the artifact on their own Mac (or friend's M4) — verifies double-click → app opens → backend ready → can download a Douyin video and translate.

## Migration / Rollout

1. Land Rust refactor + Python refactor in one PR. Windows CI must remain green.
2. Land macOS scripts + workflow in a second PR. macOS CI builds the artifact.
3. Tag `v0.1.0-mac-portable-rc1` to trigger the macOS build.
4. Maintainer downloads artifact, tests on a Mac (or sends to friend).
5. If working, document the macOS section in `README.md` and tag `v0.1.0`.

If step 4 fails, the Windows portable folder is unaffected — the failure is isolated to the macOS artifact, and the Windows build still ships from the existing local flow.

## Out of scope (deferred)

- Code signing with Apple Developer ID ($99/year) — revisit when distribution grows.
- Notarization (`xcrun notarytool`) — revisit with signing.
- Universal binary (arm64 + x86_64) — only relevant if Intel-Mac users appear.
- macOS DMG installer — not needed for "double-click" portable model.
- Linux portable — separate spec when requested.
