# Portable Windows App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows x64 portable-folder Tauri app that starts with full bundled Python/tools/models, no setup wizard, while `pnpm tauri:dev` uses the same runtime without rebuilding it on source edits.

**Architecture:** Add a Rust portable runtime resolver with explicit validation, then make backend spawn use that resolver instead of `uv`/PATH setup checks. Keep FastAPI over localhost and React polling unchanged in shape; only status variants change from setup-oriented to portable-runtime-oriented.

**Tech Stack:** Tauri 2, Rust 2021, Tokio, React/Vite/TypeScript, FastAPI/Python, pytest, Vitest.

## Global Constraints

- Release target is a copyable Windows x64 folder, not a single `.exe`.
- Bundle Python runtime/deps, FFmpeg, yt-dlp, Qwen3-ASR model files, and VoxCPM2 model/runtime files before release.
- Assume target machines have compatible NVIDIA/CUDA drivers; no CPU fallback.
- `pnpm tauri:dev` must use `vendor/portable-runtime` and keep frontend HMR + backend reload.
- No runtime/model downloading inside the app.
- Do not add new dependencies unless a task explicitly says so; use existing `tempfile`, Tokio, serde, Vitest, pytest.
- Do not commit during implementation unless the user explicitly asks for commits.

---

## File Structure

- Create `src-tauri/src/portable.rs`: pure runtime path resolution, validation, spawn environment construction, and tests.
- Modify `src-tauri/src/lib.rs`: register `portable` module and initialize `BackendState` with repo backend dir + dev flag only.
- Modify `src-tauri/src/state.rs`: keep backend child/base URL/backend dir/dev profile; add resolved runtime cache if needed.
- Modify `src-tauri/src/backend.rs`: remove `uv`/Python PATH venv detection from startup path; spawn packaged Python from `PortableRuntime`.
- Modify `src-tauri/src/commands.rs`: return `portable_missing` instead of `setup_required`; keep crash/ready semantics.
- Modify `frontend/src/lib/tauri-bridge.ts`: update status union and `waitForBackend` early-fail variants.
- Modify `frontend/src/renderer/App.tsx`: render portable package error screen instead of setup wizard for Tauri startup failures.
- Leave `frontend/src/renderer/SetupWizard.tsx` in place unless TypeScript proves it is unreachable and removable with less code.
- Modify `backend/dv_backend/vendor.py`: let `VendorResolver` prefer `DV_PORTABLE_RUNTIME_DIR/tools`.
- Modify `backend/dv_backend/runtime.py`: default vendor dir/manifest to portable runtime when env is set.
- Modify `src-tauri/tauri.conf.json`: include portable runtime resources and switch away from MSI-only output.
- Modify tests in `backend/tests/`, `frontend/tests/`, and Rust modules.
- Modify `README.md`: document portable runtime layout and dev command.

---

### Task 1: Add Rust portable runtime resolver

**Files:**
- Create: `src-tauri/src/portable.rs`
- Modify: `src-tauri/src/lib.rs`

**Interfaces:**
- Produces: `PortableRuntime`, `PortableRuntimeStatus`, `resolve_portable_runtime`, `validate_portable_runtime`, `python_executable`, `prepend_path`.
- Consumes: only stdlib paths/env and existing `cfg!(debug_assertions)` profile flag.

- [ ] **Step 1: Add module export**

Modify `src-tauri/src/lib.rs` module list to include `portable`:

```rust
pub mod backend;
pub mod commands;
pub mod portable;
pub mod setup;
pub mod state;
```

- [ ] **Step 2: Write resolver and validation tests first**

Create `src-tauri/src/portable.rs` with tests and stubbed functions that compile after Step 3:

```rust
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableRuntime {
    pub root: PathBuf,
    pub python: PathBuf,
    pub backend_dir: PathBuf,
    pub tools_dir: PathBuf,
    pub models_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PortableRuntimeStatus {
    Ready { root: PathBuf },
    Missing { root: PathBuf, missing_items: Vec<String> },
}

pub fn resolve_portable_runtime(repo_root: &Path, dev_profile: bool) -> Result<PortableRuntime, PortableRuntimeStatus> {
    let root = runtime_root(repo_root, dev_profile);
    validate_portable_runtime(&root)
}

pub fn runtime_root(repo_root: &Path, dev_profile: bool) -> PathBuf {
    if let Some(value) = env::var_os("DV_PORTABLE_RUNTIME_DIR") {
        return PathBuf::from(value);
    }
    if dev_profile {
        return repo_root.join("vendor").join("portable-runtime");
    }
    release_runtime_root()
}

pub fn validate_portable_runtime(root: &Path) -> Result<PortableRuntime, PortableRuntimeStatus> {
    let python = python_executable(root);
    let backend_dir = root.join("backend");
    let tools_dir = root.join("tools");
    let models_dir = root.join("models");
    let required = [
        (python.clone(), "python executable"),
        (backend_dir.join("dv_backend"), "backend/dv_backend"),
        (tools_dir.join("ffmpeg"), "tools/ffmpeg"),
        (tools_dir.join("yt-dlp"), "tools/yt-dlp"),
        (models_dir.join("qwen3-asr"), "models/qwen3-asr"),
        (models_dir.join("voxcpm2"), "models/voxcpm2"),
    ];
    let missing_items = required
        .into_iter()
        .filter_map(|(path, label)| (!path.exists()).then(|| format!("{} ({})", label, path.display())))
        .collect::<Vec<_>>();
    if missing_items.is_empty() {
        Ok(PortableRuntime { root: root.to_path_buf(), python, backend_dir, tools_dir, models_dir })
    } else {
        Err(PortableRuntimeStatus::Missing { root: root.to_path_buf(), missing_items })
    }
}

pub fn python_executable(root: &Path) -> PathBuf {
    let embedded = root.join("python").join("python.exe");
    if embedded.exists() {
        return embedded;
    }
    root.join(".venv").join("Scripts").join("python.exe")
}

pub fn prepend_path(dir: &Path, current: Option<OsString>) -> OsString {
    let mut paths = vec![dir.to_path_buf()];
    if let Some(current) = current {
        paths.extend(env::split_paths(&current));
    }
    env::join_paths(paths).unwrap_or_else(|_| OsString::from(dir.as_os_str()))
}

fn release_runtime_root() -> PathBuf {
    let exe = env::current_exe().unwrap_or_else(|_| PathBuf::from("."));
    let exe_dir = exe.parent().unwrap_or_else(|| Path::new("."));
    let beside_exe = exe_dir.join("portable-runtime");
    if beside_exe.exists() {
        return beside_exe;
    }
    exe_dir.join("resources").join("portable-runtime")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn make_runtime(root: &Path) {
        fs::create_dir_all(root.join(".venv/Scripts")).unwrap();
        fs::write(root.join(".venv/Scripts/python.exe"), b"").unwrap();
        fs::create_dir_all(root.join("backend/dv_backend")).unwrap();
        fs::create_dir_all(root.join("tools/ffmpeg")).unwrap();
        fs::create_dir_all(root.join("tools/yt-dlp")).unwrap();
        fs::create_dir_all(root.join("models/qwen3-asr")).unwrap();
        fs::create_dir_all(root.join("models/voxcpm2")).unwrap();
    }

    #[test]
    fn validate_accepts_complete_runtime() {
        let dir = tempdir().unwrap();
        make_runtime(dir.path());
        let runtime = validate_portable_runtime(dir.path()).unwrap();
        assert_eq!(runtime.root, dir.path());
        assert_eq!(runtime.backend_dir, dir.path().join("backend"));
    }

    #[test]
    fn validate_lists_all_missing_items() {
        let dir = tempdir().unwrap();
        let err = validate_portable_runtime(dir.path()).unwrap_err();
        match err {
            PortableRuntimeStatus::Missing { missing_items, .. } => {
                assert!(missing_items.iter().any(|item| item.contains("python executable")));
                assert!(missing_items.iter().any(|item| item.contains("backend/dv_backend")));
                assert!(missing_items.iter().any(|item| item.contains("tools/ffmpeg")));
                assert!(missing_items.iter().any(|item| item.contains("tools/yt-dlp")));
                assert!(missing_items.iter().any(|item| item.contains("models/qwen3-asr")));
                assert!(missing_items.iter().any(|item| item.contains("models/voxcpm2")));
            }
            PortableRuntimeStatus::Ready { .. } => panic!("expected missing runtime"),
        }
    }

    #[test]
    fn env_override_wins() {
        let dir = tempdir().unwrap();
        let override_dir = dir.path().join("custom-runtime");
        std::env::set_var("DV_PORTABLE_RUNTIME_DIR", &override_dir);
        let got = runtime_root(Path::new("C:/repo"), true);
        std::env::remove_var("DV_PORTABLE_RUNTIME_DIR");
        assert_eq!(got, override_dir);
    }

    #[test]
    fn dev_root_uses_vendor_portable_runtime() {
        std::env::remove_var("DV_PORTABLE_RUNTIME_DIR");
        assert_eq!(
            runtime_root(Path::new("C:/repo"), true),
            PathBuf::from("C:/repo").join("vendor").join("portable-runtime"),
        );
    }

    #[test]
    fn prepends_tool_path() {
        let got = prepend_path(Path::new("C:/rt/tools"), Some(OsString::from("C:/Windows")));
        let parts = env::split_paths(&got).collect::<Vec<_>>();
        assert_eq!(parts[0], PathBuf::from("C:/rt/tools"));
        assert_eq!(parts[1], PathBuf::from("C:/Windows"));
    }
}
```

- [ ] **Step 3: Run Rust tests for the new module**

Run: `cd src-tauri && cargo test portable::tests --lib`

Expected: PASS for five `portable::tests::*` tests.

- [ ] **Step 4: Checkpoint diff**

Run: `git diff -- src-tauri/src/lib.rs src-tauri/src/portable.rs`

Expected: diff only adds the `portable` module and the resolver file.

---

### Task 2: Spawn backend through portable runtime

**Files:**
- Modify: `src-tauri/src/backend.rs`
- Modify: `src-tauri/src/commands.rs`
- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/lib.rs`

**Interfaces:**
- Consumes: `portable::PortableRuntime`, `portable::resolve_portable_runtime`, `portable::prepend_path` from Task 1.
- Produces: `BackendStatusDto::PortableMissing { root, missing_items }` and `spawn_uvicorn(runtime, source_backend_dir, dev_profile)`.

- [ ] **Step 1: Add backend env builder test**

Append these tests to `src-tauri/src/backend.rs` `mod tests`:

```rust
#[test]
fn build_backend_command_env_uses_portable_runtime() {
    let root = PathBuf::from("C:/rt");
    let runtime = crate::portable::PortableRuntime {
        root: root.clone(),
        python: root.join(".venv/Scripts/python.exe"),
        backend_dir: root.join("backend"),
        tools_dir: root.join("tools"),
        models_dir: root.join("models"),
    };
    let envs = build_backend_env(&runtime, true);
    assert_eq!(envs.get("DV_PORTABLE_RUNTIME_DIR").unwrap(), &root.as_os_str().to_os_string());
    assert_eq!(envs.get("DV_RELOAD").unwrap(), "1");
    assert!(envs.get("PATH").unwrap().to_string_lossy().contains("tools"));
}

#[test]
fn backend_working_dir_uses_source_in_dev_and_packaged_in_release() {
    let root = PathBuf::from("C:/rt");
    let runtime = crate::portable::PortableRuntime {
        root: root.clone(),
        python: root.join(".venv/Scripts/python.exe"),
        backend_dir: root.join("backend"),
        tools_dir: root.join("tools"),
        models_dir: root.join("models"),
    };
    assert_eq!(backend_working_dir(&runtime, Path::new("C:/repo/backend"), true), PathBuf::from("C:/repo/backend"));
    assert_eq!(backend_working_dir(&runtime, Path::new("C:/repo/backend"), false), root.join("backend"));
}
```

- [ ] **Step 2: Run tests to verify missing functions fail**

Run: `cd src-tauri && cargo test backend::tests::build_backend_command_env_uses_portable_runtime backend::tests::backend_working_dir_uses_source_in_dev_and_packaged_in_release --lib`

Expected: FAIL with missing `build_backend_env` and `backend_working_dir`.

- [ ] **Step 3: Implement backend spawn helpers**

In `src-tauri/src/backend.rs`, add imports:

```rust
use std::collections::HashMap;
use std::ffi::OsString;
use crate::portable::{prepend_path, PortableRuntime};
```

Replace `spawn_uvicorn` with this signature and implementation:

```rust
pub fn backend_working_dir(runtime: &PortableRuntime, source_backend_dir: &Path, dev_profile: bool) -> PathBuf {
    if dev_profile {
        source_backend_dir.to_path_buf()
    } else {
        runtime.backend_dir.clone()
    }
}

pub fn build_backend_env(runtime: &PortableRuntime, dev_profile: bool) -> HashMap<&'static str, OsString> {
    let mut envs = HashMap::new();
    envs.insert("DV_RELOAD", OsString::from(if dev_profile { "1" } else { "0" }));
    envs.insert("DV_PORTABLE_RUNTIME_DIR", runtime.root.as_os_str().to_os_string());
    envs.insert("DV_VENDOR_DIR", runtime.tools_dir.as_os_str().to_os_string());
    envs.insert("DV_MODELS_DIR", runtime.models_dir.as_os_str().to_os_string());
    envs.insert("DV_ALLOW_PATH_TOOLS", OsString::from("0"));
    envs.insert("PATH", prepend_path(&runtime.tools_dir, std::env::var_os("PATH")));
    envs
}

pub fn spawn_uvicorn(
    runtime: &PortableRuntime,
    source_backend_dir: &Path,
    dev_profile: bool,
) -> Result<tokio::process::Child, BackendStartError> {
    use std::process::Stdio;
    let mut cmd = tokio::process::Command::new(&runtime.python);
    cmd.args(["-m", "dv_backend.main"])
        .current_dir(backend_working_dir(runtime, source_backend_dir, dev_profile))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in build_backend_env(runtime, dev_profile) {
        cmd.env(key, value);
    }
    cmd.spawn().map_err(|e| BackendStartError::Spawn(format!("{} using runtime {}", e, runtime.root.display())))
}
```

Remove the old `uv run python -m dv_backend.main` implementation.

- [ ] **Step 4: Update command status enum**

In `src-tauri/src/commands.rs`, change `BackendStatusDto` to:

```rust
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackendStatusDto {
    PortableMissing { root: String, missing_items: Vec<String> },
    Starting,
    Ready { base_url: String },
    Crashed { stderr: String },
    AlreadyRunning,
}
```

- [ ] **Step 5: Resolve runtime inside commands**

In `get_backend_status`, replace the venv detection block with:

```rust
let runtime = match crate::portable::resolve_portable_runtime(&PathBuf::from("."), state.dev_profile) {
    Ok(runtime) => runtime,
    Err(crate::portable::PortableRuntimeStatus::Missing { root, missing_items }) => {
        return Ok(BackendStatusDto::PortableMissing {
            root: root.display().to_string(),
            missing_items,
        });
    }
    Err(crate::portable::PortableRuntimeStatus::Ready { .. }) => unreachable!(),
};
```

Then change spawn calls:

```rust
let mut child = backend::spawn_uvicorn(&runtime, &state.backend_dir, state.dev_profile)
    .map_err(|e| e.to_string())?;
```

In `restart_backend`, resolve runtime the same way and call:

```rust
let mut child = backend::spawn_uvicorn(&runtime, &state.backend_dir, state.dev_profile)?;
```

- [ ] **Step 6: Keep setup command but make it inert for portable mode**

Leave `run_first_time_setup_cmd` registered for now to avoid unrelated frontend compile churn. Add this first line to the command body:

```rust
return Err(SetupError::SyncFailed("portable builds do not run first-time setup; rebuild the portable-runtime folder".into()));
```

This makes accidental setup clicks fail clearly if any old UI path remains.

- [ ] **Step 7: Run Rust tests**

Run: `cd src-tauri && cargo test --lib`

Expected: PASS. Existing venv tests in `backend.rs` may fail because venv detection is now obsolete; delete only those tests and the obsolete `VenvStatus`/`detect_venv` code if needed, then rerun.

- [ ] **Step 8: Checkpoint diff**

Run: `git diff -- src-tauri/src/backend.rs src-tauri/src/commands.rs src-tauri/src/state.rs src-tauri/src/lib.rs`

Expected: backend spawn no longer invokes `uv`; command status includes `portable_missing`.

---

### Task 3: Update frontend startup status handling

**Files:**
- Modify: `frontend/src/lib/tauri-bridge.ts`
- Modify: `frontend/src/renderer/App.tsx`
- Modify: `frontend/tests/App.test.tsx`

**Interfaces:**
- Consumes: `BackendStatusDto::PortableMissing { root, missing_items }` from Task 2.
- Produces: TypeScript `BackendStatus` union with `portable_missing` and UI that displays package errors.

- [ ] **Step 1: Update frontend status type test**

Add this test near the top of `frontend/tests/App.test.tsx` after `baseApi`:

```tsx
vi.mock("../src/lib/tauri-bridge", async () => {
  const actual = await vi.importActual<typeof import("../src/lib/tauri-bridge")>("../src/lib/tauri-bridge");
  return {
    ...actual,
    waitForBackend: vi.fn().mockResolvedValue("http://127.0.0.1:8765"),
    subscribeBackendEvents: vi.fn().mockReturnValue(() => {}),
    invokeOpenDevtools: vi.fn().mockResolvedValue(undefined),
  };
});
```

Then add a specific portable missing test:

```tsx
test("shows portable package errors when bundled runtime is incomplete", async () => {
  const bridge = await import("../src/lib/tauri-bridge");
  vi.mocked(bridge.waitForBackend).mockRejectedValueOnce({
    kind: "portable_missing",
    root: "C:/App/resources/portable-runtime",
    missing_items: ["models/qwen3-asr (C:/App/resources/portable-runtime/models/qwen3-asr)"],
  });

  render(<App api={baseApi} />);

  expect(await screen.findByText("Portable package is incomplete")).toBeInTheDocument();
  expect(screen.getByText("C:/App/resources/portable-runtime")).toBeInTheDocument();
  expect(screen.getByText(/models\/qwen3-asr/)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run frontend test to verify it fails**

Run: `pnpm --filter frontend test -- App.test.tsx -t "portable package errors"`

Expected: FAIL because `portable_missing` is not in the union/UI yet.

- [ ] **Step 3: Update Tauri bridge type and wait behavior**

In `frontend/src/lib/tauri-bridge.ts`, change `BackendStatus` to:

```ts
export type BackendStatus =
  | { kind: "portable_missing"; root: string; missing_items: string[] }
  | { kind: "starting" }
  | { kind: "ready"; base_url: string }
  | { kind: "crashed"; stderr: string }
  | { kind: "already_running" };
```

Change the early throw condition in `waitForBackend`:

```ts
if (s.kind === "portable_missing" || s.kind === "crashed") {
  throw s;
}
```

Leave `invokeSetup` and setup progress exports only if `SetupWizard.tsx` still imports them. If TypeScript reports no references after App changes, delete those exports and `SetupWizard.tsx` together.

- [ ] **Step 4: Add portable error screen in App**

In `frontend/src/renderer/App.tsx`, replace the current block:

```tsx
if (backend?.kind === "setup_required" || backend?.kind === "crashed") {
  return (
    <SetupWizard ... />
  );
}
```

with:

```tsx
if (backend?.kind === "portable_missing") {
  return (
    <div className="p-6 max-w-3xl mx-auto text-zinc-100">
      <h1 className="text-xl font-semibold text-red-400">Portable package is incomplete</h1>
      <p className="mt-2 text-zinc-300">The app could not find required bundled runtime files.</p>
      <div className="mt-4 rounded bg-zinc-900 p-3">
        <strong>Runtime path</strong>
        <code className="block mt-1 text-sm text-zinc-300">{backend.root}</code>
      </div>
      <ul className="mt-4 list-disc pl-6 text-sm text-red-200">
        {backend.missing_items.map((item) => <li key={item}>{item}</li>)}
      </ul>
      <button onClick={() => location.reload()} className="mt-4 underline">Retry after fixing the portable folder</button>
    </div>
  );
}

if (backend?.kind === "crashed") {
  return (
    <div className="p-6 max-w-3xl mx-auto text-zinc-100">
      <h1 className="text-xl font-semibold text-red-400">Backend crashed</h1>
      <pre className="mt-3 p-3 bg-zinc-900 text-zinc-100 text-sm overflow-auto rounded">
        {backend.stderr || "(no stderr captured)"}
      </pre>
      <button onClick={() => location.reload()} className="mt-4 underline">Retry</button>
    </div>
  );
}
```

Remove `SetupWizard` import if unused.

- [ ] **Step 5: Run targeted frontend test**

Run: `pnpm --filter frontend test -- App.test.tsx -t "portable package errors"`

Expected: PASS.

- [ ] **Step 6: Run full frontend tests**

Run: `pnpm --filter frontend test`

Expected: PASS. If tests expecting setup wizard fail, update them to assert the runtime blocked screen text that remains inside `App.tsx`, not the old Tauri setup wizard.

- [ ] **Step 7: Checkpoint diff**

Run: `git diff -- frontend/src/lib/tauri-bridge.ts frontend/src/renderer/App.tsx frontend/tests/App.test.tsx`

Expected: setup-required startup path replaced by portable-missing path.

---

### Task 4: Make backend prefer portable runtime tools

**Files:**
- Modify: `backend/dv_backend/vendor.py`
- Modify: `backend/dv_backend/runtime.py`
- Create or modify: `backend/tests/test_vendor.py`
- Modify: `backend/tests/test_tool_probe.py` only if imports need consolidation.

**Interfaces:**
- Consumes: Rust-set env vars `DV_PORTABLE_RUNTIME_DIR`, `DV_VENDOR_DIR`, `DV_VENDOR_MANIFEST`, `DV_ALLOW_PATH_TOOLS`.
- Produces: `VendorResolver.resolve()` checks portable `tools/` before existing vendor dir and PATH.

- [ ] **Step 1: Write backend resolver tests**

Create `backend/tests/test_vendor.py`:

```python
from pathlib import Path

from dv_backend.vendor import VendorResolver, VendorTool


def tool() -> VendorTool:
    return VendorTool(
        id="fake",
        display_name="Fake tool",
        executable="fake/fake.exe",
        dev_command="fake",
        version_args=["--version"],
        version_contains="fake",
        required=True,
        capability="test",
    )


def test_resolver_prefers_portable_runtime_tools(monkeypatch, tmp_path: Path) -> None:
    runtime = tmp_path / "portable-runtime"
    portable_exe = runtime / "tools" / "fake" / "fake.exe"
    portable_exe.parent.mkdir(parents=True)
    portable_exe.write_text("fake", encoding="utf-8")
    vendor_dir = tmp_path / "vendor"
    bundled_exe = vendor_dir / "fake" / "fake.exe"
    bundled_exe.parent.mkdir(parents=True)
    bundled_exe.write_text("vendor", encoding="utf-8")
    monkeypatch.setenv("DV_PORTABLE_RUNTIME_DIR", str(runtime))

    resolved = VendorResolver(vendor_dir, allow_path_tools=False).resolve(tool())

    assert resolved.path == portable_exe
    assert resolved.source == "portable"


def test_resolver_uses_bundled_vendor_when_no_portable_runtime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DV_PORTABLE_RUNTIME_DIR", raising=False)
    vendor_dir = tmp_path / "vendor"
    bundled_exe = vendor_dir / "fake" / "fake.exe"
    bundled_exe.parent.mkdir(parents=True)
    bundled_exe.write_text("vendor", encoding="utf-8")

    resolved = VendorResolver(vendor_dir, allow_path_tools=False).resolve(tool())

    assert resolved.path == bundled_exe
    assert resolved.source == "bundled"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_vendor.py -v`

Expected: first test FAILS because source is not `portable` yet.

- [ ] **Step 3: Implement portable tool preference**

Modify `backend/dv_backend/vendor.py` imports:

```python
import os
```

Modify `VendorResolver.resolve`:

```python
def resolve(self, tool: VendorTool) -> ResolvedTool:
    portable_root = os.environ.get("DV_PORTABLE_RUNTIME_DIR")
    if portable_root:
        portable = Path(portable_root) / "tools" / tool.executable
        if portable.is_file():
            return ResolvedTool(path=portable, source="portable")
    bundled = self.vendor_dir / tool.executable
    if bundled.is_file():
        return ResolvedTool(path=bundled, source="bundled")
    if self.allow_path_tools:
        found = shutil.which(tool.dev_command)
        if found:
            return ResolvedTool(path=Path(found), source="path")
    return ResolvedTool(path=None, source="missing")
```

- [ ] **Step 4: Make runtime service default to portable manifest when present**

Modify `backend/dv_backend/runtime.py` `default_runtime_service`:

```python
def default_runtime_service(config: AppConfig, database: Database) -> RuntimeSmokeTestService:
    project_root = Path(__file__).resolve().parents[2]
    portable_root = os.environ.get("DV_PORTABLE_RUNTIME_DIR")
    if portable_root:
        runtime_root = Path(portable_root)
        default_vendor_dir = runtime_root / "tools"
        default_manifest = runtime_root / "manifest.json"
    else:
        default_vendor_dir = project_root / "vendor"
        default_manifest = default_vendor_dir / "manifest.json"
    vendor_dir = Path(os.environ.get("DV_VENDOR_DIR", default_vendor_dir))
    manifest_path = Path(os.environ.get("DV_VENDOR_MANIFEST", default_manifest))
    return RuntimeSmokeTestService(
        config, database, manifest_path, vendor_dir,
        allow_path_tools=os.environ.get("DV_ALLOW_PATH_TOOLS", "1") == "1",
    )
```

- [ ] **Step 5: Run backend tests**

Run: `cd backend && uv run pytest tests/test_vendor.py tests/test_tool_probe.py tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 6: Checkpoint diff**

Run: `git diff -- backend/dv_backend/vendor.py backend/dv_backend/runtime.py backend/tests/test_vendor.py`

Expected: portable runtime env is the first tool source.

---

### Task 5: Configure portable Tauri bundle and docs

**Files:**
- Modify: `src-tauri/tauri.conf.json`
- Modify: `README.md`
- Modify: `package.json` only if adding helper scripts proves useful.

**Interfaces:**
- Consumes: `vendor/portable-runtime` layout from design.
- Produces: documented build/dev contract.

- [ ] **Step 1: Update Tauri config resources and bundle target**

Modify `src-tauri/tauri.conf.json` bundle section to:

```json
"bundle": {
  "active": true,
  "targets": ["nsis"],
  "resources": {
    "../vendor/portable-runtime": "portable-runtime"
  },
  "icon": [
    "icons/icon.png",
    "icons/icon.ico"
  ]
}
```

If Tauri rejects object-form resources on this version, use array form instead:

```json
"resources": ["../vendor/portable-runtime"]
```

and adjust Rust release resolver to also check `resources/vendor/portable-runtime`. Prefer whichever format `pnpm tauri:build` accepts.

- [ ] **Step 2: Update README Tauri section**

Replace the current first-run setup wording in `README.md` Tauri section with:

```markdown
## Tauri desktop app

`pnpm tauri:dev` opens the app in a Tauri window. Rust spawns the Python backend from `vendor/portable-runtime` on `127.0.0.1:8765`. The dev app does not run first-time setup; prepare `vendor/portable-runtime` once, then frontend and backend source changes reload without rebuilding the runtime.

Portable runtime layout:

```text
vendor/portable-runtime/
├── .venv/ or python/
├── backend/
├── tools/
│   ├── ffmpeg/
│   └── yt-dlp/
├── models/
│   ├── qwen3-asr/
│   └── voxcpm2/
└── manifest.json
```

Hot-reload during development:
- Edit `frontend/src/renderer/**` — Vite HMR refreshes the window.
- Edit `backend/dv_backend/**` — uvicorn reloads because `DV_RELOAD=1`.
- Edit `src-tauri/src/**` — Cargo rebuilds the affected crate, window refreshes.

`pnpm tauri:build` produces a Windows app bundle that includes the prepared portable runtime. For a folder-style portable release, copy the built executable plus its `resources/portable-runtime` directory together. Target machines must be Windows x64 with compatible NVIDIA/CUDA drivers.
```

Keep the non-Tauri `pnpm dev` wording unchanged.

- [ ] **Step 3: Validate config JSON**

Run: `pnpm tauri info`

Expected: command reads config without JSON/schema errors. If it fails on `resources`, apply the array fallback from Step 1 and rerun.

- [ ] **Step 4: Checkpoint diff**

Run: `git diff -- src-tauri/tauri.conf.json README.md package.json`

Expected: docs no longer promise first-run setup for Tauri.

---

### Task 6: End-to-end verification and cleanup

**Files:**
- Modify only files needed to fix test failures from prior tasks.
- Do not delete `SetupWizard.tsx` unless `pnpm --filter frontend test` and TypeScript show it is unused and removal reduces code.

**Interfaces:**
- Consumes all prior tasks.
- Produces verified portable startup path.

- [ ] **Step 1: Run focused Rust tests**

Run: `cd src-tauri && cargo test portable::tests backend::tests --lib`

Expected: PASS.

- [ ] **Step 2: Run focused backend tests**

Run: `cd backend && uv run pytest tests/test_vendor.py tests/test_tool_probe.py tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 3: Run focused frontend tests**

Run: `pnpm --filter frontend test -- App.test.tsx`

Expected: PASS.

- [ ] **Step 4: Run all available project tests**

Run: `pnpm test`

Expected: PASS for backend and frontend. If tests fail because local portable runtime is absent, the failing tests are too integrated; narrow them to temp runtime fixtures and rerun.

- [ ] **Step 5: Verify dev startup with missing runtime error**

Temporarily run without a prepared runtime:

```bash
DV_PORTABLE_RUNTIME_DIR=C:/missing-runtime pnpm tauri:dev
```

Expected: Tauri window shows `Portable package is incomplete` and lists missing Python/backend/tools/models. Stop the app after confirming.

- [ ] **Step 6: Verify dev startup with real runtime**

Run:

```bash
pnpm tauri:dev
```

Expected: if `vendor/portable-runtime` is prepared, backend starts from portable Python and app reaches main UI. If the runtime is not prepared on this machine, record that manual full-start verification was skipped because `vendor/portable-runtime` is absent.

- [ ] **Step 7: Final diff review**

Run: `git diff --stat` and `git diff -- docs/superpowers/specs/2026-06-27-portable-windows-app-design.md docs/superpowers/plans/2026-06-27-portable-windows-app.md src-tauri frontend backend README.md package.json`

Expected: changes match the plan; no unrelated formatting churn.

- [ ] **Step 8: Report results**

Summarize:

```text
Implemented portable runtime startup.
Checks: <commands run + pass/fail>.
Skipped: <manual runtime/build checks skipped and exact reason>.
```

No commit unless the user asks.

---

## Self-Review

- Spec coverage: runtime resolver, dev/release spawn, frontend error path, backend tool preference, bundle config, docs, and verification are covered by Tasks 1-6.
- Placeholder scan: no `TBD`, `TODO`, or unspecified edge handling remains.
- Type consistency: Rust status uses `PortableMissing` -> serialized `portable_missing`; TypeScript union matches `portable_missing`; backend env uses `DV_PORTABLE_RUNTIME_DIR`, `DV_VENDOR_DIR`, `DV_MODELS_DIR`, and `DV_ALLOW_PATH_TOOLS` consistently.
- Deliberate simplification: keep `SetupWizard.tsx` unless deleting it is clearly smaller after compiler feedback. Upgrade path is full deletion once portable mode is stable.