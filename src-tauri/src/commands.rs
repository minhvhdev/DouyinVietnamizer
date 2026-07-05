use std::path::PathBuf;
use std::time::Duration;
use serde::Serialize;
use tauri::{State, WebviewWindow};

use crate::backend::{self, BackendStartError, BackendStatusKind};
use crate::portable::{self, PortableRuntime, PortableRuntimeStatus};
use crate::setup::SetupError;
use crate::state::BackendState;

const BACKEND_START_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackendStatusDto {
    PortableMissing { root: String, missing_items: Vec<String> },
    Starting,
    Ready { base_url: String },
    Crashed { stderr: String },
    AlreadyRunning,
}

impl From<BackendStatusKind> for BackendStatusDto {
    fn from(k: BackendStatusKind) -> Self {
        match k {
            BackendStatusKind::Starting => BackendStatusDto::Starting,
            BackendStatusKind::Ready => BackendStatusDto::Ready { base_url: "http://127.0.0.1:8765".into() },
            BackendStatusKind::Crashed { stderr } => BackendStatusDto::Crashed { stderr },
            BackendStatusKind::AlreadyRunning => BackendStatusDto::AlreadyRunning,
        }
    }
}

fn repo_root_from_backend_dir(backend_dir: &PathBuf) -> PathBuf {
    backend_dir
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn resolve_runtime_for_status(backend_dir: &PathBuf, dev_profile: bool) -> Result<PortableRuntime, BackendStatusDto> {
    match portable::resolve_portable_runtime(&repo_root_from_backend_dir(backend_dir), dev_profile) {
        Ok(runtime) => Ok(runtime),
        Err(PortableRuntimeStatus::Missing { root, missing_items }) => Err(BackendStatusDto::PortableMissing {
            root: root.display().to_string(),
            missing_items,
        }),
        Err(PortableRuntimeStatus::Ready { .. }) => unreachable!(),
    }
}

fn resolve_runtime_for_start(backend_dir: &PathBuf, dev_profile: bool) -> Result<PortableRuntime, BackendStartError> {
    resolve_runtime_for_status(backend_dir, dev_profile).map_err(|status| match status {
        BackendStatusDto::PortableMissing { root, missing_items } => BackendStartError::Spawn(format!(
            "portable runtime missing at {}: {}",
            root,
            missing_items.join(", ")
        )),
        _ => BackendStartError::Spawn("portable runtime unavailable".into()),
    })
}

#[tauri::command]
pub async fn get_backend_status(
    state: State<'_, BackendState>,
) -> Result<BackendStatusDto, String> {
    let runtime = match resolve_runtime_for_status(&state.backend_dir, state.dev_profile) {
        Ok(runtime) => runtime,
        Err(status) => return Ok(status),
    };
    let mut guard = state.child.lock().await;
    if let Some(child) = guard.as_mut() {
        match child.process.try_wait() {
            Ok(Some(_status)) => {
                let stderr = backend::parse_uvicorn_stderr(&backend::buffered_stderr(child));
                *guard = None;
                return Ok(BackendStatusDto::Crashed { stderr });
            }
            Ok(None) => {
                return Ok(BackendStatusDto::Ready {
                    base_url: state.base_url.clone(),
                });
            }
            Err(e) => return Err(e.to_string()),
        }
    }
    let mut child = backend::spawn_uvicorn(&runtime, &state.backend_dir, state.dev_profile)
        .map_err(|e| e.to_string())?;
    let base_url = state.base_url.clone();
    match backend::wait_for_ready(&base_url, &mut child, BACKEND_START_TIMEOUT).await {
        Ok(()) => {
            *guard = Some(child);
            Ok(BackendStatusDto::Ready { base_url })
        }
        Err(e) => {
            let _ = child.process.start_kill();
            let stderr = match &e {
                BackendStartError::Crashed { stderr, .. } => stderr.clone(),
                BackendStartError::Timeout(_) => format!(
                    "backend did not respond within {}s; startup may still be warming up portable runtime",
                    BACKEND_START_TIMEOUT.as_secs()
                ),
                BackendStartError::Spawn(s) => s.clone(),
            };
            Ok(BackendStatusDto::Crashed { stderr })
        }
    }
}

#[tauri::command]
pub async fn run_first_time_setup_cmd() -> Result<(), SetupError> {
    Err(SetupError::SyncFailed("portable builds do not run first-time setup; rebuild the portable-runtime folder".into()))
}

#[tauri::command]
pub async fn restart_backend(
    state: State<'_, BackendState>,
) -> Result<(), BackendStartError> {
    let runtime = resolve_runtime_for_start(&state.backend_dir, state.dev_profile)?;
    {
        let mut guard = state.child.lock().await;
        if let Some(mut c) = guard.take() {
            let _ = c.process.start_kill();
        }
    }
    let mut child = backend::spawn_uvicorn(&runtime, &state.backend_dir, state.dev_profile)?;
    backend::wait_for_ready(&state.base_url, &mut child, BACKEND_START_TIMEOUT).await?;
    let mut guard = state.child.lock().await;
    *guard = Some(child);
    Ok(())
}

// Devtools are opened via the F12 key in the WebView.
// Tauri's high-level API removed the toggle; keep the command as a stub
// so the frontend button still resolves.
#[tauri::command]
pub fn open_devtools(_window: WebviewWindow) {
}

#[tauri::command]
pub fn open_folder(path: String) -> Result<(), String> {
    let folder = PathBuf::from(&path);
    if !folder.exists() {
        return Err(format!("Path does not exist: {path}"));
    }
    open::that(&folder).map_err(|e| e.to_string())
}

// BackendStatus is re-exported for any caller that needs the raw type.
pub use backend::BackendStatus as _BackendStatus;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_root_comes_from_backend_dir() {
        assert_eq!(
            repo_root_from_backend_dir(&PathBuf::from("C:/repo/backend")),
            PathBuf::from("C:/repo"),
        );
    }
}
