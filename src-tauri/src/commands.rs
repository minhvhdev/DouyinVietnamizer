use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Duration;
use serde::Serialize;
use tauri::{State, WebviewWindow};

use crate::backend::{self, BackendStartError, BackendStatusKind};
use crate::runtime::{self, AppRuntime, AppRuntimeStatus};
use crate::setup::SetupError;
use crate::state::BackendState;

const BACKEND_START_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackendStatusDto {
    EnvironmentMissing { root: String, missing_items: Vec<String> },
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

fn resolve_runtime_for_status(backend_dir: &PathBuf) -> Result<AppRuntime, BackendStatusDto> {
    match runtime::resolve_app_runtime(&repo_root_from_backend_dir(backend_dir)) {
        Ok(app_runtime) => Ok(app_runtime),
        Err(AppRuntimeStatus::Missing { root, missing_items }) => {
            Err(BackendStatusDto::EnvironmentMissing {
                root: root.display().to_string(),
                missing_items,
            })
        }
        Err(AppRuntimeStatus::Ready { .. }) => unreachable!(),
    }
}

fn resolve_runtime_for_start(backend_dir: &PathBuf) -> Result<AppRuntime, BackendStartError> {
    resolve_runtime_for_status(backend_dir).map_err(|status| match status {
        BackendStatusDto::EnvironmentMissing { root, missing_items } => BackendStartError::Spawn(format!(
            "development environment incomplete at {}: {}",
            root,
            missing_items.join(", ")
        )),
        _ => BackendStartError::Spawn("development environment unavailable".into()),
    })
}

#[tauri::command]
pub async fn get_backend_status(
    state: State<'_, BackendState>,
) -> Result<BackendStatusDto, String> {
    let app_runtime = match resolve_runtime_for_status(&state.backend_dir) {
        Ok(app_runtime) => app_runtime,
        Err(status) => return Ok(status),
    };
    let base_url = state.base_url.clone();
    let mut guard = state.child.lock().await;
    if let Some(child) = guard.as_mut() {
        match child.process.try_wait() {
            Ok(Some(_status)) => {
                let stderr = backend::parse_uvicorn_stderr(&backend::buffered_stderr(child));
                *guard = None;
                if backend::is_health_ok(&base_url).await {
                    return Ok(BackendStatusDto::Ready { base_url });
                }
                return Ok(BackendStatusDto::Crashed { stderr });
            }
            Ok(None) => {
                return Ok(BackendStatusDto::Ready { base_url });
            }
            Err(e) => return Err(e.to_string()),
        }
    }
    if backend::is_health_ok(&base_url).await {
        return Ok(BackendStatusDto::Ready { base_url });
    }
    let mut child = backend::spawn_backend(&app_runtime, state.dev_profile, true)
        .map_err(|e| e.to_string())?;
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
                    "backend did not respond within {}s; check backend/.venv and vendor tools",
                    BACKEND_START_TIMEOUT.as_secs()
                ),
                BackendStartError::Spawn(s) => s.clone(),
            };
            Ok(BackendStatusDto::Crashed { stderr })
        }
    }
}

#[tauri::command]
pub async fn run_first_time_setup_cmd(state: State<'_, BackendState>) -> Result<(), SetupError> {
    crate::setup::run_first_time_setup(&state.backend_dir, |_| {}).await
}

#[tauri::command]
pub async fn restart_backend(
    state: State<'_, BackendState>,
) -> Result<(), BackendStartError> {
    state.recovering.store(true, Ordering::SeqCst);
    let result = async {
        let app_runtime = resolve_runtime_for_start(&state.backend_dir)?;
        {
            let mut guard = state.child.lock().await;
            if let Some(mut c) = guard.take() {
                let _ = c.process.start_kill();
            }
        }
        let mut child = backend::spawn_backend(&app_runtime, state.dev_profile, true)?;
        backend::wait_for_ready(&state.base_url, &mut child, BACKEND_START_TIMEOUT).await?;
        let mut guard = state.child.lock().await;
        *guard = Some(child);
        Ok(())
    }
    .await;
    state.recovering.store(false, Ordering::SeqCst);
    result
}

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

pub use backend::BackendStatus as _BackendStatus;
