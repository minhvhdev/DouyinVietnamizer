use std::time::Duration;
use serde::Serialize;
use tauri::{AppHandle, Emitter, State, WebviewWindow};

use crate::backend::{self, BackendStartError, BackendStatus, BackendStatusKind, VenvStatus};
use crate::setup::{self, SetupError, SetupProgress};
use crate::state::BackendState;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackendStatusDto {
    SetupRequired { stage: String },
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

#[tauri::command]
pub async fn get_backend_status(
    state: State<'_, BackendState>,
) -> Result<BackendStatusDto, String> {
    let venv = backend::detect_venv(&state.backend_dir);
    if !venv.is_ready() {
        let stage = match venv {
            VenvStatus::MissingUv => "missing_uv",
            VenvStatus::MissingPython => "missing_python",
            VenvStatus::MissingVenv => "missing_venv",
            VenvStatus::Ready(_) => unreachable!(),
        };
        return Ok(BackendStatusDto::SetupRequired { stage: stage.into() });
    }
    let mut guard = state.child.lock().await;
    if let Some(child) = guard.as_mut() {
        match child.try_wait() {
            Ok(Some(_)) => {
                let stderr = backend::parse_uvicorn_stderr("");
                *guard = None;
                return Ok(BackendStatusDto::Crashed { stderr });
            }
            Ok(None) => return Ok(BackendStatusDto::Starting),
            Err(e) => return Err(e.to_string()),
        }
    }
    let mut child = backend::spawn_uvicorn(&state.backend_dir, state.dev_profile)
        .map_err(|e| e.to_string())?;
    let base_url = state.base_url.clone();
    match backend::wait_for_ready(&base_url, &mut child, Duration::from_secs(5)).await {
        Ok(()) => {
            *guard = Some(child);
            Ok(BackendStatusDto::Ready { base_url })
        }
        Err(e) => {
            let _ = child.kill();
            let stderr = match &e {
                BackendStartError::Crashed { stderr, .. } => stderr.clone(),
                BackendStartError::Timeout(_) => "backend did not respond within 5s".into(),
                BackendStartError::Spawn(s) => s.clone(),
            };
            Ok(BackendStatusDto::Crashed { stderr })
        }
    }
}

#[tauri::command]
pub async fn run_first_time_setup_cmd(
    state: State<'_, BackendState>,
    app: AppHandle,
) -> Result<(), SetupError> {
    let app2 = app.clone();
    let on_progress = move |p: SetupProgress| {
        let _ = app2.emit("setup://progress", p);
    };
    let result = setup::run_first_time_setup(&state.backend_dir, on_progress).await;
    // After setup, reset child slot so the next get_backend_status call respawns.
    if result.is_ok() {
        let mut guard = state.child.lock().await;
        *guard = None;
    }
    result
}

#[tauri::command]
pub async fn restart_backend(
    state: State<'_, BackendState>,
) -> Result<(), BackendStartError> {
    {
        let mut guard = state.child.lock().await;
        if let Some(mut c) = guard.take() {
            let _ = c.start_kill();
        }
    }
    let mut child = backend::spawn_uvicorn(&state.backend_dir, state.dev_profile)?;
    backend::wait_for_ready(&state.base_url, &mut child, Duration::from_secs(5)).await?;
    let mut guard = state.child.lock().await;
    *guard = Some(child);
    Ok(())
}

#[tauri::command]
pub fn open_devtools(window: WebviewWindow) {
    if window.is_devtools_open() {
        window.close_devtools();
    } else {
        window.open_devtools();
    }
}

// BackendStatus is re-exported for any caller that needs the raw type.
pub use backend::BackendStatus as _BackendStatus;
