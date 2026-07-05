use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::backend::{buffered_stderr, parse_uvicorn_stderr};
use crate::state::BackendState;

const WATCH_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Serialize)]
pub struct BackendCrashedPayload {
    pub stderr: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<i32>,
}

/// Poll the managed uvicorn child. When it exits unexpectedly, emit `backend://crashed`
/// so the UI can restart the backend and resume interrupted jobs.
pub async fn run(app: AppHandle) {
    let mut interval = tokio::time::interval(WATCH_INTERVAL);
    loop {
        interval.tick().await;
        let Some(state) = app.try_state::<BackendState>() else {
            continue;
        };
        let exit = {
            let mut guard = state.child.lock().await;
            let Some(child) = guard.as_mut() else {
                continue;
            };
            match child.process.try_wait() {
                Ok(Some(status)) => {
                    let stderr = parse_uvicorn_stderr(&buffered_stderr(child));
                    *guard = None;
                    Some(BackendCrashedPayload {
                        stderr,
                        code: status.code(),
                    })
                }
                Ok(None) => None,
                Err(error) => {
                    log::warn!("backend watchdog try_wait failed: {error}");
                    None
                }
            }
        };
        if let Some(payload) = exit {
            log::error!(
                "backend process exited unexpectedly (code={:?})",
                payload.code
            );
            let _ = app.emit("backend://crashed", payload);
        }
    }
}
