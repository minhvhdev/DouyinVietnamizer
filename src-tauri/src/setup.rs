use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

pub static SETUP_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, serde::Serialize)]
pub struct SetupProgress {
    pub stage: String,
    pub pct: u8,
}

#[derive(Debug, Error, serde::Serialize)]
pub enum SetupError {
    #[error("uv is not installed; see https://docs.astral.sh/uv/")]
    UvNotInstalled,
    #[error("python install failed: {0}")]
    PythonInstallFailed(String),
    #[error("uv sync failed: {0}")]
    SyncFailed(String),
}

/// Run the first-time setup. Streams `SetupProgress` via the callback as each stage
/// advances. Idempotent: re-running after a partial completion is safe.
pub async fn run_first_time_setup<F: FnMut(SetupProgress)>(
    backend_dir: &Path,
    mut on_progress: F,
) -> Result<(), SetupError> {
    if SETUP_IN_PROGRESS.swap(true, Ordering::SeqCst) {
        return Err(SetupError::SyncFailed("setup already in progress".into()));
    }
    let result = run_inner(backend_dir, &mut on_progress).await;
    SETUP_IN_PROGRESS.store(false, Ordering::SeqCst);
    result
}

async fn run_inner<F: FnMut(SetupProgress)>(
    backend_dir: &Path,
    on_progress: &mut F,
) -> Result<(), SetupError> {
    on_progress(SetupProgress { stage: "python".into(), pct: 0 });

    let mut py_install = tokio::process::Command::new("uv")
        .args(["python", "install", "3.12"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|_| SetupError::UvNotInstalled)?;
    let py_status = py_install.wait().await
        .map_err(|e| SetupError::PythonInstallFailed(e.to_string()))?;
    if !py_status.success() {
        return Err(SetupError::PythonInstallFailed(format!("exit {:?}", py_status.code())));
    }
    on_progress(SetupProgress { stage: "python".into(), pct: 50 });

    on_progress(SetupProgress { stage: "sync".into(), pct: 50 });
    let mut sync = tokio::process::Command::new("uv")
        .args(["sync", "--group", "dev"])
        .current_dir(backend_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| SetupError::SyncFailed(e.to_string()))?;
    let sync_status = sync.wait().await
        .map_err(|e| SetupError::SyncFailed(e.to_string()))?;
    if !sync_status.success() {
        return Err(SetupError::SyncFailed(format!("exit {:?}", sync_status.code())));
    }
    on_progress(SetupProgress { stage: "sync".into(), pct: 100 });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn setup_progress_serializes_to_expected_shape() {
        let p = SetupProgress { stage: "sync".into(), pct: 50 };
        let s = serde_json::to_string(&p).unwrap();
        assert!(s.contains("\"stage\":\"sync\""));
        assert!(s.contains("\"pct\":50"));
    }

    #[tokio::test]
    async fn setup_in_progress_guard_rejects_concurrent_calls() {
        SETUP_IN_PROGRESS.store(false, Ordering::SeqCst);
        let dir = std::env::temp_dir();
        let progress = Mutex::new(Vec::new());
        let cb = |p: SetupProgress| progress.lock().unwrap().push(p);

        SETUP_IN_PROGRESS.store(true, Ordering::SeqCst);
        let r = run_first_time_setup(&dir, cb).await;
        assert!(matches!(r, Err(SetupError::SyncFailed(_))));
        SETUP_IN_PROGRESS.store(false, Ordering::SeqCst);
    }
}
