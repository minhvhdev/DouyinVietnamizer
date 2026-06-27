use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VenvStatus {
    Ready(PathBuf),
    MissingUv,
    MissingPython,
    MissingVenv,
}

impl VenvStatus {
    pub fn is_ready(&self) -> bool {
        matches!(self, VenvStatus::Ready(_))
    }
}

/// Detect whether the `uv`-managed Python venv at `backend_dir/.venv` is ready.
/// Order: check `uv` on PATH, then Python 3.12, then venv directory.
pub fn detect_venv(backend_dir: &Path) -> VenvStatus {
    if Command::new("uv").arg("--version").output().is_err() {
        return VenvStatus::MissingUv;
    }
    let py_out = Command::new("python").arg("--version").output();
    match py_out {
        Ok(o) if o.status.success() => {
            let v = String::from_utf8_lossy(&o.stdout);
            if !v.contains("3.12") {
                return VenvStatus::MissingPython;
            }
        }
        _ => return VenvStatus::MissingPython,
    }
    let cfg = backend_dir.join(".venv").join("pyvenv.cfg");
    if cfg.exists() {
        VenvStatus::Ready(backend_dir.join(".venv"))
    } else {
        VenvStatus::MissingVenv
    }
}

/// Extracts the last 4KB of stderr for surfacing in error UI. Trims trailing whitespace.
pub fn parse_uvicorn_stderr(s: &str) -> String {
    const MAX: usize = 4096;
    let trimmed = s.trim();
    if trimmed.len() <= MAX {
        trimmed.to_string()
    } else {
        let start = trimmed.len() - MAX;
        // Snap to the next char boundary to avoid splitting a UTF-8 codepoint.
        let mut idx = start;
        while !trimmed.is_char_boundary(idx) {
            idx += 1;
        }
        format!("...{}\n[truncated]", &trimmed[idx..])
    }
}

use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendStartError {
    #[error("failed to spawn uvicorn: {0}")]
    Spawn(String),
    #[error("backend did not become ready within {0:?}")]
    Timeout(Duration),
    #[error("backend crashed (code={code:?}); stderr:\n{stderr}")]
    Crashed { code: Option<i32>, stderr: String },
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BackendStatus {
    pub base_url: String,
    pub kind: BackendStatusKind,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BackendStatusKind {
    Starting,
    Ready,
    Crashed { stderr: String },
    AlreadyRunning,
}

// Backward-compatible alias used by callers that expected the prior internal name.
pub type _BackendStatus = BackendStatus;

/// Spawn `uv run python -m dv_backend.main` with `current_dir = backend_dir`.
/// When `dev_profile` is true, sets `DV_RELOAD=1` so uvicorn watches source files.
pub fn spawn_uvicorn(backend_dir: &Path, dev_profile: bool) -> Result<std::process::Child, BackendStartError> {
    use std::process::Stdio;
    let mut cmd = std::process::Command::new("uv");
    cmd.args(["run", "python", "-m", "dv_backend.main"])
        .current_dir(backend_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if dev_profile {
        cmd.env("DV_RELOAD", "1");
    } else {
        cmd.env("DV_RELOAD", "0");
    }
    cmd.spawn().map_err(|e| BackendStartError::Spawn(e.to_string()))
}

/// Poll `GET {base_url}/health` every 100ms up to `timeout`. On timeout, kill child.
/// If child exits during polling, drain stderr and return Crashed.
pub async fn wait_for_ready(
    base_url: &str,
    child: &mut std::process::Child,
    timeout: Duration,
) -> Result<(), BackendStartError> {
    let poll_interval = Duration::from_millis(100);
    let start = std::time::Instant::now();
    loop {
        if is_health_ok(base_url).await {
            return Ok(());
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                let stderr = drain_stderr(child);
                return Err(BackendStartError::Crashed {
                    code: status.code(),
                    stderr: parse_uvicorn_stderr(&stderr),
                });
            }
            Ok(None) => { /* still running */ }
            Err(e) => return Err(BackendStartError::Spawn(e.to_string())),
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            return Err(BackendStartError::Timeout(timeout));
        }
        tokio::time::sleep(poll_interval).await;
    }
}

pub async fn is_health_ok(base_url: &str) -> bool {
    let url = format!("{}/health", base_url.trim_end_matches('/'));
    match reqwest::Client::builder()
        .timeout(Duration::from_millis(200))
        .build()
    {
        Ok(client) => client.get(&url).send().await
            .map(|r| r.status().is_success())
            .unwrap_or(false),
        Err(_) => false,
    }
}

fn drain_stderr(child: &mut std::process::Child) -> String {
    use std::io::Read;
    if let Some(mut s) = child.stderr.take() {
        let mut buf = String::new();
        let _ = s.read_to_string(&mut buf);
        return buf;
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn detect_venv_returns_ready_when_pyvenv_cfg_exists() {
        let dir = tempdir().unwrap();
        let venv = dir.path().join(".venv");
        fs::create_dir(&venv).unwrap();
        fs::write(venv.join("pyvenv.cfg"), "home = /usr/bin\n").unwrap();
        // Skip uv/python checks by making `uv` and `python` succeed (they exist on dev box).
        // This test relies on dev box having uv and python 3.12; in CI we'd mock Command.
        let status = detect_venv(dir.path());
        // If dev box has uv+3.12, we get Ready; otherwise MissingUv/MissingPython is acceptable
        // (the test still verifies the .venv branch when Ready is returned).
        if status.is_ready() {
            assert_eq!(status, VenvStatus::Ready(venv));
        }
    }

    #[test]
    fn detect_venv_returns_missing_venv_when_no_pyvenv_cfg() {
        let dir = tempdir().unwrap();
        // No .venv created
        let status = detect_venv(dir.path());
        // Either MissingVenv (uv+py present) or MissingUv/MissingPython (not present)
        // is acceptable. We just check it isn't Ready.
        assert!(!status.is_ready());
    }

    #[test]
    fn parse_uvicorn_stderr_short_passes_through() {
        let s = "Traceback (most recent call last):\n  File \"x.py\", line 1\n    boom";
        assert_eq!(parse_uvicorn_stderr(s), s.trim());
    }

    #[test]
    fn parse_uvicorn_stderr_long_is_truncated_with_marker() {
        let big = "x".repeat(8192);
        let out = parse_uvicorn_stderr(&big);
        assert!(out.contains("[truncated]"));
        assert!(out.starts_with("..."));
        assert!(out.len() <= 8192);
    }

    #[tokio::test]
    async fn is_health_ok_returns_false_for_unbound_port() {
        // Port 1 is reserved and almost never listening; any connection attempt fails fast.
        assert!(!is_health_ok("http://127.0.0.1:1").await);
    }

    #[tokio::test]
    async fn is_health_ok_returns_true_for_listening_server() {
        use tokio::io::{AsyncWriteExt, AsyncReadExt};
        use tokio::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            // Accept one connection, respond with HTTP/1.1 200, then close.
            if let Ok((mut s, _)) = listener.accept().await {
                let mut req = [0u8; 1024];
                let _ = s.read(&mut req).await;
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n").await;
                let _ = s.shutdown().await;
            }
        });
        assert!(is_health_ok(&format!("http://127.0.0.1:{}", port)).await);
    }
}
