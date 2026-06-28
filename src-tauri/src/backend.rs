use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

use thiserror::Error;

use crate::portable::{prepend_path, PortableRuntime};

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

#[derive(Debug, Error, serde::Serialize)]
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
pub enum BackendStatusKind {
    Starting,
    Ready,
    Crashed { stderr: String },
    AlreadyRunning,
}

// Backward-compatible alias used by callers that expected the prior internal name.
pub type _BackendStatus = BackendStatus;

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
    envs.insert("DV_VENDOR_DIR", runtime.root.as_os_str().to_os_string());
    envs.insert("DV_VENDOR_MANIFEST", runtime.root.join("manifest.json").as_os_str().to_os_string());
    envs.insert("DV_MODELS_DIR", runtime.models_dir.as_os_str().to_os_string());
    envs.insert("DV_VOXCPM_VENV", runtime.backend_dir.join("dv_backend").join(".venv-voxcpm").as_os_str().to_os_string());
    envs.insert("DV_ALLOW_PATH_TOOLS", OsString::from("0"));
    envs.insert("PATH", prepend_path(&runtime.tools_dir, std::env::var_os("PATH")));
    envs
}

/// Spawn `python -m dv_backend.main` using the bundled portable Python runtime.
/// When `dev_profile` is true, sets `DV_RELOAD=1` so uvicorn watches source files.
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

/// Poll `GET {base_url}/health` every 100ms up to `timeout`. On timeout, kill child.
/// If child exits during polling, drain stderr and return Crashed.
pub async fn wait_for_ready(
    base_url: &str,
    child: &mut tokio::process::Child,
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
                let stderr = drain_stderr(child).await;
                return Err(BackendStartError::Crashed {
                    code: status.code(),
                    stderr: parse_uvicorn_stderr(&stderr),
                });
            }
            Ok(None) => { /* still running */ }
            Err(e) => return Err(BackendStartError::Spawn(e.to_string())),
        }
        if start.elapsed() >= timeout {
            let _ = child.start_kill();
            return Err(BackendStartError::Timeout(timeout));
        }
        tokio::time::sleep(poll_interval).await;
    }
}

pub async fn is_health_ok(base_url: &str) -> bool {
    let url = format!("{}/api/health", base_url.trim_end_matches('/'));
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

async fn drain_stderr(child: &mut tokio::process::Child) -> String {
    use tokio::io::AsyncReadExt;
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

    fn runtime(root: PathBuf) -> PortableRuntime {
        PortableRuntime {
            python: root.join(".venv/Scripts/python.exe"),
            backend_dir: root.join("backend"),
            tools_dir: root.join("tools"),
            models_dir: root.join("models"),
            root,
        }
    }

    #[test]
    fn build_backend_command_env_uses_portable_runtime() {
        let root = PathBuf::from("C:/rt");
        let runtime = runtime(root.clone());
        let envs = build_backend_env(&runtime, true);
        assert_eq!(envs.get("DV_PORTABLE_RUNTIME_DIR").unwrap(), &root.as_os_str().to_os_string());
        assert_eq!(envs.get("DV_RELOAD").unwrap(), "1");
        assert!(envs.get("PATH").unwrap().to_string_lossy().contains("tools"));
    }

    #[test]
    fn build_backend_command_env_uses_portable_voxcpm_venv() {
        let root = PathBuf::from("C:/rt");
        let runtime = runtime(root);
        let envs = build_backend_env(&runtime, true);
        assert_eq!(
            envs.get("DV_VOXCPM_VENV").unwrap(),
            &PathBuf::from("C:/rt").join("backend").join("dv_backend").join(".venv-voxcpm").as_os_str().to_os_string(),
        );
    }

    #[test]
    fn backend_working_dir_uses_source_in_dev_and_packaged_in_release() {
        let root = PathBuf::from("C:/rt");
        let runtime = runtime(root.clone());
        assert_eq!(backend_working_dir(&runtime, Path::new("C:/repo/backend"), true), PathBuf::from("C:/repo/backend"));
        assert_eq!(backend_working_dir(&runtime, Path::new("C:/repo/backend"), false), root.join("backend"));
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
    async fn is_health_ok_checks_api_health_route() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            if let Ok((mut s, _)) = listener.accept().await {
                let mut req = [0u8; 1024];
                let n = s.read(&mut req).await.unwrap_or(0);
                let req = String::from_utf8_lossy(&req[..n]);
                let status = if req.starts_with("GET /api/health ") { "200 OK" } else { "404 Not Found" };
                let _ = s.write_all(format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\n\r\n").as_bytes()).await;
                let _ = s.shutdown().await;
            }
        });
        assert!(is_health_ok(&format!("http://127.0.0.1:{}", port)).await);
    }
}
