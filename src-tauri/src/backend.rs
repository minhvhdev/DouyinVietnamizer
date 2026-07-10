use std::collections::HashMap;
use std::ffi::OsString;
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};

use crate::runtime::{prepend_path, AppRuntime};

const BACKEND_PORT: u16 = 8765;
const MAX_STDERR_TAIL_BYTES: usize = 16 * 1024;

#[derive(Debug)]
pub struct ManagedChild {
    pub process: tokio::process::Child,
    stderr_tail: Arc<StdMutex<String>>,
}

impl ManagedChild {
    fn new(mut process: tokio::process::Child) -> Self {
        let stderr_tail = Arc::new(StdMutex::new(String::new()));

        if let Some(stdout) = process.stdout.take() {
            tokio::spawn(discard_stream("stdout", stdout));
        }
        if let Some(stderr) = process.stderr.take() {
            tokio::spawn(capture_stderr(stderr, stderr_tail.clone()));
        }

        Self { process, stderr_tail }
    }

    fn stderr_tail(&self) -> String {
        self.stderr_tail
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

async fn discard_stream<R>(stream_name: &'static str, reader: R)
where
    R: AsyncRead + Unpin + Send + 'static,
{
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break,
            Ok(_) => {}
            Err(error) => {
                log::debug!("backend {stream_name} drain stopped: {error}");
                break;
            }
        }
    }
}

async fn capture_stderr<R>(reader: R, stderr_tail: Arc<StdMutex<String>>)
where
    R: AsyncRead + Unpin + Send + 'static,
{
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break,
            Ok(_) => {
                let mut tail = stderr_tail
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                tail.push_str(&line);
                trim_to_last_bytes(&mut tail, MAX_STDERR_TAIL_BYTES);
            }
            Err(error) => {
                let mut tail = stderr_tail
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                tail.push_str(&format!("[stderr drain error] {error}\n"));
                trim_to_last_bytes(&mut tail, MAX_STDERR_TAIL_BYTES);
                break;
            }
        }
    }
}

fn trim_to_last_bytes(text: &mut String, max_bytes: usize) {
    if text.len() <= max_bytes {
        return;
    }
    let trim_from = text.len() - max_bytes;
    let boundary = text
        .char_indices()
        .find(|(idx, _)| *idx >= trim_from)
        .map(|(idx, _)| idx)
        .unwrap_or(trim_from);
    text.drain(..boundary);
}

fn parse_listening_pids(text: &str, port: u16) -> Vec<u32> {
    let needle = format!(":{} ", port);
    let mut pids: Vec<u32> = text
        .lines()
        .filter(|l| l.contains("LISTENING"))
        .filter(|l| l.contains(&needle))
        .filter_map(|l| l.split_whitespace().last())
        .filter_map(|s| s.parse().ok())
        .collect();
    pids.sort_unstable();
    pids.dedup();
    pids
}

#[cfg(windows)]
fn kill_port_listeners_windows(port: u16) -> std::io::Result<usize> {
    use std::process::Command;
    use std::thread;
    use std::time::Duration;

    let mut killed = 0usize;
    for _ in 0..5 {
        let out = Command::new("netstat").args(["-ano", "-p", "TCP"]).output()?;
        let pids = parse_listening_pids(&String::from_utf8_lossy(&out.stdout), port);
        if pids.is_empty() {
            break;
        }
        for pid in pids {
            let status = Command::new("taskkill")
                .args(["/F", "/T", "/PID", &pid.to_string()])
                .status()?;
            if status.success() {
                killed += 1;
            }
        }
        thread::sleep(Duration::from_millis(300));
    }
    Ok(killed)
}

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

pub fn parse_uvicorn_stderr(s: &str) -> String {
    const MAX: usize = 4096;
    let trimmed = s.trim();
    if trimmed.len() <= MAX {
        trimmed.to_string()
    } else {
        let start = trimmed.len() - MAX;
        let mut idx = start;
        while !trimmed.is_char_boundary(idx) {
            idx += 1;
        }
        format!("...{}\n[truncated]", &trimmed[idx..])
    }
}

#[derive(Debug, Error, serde::Serialize)]
pub enum BackendStartError {
    #[error("failed to spawn backend: {0}")]
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

pub type _BackendStatus = BackendStatus;

pub fn build_backend_env(runtime: &AppRuntime, dev_profile: bool) -> HashMap<&'static str, OsString> {
    let mut envs = HashMap::new();
    let _ = dev_profile;
    // Keep a single managed backend process when launched by Tauri.
    // Enabling uvicorn reload here can spawn orphan watcher/worker pairs
    // that keep port 8765 occupied and cause stale code to serve requests.
    envs.insert("DV_RELOAD", OsString::from("0"));
    envs.insert("DV_VENDOR_DIR", runtime.vendor_dir.as_os_str().to_os_string());
    envs.insert(
        "DV_VENDOR_MANIFEST",
        runtime.vendor_dir.join("manifest.json").as_os_str().to_os_string(),
    );
    envs.insert("DV_MODELS_DIR", runtime.models_dir.as_os_str().to_os_string());
    let omnivoice_venv = runtime.backend_dir.join("venvs").join("omnivoice");
    if omnivoice_venv.is_dir() {
        envs.insert("DV_OMNIVOICE_VENV", omnivoice_venv.as_os_str().to_os_string());
    }
    envs.insert("DV_ALLOW_PATH_TOOLS", OsString::from("1"));
    envs.insert("PATH", prepend_path(&runtime.vendor_dir, std::env::var_os("PATH")));
    envs
}

pub fn spawn_backend(
    runtime: &AppRuntime,
    dev_profile: bool,
    clear_port: bool,
) -> Result<ManagedChild, BackendStartError> {
    use std::process::Stdio;
    if clear_port {
        #[cfg(windows)]
        {
            let _ = kill_port_listeners_windows(BACKEND_PORT);
        }
        #[cfg(target_os = "macos")]
        {
            let _ = kill_port_listeners_macos(BACKEND_PORT);
        }
    }
    let mut cmd = if runtime.use_uv {
        let mut command = tokio::process::Command::new("uv");
        command.args(["run", "python", "-m", "dv_backend.main"]);
        command
    } else {
        let mut command = tokio::process::Command::new(&runtime.python);
        command.args(["-m", "dv_backend.main"]);
        command
    };
    cmd.current_dir(&runtime.backend_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    for (key, value) in build_backend_env(runtime, dev_profile) {
        cmd.env(key, value);
    }
    let child = cmd.spawn().map_err(|e| {
        BackendStartError::Spawn(format!(
            "{} using backend {}",
            e,
            runtime.backend_dir.display()
        ))
    })?;
    Ok(ManagedChild::new(child))
}

pub async fn wait_for_ready(
    base_url: &str,
    child: &mut ManagedChild,
    timeout: Duration,
) -> Result<(), BackendStartError> {
    let poll_interval = Duration::from_millis(100);
    let start = std::time::Instant::now();
    loop {
        if is_health_ok(base_url).await {
            return Ok(());
        }
        match child.process.try_wait() {
            Ok(Some(status)) => {
                return Err(BackendStartError::Crashed {
                    code: status.code(),
                    stderr: parse_uvicorn_stderr(&child.stderr_tail()),
                });
            }
            Ok(None) => {}
            Err(e) => return Err(BackendStartError::Spawn(e.to_string())),
        }
        if start.elapsed() >= timeout {
            let _ = child.process.start_kill();
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
        Ok(client) => client
            .get(&url)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false),
        Err(_) => false,
    }
}

pub async fn request_release_vram(base_url: &str) -> Result<(), String> {
    let url = format!("{}/api/runtime/release-vram", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(4))
        .build()
        .map_err(|error| error.to_string())?;
    let response = client
        .post(&url)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if response.status().is_success() {
        return Ok(());
    }
    let detail = response
        .text()
        .await
        .unwrap_or_else(|_| "failed to read backend response".into());
    Err(detail)
}

pub fn buffered_stderr(child: &ManagedChild) -> String {
    child.stderr_tail()
}

pub fn spawn_uvicorn(
    runtime: &AppRuntime,
    _source_backend_dir: &Path,
    dev_profile: bool,
    clear_port: bool,
) -> Result<ManagedChild, BackendStartError> {
    spawn_backend(runtime, dev_profile, clear_port)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::python_executable;

    fn runtime(root: std::path::PathBuf) -> AppRuntime {
        let backend_dir = root.join("backend");
        AppRuntime {
            repo_root: root.clone(),
            backend_dir: backend_dir.clone(),
            vendor_dir: root.join("vendor"),
            models_dir: backend_dir.join("models"),
            python: python_executable(&backend_dir),
            use_uv: false,
        }
    }

    #[test]
    fn build_backend_command_env_uses_repo_layout() {
        let root = std::path::PathBuf::from("C:/repo");
        let envs = build_backend_env(&runtime(root.clone()), true);
        assert_eq!(
            envs.get("DV_VENDOR_DIR").unwrap(),
            &root.join("vendor").as_os_str().to_os_string()
        );
        assert_eq!(envs.get("DV_RELOAD").unwrap(), "1");
    }

    #[test]
    fn parse_listening_pids_filters_by_port_and_state() {
        let sample = "\
  TCP    127.0.0.1:8765         0.0.0.0:0              LISTENING       111
  TCP    127.0.0.1:87650        0.0.0.0:0              LISTENING       222
";
        assert_eq!(parse_listening_pids(sample, 8765), vec![111]);
    }
}
