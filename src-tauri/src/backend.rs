use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};

use crate::portable::{prepend_path, PortableRuntime};

/// Port the backend listens on. Duplicated from `state::BackendState::base_url` to avoid
/// pulling tokio sync types into a pure parser.
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

/// Parse `netstat -ano -p TCP` output and return PIDs holding `LISTENING` sockets on `port`.
/// Pure function; no IO. Excludes other ports (`87650` must not match `8765`) and non-LISTENING
/// states.
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

/// On Windows, kill any leftover process bound to `port`. Best-effort: returns the number
/// of processes successfully terminated. `Err` only if `netstat` itself cannot be invoked.
#[cfg(windows)]
fn kill_port_listeners_windows(port: u16) -> std::io::Result<usize> {
    use std::process::Command;
    let out = Command::new("netstat").args(["-ano", "-p", "TCP"]).output()?;
    let pids = parse_listening_pids(&String::from_utf8_lossy(&out.stdout), port);
    let mut killed = 0usize;
    for pid in pids {
        let status = Command::new("taskkill")
            .args(["/F", "/PID", &pid.to_string()])
            .status()?;
        if status.success() {
            killed += 1;
        }
    }
    Ok(killed)
}

/// On macOS, kill any leftover process bound to `port`. Best-effort: returns the
/// number of processes successfully terminated. `Err` only if `lsof` itself
/// cannot be invoked.
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
    // vendor resolver joins DV_VENDOR_DIR + manifest executable path.
    // Portable manifests use paths like "ffmpeg/ffmpeg.exe", while binaries live
    // under runtime/tools/{ffmpeg,yt-dlp}, so DV_VENDOR_DIR must be tools_dir.
    envs.insert("DV_VENDOR_DIR", runtime.tools_dir.as_os_str().to_os_string());
    envs.insert("DV_VENDOR_MANIFEST", runtime.root.join("manifest.json").as_os_str().to_os_string());
    envs.insert("DV_MODELS_DIR", runtime.models_dir.as_os_str().to_os_string());
    let voxcpm_cli_name = if cfg!(windows) { "voxcpm2-cli.exe" } else { "voxcpm2-cli" };
    let voxcpm_cli = runtime.tools_dir.join("voxcpm2").join(voxcpm_cli_name);
    if voxcpm_cli.is_file() {
        envs.insert("DV_VOXCPM_CLI", voxcpm_cli.as_os_str().to_os_string());
    }
    let tts_server_name = if cfg!(windows) { "llama-tts-server.exe" } else { "llama-tts-server" };
    let tts_server = runtime.tools_dir.join("voxcpm2").join(tts_server_name);
    if tts_server.is_file() {
        envs.insert("DV_VOXCPM_TTS_SERVER", tts_server.as_os_str().to_os_string());
    }
    envs.insert("DV_ALLOW_PATH_TOOLS", OsString::from("0"));
    envs.insert("PATH", prepend_path(&runtime.tools_dir, std::env::var_os("PATH")));
    envs
}

/// Spawn `python -m dv_backend.main` using the bundled portable Python runtime.
/// When `dev_profile` is true, sets `DV_RELOAD=1` so uvicorn watches source files.
/// On Windows, kills any leftover process bound to `BACKEND_PORT` first so a previous
/// session's uvicorn doesn't block the bind.
pub fn spawn_uvicorn(
    runtime: &PortableRuntime,
    source_backend_dir: &Path,
    dev_profile: bool,
) -> Result<ManagedChild, BackendStartError> {
    use std::process::Stdio;
    #[cfg(windows)]
    {
        let _ = kill_port_listeners_windows(BACKEND_PORT);
    }
    #[cfg(target_os = "macos")]
    {
        let _ = kill_port_listeners_macos(BACKEND_PORT);
    }
    let mut cmd = tokio::process::Command::new(&runtime.python);
    cmd.args(["-m", "dv_backend.main"])
        .current_dir(backend_working_dir(runtime, source_backend_dir, dev_profile))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        // Hide the uvicorn console window. Without this, Rust spawns a visible
        // `python.exe` console that the user can close by accident and silently
        // kill the backend.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    for (key, value) in build_backend_env(runtime, dev_profile) {
        cmd.env(key, value);
    }
    let child = cmd
        .spawn()
        .map_err(|e| BackendStartError::Spawn(format!("{} using runtime {}", e, runtime.root.display())))?;
    Ok(ManagedChild::new(child))
}

/// Poll `GET {base_url}/health` every 100ms up to `timeout`. On timeout, kill child.
/// If child exits during polling, drain stderr and return Crashed.
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
            Ok(None) => { /* still running */ }
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
        Ok(client) => client.get(&url).send().await
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
        assert_eq!(
            envs.get("DV_VENDOR_DIR").unwrap(),
            &root.join("tools").as_os_str().to_os_string()
        );
        assert_eq!(envs.get("DV_RELOAD").unwrap(), "1");
        assert!(envs.get("PATH").unwrap().to_string_lossy().contains("tools"));
    }

    #[test]
    fn build_backend_command_env_points_voxcpm_cli_at_portable_tools() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let voxcpm_tools = root.join("tools").join("voxcpm2");
        std::fs::create_dir_all(&voxcpm_tools).unwrap();
        let cli_name = if cfg!(windows) { "voxcpm2-cli.exe" } else { "voxcpm2-cli" };
        let server_name = if cfg!(windows) { "llama-tts-server.exe" } else { "llama-tts-server" };
        std::fs::write(voxcpm_tools.join(cli_name), b"x").unwrap();
        std::fs::write(voxcpm_tools.join(server_name), b"x").unwrap();
        let runtime = runtime(root.clone());
        let envs = build_backend_env(&runtime, true);
        assert_eq!(
            envs.get("DV_VOXCPM_CLI").unwrap(),
            &voxcpm_tools.join(cli_name).as_os_str().to_os_string(),
        );
        assert_eq!(
            envs.get("DV_VOXCPM_TTS_SERVER").unwrap(),
            &voxcpm_tools.join(server_name).as_os_str().to_os_string(),
        );
        assert!(!envs.contains_key("DV_VOXCPM_VENV"));
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

    #[test]
    fn parse_listening_pids_filters_by_port_and_state() {
        // 8765 LISTENING -> kept; 87650 LISTENING -> not matched (different port);
        // 8765 TIME_WAIT -> dropped; dupes -> deduped.
        let sample = "\
  TCP    127.0.0.1:8765         0.0.0.0:0              LISTENING       111
  TCP    127.0.0.1:87650        0.0.0.0:0              LISTENING       222
  TCP    127.0.0.1:8765         127.0.0.1:55555        TIME_WAIT       0
  TCP    0.0.0.0:8765           0.0.0.0:0              LISTENING       111
";
        assert_eq!(parse_listening_pids(sample, 8765), vec![111]);
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

    #[cfg(target_os = "macos")]
    fn parse_lsof_pids(text: &str) -> Vec<u32> {
        text.lines()
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn parse_lsof_pids_extracts_unique_pids() {
        let sample = "111\n222\n111\nabc\n";
        let mut got = parse_lsof_pids(sample);
        got.sort_unstable();
        got.dedup();
        assert_eq!(got, vec![111, 222]);
    }
}
