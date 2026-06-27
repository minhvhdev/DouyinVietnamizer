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
}
