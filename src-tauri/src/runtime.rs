use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppRuntime {
    pub repo_root: PathBuf,
    pub backend_dir: PathBuf,
    pub vendor_dir: PathBuf,
    pub models_dir: PathBuf,
    pub python: PathBuf,
    pub use_uv: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AppRuntimeStatus {
    Ready { root: PathBuf },
    Missing { root: PathBuf, missing_items: Vec<String> },
}

pub fn resolve_app_runtime(repo_root: &Path) -> Result<AppRuntime, AppRuntimeStatus> {
    let repo_root = repo_root.to_path_buf();
    let backend_dir = repo_root.join("backend");
    let vendor_dir = repo_root.join("vendor");
    let models_dir = backend_dir.join("models");
    let python = python_executable(&backend_dir);
    // Prefer project-local backend/.venv when it exists so runtime behavior
    // matches the environment used during local backend testing.
    let use_uv = !python.is_file() && uv_available();

    let mut missing_items = Vec::new();
    if !backend_dir.join("dv_backend").is_dir() {
        missing_items.push(format!(
            "backend/dv_backend ({})",
            backend_dir.join("dv_backend").display()
        ));
    }
    if !python.is_file() && !use_uv {
        missing_items.push(format!(
            "backend virtualenv python ({}) or uv on PATH",
            python.display()
        ));
    }
    if !vendor_dir.join("manifest.json").is_file() {
        missing_items.push(format!(
            "vendor/manifest.json ({})",
            vendor_dir.join("manifest.json").display()
        ));
    }

    if missing_items.is_empty() {
        Ok(AppRuntime {
            repo_root,
            backend_dir,
            vendor_dir,
            models_dir,
            python,
            use_uv,
        })
    } else {
        Err(AppRuntimeStatus::Missing {
            root: repo_root,
            missing_items,
        })
    }
}

pub fn python_executable(backend_dir: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        backend_dir.join(".venv").join("Scripts").join("python.exe")
    }
    #[cfg(not(windows))]
    {
        backend_dir.join(".venv").join("bin").join("python")
    }
}

pub fn uv_available() -> bool {
    std::process::Command::new("uv")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

pub fn prepend_path(dir: &Path, current: Option<OsString>) -> OsString {
    let mut paths = vec![dir.to_path_buf()];
    if let Some(current) = current {
        paths.extend(env::split_paths(&current));
    }
    env::join_paths(paths).unwrap_or_else(|_| OsString::from(dir.as_os_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn make_backend(root: &Path) {
        let backend = root.join("backend");
        let python = python_executable(&backend);
        fs::create_dir_all(python.parent().unwrap()).unwrap();
        fs::write(python, b"").unwrap();
        fs::create_dir_all(backend.join("dv_backend")).unwrap();
        fs::create_dir_all(root.join("vendor")).unwrap();
        fs::write(root.join("vendor/manifest.json"), b"{}").unwrap();
    }

    #[test]
    fn validate_accepts_complete_dev_layout() {
        let dir = tempdir().unwrap();
        make_backend(dir.path());
        let runtime = resolve_app_runtime(dir.path()).unwrap();
        assert_eq!(runtime.backend_dir, dir.path().join("backend"));
        assert_eq!(runtime.vendor_dir, dir.path().join("vendor"));
    }

    #[test]
    fn validate_lists_missing_items() {
        let dir = tempdir().unwrap();
        let err = resolve_app_runtime(dir.path()).unwrap_err();
        match err {
            AppRuntimeStatus::Missing { missing_items, .. } => {
                assert!(missing_items.iter().any(|item| item.contains("backend/dv_backend")));
                assert!(missing_items.iter().any(|item| item.contains("manifest.json")));
            }
            AppRuntimeStatus::Ready { .. } => panic!("expected missing runtime"),
        }
    }

    #[test]
    fn prepends_tool_path() {
        let dir = tempdir().unwrap();
        let tools = dir.path().join("vendor");
        let system = dir.path().join("system");
        let current = env::join_paths([system.clone()]).unwrap();
        let got = prepend_path(&tools, Some(current));
        let parts = env::split_paths(&got).collect::<Vec<_>>();
        assert_eq!(parts[0], tools);
        assert_eq!(parts[1], system);
    }
}
