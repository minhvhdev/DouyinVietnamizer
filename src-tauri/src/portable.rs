use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableRuntime {
    pub root: PathBuf,
    pub python: PathBuf,
    pub backend_dir: PathBuf,
    pub tools_dir: PathBuf,
    pub models_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PortableRuntimeStatus {
    Ready { root: PathBuf },
    Missing { root: PathBuf, missing_items: Vec<String> },
}

pub fn resolve_portable_runtime(
    repo_root: &Path,
    dev_profile: bool,
) -> Result<PortableRuntime, PortableRuntimeStatus> {
    let root = runtime_root(repo_root, dev_profile);
    validate_portable_runtime(&root)
}

pub fn runtime_root(repo_root: &Path, dev_profile: bool) -> PathBuf {
    if let Some(value) = env::var_os("DV_PORTABLE_RUNTIME_DIR") {
        return PathBuf::from(value);
    }
    if dev_profile {
        return repo_root.join("vendor").join("portable-runtime");
    }
    release_runtime_root()
}

pub fn validate_portable_runtime(root: &Path) -> Result<PortableRuntime, PortableRuntimeStatus> {
    let python = python_executable(root);
    let backend_dir = root.join("backend");
    let tools_dir = root.join("tools");
    let models_dir = root.join("models");
    let required = [
        (python.clone(), "python executable"),
        (backend_dir.join("dv_backend"), "backend/dv_backend"),
        (tools_dir.join("ffmpeg"), "tools/ffmpeg"),
        (models_dir.join("qwen3-asr"), "models/qwen3-asr"),
        (models_dir.join("voxcpm2"), "models/voxcpm2"),
    ];
    let missing_items = required
        .into_iter()
        .filter_map(|(path, label)| (!path.exists()).then(|| format!("{} ({})", label, path.display())))
        .collect::<Vec<_>>();
    if missing_items.is_empty() {
        Ok(PortableRuntime {
            root: root.to_path_buf(),
            python,
            backend_dir,
            tools_dir,
            models_dir,
        })
    } else {
        Err(PortableRuntimeStatus::Missing {
            root: root.to_path_buf(),
            missing_items,
        })
    }
}

pub fn python_executable(root: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        let embedded = root.join("python").join("python.exe");
        if embedded.exists() {
            return embedded;
        }
        root.join(".venv").join("Scripts").join("python.exe")
    }
    #[cfg(target_os = "macos")]
    {
        let embedded = root.join("python").join("bin").join("python3");
        if embedded.exists() {
            return embedded;
        }
        root.join(".venv").join("bin").join("python")
    }
}

pub fn prepend_path(dir: &Path, current: Option<OsString>) -> OsString {
    let mut paths = vec![dir.to_path_buf()];
    if let Some(current) = current {
        paths.extend(env::split_paths(&current));
    }
    env::join_paths(paths).unwrap_or_else(|_| OsString::from(dir.as_os_str()))
}

fn macos_bundle_runtime_roots(exe_dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let contents_dir = exe_dir.parent()?;
    let app_dir = contents_dir.parent()?;
    let app_parent = app_dir.parent()?;
    Some((
        app_parent.join("portable-runtime"),
        contents_dir.join("Resources").join("portable-runtime"),
    ))
}

fn release_runtime_root() -> PathBuf {
    let exe = env::current_exe().unwrap_or_else(|_| PathBuf::from("."));
    let exe_dir = exe.parent().unwrap_or_else(|| Path::new("."));
    let beside_exe = exe_dir.join("portable-runtime");
    if beside_exe.exists() {
        return beside_exe;
    }
    #[cfg(target_os = "macos")]
    if let Some((bundle_sibling, bundled_resource)) = macos_bundle_runtime_roots(exe_dir) {
        if bundle_sibling.exists() {
            return bundle_sibling;
        }
        if bundled_resource.exists() {
            return bundled_resource;
        }
        return bundle_sibling;
    }
    exe_dir.join("resources").join("portable-runtime")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    use std::sync::Mutex;
    use tempfile::tempdir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn make_runtime(root: &Path) {
        let python = test_python(root);
        fs::create_dir_all(python.parent().unwrap()).unwrap();
        fs::write(python, b"").unwrap();
        fs::create_dir_all(root.join("backend/dv_backend")).unwrap();
        fs::create_dir_all(root.join("tools/ffmpeg")).unwrap();
        fs::create_dir_all(root.join("models/qwen3-asr")).unwrap();
        fs::create_dir_all(root.join("models/voxcpm2")).unwrap();
    }

    #[cfg(windows)]
    fn test_python(root: &Path) -> PathBuf {
        root.join(".venv/Scripts/python.exe")
    }

    #[cfg(not(windows))]
    fn test_python(root: &Path) -> PathBuf {
        root.join(".venv/bin/python")
    }

    #[test]
    fn validate_accepts_complete_runtime() {
        let dir = tempdir().unwrap();
        make_runtime(dir.path());
        let runtime = validate_portable_runtime(dir.path()).unwrap();
        assert_eq!(runtime.root, dir.path());
        assert_eq!(runtime.backend_dir, dir.path().join("backend"));
    }

    #[test]
    fn validate_lists_all_missing_items() {
        let dir = tempdir().unwrap();
        let err = validate_portable_runtime(dir.path()).unwrap_err();
        match err {
            PortableRuntimeStatus::Missing { missing_items, .. } => {
                assert!(missing_items.iter().any(|item| item.contains("python executable")));
                assert!(missing_items.iter().any(|item| item.contains("backend/dv_backend")));
                assert!(missing_items.iter().any(|item| item.contains("tools/ffmpeg")));
                assert!(missing_items.iter().any(|item| item.contains("models/qwen3-asr")));
                assert!(missing_items.iter().any(|item| item.contains("models/voxcpm2")));
            }
            PortableRuntimeStatus::Ready { .. } => panic!("expected missing runtime"),
        }
    }

    #[test]
    fn env_override_wins() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempdir().unwrap();
        let override_dir = dir.path().join("custom-runtime");
        std::env::set_var("DV_PORTABLE_RUNTIME_DIR", &override_dir);
        let got = runtime_root(Path::new("C:/repo"), true);
        std::env::remove_var("DV_PORTABLE_RUNTIME_DIR");
        assert_eq!(got, override_dir);
    }

    #[test]
    fn dev_root_uses_vendor_portable_runtime() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("DV_PORTABLE_RUNTIME_DIR");
        assert_eq!(
            runtime_root(Path::new("C:/repo"), true),
            PathBuf::from("C:/repo").join("vendor").join("portable-runtime"),
        );
    }

    #[test]
    fn prepends_tool_path() {
        let dir = tempdir().unwrap();
        let tools = dir.path().join("tools");
        let system = dir.path().join("system");
        let current = env::join_paths([system.clone()]).unwrap();
        let got = prepend_path(&tools, Some(current));
        let parts = env::split_paths(&got).collect::<Vec<_>>();
        assert_eq!(parts[0], tools);
        assert_eq!(parts[1], system);
    }

    #[test]
    fn macos_bundle_runtime_roots_use_app_sibling_and_contents_resources() {
        let exe_dir = Path::new("/tmp/DouyinVietnamizer.app/Contents/MacOS");
        let (sibling, resource) = macos_bundle_runtime_roots(exe_dir).unwrap();
        assert_eq!(sibling, PathBuf::from("/tmp/portable-runtime"));
        assert_eq!(resource, PathBuf::from("/tmp/DouyinVietnamizer.app/Contents/Resources/portable-runtime"));
    }

    #[cfg(target_os = "macos")]
    fn make_runtime_macos(root: &Path) {
        fs::create_dir_all(root.join(".venv/bin")).unwrap();
        fs::write(root.join(".venv/bin/python"), b"").unwrap();
        fs::create_dir_all(root.join("backend/dv_backend")).unwrap();
        fs::create_dir_all(root.join("tools/ffmpeg")).unwrap();
        fs::create_dir_all(root.join("models/qwen3-asr")).unwrap();
        fs::create_dir_all(root.join("models/voxcpm2")).unwrap();
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn python_executable_picks_venv_bin_python_on_macos() {
        let dir = tempdir().unwrap();
        make_runtime_macos(dir.path());
        let p = python_executable(dir.path());
        assert_eq!(p, dir.path().join(".venv").join("bin").join("python"));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn python_executable_prefers_embedded_python3_when_present() {
        let dir = tempdir().unwrap();
        make_runtime_macos(dir.path());
        fs::create_dir_all(dir.path().join("python/bin")).unwrap();
        fs::write(dir.path().join("python/bin/python3"), b"").unwrap();
        let p = python_executable(dir.path());
        assert_eq!(p, dir.path().join("python").join("bin").join("python3"));
    }
}
