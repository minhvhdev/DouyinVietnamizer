use std::path::PathBuf;
use tokio::process::Child;
use tokio::sync::Mutex;

pub struct BackendState {
    pub child: Mutex<Option<Child>>,
    pub base_url: String,
    pub backend_dir: PathBuf,
    pub dev_profile: bool,
}

impl BackendState {
    pub fn new(backend_dir: PathBuf, dev_profile: bool) -> Self {
        Self {
            child: Mutex::new(None),
            base_url: "http://127.0.0.1:8765".into(),
            backend_dir,
            dev_profile,
        }
    }
}
