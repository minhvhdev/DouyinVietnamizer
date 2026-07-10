use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use tokio::sync::Mutex;

use crate::backend::ManagedChild;

pub struct BackendState {
    pub child: Mutex<Option<ManagedChild>>,
    pub base_url: String,
    pub backend_dir: PathBuf,
    pub dev_profile: bool,
    pub shutdown_requested: AtomicBool,
    /// Suppress false crash events while `restart_backend` is in flight.
    pub recovering: AtomicBool,
}

impl BackendState {
    pub fn new(backend_dir: PathBuf, dev_profile: bool) -> Self {
        Self {
            child: Mutex::new(None),
            base_url: "http://127.0.0.1:8765".into(),
            backend_dir,
            dev_profile,
            shutdown_requested: AtomicBool::new(false),
            recovering: AtomicBool::new(false),
        }
    }
}
