pub mod backend;
pub mod commands;
pub mod portable;
pub mod setup;
pub mod state;

use std::path::PathBuf;
use state::BackendState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let dev_profile = cfg!(debug_assertions);
    let backend_dir = if dev_profile {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("backend")
    } else {
        PathBuf::from("backend")
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .manage(BackendState::new(backend_dir, dev_profile))
        .invoke_handler(tauri::generate_handler![
            commands::get_backend_status,
            commands::run_first_time_setup_cmd,
            commands::restart_backend,
            commands::open_devtools,
        ])
        .setup(|_app| Ok(()))
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
