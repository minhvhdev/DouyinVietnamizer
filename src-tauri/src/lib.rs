pub mod backend;
pub mod commands;
pub mod portable;
pub mod setup;
pub mod state;
pub mod watchdog;

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use state::BackendState;
use tauri::Manager;

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
            commands::open_folder,
        ])
        .on_window_event(|window, event| {
            let tauri::WindowEvent::CloseRequested { api, .. } = event else {
                return;
            };
            let app = window.app_handle().clone();
            let Some(state) = app.try_state::<BackendState>() else {
                return;
            };
            if state.shutdown_requested.swap(true, Ordering::SeqCst) {
                return;
            }
            api.prevent_close();
            let label = window.label().to_string();
            tauri::async_runtime::spawn(async move {
                if let Some(state) = app.try_state::<BackendState>() {
                    let _ = crate::backend::request_release_vram(&state.base_url).await;
                    let mut guard = state.child.lock().await;
                    if let Some(mut child) = guard.take() {
                        let _ = child.process.start_kill();
                    }
                }
                if let Some(next_window) = app.get_webview_window(&label) {
                    let _ = next_window.close();
                }
            });
        })
        .setup(|app| {
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                watchdog::run(handle).await;
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
