import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export type BackendStatus =
  | { kind: "setup_required"; stage: string }
  | { kind: "starting" }
  | { kind: "ready"; base_url: string }
  | { kind: "crashed"; stderr: string }
  | { kind: "already_running" };

const DEFAULT_INTERVAL = 200;
const DEFAULT_TIMEOUT = 30_000;

export async function waitForBackend(
  opts: { intervalMs?: number; timeoutMs?: number } = {},
): Promise<string> {
  const interval = opts.intervalMs ?? DEFAULT_INTERVAL;
  const timeout = opts.timeoutMs ?? DEFAULT_TIMEOUT;
  const deadline = Date.now() + timeout;
  let last: BackendStatus | null = null;
  while (Date.now() < deadline) {
    const s = (await invoke("get_backend_status")) as BackendStatus;
    last = s;
    if (s.kind === "ready") return s.base_url;
    if (s.kind === "setup_required" || s.kind === "crashed") {
      throw s;
    }
    await new Promise((r) => setTimeout(r, interval));
  }
  throw last ?? { kind: "crashed", stderr: "timed out waiting for backend" };
}

export function subscribeBackendEvents(handlers: {
  onReady?: (baseUrl: string) => void;
  onCrashed?: (stderr: string) => void;
}): () => void {
  const unsubs: UnlistenFn[] = [];
  listen<{ base_url: string }>("backend://ready", (e) => {
    handlers.onReady?.(e.payload.base_url);
  }).then((u) => unsubs.push(u));
  listen<{ stderr: string }>("backend://crashed", (e) => {
    handlers.onCrashed?.(e.payload.stderr);
  }).then((u) => unsubs.push(u));
  return () => unsubs.forEach((u) => u());
}

export async function invokeRestart(): Promise<void> {
  await invoke("restart_backend");
}

export async function invokeSetup(): Promise<void> {
  await invoke("run_first_time_setup_cmd");
}

export async function invokeOpenDevtools(): Promise<void> {
  await invoke("open_devtools");
}

export type SetupProgress = { stage: string; pct: number };

export async function subscribeSetupProgress(
  onProgress: (p: SetupProgress) => void,
): Promise<UnlistenFn> {
  return await listen<SetupProgress>("setup://progress", (e) => onProgress(e.payload));
}
