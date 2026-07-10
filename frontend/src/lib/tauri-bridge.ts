import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export type BackendStatus =
  | { kind: "environment_missing"; root: string; missing_items: string[] }
  | { kind: "starting" }
  | { kind: "ready"; base_url: string }
  | { kind: "crashed"; stderr: string }
  | { kind: "already_running" };

const DEFAULT_INTERVAL = 200;
const DEFAULT_TIMEOUT = 30_000;
export const BACKEND_BASE = "http://127.0.0.1:8765";

export type BackendConnectionState = "checking" | "online" | "offline" | "restarting";

function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export async function probeBackendHealth(
  baseUrl: string = BACKEND_BASE,
  timeoutMs = 3_000,
): Promise<boolean> {
  try {
    const res = await fetch(`${baseUrl}/api/health`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function inspectBackend(
  baseUrl: string = BACKEND_BASE,
): Promise<BackendStatus | "unreachable"> {
  if (await probeBackendHealth(baseUrl)) {
    return { kind: "ready", base_url: baseUrl };
  }
  if (!isTauri()) {
    return "unreachable";
  }
  return (await invoke("get_backend_status")) as BackendStatus;
}

export async function waitForBackend(
  opts: { intervalMs?: number; timeoutMs?: number } = {},
): Promise<string> {
  const interval = opts.intervalMs ?? DEFAULT_INTERVAL;
  const timeout = opts.timeoutMs ?? DEFAULT_TIMEOUT;
  const deadline = Date.now() + timeout;
  let last: BackendStatus | null = null;
  while (Date.now() < deadline) {
    if (!isTauri()) {
      if (await probeBackendHealth()) return BACKEND_BASE;
      last = { kind: "starting" };
      await new Promise((r) => setTimeout(r, interval));
      continue;
    }
    const s = (await invoke("get_backend_status")) as BackendStatus;
    last = s;
    if (s.kind === "ready" || s.kind === "already_running") return BACKEND_BASE;
    if (s.kind === "environment_missing" || s.kind === "crashed") {
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
  if (!isTauri()) return () => {};
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
  if (!isTauri()) return;
  await invoke("restart_backend");
}

export async function invokeSetup(): Promise<void> {
  if (!isTauri()) return;
  await invoke("run_first_time_setup_cmd");
}

export async function invokeOpenDevtools(): Promise<void> {
  if (!isTauri()) return;
  await invoke("open_devtools");
}

export async function invokeOpenFolder(path: string): Promise<void> {
  if (!isTauri()) {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(path);
    }
    throw new Error(`Đường dẫn đã được sao chép: ${path}`);
  }
  await invoke("open_folder", { path });
}

export type SetupProgress = { stage: string; pct: number };

export async function subscribeSetupProgress(
  onProgress: (p: SetupProgress) => void,
): Promise<UnlistenFn> {
  if (!isTauri()) return () => {};
  return await listen<SetupProgress>("setup://progress", (e) => onProgress(e.payload));
}
