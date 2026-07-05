import { expect, test, vi } from "vitest";

vi.unmock("../src/lib/tauri-bridge");

test("probeBackendHealth uses HTTP health endpoint", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));

  const { probeBackendHealth } = await import("../src/lib/tauri-bridge");

  await expect(probeBackendHealth()).resolves.toBe(true);
  expect(fetch).toHaveBeenCalledWith(
    "http://127.0.0.1:8765/api/health",
    expect.objectContaining({ signal: expect.any(AbortSignal) }),
  );
});

test("waitForBackend uses HTTP health outside Tauri", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));

  const { waitForBackend } = await import("../src/lib/tauri-bridge");

  await expect(waitForBackend({ intervalMs: 1, timeoutMs: 20 })).resolves.toBe("http://127.0.0.1:8765");
  expect(fetch).toHaveBeenCalledWith(
    "http://127.0.0.1:8765/api/health",
    expect.objectContaining({ signal: expect.any(AbortSignal) }),
  );
});
