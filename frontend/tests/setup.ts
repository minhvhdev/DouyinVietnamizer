import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

vi.mock("../src/lib/tauri-bridge", () => ({
  waitForBackend: vi.fn().mockResolvedValue("http://127.0.0.1:8765"),
  subscribeBackendEvents: vi.fn().mockReturnValue(() => {}),
  invokeOpenDevtools: vi.fn().mockResolvedValue(undefined),
  invokeOpenFolder: vi.fn().mockResolvedValue(undefined),
  invokeSetup: vi.fn().mockResolvedValue(undefined),
  invokeRestart: vi.fn().mockResolvedValue(undefined),
  subscribeSetupProgress: vi.fn().mockResolvedValue(() => {}),
}));

