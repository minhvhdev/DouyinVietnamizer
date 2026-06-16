import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";

import { App } from "../src/renderer/App";

const job = {
  id: "job-1",
  source_url: "https://v.douyin.com/demo/",
  status: "queued",
  current_step: null,
  created_at: "2026-06-13T00:00:00Z",
  updated_at: "2026-06-13T00:00:00Z",
  steps: Array.from({ length: 12 }, (_, position) => ({
    name: `step-${position}`,
    position,
    status: "pending"
  }))
};
const readyRuntime = {
  status: "ready",
  checked_at: "2026-06-13T00:00:00Z",
  checks: [{ id: "storage", display_name: "Local storage", status: "ready", required: true, message: "Local storage is writable.", action: "No action required." }]
};
import { JobsApi } from "../src/shared/contracts";

const baseApi: JobsApi = {
  listJobs: vi.fn().mockResolvedValue([]),
  createJob: vi.fn().mockResolvedValue(job),
  runtimeStatus: vi.fn().mockResolvedValue(readyRuntime),
  runSmokeTest: vi.fn().mockResolvedValue(readyRuntime),
  startJob: vi.fn().mockResolvedValue({ status: "started" }),
  cancelJob: vi.fn().mockResolvedValue({ status: "cancelled" }),
  selectVideo: vi.fn().mockResolvedValue({ status: "selected" }),
  getCheckpoint: vi.fn().mockResolvedValue(null),
  getSettings: vi.fn().mockResolvedValue({}),
  updateSettings: vi.fn().mockResolvedValue({}),
  getEvents: vi.fn().mockResolvedValue([]),
  listOutputs: vi.fn().mockResolvedValue([])
};

test("shows an actionable backend connection error", async () => {
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockRejectedValue(new Error("Backend unavailable")) };
  render(<App api={api} />);

  expect(await screen.findByText("Backend unavailable")).toBeInTheDocument();
  expect(screen.getByText(/Check logs or configuration/)).toBeInTheDocument();
});

test("creates a job and shows it on the dashboard", async () => {
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([]), createJob: vi.fn().mockResolvedValue(job) };
  render(<App api={api} />);

  fireEvent.change(await screen.findByPlaceholderText(/Paste a Douyin/), {
    target: { value: job.source_url }
  });
  fireEvent.click(screen.getByRole("button", { name: "Create job" }));

  await waitFor(() => expect(screen.getByText(job.source_url)).toBeInTheDocument());
});

test("shows runtime checks and reruns the smoke test", async () => {
  const warning = { ...readyRuntime, status: "warning", checks: [{ ...readyRuntime.checks[0], status: "warning", action: "Bundle this tool before release." }] };
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([]), runtimeStatus: vi.fn().mockResolvedValue(warning), runSmokeTest: vi.fn().mockResolvedValue(readyRuntime) };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: /Runtime/ }));
  expect(screen.getByText("Bundle this tool before release.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Run smoke test" }));

  await waitFor(() => expect(api.runSmokeTest).toHaveBeenCalled());
  expect((await screen.findAllByText("Runtime ready")).length).toBe(2);
});

test("disables job creation when runtime is blocked", async () => {
  const blocked = { ...readyRuntime, status: "blocked" };
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([]), runtimeStatus: vi.fn().mockResolvedValue(blocked) };
  render(<App api={api} />);

  expect(await screen.findByRole("button", { name: "Create job" })).toBeDisabled();
});

test("shows free translation Edge TTS and browser cookie disclosure", async () => {
  const api: JobsApi = {
    ...baseApi,
    getSettings: vi.fn().mockResolvedValue({
      cookies_browser: "none",
      translation_backend: "google_free",
      asr_backend: "whisper_cpu",
      whisper_model_path: "",
      tts_backend: "edge",
      edge_tts_voice: "vi-VN-HoaiMyNeural",
      edge_tts_rate: "+0%"
    })
  };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: "Settings" }));

  expect(screen.getAllByText("Google Translate Free").length).toBeGreaterThan(0);
  expect(screen.getAllByText("Microsoft Edge TTS").length).toBeGreaterThan(0);
  expect(screen.getByText(/cookies may contain sensitive session data/i)).toBeInTheDocument();
});

test("manages Gemini API key pool from settings", async () => {
  const updateSettings = vi.fn().mockImplementation(async (payload) => {
    if (payload.gemini_api_key_update) {
      return {
        gemini_api_keys: [{ id: "key-1", masked: "AIza...7890", label: payload.gemini_api_key_update.label }]
      };
    }
    return {
      gemini_api_keys: [{ id: "key-1", masked: "AIza...7890", label: "Studio quota 1" }]
    };
  });
  const api: JobsApi = {
    ...baseApi,
    updateSettings,
    getSettings: vi.fn().mockResolvedValue({
      cookies_browser: "none",
      translation_backend: "gemini",
      tts_backend: "gemini",
      gemini_api_keys: [{ id: "key-1", masked: "AIza...7890", label: "Studio quota 1" }],
      gemini_translation_model: "gemini-2.5-flash",
      gemini_tts_model: "gemini-2.5-flash-preview-tts",
      gemini_tts_voice: "Zephyr"
    })
  };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: "Settings" }));

  expect(await screen.findByText("Google AI Studio / Gemini API Keys")).toBeInTheDocument();
  expect(screen.getAllByText("Gemini").length).toBeGreaterThan(0);
  expect(screen.getByText("AIza...7890")).toBeInTheDocument();
  expect(screen.getByDisplayValue("Studio quota 1")).toBeInTheDocument();

  fireEvent.change(screen.getByPlaceholderText(/Paste Google AI Studio API key/i), {
    target: { value: "AIzaSyNewSecret" }
  });
  fireEvent.click(screen.getByRole("button", { name: "Add Gemini key" }));
  await waitFor(() => expect(updateSettings).toHaveBeenCalledWith({ gemini_api_key_add: "AIzaSyNewSecret" }));

  fireEvent.change(screen.getByLabelText("Edit label for Gemini key AIza...7890"), {
    target: { value: "backup key" }
  });
  fireEvent.click(screen.getByRole("button", { name: "Save label for Gemini key AIza...7890" }));
  await waitFor(() => expect(updateSettings).toHaveBeenCalledWith({
    gemini_api_key_update: { id: "key-1", label: "backup key" }
  }));

  fireEvent.click(screen.getByRole("button", { name: "Remove Gemini key AIza...7890" }));
  await waitFor(() => expect(updateSettings).toHaveBeenCalledWith({ gemini_api_key_remove: "key-1" }));

  expect(screen.queryByText("AIzaSyNewSecret")).not.toBeInTheDocument();
});

