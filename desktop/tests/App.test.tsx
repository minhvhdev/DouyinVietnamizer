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

test("shows an actionable backend connection error", async () => {
  const api = { listJobs: vi.fn().mockRejectedValue(new Error("Backend unavailable")), createJob: vi.fn(), runtimeStatus: vi.fn().mockResolvedValue(readyRuntime), runSmokeTest: vi.fn() };
  render(<App api={api} />);

  expect(await screen.findByText("Backend unavailable")).toBeInTheDocument();
  expect(screen.getByText(/Start the local backend/)).toBeInTheDocument();
});

test("creates a job and shows it on the dashboard", async () => {
  const api = { listJobs: vi.fn().mockResolvedValue([]), createJob: vi.fn().mockResolvedValue(job), runtimeStatus: vi.fn().mockResolvedValue(readyRuntime), runSmokeTest: vi.fn() };
  render(<App api={api} />);

  fireEvent.change(await screen.findByPlaceholderText(/Paste a Douyin/), {
    target: { value: job.source_url }
  });
  fireEvent.click(screen.getByRole("button", { name: "Create job" }));

  await waitFor(() => expect(screen.getByText(job.source_url)).toBeInTheDocument());
});

test("shows runtime checks and reruns the smoke test", async () => {
  const warning = { ...readyRuntime, status: "warning", checks: [{ ...readyRuntime.checks[0], status: "warning", action: "Bundle this tool before release." }] };
  const api = { listJobs: vi.fn().mockResolvedValue([]), createJob: vi.fn(), runtimeStatus: vi.fn().mockResolvedValue(warning), runSmokeTest: vi.fn().mockResolvedValue(readyRuntime) };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: /Runtime/ }));
  expect(screen.getByText("Bundle this tool before release.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Run smoke test" }));

  await waitFor(() => expect(api.runSmokeTest).toHaveBeenCalled());
  expect((await screen.findAllByText("Runtime ready")).length).toBe(2);
});

test("disables job creation when runtime is blocked", async () => {
  const blocked = { ...readyRuntime, status: "blocked" };
  const api = { listJobs: vi.fn().mockResolvedValue([]), createJob: vi.fn(), runtimeStatus: vi.fn().mockResolvedValue(blocked), runSmokeTest: vi.fn() };
  render(<App api={api} />);

  expect(await screen.findByRole("button", { name: "Create job" })).toBeDisabled();
});

