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

test("shows an actionable backend connection error", async () => {
  const api = { listJobs: vi.fn().mockRejectedValue(new Error("Backend unavailable")), createJob: vi.fn() };
  render(<App api={api} />);

  expect(await screen.findByText("Backend unavailable")).toBeInTheDocument();
  expect(screen.getByText(/Start the local backend/)).toBeInTheDocument();
});

test("creates a job and shows it on the dashboard", async () => {
  const api = { listJobs: vi.fn().mockResolvedValue([]), createJob: vi.fn().mockResolvedValue(job) };
  render(<App api={api} />);

  fireEvent.change(await screen.findByPlaceholderText(/Paste a Douyin/), {
    target: { value: job.source_url }
  });
  fireEvent.click(screen.getByRole("button", { name: "Create job" }));

  await waitFor(() => expect(screen.getByText(job.source_url)).toBeInTheDocument());
});

