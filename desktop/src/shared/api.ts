import type { Job, JobsApi } from "./contracts";

const baseUrl = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8765";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers }
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error?.message ?? "Backend unavailable");
  return body;
}

export const api: JobsApi = {
  listJobs: () => request<Job[]>("/api/jobs"),
  createJob: (sourceUrl) =>
    request<Job>("/api/jobs", { method: "POST", body: JSON.stringify({ source_url: sourceUrl }) })
};

