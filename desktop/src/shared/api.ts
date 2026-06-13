import type { Job, JobsApi, RuntimeReport } from "./contracts";

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
    request<Job>("/api/jobs", { method: "POST", body: JSON.stringify({ source_url: sourceUrl }) }),
  runtimeStatus: () => request<RuntimeReport>("/api/runtime/status"),
  runSmokeTest: () => request<RuntimeReport>("/api/runtime/smoke-test", { method: "POST" }),
  startJob: (jobId) => request(`/api/jobs/${jobId}/start`, { method: "POST" }),
  cancelJob: (jobId) => request(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
  selectVideo: (jobId, index) =>
    request(`/api/jobs/${jobId}/select-video`, { method: "POST", body: JSON.stringify({ index }) }),
  getCheckpoint: (jobId, stepName) => request(`/api/jobs/${jobId}/checkpoint/${stepName}`),
  getSettings: () => request("/api/settings"),
  updateSettings: (payload) => request("/api/settings", { method: "PUT", body: JSON.stringify(payload) }),
  getEvents: () => request("/api/events"),
  listOutputs: () => request("/api/outputs")
};

