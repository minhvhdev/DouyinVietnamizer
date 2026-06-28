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
  importJob: async (file: File, title?: string) => {
    const formData = new FormData();
    formData.append("file", file);
    if (title && title.trim()) formData.append("title", title.trim());
    const response = await fetch(`${baseUrl}/api/jobs/import`, {
      method: "POST",
      body: formData
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message ?? "Failed to import video");
    return body as Job;
  },
  runtimeStatus: () => request<RuntimeReport>("/api/runtime/status"),
  runSmokeTest: () => request<RuntimeReport>("/api/runtime/smoke-test", { method: "POST" }),
  startJob: (jobId) => request(`/api/jobs/${jobId}/start`, { method: "POST" }),
  cancelJob: (jobId) => request(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
  deleteJob: (jobId) => request(`/api/jobs/${jobId}`, { method: "DELETE" }),
  selectVideo: (jobId, index) =>
    request(`/api/jobs/${jobId}/select-video`, { method: "POST", body: JSON.stringify({ index }) }),
  getCheckpoint: (jobId, stepName) => request(`/api/jobs/${jobId}/checkpoint/${stepName}`),
  getSettings: () => request("/api/settings"),
  updateSettings: (payload) => request("/api/settings", { method: "PUT", body: JSON.stringify(payload) }),
  getEvents: () => request("/api/events"),
  listOutputs: () => request("/api/outputs"),
  listClonedVoices: () => request<any[]>("/api/cloned-voices"),
  createClonedVoice: async (name, file) => {
    const formData = new FormData();
    formData.append("name", name);
    formData.append("file", file);
    const response = await fetch(`${baseUrl}/api/cloned-voices`, {
      method: "POST",
      body: formData
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message ?? "Failed to create cloned voice");
    return body;
  },
  deleteClonedVoice: (voiceId) => request(`/api/cloned-voices/${voiceId}`, { method: "DELETE" }),
  testClonedVoice: async (voiceId, text, mode = "reference") => {
    const response = await fetch(`${baseUrl}/api/cloned-voices/${voiceId}/test`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode })
    });
    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.error?.message ?? "Failed to test cloned voice");
    }
    return response.blob();
  },
  previewPresetVoice: async (voice, text) => {
    const response = await fetch(`${baseUrl}/api/voices/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice, text })
    });
    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.error?.message ?? "Failed to preview preset voice");
    }
    return response.blob();
  },
  rerunJob: (jobId, keepSteps) =>
    request(`/api/jobs/${jobId}/rerun`, { method: "POST", body: JSON.stringify({ keep_steps: keepSteps }) }),
  redubJob: (jobId) => request(`/api/jobs/${jobId}/redub`, { method: "POST" }),
  getJobFiles: (jobId) => request(`/api/jobs/${jobId}/files`),
  detectHardware: () => request("/api/runtime/detect-hardware"),
  bootstrapVendor: (profile) => request("/api/runtime/bootstrap-vendor", { method: "POST", body: JSON.stringify({ profile }) }),
  bootstrapPyannote: () => request("/api/runtime/bootstrap-pyannote", { method: "POST" }),
  bootstrapProgress: () => request("/api/runtime/bootstrap-progress")
};
