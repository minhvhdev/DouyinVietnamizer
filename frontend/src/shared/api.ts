import type { Job, JobFolder, JobsApi, RuntimeReport } from "./contracts";

const baseUrl = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8765";

export function formatApiError(body: { error?: { message?: string; action?: string; detail?: string } }): string {
  const err = body.error;
  const parts = [err?.message, err?.action, err?.detail ? `Chi tiết: ${err.detail}` : null].filter(Boolean);
  return parts.join("\n\n") || "Backend unavailable";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers }
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(formatApiError(body));
  }
  return body;
}

export const api: JobsApi = {
  listJobs: () => request<Job[]>("/api/jobs"),
  createJob: (sourceUrl) =>
    request<Job>("/api/jobs", { method: "POST", body: JSON.stringify({ source_url: sourceUrl }) }),
  selectVideo: (jobId, index) =>
    request(`/api/jobs/${jobId}/select-video`, { method: "POST", body: JSON.stringify({ index }) }),
  updateYtDlp: () => request<{ status: string; version: string; previous_version: string; method: string }>(
    "/api/runtime/update-yt-dlp",
    { method: "POST" },
  ),
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
  releaseVram: () => request("/api/runtime/release-vram", { method: "POST" }),
  startJob: (jobId) => request(`/api/jobs/${jobId}/start`, { method: "POST" }),
  cancelJob: (jobId) => request(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
  deleteJob: (jobId) => request(`/api/jobs/${jobId}`, { method: "DELETE" }),
  getCheckpoint: (jobId, stepName) => request(`/api/jobs/${jobId}/checkpoint/${stepName}`),
  getSettings: () => request("/api/settings"),
  updateSettings: (payload) => request("/api/settings", { method: "PUT", body: JSON.stringify(payload) }),
  getEvents: () => request("/api/events"),
  listOutputs: () => request("/api/outputs"),
  listClonedVoices: (backend = "omnivoice") => request<any[]>(`/api/cloned-voices?backend=${encodeURIComponent(backend)}`),
  createClonedVoice: async (name, file, backend = "omnivoice", refText) => {
    const formData = new FormData();
    formData.append("name", name);
    formData.append("file", file);
    formData.append("backend", backend);
    if (refText?.trim()) {
      formData.append("ref_text", refText.trim());
    }
    const response = await fetch(`${baseUrl}/api/cloned-voices`, {
      method: "POST",
      body: formData
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message ?? "Failed to create cloned voice");
    return body;
  },
  deleteClonedVoice: (voiceId, backend = "omnivoice") => request(`/api/cloned-voices/${voiceId}?backend=${encodeURIComponent(backend)}`, { method: "DELETE" }),
  testClonedVoice: async (voiceId, text, mode = "reference", backend = "omnivoice") => {
    const response = await fetch(`${baseUrl}/api/cloned-voices/${voiceId}/test`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode, backend }),
      signal: AbortSignal.timeout(180_000),
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
  listTtsVoices: async (backend, locale) => {
    const params = new URLSearchParams({ backend });
    if (locale) params.set("locale", locale);
    return request(`/api/tts/voices?${params.toString()}`);
  },
  listOpenAiModels: async (options = {}) =>
    request<Array<{ id: string; name: string }>>("/api/translation/openai-models", {
      method: "POST",
      body: JSON.stringify({
        base_url: options.baseUrl,
        api_key: options.apiKey,
      }),
    }),
  previewTts: async (text, options = {}) => {
    const response = await fetch(`${baseUrl}/api/tts/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        backend: options.backend,
        voice: options.voice,
        settings: options.settings,
      }),
      signal: AbortSignal.timeout(180_000),
    });
    if (!response.ok) {
      const body = await response.json();
      throw new Error(formatApiError(body));
    }
    return response.blob();
  },
  rerunJob: (jobId, keepSteps) =>
    request(`/api/jobs/${jobId}/rerun`, { method: "POST", body: JSON.stringify({ keep_steps: keepSteps }) }),
  redubJob: (jobId) => request(`/api/jobs/${jobId}/redub`, { method: "POST" }),
  getJobFiles: (jobId) => request(`/api/jobs/${jobId}/files`),
  getJobFolder: (jobId) => request<JobFolder>(`/api/jobs/${jobId}/folder`),
  detectHardware: () => request("/api/runtime/detect-hardware"),
  bootstrapVendor: (profile) => request("/api/runtime/bootstrap-vendor", { method: "POST", body: JSON.stringify({ profile }) }),
  bootstrapProgress: () => request("/api/runtime/bootstrap-progress")
};
