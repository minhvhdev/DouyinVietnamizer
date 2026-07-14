import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";

import { App } from "../src/renderer/App";

vi.mock("../src/lib/tauri-bridge", async () => {
  const actual = await vi.importActual<typeof import("../src/lib/tauri-bridge")>("../src/lib/tauri-bridge");
  return {
    ...actual,
    waitForBackend: vi.fn().mockResolvedValue("http://127.0.0.1:8765"),
    subscribeBackendEvents: vi.fn().mockReturnValue(() => {}),
    invokeRestart: vi.fn().mockResolvedValue(undefined),
    invokeOpenDevtools: vi.fn().mockResolvedValue(undefined),
    invokeOpenFolder: vi.fn().mockResolvedValue(undefined),
  };
});

const job = {
  id: "job-1",
  source_url: "import://demo.mp4",
  title: "demo",
  title_vi: null,
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
  selectVideo: vi.fn().mockResolvedValue({ status: "selected", video: {} }),
  updateYtDlp: vi.fn().mockResolvedValue({ status: "updated", version: "2026.01.01", previous_version: "2025.01.01", method: "binary_replace" }),
  importJob: vi.fn().mockResolvedValue(job),
  runtimeStatus: vi.fn().mockResolvedValue(readyRuntime),
  runSmokeTest: vi.fn().mockResolvedValue(readyRuntime),
  releaseVram: vi.fn().mockResolvedValue({ status: "ok", released: [], terminated_processes: [], errors: [], gpu: { cuda_supported: false, active_omnivoice_clients: 0, resident_models: [], helper_processes: [] } }),
  startJob: vi.fn().mockResolvedValue({ status: "started" }),
  cancelJob: vi.fn().mockResolvedValue({ status: "cancelled" }),
  deleteJob: vi.fn().mockResolvedValue({ status: "deleted" }),
  getCheckpoint: vi.fn().mockResolvedValue(null),
  getSettings: vi.fn().mockResolvedValue({}),
  updateSettings: vi.fn().mockResolvedValue({}),
  getEvents: vi.fn().mockResolvedValue([]),
  listOutputs: vi.fn().mockResolvedValue([]),
  listClonedVoices: vi.fn().mockResolvedValue([]),
  createClonedVoice: vi.fn().mockResolvedValue({ id: "voice-1", name: "Voice 1", wav_filename: "v1.wav", wav_path: "", transcript: "", transcribed: false, created_at: "" }),
  deleteClonedVoice: vi.fn().mockResolvedValue({ status: "deleted" }),
  startVoiceCalibration: vi.fn().mockResolvedValue({ voice_id: "voice-1", status: "idle" }),
  getVoiceCalibration: vi.fn().mockResolvedValue({ voice_id: "voice-1", status: "idle" }),
  cancelVoiceCalibration: vi.fn().mockResolvedValue({ status: "cancelled", voice_id: "voice-1" }),
  resumeVoiceCalibration: vi.fn().mockResolvedValue({ voice_id: "voice-1", status: "idle" }),
  resetVoiceDurationProfile: vi.fn().mockResolvedValue({ status: "reset", voice_id: "voice-1" }),
  testClonedVoice: vi.fn().mockResolvedValue(new Blob()),
  previewPresetVoice: vi.fn().mockResolvedValue(new Blob()),
  listTtsVoices: vi.fn().mockResolvedValue([
    { id: "vi-VN-HoaiMyNeural", name: "Hoài My (Nữ)" },
    { id: "vi-VN-NamMinhNeural", name: "Nam Minh (Nam)" },
  ]),
  listOpenAiModels: vi.fn().mockResolvedValue([
    { id: "gpt-4o", name: "gpt-4o" },
    { id: "gpt-4o-mini", name: "gpt-4o-mini" },
  ]),
  previewTts: vi.fn().mockResolvedValue(new Blob()),
  rerunJob: vi.fn().mockResolvedValue({ status: "queued", job }),
  redubJob: vi.fn().mockResolvedValue({ status: "queued", job }),
  getJobFiles: vi.fn().mockResolvedValue([]),
  getJobFolder: vi.fn().mockResolvedValue({ path: "C:/data/jobs/job-1", exists: true }),
  detectHardware: vi.fn().mockResolvedValue({ cuda_supported: true, vulkan_supported: true, avx2_supported: true, espeak_installed: true, recommendation: "gpu_cuda" }),
  bootstrapVendor: vi.fn().mockResolvedValue({ status: "started" }),
  bootstrapProgress: vi.fn().mockResolvedValue({ status: "idle", current_task: "", download_percent: 0, download_speed_kb: 0, downloaded_bytes: 0, total_bytes: 0, error_message: "", logs: [] })
};

test("keeps the dashboard grid limited to sidebar and main content", async () => {
  const { container } = render(<App api={baseApi} />);

  await screen.findByText("Bảng điều khiển tiến trình");

  const shell = container.querySelector(".shell");
  expect(Array.from(shell?.children ?? []).map((child) => child.tagName)).toEqual(["ASIDE", "MAIN"]);
});

test("shows environment errors when dev layout is incomplete", async () => {
  const bridge = await import("../src/lib/tauri-bridge");
  vi.mocked(bridge.waitForBackend).mockRejectedValueOnce({
    kind: "environment_missing",
    root: "C:/repo",
    missing_items: [
      "backend/dv_backend (C:/repo/backend/dv_backend)",
      "vendor/manifest.json (C:/repo/vendor/manifest.json)",
    ],
  });

  render(<App api={baseApi} />);

  expect(await screen.findByText("Môi trường dev chưa sẵn sàng")).toBeInTheDocument();
  expect(screen.getByText("C:/repo")).toBeInTheDocument();
  expect(screen.getByText(/backend\/dv_backend/)).toBeInTheDocument();
  expect(screen.getByText(/vendor\/manifest.json/)).toBeInTheDocument();
});

test("shows an actionable backend connection error", async () => {
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockRejectedValue(new Error("Backend unavailable")) };
  render(<App api={api} />);

  const errorsButton = await screen.findByRole("button", { name: /Lỗi và thông báo \(1\)/i });
  fireEvent.click(errorsButton);

  expect(await screen.findByText("Backend unavailable")).toBeInTheDocument();
  expect(screen.getByText(/Lỗi thao tác/i)).toBeInTheDocument();
});

test("creates a job from a local video file", async () => {
  const importJob = vi.fn().mockResolvedValue(job);
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([]), importJob };
  render(<App api={api} />);

  const file = new File(["video-bytes"], "demo.mp4", { type: "video/mp4" });
  const input = await screen.findByLabelText(/Chọn video từ máy tính/i);
  fireEvent.change(input, { target: { files: [file] } });
  const form = input.closest("form");
  expect(form).not.toBeNull();
  fireEvent.submit(form!);

  await waitFor(() => expect(importJob).toHaveBeenCalledWith(file));
  expect(await screen.findByText("demo")).toBeInTheDocument();
  expect(screen.queryByText("Chi tiết tiến trình")).not.toBeInTheDocument();
});

test("creates jobs from multiple local video files", async () => {
  const importJob = vi.fn()
    .mockResolvedValueOnce({ ...job, id: "job-1", title: "demo-a" })
    .mockResolvedValueOnce({ ...job, id: "job-2", title: "demo-b" });
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([]), importJob };
  render(<App api={api} />);

  const files = [
    new File(["video-a"], "demo-a.mp4", { type: "video/mp4" }),
    new File(["video-b"], "demo-b.mp4", { type: "video/mp4" }),
  ];
  const input = await screen.findByLabelText(/Chọn video từ máy tính/i);
  fireEvent.change(input, { target: { files } });
  fireEvent.submit(input.closest("form")!);

  await waitFor(() => expect(importJob).toHaveBeenCalledTimes(2));
  expect(importJob).toHaveBeenNthCalledWith(1, files[0]);
  expect(importJob).toHaveBeenNthCalledWith(2, files[1]);
  expect(await screen.findByText("demo-a")).toBeInTheDocument();
  expect(await screen.findByText("demo-b")).toBeInTheDocument();
});

test("creates a job from a douyin link", async () => {
  const createJob = vi.fn().mockResolvedValue({ ...job, source_url: "https://www.douyin.com/video/123" });
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([]), createJob };
  render(<App api={api} />);

  const input = await screen.findByLabelText(/Dán liên kết video Douyin hoặc Bilibili/i);
  fireEvent.change(input, { target: { value: "https://www.douyin.com/video/123" } });
  fireEvent.submit(input.closest("form")!);

  await waitFor(() => expect(createJob).toHaveBeenCalledWith("https://www.douyin.com/video/123"));
});

test("job details shows live elapsed time for running steps", async () => {
  const dateNow = vi.spyOn(Date, "now").mockReturnValue(new Date("2026-06-13T00:00:05Z").getTime());
  try {
    const runningJob = {
      ...job,
      status: "running",
      steps: [{ name: "asr", position: 0, status: "running", started_at: "2026-06-13T00:00:00Z", duration_ms: null }]
    };
    const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([runningJob]) };
    render(<App api={api} />);

    fireEvent.click(await screen.findByText("demo"));

    expect(await screen.findByText(/Thời gian: 5.0 s/)).toBeInTheDocument();
  } finally {
    dateNow.mockRestore();
  }
});


test("job details opens only by click and closes without leaving selection active", async () => {
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([job]) };
  const { container } = render(<App api={api} />);

  fireEvent.click(await screen.findByText("demo"));
  expect(await screen.findByText("Chi tiết tiến trình")).toBeInTheDocument();

  fireEvent.click(screen.getByLabelText("Close job details"));

  await waitFor(() => expect(screen.queryByText("Chi tiết tiến trình")).not.toBeInTheDocument());
  expect(container.querySelector(".selected-card")).toBeNull();
});

test("deletes completed jobs from the dashboard", async () => {
  const completedJob = { ...job, status: "completed" };
  const deleteJob = vi.fn().mockResolvedValue({ status: "deleted" });
  const api: JobsApi = {
    ...baseApi,
    deleteJob,
    listJobs: vi.fn()
      .mockResolvedValueOnce([completedJob])
      .mockResolvedValueOnce([])
  };
  render(<App api={api} />);

  expect(await screen.findByText("demo")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: `Xóa tiến trình ${job.id}` }));

  await waitFor(() => expect(deleteJob).toHaveBeenCalledWith(job.id));
  await waitFor(() => expect(screen.queryByText("demo")).not.toBeInTheDocument());
});

test("shows runtime checks and reruns the smoke test", async () => {
  const warning = { ...readyRuntime, status: "warning", checks: [{ ...readyRuntime.checks[0], status: "warning", action: "Install the tool under vendor/ or add it to PATH, then retry." }] };
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([]), runtimeStatus: vi.fn().mockResolvedValue(warning), runSmokeTest: vi.fn().mockResolvedValue(readyRuntime) };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: /Môi trường:/i }));
  expect(screen.getByText("Install the tool under vendor/ or add it to PATH, then retry.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Chạy thử nghiệm hệ thống" }));

  await waitFor(() => expect(api.runSmokeTest).toHaveBeenCalled());
  expect(await screen.findByText("Đã sẵn sàng")).toBeInTheDocument();
});

test("opens the runtime panel and retries when the first status fetch fails", async () => {
  const runtimeStatus = vi.fn()
    .mockRejectedValueOnce(new Error("Backend unavailable"))
    .mockResolvedValueOnce(readyRuntime);
  const api: JobsApi = {
    ...baseApi,
    listJobs: vi.fn().mockResolvedValue([]),
    runtimeStatus,
  };
  render(<App api={api} />);

  const button = await screen.findByRole("button", { name: /Môi trường:/i });
  fireEvent.click(button);

  expect(await screen.findByText("Môi trường thực thi")).toBeInTheDocument();
  await waitFor(() => expect(runtimeStatus).toHaveBeenCalledTimes(2));
  expect(await screen.findByText("Local storage is writable.")).toBeInTheDocument();
});

test("keeps the main app visible when runtime stays blocked", async () => {
  const blocked = {
    ...readyRuntime,
    status: "blocked",
    checks: [{ ...readyRuntime.checks[0], status: "blocked", message: "Dev environment check failed.", action: "Open diagnostics." }]
  };
  const api: JobsApi = {
    ...baseApi,
    listJobs: vi.fn().mockResolvedValue([]),
    runtimeStatus: vi.fn().mockResolvedValue(blocked),
    runSmokeTest: vi.fn().mockResolvedValue(blocked)
  };
  render(<App api={api} />);

  expect(await screen.findByText("Bảng điều khiển tiến trình")).toBeInTheDocument();
  expect(screen.queryByText("Thiết lập môi trường ứng dụng")).not.toBeInTheDocument();
});

test("runtime panel distinguishes required and optional warnings", async () => {
  const warning = {
    ...readyRuntime,
    status: "warning",
    checks: [
      { id: "storage", display_name: "Local storage", status: "ready", required: true, message: "Local storage is writable.", action: "No action required." },
      { id: "ffmpeg", display_name: "FFmpeg", status: "warning", required: true, message: "FFmpeg returned unrecognized version output.", action: "Verify that the executable matches the vendor manifest." },
      { id: "legacy_tool", display_name: "Legacy tool", status: "warning", required: false, message: "Legacy tool was not found.", action: "Install the tool under vendor/ or add it to PATH, then retry." }
    ]
  };
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([]), runtimeStatus: vi.fn().mockResolvedValue(warning) };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: /Môi trường:/i }));

  expect(await screen.findByText("FFmpeg")).toBeInTheDocument();
  expect(screen.getAllByText("Bắt buộc").length).toBeGreaterThan(0);
  expect(screen.getByText("Tuỳ chọn")).toBeInTheDocument();
  expect(screen.getByText("Cảnh báo")).toBeInTheDocument();
  expect(screen.getByText("Thiếu tuỳ chọn")).toBeInTheDocument();
});

test("shows the VRAM summary on the sidebar button", async () => {
  const runtimeWithGpu = {
    ...readyRuntime,
    gpu: {
      cuda_supported: true,
      device_name: "NVIDIA GeForce RTX 5060",
      total_vram_mb: 8192,
      used_vram_mb: 1126.4,
      free_vram_mb: 7065.6,
      torch_allocated_mb: 0,
      torch_reserved_mb: 0,
      torch_peak_mb: 0,
      active_omnivoice_clients: 0,
      resident_models: [],
      helper_processes: []
    }
  };
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([]), runtimeStatus: vi.fn().mockResolvedValue(runtimeWithGpu) };
  render(<App api={api} />);

  expect(await screen.findByText("Môi trường: Đủ")).toBeInTheDocument();
  expect(await screen.findByText("VRAM 1.10 GB / 8.00 GB")).toBeInTheDocument();
});

test("shows translation and OmniVoice settings", async () => {
  const api: JobsApi = {
    ...baseApi,
    getSettings: vi.fn().mockResolvedValue({
      translation_backend: "google_free",
      tts_backend: "omnivoice",
    })
  };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: "Cài đặt" }));

  expect(await screen.findByRole("tab", { name: "Dịch thuật" })).toBeInTheDocument();
  expect(await screen.findByText("Engine lồng tiếng")).toBeInTheDocument();
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
      translation_backend: "gemini",
      tts_backend: "omnivoice",
      gemini_api_keys: [{ id: "key-1", masked: "AIza...7890", label: "Studio quota 1" }],
      gemini_translation_model: "gemini-2.5-flash",
    })
  };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: "Cài đặt" }));
  fireEvent.click(await screen.findByRole("tab", { name: "Dịch thuật" }));
  expect(await screen.findByText("Quản lý khóa Gemini")).toBeInTheDocument();
  expect(screen.getByText("AIza...7890")).toBeInTheDocument();
  expect(screen.getByDisplayValue("Studio quota 1")).toBeInTheDocument();

  fireEvent.change(screen.getByPlaceholderText(/Dán khóa API Google AI Studio/i), {
    target: { value: "AIzaSyNewSecret" }
  });
  fireEvent.click(screen.getByRole("button", { name: "Thêm khóa" }));
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

test("saves a pending Gemini key when a field loses focus", async () => {
  const updateSettings = vi.fn().mockResolvedValue({
    gemini_api_keys: [{ id: "key-1", masked: "AIza...7890", label: "AIza...7890" }]
  });
  const api: JobsApi = {
    ...baseApi,
    updateSettings,
    getSettings: vi.fn().mockResolvedValue({
      translation_backend: "gemini",
      tts_backend: "omnivoice",
      gemini_api_keys: []
    })
  };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: "Cài đặt" }));
  fireEvent.click(await screen.findByRole("tab", { name: "Dịch thuật" }));
  fireEvent.change(screen.getByPlaceholderText(/Dán khóa API Google AI Studio/i), {
    target: { value: "AIzaSySecret1234567890" }
  });
  fireEvent.blur(screen.getByPlaceholderText(/Dán khóa API Google AI Studio/i));

  await waitFor(() =>
    expect(updateSettings).toHaveBeenCalledWith(expect.objectContaining({
      gemini_api_key_add: "AIzaSySecret1234567890"
    }))
  );
  expect(await screen.findByText("AIza...7890")).toBeInTheDocument();
  expect(screen.getByPlaceholderText(/Dán khóa API Google AI Studio/i)).toHaveValue("");
});

test("does not resend stale Gemini key command fields on autosave", async () => {
  const updateSettings = vi.fn().mockResolvedValue({
    gemini_api_keys: []
  });
  const api: JobsApi = {
    ...baseApi,
    updateSettings,
    getSettings: vi.fn().mockResolvedValue({
      translation_backend: "gemini",
      tts_backend: "omnivoice",
      gemini_api_keys: [],
      gemini_api_key_add: "AIzaSyStaleSecret1234567890"
    })
  };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: "Cài đặt" }));
  fireEvent.click(await screen.findByRole("tab", { name: "Dịch thuật" }));
  const backendSelect = await screen.findByLabelText(/Bộ dịch thuật/i);
  fireEvent.change(backendSelect, { target: { value: "openai" } });
  fireEvent.blur(backendSelect);

  await waitFor(() =>
    expect(updateSettings).toHaveBeenCalledWith(expect.objectContaining({
      translation_backend: "openai",
    }))
  );
  expect(updateSettings.mock.calls[0][0]).not.toHaveProperty("gemini_api_key_add");
});

test("auto-saves settings when an input loses focus", async () => {
  const updateSettings = vi.fn().mockResolvedValue({
    translation_backend: "openai",
    tts_backend: "omnivoice",
  });
  const api: JobsApi = {
    ...baseApi,
    updateSettings,
    getSettings: vi.fn().mockResolvedValue({
      translation_backend: "google_free",
      tts_backend: "omnivoice",
    })
  };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: "Cài đặt" }));
  fireEvent.click(await screen.findByRole("tab", { name: "Dịch thuật" }));
  const backendSelect = await screen.findByLabelText(/Bộ dịch thuật/i);
  fireEvent.change(backendSelect, { target: { value: "openai" } });
  fireEvent.blur(backendSelect);

  await waitFor(() =>
    expect(updateSettings).toHaveBeenCalledWith(expect.objectContaining({
      translation_backend: "openai",
    }))
  );
});

test("navigates to Clone Giong tab and lists cloned voices", async () => {
  const clonedVoices = [
    { id: "voice-1", name: "Giọng Anh", wav_filename: "v1.wav", wav_path: "/path/v1.wav", transcript: "xin chào", transcribed: true, created_at: "2026-06-16T12:00:00Z" }
  ];
  const api: JobsApi = {
    ...baseApi,
    listClonedVoices: vi.fn().mockResolvedValue(clonedVoices),
    deleteClonedVoice: vi.fn().mockResolvedValue({ status: "deleted" })
  };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: "Clone Giọng" }));

  expect(await screen.findByText("Giọng Anh")).toBeInTheDocument();
  expect(screen.getByText("OFFLINE CLONE")).toBeInTheDocument();
});

