import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";

import { App } from "../src/renderer/App";

vi.mock("../src/lib/tauri-bridge", async () => {
  const actual = await vi.importActual<typeof import("../src/lib/tauri-bridge")>("../src/lib/tauri-bridge");
  return {
    ...actual,
    waitForBackend: vi.fn().mockResolvedValue("http://127.0.0.1:8765"),
    subscribeBackendEvents: vi.fn().mockReturnValue(() => {}),
    invokeOpenDevtools: vi.fn().mockResolvedValue(undefined),
  };
});

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
  importJob: vi.fn().mockResolvedValue(job),
    bootstrapPyannote: vi.fn().mockResolvedValue({ status: "started" }),

  runtimeStatus: vi.fn().mockResolvedValue(readyRuntime),
  runSmokeTest: vi.fn().mockResolvedValue(readyRuntime),
  startJob: vi.fn().mockResolvedValue({ status: "started" }),
  cancelJob: vi.fn().mockResolvedValue({ status: "cancelled" }),
  deleteJob: vi.fn().mockResolvedValue({ status: "deleted" }),
  selectVideo: vi.fn().mockResolvedValue({ status: "selected" }),
  getCheckpoint: vi.fn().mockResolvedValue(null),
  getSettings: vi.fn().mockResolvedValue({}),
  updateSettings: vi.fn().mockResolvedValue({}),
  getEvents: vi.fn().mockResolvedValue([]),
  listOutputs: vi.fn().mockResolvedValue([]),
  listClonedVoices: vi.fn().mockResolvedValue([]),
  createClonedVoice: vi.fn().mockResolvedValue({ id: "voice-1", name: "Voice 1", wav_filename: "v1.wav", wav_path: "", transcript: "", transcribed: false, created_at: "" }),
  deleteClonedVoice: vi.fn().mockResolvedValue({ status: "deleted" }),
  testClonedVoice: vi.fn().mockResolvedValue(new Blob()),
  previewPresetVoice: vi.fn().mockResolvedValue(new Blob()),
  rerunJob: vi.fn().mockResolvedValue({ status: "queued", job }),
  redubJob: vi.fn().mockResolvedValue({ status: "queued", job }),
  getJobFiles: vi.fn().mockResolvedValue([]),
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

test("shows portable package errors when bundled runtime is incomplete", async () => {
  const bridge = await import("../src/lib/tauri-bridge");
  vi.mocked(bridge.waitForBackend).mockRejectedValueOnce({
    kind: "portable_missing",
    root: "C:/App/resources/portable-runtime",
    missing_items: [
      "models/qwen3-asr (C:/App/resources/portable-runtime/models/qwen3-asr)",
      "tools/ffmpeg (C:/App/resources/portable-runtime/tools/ffmpeg)",
    ],
  });

  render(<App api={baseApi} />);

  expect(await screen.findByText("Portable package is incomplete")).toBeInTheDocument();
  expect(screen.getByText("C:/App/resources/portable-runtime")).toBeInTheDocument();
  expect(screen.getByText(/models\/qwen3-asr/)).toBeInTheDocument();
  expect(screen.getByText(/tools\/ffmpeg/)).toBeInTheDocument();
});

test("shows an actionable backend connection error", async () => {
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockRejectedValue(new Error("Backend unavailable")) };
  render(<App api={api} />);

  expect(await screen.findByText("Backend unavailable")).toBeInTheDocument();
  expect(screen.getByText(/kiểm tra nhật ký hoạt động hoặc cấu hình/i)).toBeInTheDocument();
});

test("creates a job and shows it on the dashboard", async () => {
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([]), createJob: vi.fn().mockResolvedValue(job) };
  render(<App api={api} />);

  fireEvent.change(await screen.findByPlaceholderText(/Dán liên kết video hoặc kênh Douyin/), {
    target: { value: job.source_url }
  });
  fireEvent.click(screen.getByRole("button", { name: "Tạo tiến trình" }));

  await waitFor(() => expect(screen.getByText(job.source_url)).toBeInTheDocument());
  expect(screen.queryByText("Chi tiết tiến trình")).not.toBeInTheDocument();
});

test("job details opens only by click and closes without leaving selection active", async () => {
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([job]) };
  const { container } = render(<App api={api} />);

  fireEvent.click(await screen.findByText(job.source_url));
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

  expect(await screen.findByText(job.source_url)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: `Xóa tiến trình ${job.id}` }));

  await waitFor(() => expect(deleteJob).toHaveBeenCalledWith(job.id));
  await waitFor(() => expect(screen.queryByText(job.source_url)).not.toBeInTheDocument());
});

test("shows runtime checks and reruns the smoke test", async () => {
  const warning = { ...readyRuntime, status: "warning", checks: [{ ...readyRuntime.checks[0], status: "warning", action: "Install the tool under vendor/ or add it to PATH, then retry." }] };
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([]), runtimeStatus: vi.fn().mockResolvedValue(warning), runSmokeTest: vi.fn().mockResolvedValue(readyRuntime) };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: /Môi trường/ }));
  expect(screen.getByText("Install the tool under vendor/ or add it to PATH, then retry.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Chạy thử nghiệm hệ thống" }));

  await waitFor(() => expect(api.runSmokeTest).toHaveBeenCalled());
  expect(await screen.findByText("Đã sẵn sàng")).toBeInTheDocument();
});

test("shows environment setup wizard when runtime is blocked", async () => {
  const blocked = { ...readyRuntime, status: "blocked" };
  const api: JobsApi = { ...baseApi, listJobs: vi.fn().mockResolvedValue([]), runtimeStatus: vi.fn().mockResolvedValue(blocked) };
  render(<App api={api} />);

  expect(await screen.findByText("Thiết lập môi trường ứng dụng")).toBeInTheDocument();
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

  fireEvent.click(await screen.findByRole("button", { name: /Môi trường/ }));

  expect(await screen.findByText("FFmpeg")).toBeInTheDocument();
  expect(screen.getAllByText("Bắt buộc").length).toBeGreaterThan(0);
  expect(screen.getByText("Tuỳ chọn")).toBeInTheDocument();
  expect(screen.getByText("Cảnh báo")).toBeInTheDocument();
  expect(screen.getByText("Thiếu tuỳ chọn")).toBeInTheDocument();
});

test("shows translation and VoxCPM2 settings with browser cookie disclosure", async () => {
  const api: JobsApi = {
    ...baseApi,
    getSettings: vi.fn().mockResolvedValue({
      cookies_browser: "none",
      translation_backend: "google_free",
      tts_backend: "voxcpm",
    })
  };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: "Cài đặt" }));

  expect((await screen.findAllByText("Google Dịch Miễn Phí")).length).toBeGreaterThan(0);
  expect(await screen.findByText(/cookie dùng cho yt-dlp/i)).toBeInTheDocument();
  expect(await screen.findByText("Lồng tiếng VoxCPM2")).toBeInTheDocument();
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
      tts_backend: "voxcpm",
      gemini_api_keys: [{ id: "key-1", masked: "AIza...7890", label: "Studio quota 1" }],
      gemini_translation_model: "gemini-2.5-flash",
    })
  };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: "Cài đặt" }));
  expect(await screen.findByText("Google AI Studio / Khóa API Gemini")).toBeInTheDocument();
  expect(screen.getByText("AIza...7890")).toBeInTheDocument();
  expect(screen.getByDisplayValue("Studio quota 1")).toBeInTheDocument();

  fireEvent.change(screen.getByPlaceholderText(/Dán khóa API Google AI Studio/i), {
    target: { value: "AIzaSyNewSecret" }
  });
  fireEvent.click(screen.getByRole("button", { name: "Thêm khóa Gemini" }));
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

test("saves a pending Gemini key when saving settings", async () => {
  const updateSettings = vi.fn().mockResolvedValue({
    gemini_api_keys: [{ id: "key-1", masked: "AIza...7890", label: "AIza...7890" }]
  });
  const api: JobsApi = {
    ...baseApi,
    updateSettings,
    getSettings: vi.fn().mockResolvedValue({
      cookies_browser: "none",
      translation_backend: "gemini",
      tts_backend: "voxcpm",
      gemini_api_keys: []
    })
  };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: "Cài đặt" }));
  fireEvent.change(screen.getByPlaceholderText(/Dán khóa API Google AI Studio/i), {
    target: { value: "AIzaSySecret1234567890" }
  });
  fireEvent.click(screen.getByRole("button", { name: "Lưu Cài Đặt" }));

  await waitFor(() =>
    expect(updateSettings).toHaveBeenCalledWith(expect.objectContaining({
      gemini_api_key_add: "AIzaSySecret1234567890"
    }))
  );
  expect(await screen.findByText("AIza...7890")).toBeInTheDocument();
  expect(screen.getByPlaceholderText(/Dán khóa API Google AI Studio/i)).toHaveValue("");
});

test("does not resend stale Gemini key command fields from settings", async () => {
  const updateSettings = vi.fn().mockResolvedValue({
    gemini_api_keys: []
  });
  const api: JobsApi = {
    ...baseApi,
    updateSettings,
    getSettings: vi.fn().mockResolvedValue({
      cookies_browser: "none",
      translation_backend: "gemini",
      tts_backend: "voxcpm",
      gemini_api_keys: [],
      gemini_api_key_add: "AIzaSyStaleSecret1234567890"
    })
  };
  render(<App api={api} />);

  fireEvent.click(await screen.findByRole("button", { name: "Cài đặt" }));
  fireEvent.click(screen.getByRole("button", { name: "Lưu Cài Đặt" }));

  await waitFor(() => expect(updateSettings).toHaveBeenCalled());
  expect(updateSettings.mock.calls[0][0]).not.toHaveProperty("gemini_api_key_add");
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

