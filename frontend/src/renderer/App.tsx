import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import {
  Activity,
  CircleAlert,
  Clock3,
  Info,
  Plus,
  Radio,
  RefreshCw,
  Settings2,
  X,
  Play,
  FileVideo,
  FileAudio,
  AlertTriangle,
  CheckCircle2,
  Save,
  ChevronRight,
  Trash2,
  Mic,
  Volume2,
  Upload,
  Link2,
  FolderOpen,
  Server,
} from "lucide-react";

import type { ClonedVoice, Job, JobsApi, OutputItem, ReleaseVramResult, RuntimeCheck, RuntimeReport } from "../shared/contracts";
import { api as defaultApi } from "../shared/api";
import { invokeOpenFolder, invokeRestart, probeBackendHealth, subscribeBackendEvents, waitForBackend, type BackendConnectionState, type BackendStatus, BACKEND_BASE } from "../lib/tauri-bridge";
import "./styles.css";
import "./runtime.css";

const PIPELINE_STEPS = [
  "resolve",
  "download",
  "extract_audio",
  "vad",
  "asr",
  "normalize_segments",
  "translate",
  "tts",
  "duration_repair",
  "mix",
  "render",
  "qc",
] as const;

const PRESET_VOICES = [] as const;

const SETTINGS_TABS = [
  { id: "download", label: "Tải video" },
  { id: "translation", label: "Dịch thuật" },
  { id: "audio", label: "Âm thanh" },
  { id: "tts", label: "Lồng tiếng" },
  { id: "subtitles", label: "Phụ đề" },
] as const;

type SettingsTabId = (typeof SETTINGS_TABS)[number]["id"];

function SettingHint({ text }: { text: string }) {
  return (
    <span className="setting-hint">
      <button
        type="button"
        className="setting-hint__trigger"
        aria-label={text}
        title={text}
      >
        <Info size={14} aria-hidden="true" />
      </button>
      <span className="setting-hint__tooltip" role="tooltip">
        {text}
      </span>
    </span>
  );
}

function SettingsFieldLabel({ label, hint }: { label: string; hint: string }) {
  return (
    <span className="settings-label__title">
      {label}
      <SettingHint text={hint} />
    </span>
  );
}

function SettingsCheckboxField({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="settings-label settings-label--inline">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <SettingsFieldLabel label={label} hint={hint} />
    </label>
  );
}

const TTS_BACKEND_OPTIONS = [
  {
    id: "voxcpm",
    label: "VoxCPM2",
    hint: "Clone giọng offline, cần GPU",
  },
  {
    id: "edge_tts",
    label: "Edge TTS",
    hint: "Microsoft neural, miễn phí, cần mạng",
  },
  {
    id: "google_tts",
    label: "Google TTS",
    hint: "Google Cloud Standard, 4 giọng vi-VN",
  },
  {
    id: "gemini_tts",
    label: "Gemini TTS",
    hint: "Google AI Studio, cần khóa API",
  },
] as const;

const GOOGLE_TTS_VOICES = [
  { id: "vi-VN-Standard-A", name: "Standard A — Nữ" },
  { id: "vi-VN-Standard-B", name: "Standard B — Nam" },
  { id: "vi-VN-Standard-C", name: "Standard C — Nữ" },
  { id: "vi-VN-Standard-D", name: "Standard D — Nam" },
] as const;

const GEMINI_TTS_VOICES = [
  { id: "Zephyr", name: "Zephyr (Sáng)" },
  { id: "Puck", name: "Puck (Vui)" },
  { id: "Charon", name: "Charon (Trầm)" },
  { id: "Kore", name: "Kore (Chắc)" },
  { id: "Fenrir", name: "Fenrir (Sôi nổi)" },
  { id: "Aoede", name: "Aoede (Nhẹ)" },
] as const;

const translateStatus = (status?: string) => {
  if (!status) return "đang kiểm tra";
  switch (status.toLowerCase()) {
    case "ready": return "sẵn sàng";
    case "blocked": return "bị chặn";
    case "warning": return "cảnh báo";
    case "checking": return "đang kiểm tra";
    case "loading": return "đang tải";
    default: return status;
  }
};

const translateJobStatus = (status: string) => {
  switch (status.toLowerCase()) {
    case "queued": return "đang chờ";
    case "idle": return "chờ chạy";
    case "running": return "đang chạy";
    case "waiting_for_selection": return "chờ chọn video";
    case "completed": return "đã hoàn thành";
    case "failed": return "thất bại";
    case "interrupted": return "bị gián đoạn";
    default: return status.replaceAll("_", " ");
  }
};

const translateStepName = (name: string) => {
  switch (name.toLowerCase()) {
    case "resolve": return "Phân tích liên kết";
    case "download": return "Tải video";
    case "extract_audio": return "Tách âm thanh";
    case "vad": return "Nhận diện giọng nói (VAD)";
    case "asr": return "Nhận dạng tiếng Trung (ASR)";
    case "normalize_segments": return "Chuẩn hóa phân đoạn";
    case "translate": return "Dịch thuật";
    case "tts": return "Lồng tiếng tiếng Việt (TTS)";
    case "duration_repair": return "Khớp độ dài âm thanh";
    case "mix": return "Trộn nhạc nền & giọng nói";
    case "render": return "Xuất video thành phẩm";
    case "qc": return "Kiểm định chất lượng (QC)";
    default: return name.replaceAll("_", " ");
  }
};

const translateStepStatus = (status: string) => {
  switch (status.toLowerCase()) {
    case "completed": return "Đã hoàn thành";
    case "running": return "Đang chạy";
    case "failed": return "Thất bại";
    case "pending": return "Chờ xử lý";
    default: return status;
  }
};

const translateCheckStatus = (status: string) => {
  switch (status.toLowerCase()) {
    case "ready":
    case "pass": return "Đạt";
    case "warning": return "Cảnh báo";
    case "fail": return "Không đạt";
    default: return status;
  }
};

const runtimeCheckKindLabel = (check: RuntimeCheck) => (check.required ? "Bắt buộc" : "Tuỳ chọn");

const runtimeCheckStatusLabel = (check: RuntimeCheck) => {
  const status = check.status.toLowerCase();
  const action = check.action.toLowerCase();
  if (status === "ready" || status === "pass") return "Đã sẵn sàng";
  if (status === "fail" || status === "blocked") return "Cần xử lý";
  if (status === "warning" && check.required && action.includes("path")) return "Dùng PATH";
  if (status === "warning" && !check.required) return "Thiếu tuỳ chọn";
  if (status === "warning") return "Cảnh báo";
  return translateCheckStatus(check.status);
};

const runtimeCheckTone = (check: RuntimeCheck) => {
  const status = check.status.toLowerCase();
  if (status === "ready" || status === "pass") return "ready";
  if (status === "fail" || status === "blocked") return "blocked";
  if (!check.required) return "optional";
  return "warning";
};

function RuntimeCheckIcon({ check }: { check: RuntimeCheck }) {
  const status = check.status.toLowerCase();
  if (status === "ready" || status === "pass") return <CheckCircle2 size={19} />;
  if (status === "fail" || status === "blocked") return <CircleAlert size={19} />;
  return <AlertTriangle size={19} />;
}

const isDeletableJob = (job: Job) => ["completed", "failed", "interrupted"].includes(job.status);

function formatJobLabel(job: Job): string {
  if (job.title_vi || job.title) {
    return job.title_vi || job.title || "";
  }
  if (job.source_url.startsWith("import://")) {
    return job.source_url.slice("import://".length);
  }
  return job.source_url;
}

function formatVram(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return "N/A";
  if (value >= 1024) return `${(value / 1024).toFixed(2)} GB`;
  return `${value.toFixed(0)} MB`;
}

function formatSidebarVram(runtime: RuntimeReport | null): string {
  if (!runtime?.gpu?.cuda_supported) return "N/A";
  return `${formatVram(runtime.gpu.used_vram_mb)} / ${formatVram(runtime.gpu.total_vram_mb)}`;
}

function formatSidebarEnvironment(runtime: RuntimeReport | null): "Đủ" | "Thiếu" {
  const status = runtime?.status?.toLowerCase();
  return status === "ready" || status === "warning" ? "Đủ" : "Thiếu";
}

function summarizeVramRelease(result: ReleaseVramResult): string {
  const released = result.released.length ? `Đã dọn: ${result.released.join(", ")}.` : "Không có cache nội bộ nào cần dọn.";
  const terminated = result.terminated_processes.length
    ? `Đã dừng ${result.terminated_processes.length} tiến trình helper GPU.`
    : "Không phát hiện tiến trình helper GPU tồn đọng.";
  const suffix = result.errors.length ? ` Có ${result.errors.length} cảnh báo.` : "";
  return `${released} ${terminated}${suffix}`;
}

export function App({ api = defaultApi }: { api?: JobsApi }) {
  const [activeTab, setActiveTab] = useState<"jobs" | "outputs" | "settings" | "cloning">("jobs");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [videoFiles, setVideoFiles] = useState<File[]>([]);
  const [sourceUrl, setSourceUrl] = useState("");
  const [creatingLinkJob, setCreatingLinkJob] = useState(false);
  const [importingVideo, setImportingVideo] = useState(false);
  const [importProgress, setImportProgress] = useState<string | null>(null);
  const [resolveCp, setResolveCp] = useState<any | null>(null);
  const [ytDlpUpdating, setYtDlpUpdating] = useState(false);
  const [ytDlpNotice, setYtDlpNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<RuntimeReport | null>(null);
  const [runtimeOpen, setRuntimeOpen] = useState(false);
  const [runtimeLoading, setRuntimeLoading] = useState(false);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [testingRuntime, setTestingRuntime] = useState(false);
  const [releasingVram, setReleasingVram] = useState(false);
  const [vramNotice, setVramNotice] = useState<string | null>(null);

  // Tauri-side backend status (Python server managed by the Rust shell)
  const [backend, setBackend] = useState<BackendStatus | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [backendNotice, setBackendNotice] = useState<string | null>(null);
  const [backendConnection, setBackendConnection] = useState<BackendConnectionState>("checking");
  const [backendLastOkAt, setBackendLastOkAt] = useState<number | null>(null);
  const [backendCheckedAt, setBackendCheckedAt] = useState<number | null>(null);
  const recoveringBackendRef = useRef(false);
  const backendPollFailuresRef = useRef(0);

  // Selected job details modal state
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [drawerTab, setDrawerTab] = useState<"steps" | "segments" | "files">("steps");
  const [durationRepairCp, setDurationRepairCp] = useState<any | null>(null);
  const [jobFiles, setJobFiles] = useState<any[]>([]);
  const [rerunModalOpen, setRerunModalOpen] = useState(false);
  const [rerunKeepSteps, setRerunKeepSteps] = useState<string[]>([]);
  const [rerunSubmitting, setRerunSubmitting] = useState(false);

  // Outputs gallery state
  const [outputs, setOutputs] = useState<OutputItem[]>([]);
  const [activeVideoUrl, setActiveVideoUrl] = useState<string | null>(null);

  // Settings form state
  const [settings, setSettings] = useState<Record<string, any>>({
    translation_backend: "google_free",
    translation_source_language: "zh-CN",
    translation_target_language: "vi",
    subtitles_enabled: true,
    subtitle_font_size: 48,
    subtitle_font_color: "#FFFFFF",
    subtitle_background_color: "#000000",
    subtitle_background_opacity: 95,
    subtitle_background_padding: 12,
    subtitle_edge_margin: 80,
    subtitle_position: "bottom",
    gemini_api_keys: [],
    gemini_translation_model: "gemini-2.5-flash",
    openai_api_base: "https://api.openai.com/v1",
    openai_translation_model: "",
    gemini_tts_model: "gemini-2.5-flash-preview-tts",
    gemini_tts_voice: "Zephyr",
    tts_backend: "voxcpm",
    edge_tts_voice: "vi-VN-HoaiMyNeural",
    google_tts_voice: "vi-VN-Standard-A",
    google_tts_speaking_rate: 1,
    voxcpm_clone_mode: "reference",
  });
  const [newGeminiKey, setNewGeminiKey] = useState("");
  const [newGoogleTtsKey, setNewGoogleTtsKey] = useState("");
  const [newOpenAiKey, setNewOpenAiKey] = useState("");
  const [settingsSuccess, setSettingsSuccess] = useState(false);
  const [settingsTab, setSettingsTab] = useState<SettingsTabId>("tts");
  const [ttsPreviewText, setTtsPreviewText] = useState("Xin chào, đây là bản nghe thử giọng đọc tiếng Việt.");
  const [ttsPreviewLoading, setTtsPreviewLoading] = useState(false);
  const [ttsPreviewUrl, setTtsPreviewUrl] = useState<string | null>(null);
  const [edgeTtsVoices, setEdgeTtsVoices] = useState<Array<{ id: string; name: string }>>([]);
  const [openAiModels, setOpenAiModels] = useState<Array<{ id: string; name: string }>>([]);
  const [openAiModelsLoading, setOpenAiModelsLoading] = useState(false);
  const ttsPreviewUrlRef = useRef<string | null>(null);

  // Cloned voices states
  const [clonedVoices, setClonedVoices] = useState<ClonedVoice[]>([]);
  const [voiceName, setVoiceName] = useState("");
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [voiceUploading, setVoiceUploading] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [voiceNotice, setVoiceNotice] = useState<string | null>(null);
  const [testText, setTestText] = useState<Record<string, string>>({});
  const [testSynthesizing, setTestSynthesizing] = useState<Record<string, boolean>>({});
  const [testAudioUrls, setTestAudioUrls] = useState<Record<string, string>>({});
  const selectedClonedVoice = clonedVoices.find((voice) => voice.wav_path === settings.voxcpm_ref_audio);
  const activeTtsBackend = settings.tts_backend ?? "voxcpm";

  const loadEdgeTtsVoices = useCallback(async () => {
    try {
      const voices = await api.listTtsVoices("edge_tts");
      setEdgeTtsVoices(voices);
    } catch {
      setEdgeTtsVoices([
        { id: "vi-VN-HoaiMyNeural", name: "Hoài My (Nữ)" },
        { id: "vi-VN-NamMinhNeural", name: "Nam Minh (Nam)" },
      ]);
    }
  }, [api]);

  const loadOpenAiModels = useCallback(async () => {
    const baseUrl = String(settings.openai_api_base ?? "").trim();
    const pendingKey = newOpenAiKey.trim();
    const hasSavedKey = Boolean(settings.openai_api_key_configured);
    if (!baseUrl || (!pendingKey && !hasSavedKey)) {
      setOpenAiModels([]);
      return;
    }
    setOpenAiModelsLoading(true);
    try {
      const models = await api.listOpenAiModels({
        baseUrl,
        apiKey: pendingKey || undefined,
      });
      setOpenAiModels(models);
      setSettings((current) => {
        if (
          current.openai_translation_model &&
          models.some((model) => model.id === current.openai_translation_model)
        ) {
          return current;
        }
        return { ...current, openai_translation_model: models[0]?.id ?? "" };
      });
    } catch (cause) {
      setOpenAiModels([]);
      setError(cause instanceof Error ? cause.message : "Không thể tải danh sách model");
    } finally {
      setOpenAiModelsLoading(false);
    }
  }, [api, newOpenAiKey, settings.openai_api_base, settings.openai_api_key_configured]);

  useEffect(() => {
    return () => {
      if (ttsPreviewUrlRef.current) {
        URL.revokeObjectURL(ttsPreviewUrlRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (activeTab === "settings" && settingsTab === "tts" && activeTtsBackend === "edge_tts") {
      void loadEdgeTtsVoices();
    }
  }, [activeTab, settingsTab, activeTtsBackend, loadEdgeTtsVoices]);

  useEffect(() => {
    if (activeTab === "settings" && settingsTab === "translation" && settings.translation_backend === "openai") {
      void loadOpenAiModels();
    }
  }, [activeTab, settingsTab, settings.translation_backend, loadOpenAiModels]);

  const recoverBackend = useCallback(async (reason: string) => {
    if (recoveringBackendRef.current) return;
    recoveringBackendRef.current = true;
    setBackendError(null);
    setBackend({ kind: "starting" });
    setBackendConnection("restarting");
    setBackendNotice(reason);
    try {
      await invokeRestart();
      const baseUrl = await waitForBackend({ timeoutMs: 90_000 });
      setBackend({ kind: "ready", base_url: baseUrl });
      setBackendConnection("online");
      setBackendLastOkAt(Date.now());
      setBackendNotice(null);
      backendPollFailuresRef.current = 0;
      const newJobs = await api.listJobs();
      setJobs(newJobs);
      if (selectedJobId) {
        const updated = newJobs.find((j) => j.id === selectedJobId);
        if (updated) {
          setSelectedJob(updated);
          fetchJobCheckpoints(updated.id);
          fetchJobFiles(updated.id);
        }
      }
      void refreshRuntime();
    } catch (cause) {
      if (cause && typeof cause === "object" && "kind" in cause) {
        setBackend(cause as BackendStatus);
        setBackendConnection((cause as BackendStatus).kind === "crashed" ? "offline" : "checking");
      } else {
        setBackend({
          kind: "crashed",
          stderr: cause instanceof Error ? cause.message : String(cause),
        });
        setBackendConnection("offline");
      }
      setBackendNotice(null);
    } finally {
      recoveringBackendRef.current = false;
    }
  }, [api, selectedJobId]);

  // Fetch initial data
  useEffect(() => {
    refreshJobs();
    refreshOutputs();
    loadSettings();
    if (activeTab === "cloning" || activeTab === "settings") {
      refreshClonedVoices();
    }

    // Auto-refresh running jobs every 2 seconds
    const interval = setInterval(() => {
      api.listJobs().then((newJobs) => {
        setJobs(newJobs);
        if (selectedJobId) {
          const updated = newJobs.find((j) => j.id === selectedJobId);
          if (updated) {
            setSelectedJob(updated);
            if (updated.status === "waiting_for_selection" || updated.current_step === "download") {
              fetchResolveCheckpoint(updated.id);
            }
            fetchJobCheckpoints(updated.id);
            fetchJobFiles(updated.id);
          }
        }
      }).catch(() => {
        // Connection recovery is handled by the dedicated backend health poll.
      });
      if (activeTab === "outputs") {
        refreshOutputs();
      }
      if (activeTab === "cloning" || activeTab === "settings") {
        refreshClonedVoices();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [api, selectedJobId, activeTab, recoverBackend]);

  // Tauri bridge: wait for Python backend to be ready, subscribe to crash events
  useEffect(() => {
    waitForBackend({ timeoutMs: 30_000 })
      .then((baseUrl) => {
        setBackend({ kind: "ready", base_url: baseUrl });
        setBackendConnection("online");
        setBackendLastOkAt(Date.now());
        void refreshRuntime();
      })
      .catch((e) => {
        if (e && typeof e === "object" && "kind" in e) {
          setBackend(e as BackendStatus);
        } else {
          setBackendError(String(e));
        }
      });
    const unlisten = subscribeBackendEvents({
      onCrashed: () => {
        void recoverBackend("Backend dừng đột ngột. Đang khởi động lại và tiếp tục job…");
      },
    });
    return () => { unlisten?.(); };
  }, [recoverBackend]);

  // Poll /api/health so we can surface backend drops while the UI stays open.
  useEffect(() => {
    if (!backend || backend.kind === "starting" || backend.kind === "crashed" || backend.kind === "portable_missing") {
      return;
    }
    let cancelled = false;
    const baseUrl = backend?.kind === "ready" ? backend.base_url : BACKEND_BASE;

    const tick = async () => {
      if (recoveringBackendRef.current) {
        return;
      }
      const ok = await probeBackendHealth(baseUrl);
      if (cancelled) {
        return;
      }
      const now = Date.now();
      setBackendCheckedAt(now);
      if (ok) {
        backendPollFailuresRef.current = 0;
        setBackendConnection("online");
        setBackendLastOkAt(now);
        setBackendNotice(null);
        if (backend?.kind !== "ready") {
          setBackend({ kind: "ready", base_url: baseUrl });
        }
        return;
      }
      backendPollFailuresRef.current += 1;
      setBackendConnection((current) => (current === "restarting" ? "restarting" : "offline"));
      if (backendPollFailuresRef.current >= 2) {
        setBackendNotice("Backend không phản hồi. Ứng dụng sẽ không tự tắt backend; hãy dùng nút khởi động lại nếu cần.");
      }
    };

    void tick();
    const id = setInterval(() => { void tick(); }, 2_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [backend, recoverBackend]);

  async function refreshClonedVoices() {
    try {
      setClonedVoices(await api.listClonedVoices());
    } catch (cause) {
      setVoiceError(cause instanceof Error ? cause.message : "Không thể tải danh sách giọng clone");
    }
  }

  async function handleUploadVoice(e: FormEvent) {
    e.preventDefault();
    if (!voiceName || !voiceFile) {
      setVoiceError("Vui lòng nhập tên giọng và chọn tệp âm thanh mẫu (.wav hoặc .mp3).");
      return;
    }
    setVoiceUploading(true);
    setVoiceError(null);
    setVoiceNotice(null);
    try {
      const created = await api.createClonedVoice(voiceName, voiceFile);
      const transcriptLength = (created.transcript ?? "").length;
      setVoiceNotice(
        transcriptLength > 0
          ? `Đã upload và transcript ${transcriptLength} ký tự. Giọng đã sẵn sàng cho ultimate clone.`
          : "Đã upload giọng, nhưng ASR chưa tạo được transcript. Ultimate clone cần file .txt cạnh WAV."
      );
      setVoiceName("");
      setVoiceFile(null);
      const fileInput = document.getElementById("voice-file-input") as HTMLInputElement;
      if (fileInput) fileInput.value = "";
      await refreshClonedVoices();
    } catch (cause) {
      setVoiceError(cause instanceof Error ? cause.message : "Tải lên tệp mẫu thất bại");
    } finally {
      setVoiceUploading(false);
    }
  }

  async function handleDeleteVoice(id: string) {
    if (!confirm("Bạn có chắc chắn muốn xóa giọng nói clone này không?")) return;
    setVoiceError(null);
    try {
      await api.deleteClonedVoice(id);
      await refreshClonedVoices();
    } catch (cause) {
      setVoiceError(cause instanceof Error ? cause.message : "Xóa giọng nói thất bại");
    }
  }

  async function handleTestVoice(id: string) {
    const text = testText[id] || "Chào bạn, đây là thử nghiệm giọng nói clone offline của tôi.";
    setTestSynthesizing(prev => ({ ...prev, [id]: true }));
    setVoiceError(null);
    try {
      const mode = settings.voxcpm_clone_mode === "ultimate" ? "ultimate" : "reference";
      const blob = await api.testClonedVoice(id, text, mode);
      const url = URL.createObjectURL(blob);
      setTestAudioUrls(prev => {
        if (prev[id]) URL.revokeObjectURL(prev[id]);
        return { ...prev, [id]: url };
      });
    } catch (cause) {
      setVoiceError(cause instanceof Error ? cause.message : "Tổng hợp âm thanh thử nghiệm thất bại");
    } finally {
      setTestSynthesizing(prev => ({ ...prev, [id]: false }));
    }
  }

  async function refreshJobs() {
    try {
      setJobs(await api.listJobs());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể tải danh sách tiến trình");
    }
  }

  async function refreshRuntime() {
    setRuntimeLoading(true);
    setRuntimeError(null);
    try {
      const latest = await api.runtimeStatus();
      if (latest.status !== "blocked") {
        setRuntime(latest);
        return latest;
      }
      try {
        const checked = await api.runSmokeTest();
        setRuntime(checked);
        return checked;
      } catch {
        setRuntime(latest);
        return latest;
      }
    } catch (cause) {
      setRuntimeError(cause instanceof Error ? cause.message : "Không thể tải trạng thái môi trường thực thi");
      return null;
    } finally {
      setRuntimeLoading(false);
    }
  }

  async function refreshOutputs() {
    try {
      setOutputs(await api.listOutputs());
    } catch (cause) {
      console.error(cause);
    }
  }

  async function loadSettings() {
    try {
      const data = await api.getSettings();
      setSettings(data);
    } catch (cause) {
      console.error(cause);
    }
  }

  async function fetchJobCheckpoints(jobId: string) {
    try {
      const cp = await api.getCheckpoint(jobId, "duration_repair");
      setDurationRepairCp(cp);
    } catch {
      try {
        const cp = await api.getCheckpoint(jobId, "translate");
        setDurationRepairCp(cp);
      } catch {
        setDurationRepairCp(null);
      }
    }
  }

  async function importVideoJob(event: FormEvent) {
    event.preventDefault();
    if (videoFiles.length === 0) {
      setError("Vui lòng chọn ít nhất một video từ máy tính.");
      return;
    }
    setError(null);
    setImportingVideo(true);
    setImportProgress(null);
    const createdJobs: Job[] = [];
    const failures: string[] = [];
    try {
      for (let index = 0; index < videoFiles.length; index += 1) {
        const file = videoFiles[index];
        setImportProgress(`Đang tải lên ${index + 1}/${videoFiles.length}: ${file.name}`);
        try {
          createdJobs.push(await api.importJob(file));
        } catch (cause) {
          failures.push(
            `${file.name}: ${cause instanceof Error ? cause.message : "Không thể tạo tiến trình"}`,
          );
        }
      }
      if (createdJobs.length > 0) {
        setJobs((current) => [...createdJobs, ...current]);
        setVideoFiles([]);
        const fileInput = document.getElementById("video-file-input") as HTMLInputElement | null;
        if (fileInput) fileInput.value = "";
      }
      if (failures.length > 0) {
        const prefix = createdJobs.length > 0
          ? `Đã tạo ${createdJobs.length} tiến trình. `
          : "";
        setError(`${prefix}Không thể import ${failures.length} file:\n${failures.join("\n")}`);
      }
    } finally {
      setImportingVideo(false);
      setImportProgress(null);
    }
  }

  async function createLinkJob(event: FormEvent) {
    event.preventDefault();
    const trimmed = sourceUrl.trim();
    if (!trimmed) {
      setError("Vui lòng dán liên kết video Douyin hoặc Bilibili.");
      return;
    }
    setError(null);
    setCreatingLinkJob(true);
    try {
      const job = await api.createJob(trimmed);
      setJobs((current) => [job, ...current]);
      setSourceUrl("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể tạo tiến trình từ liên kết");
    } finally {
      setCreatingLinkJob(false);
    }
  }

  async function fetchResolveCheckpoint(jobId: string) {
    try {
      const checkpoint = await api.getCheckpoint(jobId, "resolve");
      setResolveCp(checkpoint);
    } catch {
      setResolveCp(null);
    }
  }

  async function handleSelectPlaylistVideo(jobId: string, index: number) {
    try {
      await api.selectVideo(jobId, index);
      setResolveCp(null);
      await refreshJobs();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể chọn video");
    }
  }

  async function handleUpdateYtDlp() {
    setYtDlpUpdating(true);
    setYtDlpNotice(null);
    try {
      const result = await api.updateYtDlp();
      setYtDlpNotice(`Đã cập nhật yt-dlp: ${result.previous_version} → ${result.version}`);
      await refreshRuntime();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể cập nhật yt-dlp");
    } finally {
      setYtDlpUpdating(false);
    }
  }

  async function runSmokeTest() {
    setTestingRuntime(true);
    setRuntimeError(null);
    try {
      setRuntime(await api.runSmokeTest());
      setVramNotice(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Thử nghiệm hệ thống thất bại");
    } finally {
      setTestingRuntime(false);
    }
  }

  function openRuntimePanel() {
    setRuntimeOpen(true);
    if (!runtime && !runtimeLoading) {
      void refreshRuntime();
    }
  }

  async function handleReleaseVram() {
    setReleasingVram(true);
    setError(null);
    try {
      const result = await api.releaseVram();
      setVramNotice(summarizeVramRelease(result));
      await refreshRuntime();
      await refreshJobs();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể giải phóng VRAM");
    } finally {
      setReleasingVram(false);
    }
  }

  async function fetchJobFiles(jobId: string) {
    try {
      const files = await api.getJobFiles(jobId);
      setJobFiles(files);
    } catch {
      setJobFiles([]);
    }
  }

  async function handleSelectJob(job: Job) {
    setSelectedJobId(job.id);
    setSelectedJob(job);
    setResolveCp(null);
    setDurationRepairCp(null);
    setJobFiles([]);
    setDrawerTab("steps");
    fetchResolveCheckpoint(job.id);
    fetchJobCheckpoints(job.id);
    fetchJobFiles(job.id);
  }

  function closeSelectedJob() {
    setSelectedJobId(null);
    setSelectedJob(null);
    setResolveCp(null);
    setDurationRepairCp(null);
    setJobFiles([]);
    closeRerunModal();
  }

  function getDefaultKeepSteps(job: Job): string[] {
    const completed = new Set(
      job.steps.filter((step) => step.status === "completed").map((step) => step.name),
    );
    const kept: string[] = [];
    for (const step of PIPELINE_STEPS) {
      if (completed.has(step)) {
        kept.push(step);
      } else {
        break;
      }
    }
    if (kept.length === PIPELINE_STEPS.length) {
      const translateIndex = PIPELINE_STEPS.indexOf("translate");
      return PIPELINE_STEPS.slice(0, translateIndex + 1);
    }
    return kept;
  }

  function openRerunModal(job: Job) {
    setRerunKeepSteps(getDefaultKeepSteps(job));
    setRerunModalOpen(true);
  }

  function closeRerunModal() {
    setRerunModalOpen(false);
    setRerunKeepSteps([]);
  }

  function handleRerunBoundarySelect(firstRerunIndex: number) {
    setRerunKeepSteps(PIPELINE_STEPS.slice(0, firstRerunIndex));
  }

  async function handleRerunJob() {
    if (!selectedJob) return;
    if (rerunKeepSteps.length >= PIPELINE_STEPS.length) {
      setError("Hãy bỏ chọn ít nhất một bước để chạy lại.");
      return;
    }

    setRerunSubmitting(true);
    setError(null);
    try {
      await api.rerunJob(selectedJob.id, rerunKeepSteps);
      closeRerunModal();
      refreshJobs();
      closeSelectedJob();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Chạy lại thất bại");
    } finally {
      setRerunSubmitting(false);
    }
  }

  async function startJob(jobId: string) {
    try {
      await api.startJob(jobId);
      refreshJobs();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể bắt đầu tiến trình");
    }
  }

  async function cancelJob(jobId: string) {
    try {
      await api.cancelJob(jobId);
      refreshJobs();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể hủy tiến trình");
    }
  }

  async function deleteJob(jobId: string) {
    try {
      await api.deleteJob(jobId);
      setJobs((current) => current.filter((job) => job.id !== jobId));
      if (selectedJobId === jobId) {
        closeSelectedJob();
      }
      refreshJobs();
      refreshOutputs();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể xóa tiến trình");
    }
  }

  async function openJobFolder(jobId: string) {
    try {
      const folder = await api.getJobFolder(jobId);
      if (!folder.exists) {
        setError("Thư mục tiến trình chưa được tạo trên máy.");
        return;
      }
      await invokeOpenFolder(folder.path);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể mở thư mục tiến trình");
    }
  }

  async function handleSaveSettings(event: FormEvent) {
    event.preventDefault();
    setSettingsSuccess(false);
    try {
      const {
        gemini_api_keys,
        gemini_api_key_add,
        gemini_api_key_remove,
        gemini_api_key_update,
        ...savePayload
      } = settings;
      const pendingGeminiKey = newGeminiKey.trim();
      const pendingGoogleTtsKey = newGoogleTtsKey.trim();
      const pendingOpenAiKey = newOpenAiKey.trim();
      if (pendingGeminiKey) {
        savePayload.gemini_api_key_add = pendingGeminiKey;
      }
      if (pendingGoogleTtsKey) {
        savePayload.google_tts_api_key = pendingGoogleTtsKey;
      }
      if (pendingOpenAiKey) {
        savePayload.openai_api_key = pendingOpenAiKey;
      }
      const updated = await api.updateSettings(savePayload);
      setSettings(updated);
      if (pendingGeminiKey) {
        setNewGeminiKey("");
      }
      if (pendingGoogleTtsKey) {
        setNewGoogleTtsKey("");
      }
      if (pendingOpenAiKey) {
        setNewOpenAiKey("");
      }
      setSettingsSuccess(true);
      setTimeout(() => setSettingsSuccess(false), 3000);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể lưu cài đặt");
    }
  }

  async function handleAddGeminiKey() {
    const key = newGeminiKey.trim();
    if (!key) return;
    setSettingsSuccess(false);
    try {
      const updated = await api.updateSettings({ gemini_api_key_add: key });
      setSettings(updated);
      setNewGeminiKey("");
      setSettingsSuccess(true);
      setTimeout(() => setSettingsSuccess(false), 3000);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể thêm khóa API Gemini");
    }
  }

  async function handleRemoveGeminiKey(id: string) {
    setSettingsSuccess(false);
    try {
      const updated = await api.updateSettings({ gemini_api_key_remove: id });
      setSettings(updated);
      setSettingsSuccess(true);
      setTimeout(() => setSettingsSuccess(false), 3000);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể xóa khóa API Gemini");
    }
  }

  function updateGeminiKeyLabel(id: string, label: string) {
    setSettings({
      ...settings,
      gemini_api_keys: (settings.gemini_api_keys ?? []).map((item: any) =>
        item.id === id ? { ...item, label } : item
      )
    });
  }

  async function handleSaveGeminiKeyLabel(id: string, label: string) {
    setSettingsSuccess(false);
    try {
      const updated = await api.updateSettings({
        gemini_api_key_update: { id, label }
      });
      setSettings(updated);
      setSettingsSuccess(true);
      setTimeout(() => setSettingsSuccess(false), 3000);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể cập nhật nhãn khóa API Gemini");
    }
  }

  async function handlePreviewTts() {
    const text = ttsPreviewText.trim();
    if (!text) {
      setError("Nhập nội dung để nghe thử giọng đọc.");
      return;
    }
    setTtsPreviewLoading(true);
    setError(null);
    try {
      const backend = activeTtsBackend;
      const voice =
        backend === "edge_tts"
          ? settings.edge_tts_voice
          : backend === "google_tts"
            ? settings.google_tts_voice
          : backend === "gemini_tts"
            ? settings.gemini_tts_voice
            : undefined;
      const blob = await api.previewTts(text, {
        backend,
        voice,
        settings: {
          tts_backend: backend,
          edge_tts_voice: settings.edge_tts_voice,
          google_tts_voice: settings.google_tts_voice,
          google_tts_speaking_rate: settings.google_tts_speaking_rate,
          google_tts_api_key: newGoogleTtsKey.trim() || undefined,
          gemini_tts_model: settings.gemini_tts_model,
          gemini_tts_voice: settings.gemini_tts_voice,
          voxcpm_ref_audio: settings.voxcpm_ref_audio,
          voxcpm_instruct: settings.voxcpm_instruct,
          voxcpm_clone_mode: settings.voxcpm_clone_mode,
          voxcpm_auto_voice: settings.voxcpm_auto_voice,
        },
      });
      if (ttsPreviewUrlRef.current) {
        URL.revokeObjectURL(ttsPreviewUrlRef.current);
      }
      const url = URL.createObjectURL(blob);
      ttsPreviewUrlRef.current = url;
      setTtsPreviewUrl(url);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể nghe thử giọng đọc");
    } finally {
      setTtsPreviewLoading(false);
    }
  }

  function formatDuration(sec?: number): string {
    if (sec === undefined || sec === null) return "--:--";
    const minutes = Math.floor(sec / 60);
    const seconds = Math.floor(sec % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  }

  function formatStepDuration(ms?: number | null, startedAt?: string | null, status?: string): string {
    const elapsedMs = status === "running" && startedAt ? Date.now() - Date.parse(startedAt) : null;
    const value = ms ?? (elapsedMs !== null && Number.isFinite(elapsedMs) && elapsedMs >= 0 ? elapsedMs : null);
    if (value === null) return "--";
    if (value < 1000) return `${Math.round(value)} ms`;
    return `${(value / 1000).toFixed(value < 10_000 ? 1 : 0)} s`;
  }

  function formatBytes(bytes?: number): string {
    if (!bytes) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function formatBackendClock(timestamp: number | null): string {
    if (!timestamp) return "chưa có";
    return new Date(timestamp).toLocaleTimeString("vi-VN", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  function backendConnectionLabel(state: BackendConnectionState): string {
    switch (state) {
      case "online":
        return "Đang chạy";
      case "offline":
        return "Không phản hồi";
      case "restarting":
        return "Đang khởi động lại";
      default:
        return "Đang kiểm tra";
    }
  }

  function renderBackendBanner() {
    if (backendConnection !== "offline" && backendConnection !== "restarting") {
      return null;
    }
    return (
      <div className={`backend-banner backend-banner--${backendConnection}`}>
        <AlertTriangle size={20} />
        <div>
          <strong>
            {backendConnection === "restarting"
              ? "Backend đang được khởi động lại"
              : "Backend không phản hồi"}
          </strong>
          <span>
            {backendNotice
              ?? (backendConnection === "restarting"
                ? "Tiến trình Python có thể đã dừng giữa chừng. Đang thử kết nối lại…"
                : "Job có thể bị treo nếu backend đã dừng. Kiểm tra lần cuối lúc "
                  + formatBackendClock(backendCheckedAt) + ".")}
          </span>
        </div>
        {backendConnection === "offline" && (
          <button type="button" onClick={() => void recoverBackend("Đang khởi động lại backend…")}>
            Khởi động lại
          </button>
        )}
      </div>
    );
  }

  if (backend?.kind === "portable_missing") {
    return (
      <div className="p-6 max-w-3xl mx-auto text-zinc-100">
        <h1 className="text-xl font-semibold text-red-400">Portable package is incomplete</h1>
        <p className="mt-2 text-zinc-300">The app could not find required bundled runtime files.</p>
        <div className="mt-4 rounded bg-zinc-900 p-3">
          <strong>Runtime path</strong>
          <code className="block mt-1 text-sm text-zinc-300">{backend.root}</code>
        </div>
        <ul className="mt-4 list-disc pl-6 text-sm text-red-200">
          {backend.missing_items.map((item) => <li key={item}>{item}</li>)}
        </ul>
        <button onClick={() => location.reload()} className="mt-4 underline">Retry after fixing the portable folder</button>
      </div>
    );
  }

  if (backend?.kind === "crashed") {
    return (
      <div className="p-6 max-w-3xl mx-auto text-zinc-100">
        <h1 className="text-xl font-semibold text-red-400">Backend dừng đột ngột</h1>
        <p className="mt-2 text-sm text-zinc-300">
          Cửa sổ app vẫn mở nhưng tiến trình xử lý Python có thể đã bị dừng (GPU, bộ nhớ, hoặc lỗi hệ thống).
          Bạn không cần đóng app — nhấn nút bên dưới để khởi động lại và tiếp tục job từ checkpoint.
        </p>
        <pre className="mt-3 p-3 bg-zinc-900 text-zinc-100 text-sm overflow-auto rounded">
          {backend.stderr || "(no stderr captured)"}
        </pre>
        <button
          type="button"
          onClick={() => void recoverBackend("Đang khởi động lại backend…")}
          className="mt-4 underline"
        >
          Khởi động lại backend
        </button>
      </div>
    );
  }

  if (backendError) {
    return (
      <div className="p-6 text-red-600">
        Could not reach backend: {backendError}
        <button onClick={() => location.reload()} className="ml-3 underline">Retry</button>
      </div>
    );
  }

  if (!backend || backend.kind === "starting") {
    return (
      <div className="p-6 text-zinc-600">
        {backendNotice ?? "Starting backend…"}
      </div>
    );
  }

  return (
    <>
      <div className="shell">
      <aside>
        <div className="brand">
          <span>DV</span>
          <div>
            <strong>Douyin</strong>
            <br />
            <small style={{ color: "#8170ff", fontWeight: 600 }}>Vietnamizer</small>
          </div>
        </div>
        <nav>
          <button className={activeTab === "jobs" ? "active" : ""} onClick={() => setActiveTab("jobs")}>
            <Activity size={18} /> Tiến trình
          </button>
          <button className={activeTab === "cloning" ? "active" : ""} onClick={() => setActiveTab("cloning")}>
            <Mic size={18} /> Clone Giọng
          </button>
          <button className={activeTab === "outputs" ? "active" : ""} onClick={() => setActiveTab("outputs")}>
            <Radio size={18} /> Thành phẩm
          </button>
          <button className={activeTab === "settings" ? "active" : ""} onClick={() => setActiveTab("settings")}>
            <Settings2 size={18} /> Cài đặt
          </button>
        </nav>
        <div className="sidebar-foot">
          <div
            className={`backend-status backend-status--${backendConnection}`}
            title={
              backendConnection === "online"
                ? `Backend phản hồi ổn định. Lần kiểm tra gần nhất: ${formatBackendClock(backendCheckedAt)}`
                : `Backend ${backendConnectionLabel(backendConnection).toLowerCase()}`
            }
          >
            <i />
            <div>
              <strong>
                <Server size={14} style={{ display: "inline", marginRight: 6, verticalAlign: "-2px" }} />
                Backend: {backendConnectionLabel(backendConnection)}
              </strong>
              <small>
                {backendConnection === "online"
                  ? `OK lúc ${formatBackendClock(backendLastOkAt)}`
                  : backendConnection === "restarting"
                    ? (backendNotice ?? "Đang kết nối lại…")
                    : `Mất kết nối · kiểm tra ${formatBackendClock(backendCheckedAt)}`}
              </small>
            </div>
          </div>
          <button
            className={`runtime ${runtime?.status ?? (runtimeError ? "warning" : "loading")}`}
            onClick={openRuntimePanel}
          >
            <i />
            <div>
              <strong>Môi trường: {formatSidebarEnvironment(runtime)}</strong>
              <small>VRAM {formatSidebarVram(runtime)}</small>
            </div>
          </button>
        </div>
      </aside>

      <main>
        {renderBackendBanner()}
        {activeTab === "jobs" && (
          <>
            <header>
              <div>
                <h1>Bảng điều khiển tiến trình</h1>
              </div>
              <span className="phase">Tiến trình hoạt động</span>
            </header>

            {error && (
              <div className="error">
                <CircleAlert size={22} />
                <div>
                  <strong>{error}</strong>
                  <span>Vui lòng kiểm tra nhật ký hoạt động hoặc cấu hình, sau đó thử lại.</span>
                </div>
                <button onClick={() => setError(null)} style={{ background: "transparent", color: "inherit", marginLeft: "auto" }}>
                  <X size={18} />
                </button>
              </div>
            )}

            <section className="new-job">
              <div>
                <h2>Tạo tiến trình lồng tiếng mới</h2>
                <p>Tải từ liên kết Douyin/Bilibili hoặc chọn video local — nhiều file local sẽ được xếp hàng.</p>
              </div>

              <form onSubmit={createLinkJob} className="link-job-form">
                <label>
                  <Link2 size={18} />
                  <input
                    aria-label="Dán liên kết video Douyin hoặc Bilibili"
                    value={sourceUrl}
                    onChange={(event) => setSourceUrl(event.target.value)}
                    placeholder="https://www.douyin.com/video/... hoặc https://www.bilibili.com/video/..."
                  />
                </label>
                <button type="submit" disabled={runtime?.status === "blocked" || creatingLinkJob || !sourceUrl.trim()}>
                  <Link2 size={18} /> {creatingLinkJob ? "Đang tạo..." : "Tạo từ liên kết"}
                </button>
              </form>

              <form onSubmit={importVideoJob} className="import-job">
                <label>
                  <FileVideo size={18} />
                  <input
                    id="video-file-input"
                    aria-label="Chọn video từ máy tính"
                    multiple
                    type="file"
                    accept=".mp4,.mov,.m4v,.mkv,.webm,.flv,.avi,.ts,.mp3,.wav,.m4a,.ogg,.opus,video/*,audio/*"
                    onChange={(event) => {
                      const files = Array.from(event.target.files ?? []);
                      setVideoFiles(files);
                    }}
                  />
                </label>
                {videoFiles.length > 0 && (
                  <p className="import-file-count">
                    Đã chọn {videoFiles.length} file: {videoFiles.map((file) => file.name).join(", ")}
                  </p>
                )}
                {importProgress && (
                  <p className="import-file-count">{importProgress}</p>
                )}
                <button type="submit" disabled={runtime?.status === "blocked" || importingVideo || videoFiles.length === 0}>
                  <Upload size={18} />{" "}
                  {importingVideo
                    ? "Đang tải lên..."
                    : videoFiles.length > 1
                      ? `Tạo ${videoFiles.length} tiến trình`
                      : "Tạo từ file local"}
                </button>
              </form>
            </section>

            <section className="jobs">
              <div className="section-title">
                <h2>Tiến trình gần đây</h2>
                <span>Tổng cộng {jobs.length}</span>
              </div>
              {jobs.length === 0 && !error && (
                <div className="empty">
                  <Clock3 size={32} />
                  <h3>Chưa có tiến trình nào</h3>
                  <p>Chọn video local hoặc dán liên kết Douyin/Bilibili ở trên để bắt đầu.</p>
                </div>
              )}
              <div style={{ display: "grid", gap: "12px" }}>
                {jobs.map((job) => {
                  const completedSteps = job.steps.filter((s) => s.status === "completed").length;
                  return (
                    <article
                      key={job.id}
                      onClick={() => handleSelectJob(job)}
                      style={{ cursor: "pointer", transition: "border-color 0.2s" }}
                      className={`job-card ${selectedJobId === job.id ? "selected-card" : ""}`}
                    >
                      <div className="job-top">
                        <div style={{ flex: 1, marginRight: "12px" }}>
                          <span className={`status ${job.status}`}>{translateJobStatus(job.status)}</span>
                          <h3 style={{ margin: "8px 0 4px", fontSize: "16px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {formatJobLabel(job)}
                          </h3>
                          {job.title_vi && job.title && job.title_vi !== job.title && (
                            <small style={{ color: "#747d90", display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              {job.title}
                            </small>
                          )}
                          {job.last_error_code && job.status === "failed" && (
                            <small style={{ color: "#f16f7e", display: "block", marginTop: "6px" }}>
                              {job.last_error_code}: {job.last_error_message}
                            </small>
                          )}
                          <small style={{ fontFamily: "monospace", color: "#626b7d" }}>{job.id}</small>
                        </div>
                        <div style={{ textAlign: "right" }}>
                          <b>{completedSteps} / {job.steps.length} bước</b>
                          <br />
                          <small style={{ color: "#747d90" }}>{new Date(job.created_at).toLocaleTimeString()}</small>
                          {isDeletableJob(job) && (
                            <button
                              aria-label={`Xóa tiến trình ${job.id}`}
                              className="danger-button"
                              style={{ marginTop: "8px", padding: "7px 10px" }}
                              onClick={(event) => {
                                event.stopPropagation();
                                deleteJob(job.id);
                              }}
                            >
                              <Trash2 size={14} /> Xóa
                            </button>
                          )}
                        </div>
                      </div>
                      <div className="timeline">
                        {job.steps.map((step) => (
                          <div key={step.name} title={`${translateStepName(step.name)}: ${translateStepStatus(step.status)}`} className={step.status} />
                        ))}
                      </div>
                    </article>
                  );
                })}
              </div>
            </section>
          </>
        )}

        {activeTab === "cloning" && (
          <>
            <header className="settings-header">
              <div>
                <h1>Quản lý giọng nói nhân bản (Voice Cloning)</h1>
                <p className="settings-subtitle">Thêm giọng từ .wav hoặc .mp3; backend tự transcript ra file .txt cạnh WAV để dùng ultimate clone.</p>
              </div>
            </header>

            {voiceError && (
              <div className="error" style={{ marginBottom: "20px" }}>
                <CircleAlert size={22} />
                <div>
                  <strong>Lỗi quản lý giọng đọc</strong>
                  <span>{voiceError}</span>
                </div>
                <button onClick={() => setVoiceError(null)} style={{ background: "transparent", color: "inherit", marginLeft: "auto" }}>
                  <X size={18} />
                </button>
              </div>
            )}
            {voiceNotice && (
              <div className="success" style={{ marginBottom: "20px" }}>
                <CheckCircle2 size={18} />
                <span>{voiceNotice}</span>
              </div>
            )}

            <div className="settings-page-layout">
              {/* Cột Trái: Tải lên giọng đọc mẫu */}
              <div className="settings-column">
                <section className="settings-card">
                  <div className="card-header-accent">
                    <span className="accent-bar"></span>
                    <h3>Thêm Giọng Nhân Bản Mới</h3>
                  </div>
                  <p className="card-description">
                    Tải lên một tệp âm thanh mẫu (.wav hoặc .mp3, dài 3-10 giây). Backend sẽ chuyển MP3 sang WAV và tự tạo transcript .txt cạnh file audio.
                  </p>
                  
                  <form onSubmit={handleUploadVoice} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                    <label className="settings-label">
                      <span>Tên giọng nói (ví dụ: Giọng Anh Minh)</span>
                      <input
                        required
                        className="settings-input"
                        placeholder="Nhập tên giọng..."
                        value={voiceName}
                        onChange={(e) => setVoiceName(e.target.value)}
                      />
                    </label>

                    <label className="settings-label">
                      <span>Chọn tệp mẫu .wav hoặc .mp3</span>
                      <input
                        id="voice-file-input"
                        required
                        type="file"
                        accept=".wav,.mp3,audio/wav,audio/x-wav,audio/mpeg,audio/mp3"
                        className="settings-input"
                        onChange={(e) => setVoiceFile(e.target.files?.[0] || null)}
                        style={{ padding: "10px" }}
                      />
                    </label>

                    <button
                      type="submit"
                      disabled={voiceUploading}
                      className="gradient-button"
                      style={{ justifyContent: "center", marginTop: "10px" }}
                    >
                      <Upload size={18} /> {voiceUploading ? "Đang tải lên..." : "Tải lên & Lưu giọng đọc"}
                    </button>
                  </form>
                </section>
              </div>

              {/* Cột Phải: Danh sách giọng đọc & Thử nghiệm */}
              <div className="settings-column">
                <section className="settings-card">
                  <div className="card-header-accent">
                    <span className="accent-bar"></span>
                    <h3>Danh sách giọng clone</h3>
                  </div>
                  <p className="card-description">Danh sách các giọng đọc đã được đăng ký và sẵn sàng sử dụng.</p>

                  <div className="gemini-keys-list" style={{ maxHeight: "650px", overflowY: "auto" }}>
                    {clonedVoices.length === 0 ? (
                      <div className="empty-keys-placeholder">Chưa có giọng nhân bản nào. Hãy tải lên một tệp mẫu ở cột bên trái.</div>
                    ) : (
                      clonedVoices.map((voice) => (
                        <div key={voice.id} className="gemini-key-card" style={{ gap: "12px" }}>
                          <div className="key-card-header">
                            <div className="key-info">
                              <span className="key-badge">OFFLINE CLONE</span>
                              <code className="key-masked" style={{ fontWeight: 600, color: "#fff" }}>{voice.name}</code>
                            </div>
                            <button
                              type="button"
                              onClick={() => handleDeleteVoice(voice.id)}
                              className="key-action-btn delete-btn"
                            >
                              <Trash2 size={13} /> Xóa
                            </button>
                          </div>

                          <div className="key-card-body" style={{ display: "flex", flexDirection: "column", gap: "12px", borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: "12px" }}>
                            <div className={`alert-info-box ${voice.transcribed ? "info" : "warning"}`}>
                              {voice.transcribed ? <CheckCircle2 size={14} /> : <CircleAlert size={14} />}
                              <span>
                                {voice.transcribed
                                  ? `Transcript sẵn sàng (${(voice.transcript ?? "").length} ký tự) cho ultimate clone.`
                                  : "Chưa có transcript .txt; ultimate clone sẽ cần transcript chính xác."}
                              </span>
                            </div>

                            {/* Play reference audio */}
                            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                              <span style={{ fontSize: "12px", color: "#8b949e" }}>Âm thanh mẫu:</span>
                              <audio
                                controls
                                src={`http://127.0.0.1:8765/api/cloned-voices/${voice.id}/wav`}
                                style={{ height: "30px", width: "100%", maxWidth: "300px" }}
                              />
                            </div>

                            {/* Test synthesis */}
                            <div style={{ display: "flex", flexDirection: "column", gap: "8px", background: "rgba(255,255,255,0.015)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: "8px", padding: "10px" }}>
                              <span style={{ fontSize: "11px", color: "#a79aff", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.03em" }}>Chạy thử giọng (Synthesize Test)</span>
                              <div style={{ display: "flex", gap: "8px" }}>
                                <input
                                  className="settings-input"
                                  placeholder="Nhập nội dung cần đọc thử..."
                                  style={{ height: "32px", fontSize: "13px", padding: "6px 10px" }}
                                  value={testText[voice.id] ?? ""}
                                  onChange={(e) => setTestText(prev => ({ ...prev, [voice.id]: e.target.value }))}
                                />
                                <button
                                  type="button"
                                  disabled={testSynthesizing[voice.id]}
                                  onClick={() => handleTestVoice(voice.id)}
                                  className="key-action-btn save-btn"
                                  style={{ height: "32px", whiteSpace: "nowrap" }}
                                >
                                  {testSynthesizing[voice.id] ? "Đang tạo..." : "Thử giọng"}
                                </button>
                              </div>

                              {testAudioUrls[voice.id] && (
                                <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "6px", borderTop: "1px dashed rgba(255,255,255,0.05)", paddingTop: "6px" }}>
                                  <Volume2 size={14} style={{ color: "#5bdd9a" }} />
                                  <span style={{ fontSize: "12px", color: "#5bdd9a" }}>Kết quả:</span>
                                  <audio
                                    controls
                                    autoPlay
                                    src={testAudioUrls[voice.id]}
                                    style={{ height: "28px", flex: 1 }}
                                  />
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </section>
              </div>
            </div>
          </>
        )}

        {activeTab === "outputs" && (
          <>
            <header>
              <div>
                <h1>Thư viện thành phẩm</h1>
              </div>
            </header>
            <section style={{ marginTop: "24px" }}>
              {outputs.length === 0 ? (
                <div className="empty">
                  <FileVideo size={40} />
                  <h3>Chưa có thành phẩm nào</h3>
                  <p>Video lồng tiếng sẽ xuất hiện ở đây sau khi tiến trình hoàn thành.</p>
                </div>
              ) : (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: "20px" }}>
                  {outputs.map((out) => (
                    <div key={out.job_id} className="output-card" style={{ background: "#12151c", border: "1px solid #292f3b", borderRadius: "14px", padding: "18px", display: "flex", flexDirection: "column" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
                        <FileVideo size={28} style={{ color: "#8170ff" }} />
                        <div style={{ overflow: "hidden" }}>
                          <h3 style={{ margin: 0, fontSize: "15px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                            {out.title_vi || out.title}
                          </h3>
                          {out.title_vi && out.title_vi !== out.title && (
                            <small style={{ color: "#747d90", display: "block", marginTop: "4px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                              {out.title}
                            </small>
                          )}
                          <small style={{ color: "#626b7d", display: "block", marginTop: "4px" }}>{formatBytes(out.file_size)}</small>
                        </div>
                      </div>
                      <p style={{ color: "#8f97a6", fontSize: "13px", margin: "0 0 16px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        Nguồn: {out.source_url.startsWith("import://")
                          ? out.source_url.slice("import://".length)
                          : <a href={out.source_url} target="_blank" rel="noreferrer" style={{ color: "#8170ff" }}>{out.source_url}</a>}
                      </p>
                      <button
                        className="smoke-button"
                        style={{ marginTop: "auto", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}
                        onClick={() => setActiveVideoUrl(`http://127.0.0.1:8765/api/jobs/${out.job_id}/output`)}
                      >
                        <Play size={16} /> Phát Video Lồng Tiếng
                      </button>
                      <button
                        className="smoke-button"
                        style={{ marginTop: "8px", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", background: "#20242e" }}
                        onClick={() => openJobFolder(out.job_id)}
                      >
                        <FolderOpen size={16} /> Mở thư mục
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </>
        )}

        {activeTab === "settings" && (
          <>
            <header className="settings-header settings-header-compact">
              <div>
                <h1>Cài đặt ứng dụng</h1>
                <p className="settings-subtitle">
                  Cấu hình theo từng nhóm: tải video, dịch thuật, xử lý âm thanh, lồng tiếng và phụ đề.
                </p>
              </div>
            </header>
            <form onSubmit={handleSaveSettings} className="settings-page-layout settings-page-layout--tabbed">
              <div className="settings-shell">
                <div className="settings-toolbar">
                  <div className="settings-tabs settings-tabs--grow" role="tablist" aria-label="Nhóm cài đặt">
                    {SETTINGS_TABS.map((tab) => (
                      <button
                        key={tab.id}
                        type="button"
                        role="tab"
                        aria-selected={settingsTab === tab.id}
                        className={`settings-tab-btn${settingsTab === tab.id ? " active" : ""}`}
                        onClick={() => setSettingsTab(tab.id)}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </div>
                  <div className="settings-save-slot">
                    <button type="submit" className="save-settings-button">
                      <Save size={18} /> Lưu cài đặt
                    </button>
                    {settingsSuccess && (
                      <span className="save-success-badge">
                        <CheckCircle2 size={18} /> Đã lưu
                      </span>
                    )}
                  </div>
                </div>

                {settingsTab === "download" && (
                  <div className="settings-tab-panel">
                    <section className="settings-card settings-card--full">
                      <div className="card-header-accent">
                        <span className="accent-bar"></span>
                        <h3>Tải video (Douyin / Bilibili)</h3>
                      </div>
                      <p className="card-description">
                        Tự lấy cookie từ Chrome, rồi lần lượt thử Edge → Firefox → Brave nếu cần.
                        Đăng nhập Douyin/Bilibili trên Chrome trước khi tạo job từ liên kết.
                      </p>
                      <button
                        type="button"
                        className="gradient-button"
                        disabled={ytDlpUpdating}
                        onClick={() => void handleUpdateYtDlp()}
                        style={{ justifyContent: "center", maxWidth: "320px" }}
                      >
                        <RefreshCw size={16} /> {ytDlpUpdating ? "Đang cập nhật yt-dlp..." : "Cập nhật yt-dlp"}
                      </button>
                      {ytDlpNotice && (
                        <p className="import-file-count" style={{ marginTop: "10px", color: "#9be7a8" }}>{ytDlpNotice}</p>
                      )}
                    </section>
                  </div>
                )}

                {settingsTab === "translation" && (
                  <div className="settings-tab-panel">
                    <section className="settings-card">
                      <div className="card-header-accent">
                        <span className="accent-bar"></span>
                        <h3>Dịch thuật</h3>
                      </div>
                      <p className="card-description card-description--compact">
                        Chọn bộ dịch cho phụ đề và nội dung lồng tiếng.
                      </p>
                      <label className="settings-label">
                        <span>Bộ dịch thuật</span>
                        <select
                          className="settings-input"
                          value={settings.translation_backend ?? "google_free"}
                          onChange={(e) => setSettings({ ...settings, translation_backend: e.target.value })}
                        >
                          <option value="google_free">Google Dịch Miễn Phí</option>
                          <option value="gemini">Gemini</option>
                          <option value="openai">OpenAPI (tương thích OpenAI)</option>
                        </select>
                      </label>
                      {settings.translation_backend === "gemini" && (
                        <label className="settings-label">
                          <span>Mô hình dịch Gemini</span>
                          <input
                            className="settings-input"
                            value={settings.gemini_translation_model ?? "gemini-2.5-flash"}
                            onChange={(e) => setSettings({ ...settings, gemini_translation_model: e.target.value })}
                          />
                        </label>
                      )}
                      {settings.translation_backend === "openai" && (
                        <div className="settings-field-grid settings-field-grid--2">
                          <label className="settings-label settings-field-grid__span-2">
                            <span>Base URL</span>
                            <input
                              className="settings-input"
                              placeholder="https://api.openai.com/v1"
                              value={settings.openai_api_base ?? "https://api.openai.com/v1"}
                              onChange={(e) => setSettings({ ...settings, openai_api_base: e.target.value })}
                            />
                          </label>
                          <label className="settings-label settings-field-grid__span-2">
                            <span>API key</span>
                            <input
                              className="settings-input"
                              type="password"
                              placeholder="sk-..."
                              value={newOpenAiKey}
                              onChange={(e) => setNewOpenAiKey(e.target.value)}
                            />
                          </label>
                          {settings.openai_api_key_configured && (
                            <div className="alert-info-box info settings-field-grid__span-2">
                              <CheckCircle2 size={14} />
                              <span>Đã lưu khóa: {settings.openai_api_key_masked}</span>
                            </div>
                          )}
                          <label className="settings-label settings-field-grid__span-2">
                            <span>Model dịch</span>
                            <div className="input-with-button">
                              <select
                                className="settings-input"
                                value={settings.openai_translation_model ?? ""}
                                onChange={(e) => setSettings({ ...settings, openai_translation_model: e.target.value })}
                                disabled={openAiModelsLoading || openAiModels.length === 0}
                              >
                                {openAiModels.length === 0 ? (
                                  <option value="">
                                    {openAiModelsLoading ? "Đang tải model..." : "Chưa có model — nhập API key và tải lại"}
                                  </option>
                                ) : (
                                  openAiModels.map((model) => (
                                    <option key={model.id} value={model.id}>
                                      {model.name}
                                    </option>
                                  ))
                                )}
                              </select>
                              <button
                                type="button"
                                className="key-action-btn save-btn"
                                disabled={openAiModelsLoading}
                                onClick={() => void loadOpenAiModels()}
                              >
                                <RefreshCw size={13} /> {openAiModelsLoading ? "Đang tải..." : "Tải model"}
                              </button>
                            </div>
                          </label>
                          {!settings.openai_api_key_configured && !newOpenAiKey.trim() && (
                            <div className="alert-info-box warning settings-field-grid__span-2">
                              <CircleAlert size={14} />
                              <span>Nhập API key và lưu cài đặt trước khi chạy job dịch OpenAPI.</span>
                            </div>
                          )}
                        </div>
                      )}
                    </section>

                    <section className="settings-card">
                      <div className="card-header-accent">
                        <span className="accent-bar"></span>
                        <h3>Google AI Studio / Khóa API Gemini</h3>
                      </div>
                      <p className="card-description card-description--compact">
                        Dùng cho dịch Gemini và Gemini TTS. Thêm nhiều khóa để luân phiên khi hết quota.
                      </p>
                      <div className="add-key-container">
                        <label className="settings-label" style={{ flex: 1 }}>
                          <span>Khóa API Gemini mới</span>
                          <input
                            className="settings-input"
                            type="password"
                            placeholder="Dán khóa API Google AI Studio"
                            value={newGeminiKey}
                            onChange={(e) => setNewGeminiKey(e.target.value)}
                          />
                        </label>
                        <button type="button" className="gradient-button" onClick={handleAddGeminiKey}>
                          <Plus size={16} /> Thêm khóa
                        </button>
                      </div>
                      <div className="gemini-keys-list">
                        {(settings.gemini_api_keys ?? []).length === 0 ? (
                          <div className="empty-keys-placeholder">Chưa có khóa API Gemini nào.</div>
                        ) : (
                          (settings.gemini_api_keys ?? []).map((item: any) => (
                            <div key={item.id} className="gemini-key-card">
                              <div className="key-card-header">
                                <div className="key-info">
                                  <span className="key-badge">API KEY</span>
                                  <code className="key-masked">{item.masked ?? item.label}</code>
                                </div>
                                <button
                                  type="button"
                                  aria-label={`Remove Gemini key ${item.masked ?? item.label}`}
                                  onClick={() => handleRemoveGeminiKey(item.id)}
                                  className="key-action-btn delete-btn"
                                >
                                  <Trash2 size={13} /> Gỡ bỏ
                                </button>
                              </div>
                              <div className="key-card-body">
                                <div className="key-label-wrapper">
                                  <span className="input-label-small">Nhãn</span>
                                  <div className="input-with-button">
                                    <input
                                      aria-label={`Edit label for Gemini key ${item.masked ?? item.label}`}
                                      className="settings-input key-label-input"
                                      placeholder="Ví dụ: Khóa dự phòng"
                                      value={item.label ?? item.masked ?? ""}
                                      onChange={(event) => updateGeminiKeyLabel(item.id, event.target.value)}
                                    />
                                    <button
                                      type="button"
                                      aria-label={`Save label for Gemini key ${item.masked ?? item.label}`}
                                      onClick={() => handleSaveGeminiKeyLabel(item.id, item.label ?? "")}
                                      className="key-action-btn save-btn"
                                    >
                                      <Save size={13} /> Lưu nhãn
                                    </button>
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                    </section>
                  </div>
                )}

                {settingsTab === "audio" && (
                  <div className="settings-tab-panel">
                    <section className="settings-card settings-card--full">
                      <div className="card-header-accent">
                        <span className="accent-bar"></span>
                        <h3>Nhận diện giọng nói (VAD)</h3>
                      </div>
                      <p className="card-description card-description--compact">
                        Xác định vùng có lời nói trước khi nhận dạng tiếng Trung. Thay đổi có hiệu lực từ bước VAD trở đi — chạy lại job từ VAD hoặc ASR.
                      </p>
                      <div className="settings-field-grid settings-field-grid--2">
                        <label className="settings-label settings-field-grid__span-2">
                          <SettingsFieldLabel
                            label="Engine VAD"
                            hint="Silero dùng mô hình neural, chính xác hơn trên Windows/macOS. FFmpeg silencedetect là phương án cũ để rollback khi cần."
                          />
                          <select
                            className="settings-input"
                            value={settings.vad_engine ?? "silero"}
                            onChange={(e) => setSettings({ ...settings, vad_engine: e.target.value })}
                          >
                            <option value="silero">Silero VAD (khuyến nghị)</option>
                            <option value="silencedetect">FFmpeg silencedetect (legacy)</option>
                          </select>
                        </label>
                        <SettingsCheckboxField
                          label="Lọc ASR do VAD nhầm"
                          hint="Loại bỏ đoạn ASR trùng lặp hoặc rỗng khi VAD nhầm nhạc nền là lời nói. Bật mặc định; tắt nếu thấy mất câu hợp lệ."
                          checked={settings.vad_false_positive_filter_enabled ?? true}
                          onChange={(checked) => setSettings({ ...settings, vad_false_positive_filter_enabled: checked })}
                        />
                        <SettingsCheckboxField
                          label="Lọc theo năng lượng giọng/nhạc"
                          hint="So sánh stem vocals và nhạc nền (Demucs) để loại vùng nghi ngờ chỉ có nhạc. Cần mix_mode background_only để có stem."
                          checked={settings.vad_energy_filter_enabled ?? true}
                          onChange={(checked) => setSettings({ ...settings, vad_energy_filter_enabled: checked })}
                        />
                        <label className="settings-label settings-field-grid__span-2">
                          <SettingsFieldLabel
                            label={`Tỷ lệ giọng/nhạc tối thiểu (${settings.vad_energy_min_vocal_ratio ?? 1.15})`}
                            hint="Vùng có tỷ lệ năng lượng vocals so với nhạc nền thấp hơn ngưỡng này sẽ bị loại. Tăng nếu còn nhầm nhạc; giảm nếu mất câu hợp lệ."
                          />
                          <input
                            className="settings-input"
                            type="range"
                            min={0.8}
                            max={2.5}
                            step={0.05}
                            value={settings.vad_energy_min_vocal_ratio ?? 1.15}
                            onChange={(e) => setSettings({ ...settings, vad_energy_min_vocal_ratio: Number(e.target.value) })}
                            disabled={!(settings.vad_energy_filter_enabled ?? true)}
                          />
                        </label>
                      </div>
                    </section>

                    {(settings.vad_engine ?? "silero") === "silero" && (
                      <section className="settings-card">
                        <div className="card-header-accent">
                          <span className="accent-bar"></span>
                          <h3>Silero VAD</h3>
                        </div>
                        <div className="settings-field-grid settings-field-grid--2">
                          <label className="settings-label settings-field-grid__span-2">
                            <SettingsFieldLabel
                              label={`Ngưỡng phát hiện giọng (${settings.silero_vad_threshold ?? 0.5})`}
                              hint="Xác suất tối thiểu để coi là có giọng (0–1). Cao hơn = ít đoạn giả từ nhạc nền, nhưng có thể bỏ sót lời nói nhỏ."
                            />
                            <input
                              className="settings-input"
                              type="range"
                              min={0.1}
                              max={0.9}
                              step={0.05}
                              value={settings.silero_vad_threshold ?? 0.5}
                              onChange={(e) => setSettings({ ...settings, silero_vad_threshold: Number(e.target.value) })}
                            />
                          </label>
                          <label className="settings-label">
                            <SettingsFieldLabel
                              label="Đoạn nói tối thiểu (ms)"
                              hint="Đoạn ngắn hơn giá trị này sẽ bị bỏ qua. Tăng nếu còn nhiễu ngắn; giảm nếu mất tiếng lẩm nhẩm."
                            />
                            <input
                              className="settings-input"
                              type="number"
                              min={0}
                              max={5000}
                              step={50}
                              value={settings.silero_vad_min_speech_duration_ms ?? 250}
                              onChange={(e) => setSettings({ ...settings, silero_vad_min_speech_duration_ms: Number(e.target.value) })}
                            />
                          </label>
                          <label className="settings-label">
                            <SettingsFieldLabel
                              label="Im lặng tối thiểu (ms)"
                              hint="Khoảng im lặng liên tiếp tối thiểu để tách hai đoạn nói. Tăng nếu một câu bị cắt thành nhiều mảnh."
                            />
                            <input
                              className="settings-input"
                              type="number"
                              min={0}
                              max={5000}
                              step={50}
                              value={settings.silero_vad_min_silence_duration_ms ?? 300}
                              onChange={(e) => setSettings({ ...settings, silero_vad_min_silence_duration_ms: Number(e.target.value) })}
                            />
                          </label>
                          <label className="settings-label settings-field-grid__span-2">
                            <SettingsFieldLabel
                              label="Đệm quanh đoạn nói (ms)"
                              hint="Thêm mili giây trước và sau mỗi vùng nói để tránh cắt mất đầu/cuối âm tiết."
                            />
                            <input
                              className="settings-input"
                              type="number"
                              min={0}
                              max={2000}
                              step={25}
                              value={settings.silero_vad_speech_pad_ms ?? 150}
                              onChange={(e) => setSettings({ ...settings, silero_vad_speech_pad_ms: Number(e.target.value) })}
                            />
                          </label>
                        </div>
                      </section>
                    )}

                    {(settings.vad_engine ?? "silero") === "silencedetect" && (
                      <section className="settings-card">
                        <div className="card-header-accent">
                          <span className="accent-bar"></span>
                          <h3>FFmpeg silencedetect</h3>
                        </div>
                        <div className="settings-field-grid settings-field-grid--2">
                          <label className="settings-label">
                            <SettingsFieldLabel
                              label="Ngưỡng im lặng (dB)"
                              hint="Mức âm lượng coi là im lặng (âm dB, ví dụ -30). Giá trị thấp hơn (gần 0) = nhạy hơn, dễ nhận nhạc nền là lời nói."
                            />
                            <input
                              className="settings-input"
                              type="number"
                              min={-90}
                              max={0}
                              step={1}
                              value={settings.silencedetect_noise_db ?? -30}
                              onChange={(e) => setSettings({ ...settings, silencedetect_noise_db: Number(e.target.value) })}
                            />
                          </label>
                          <label className="settings-label">
                            <SettingsFieldLabel
                              label="Im lặng tối thiểu (giây)"
                              hint="Thời gian im lặng liên tiếp tối thiểu để FFmpeg tách hai đoạn nói."
                            />
                            <input
                              className="settings-input"
                              type="number"
                              min={0.05}
                              max={5}
                              step={0.05}
                              value={settings.silencedetect_min_silence_sec ?? 0.5}
                              onChange={(e) => setSettings({ ...settings, silencedetect_min_silence_sec: Number(e.target.value) })}
                            />
                          </label>
                        </div>
                      </section>
                    )}

                    <section className="settings-card settings-card--full">
                      <div className="card-header-accent">
                        <span className="accent-bar"></span>
                        <h3>ASR thưa (nâng cao)</h3>
                      </div>
                      <p className="card-description card-description--compact">
                        Chỉ chạy nhận dạng trên vùng có giọng khi video có nhiều im lặng — tiết kiệm thời gian trên clip dài.
                      </p>
                      <SettingsCheckboxField
                        label="Bật ASR thưa"
                        hint="Khi bật, ASR chỉ xử lý các chunk có giọng thay vì toàn bộ audio. Hữu ích khi tỷ lệ im lặng cao."
                        checked={settings.sparse_asr_enabled ?? false}
                        onChange={(checked) => setSettings({ ...settings, sparse_asr_enabled: checked })}
                      />
                      <div className="settings-field-grid settings-field-grid--2">
                        <label className="settings-label">
                          <SettingsFieldLabel
                            label={`Khoảng gộp VAD (giây)`}
                            hint="Gộp các vùng VAD cách nhau không quá giá trị này thành một chunk ASR. Tăng nếu câu bị tách quá nhỏ."
                          />
                          <input
                            className="settings-input"
                            type="number"
                            min={0}
                            max={2}
                            step={0.05}
                            value={settings.sparse_asr_merge_gap_sec ?? 0.25}
                            onChange={(e) => setSettings({ ...settings, sparse_asr_merge_gap_sec: Number(e.target.value) })}
                            disabled={!settings.sparse_asr_enabled}
                          />
                        </label>
                        <label className="settings-label">
                          <SettingsFieldLabel
                            label={`Tỷ lệ im lặng tối thiểu (${settings.sparse_asr_min_silence_ratio ?? 0.35})`}
                            hint="Chỉ kích hoạt ASR thưa khi tỷ lệ im lặng trong audio ≥ giá trị này (0–0.95)."
                          />
                          <input
                            className="settings-input"
                            type="range"
                            min={0}
                            max={0.95}
                            step={0.05}
                            value={settings.sparse_asr_min_silence_ratio ?? 0.35}
                            onChange={(e) => setSettings({ ...settings, sparse_asr_min_silence_ratio: Number(e.target.value) })}
                            disabled={!settings.sparse_asr_enabled}
                          />
                        </label>
                        <label className="settings-label">
                          <SettingsFieldLabel
                            label="Chunk tối đa (giây)"
                            hint="Độ dài tối đa mỗi đoạn gửi ASR khi chạy chế độ thưa."
                          />
                          <input
                            className="settings-input"
                            type="number"
                            min={5}
                            max={120}
                            step={1}
                            value={settings.sparse_asr_chunk_sec ?? 25}
                            onChange={(e) => setSettings({ ...settings, sparse_asr_chunk_sec: Number(e.target.value) })}
                            disabled={!settings.sparse_asr_enabled}
                          />
                        </label>
                        <label className="settings-label">
                          <SettingsFieldLabel
                            label="Đệm chunk (ms)"
                            hint="Thêm mili giây trước/sau mỗi chunk ASR để không cắt mất biên âm tiết."
                          />
                          <input
                            className="settings-input"
                            type="number"
                            min={0}
                            max={1000}
                            step={25}
                            value={settings.sparse_asr_padding_ms ?? 200}
                            onChange={(e) => setSettings({ ...settings, sparse_asr_padding_ms: Number(e.target.value) })}
                            disabled={!settings.sparse_asr_enabled}
                          />
                        </label>
                      </div>
                    </section>
                  </div>
                )}

                {settingsTab === "tts" && (
                  <div className="settings-tab-panel">
                    <section className="settings-card settings-card--full">
                      <div className="card-header-accent">
                        <span className="accent-bar"></span>
                        <h3>Khớp thời lượng lồng tiếng</h3>
                      </div>
                      <p className="card-description card-description--compact">
                        Điều chỉnh bước <strong>duration_repair</strong> khi giọng TTS ngắn hoặc dài hơn timeline gốc.
                        Khi cần rút ngắn/kéo dài câu, pipeline dùng cùng backend dịch đã chọn ở tab Dịch thuật (OpenAPI hoặc Gemini).
                        Chạy lại job từ duration_repair hoặc TTS để áp dụng.
                      </p>
                      <SettingsCheckboxField
                        label="Bật khớp thời lượng chính xác"
                        hint="Khi bật, pipeline cố căn thời lượng từng đoạn TTS với slot gốc (kéo dài, rút ngắn hoặc thêm im lặng)."
                        checked={settings.exact_timing_enabled ?? true}
                        onChange={(checked) => setSettings({ ...settings, exact_timing_enabled: checked })}
                      />
                      <div className="settings-field-grid settings-field-grid--2">
                        <label className="settings-label settings-field-grid__span-2">
                          <SettingsFieldLabel
                            label={`Tốc độ TTS toàn cục (${(settings.tts_global_speed ?? 1).toFixed(2)}×)`}
                            hint="Nhân tốc độ phát audio TTS trước khi khớp timeline. Tăng nhẹ (vd. 1.05) nếu lồng tiếng vẫn dài hơn giọng gốc; giảm nếu quá nhanh."
                          />
                          <input
                            className="settings-input"
                            type="range"
                            min={0.9}
                            max={1.3}
                            step={0.01}
                            value={settings.tts_global_speed ?? 1}
                            onChange={(e) => setSettings({ ...settings, tts_global_speed: Number(e.target.value) })}
                            disabled={!settings.exact_timing_enabled}
                          />
                        </label>
                        <label className="settings-label settings-field-grid__span-2">
                          <SettingsFieldLabel
                            label={`Tốc độ nói tiếng Việt ước lượng (${(settings.vietnamese_speaking_rate_wps ?? 3.2).toFixed(2)} từ/giây)`}
                            hint="Dùng khi dịch để ước lượng độ dài câu. Tự hiệu chỉnh sau mỗi job TTS; chỉnh tay nếu giọng đọc nhanh/chậm hơn mặc định."
                          />
                          <input
                            className="settings-input"
                            type="range"
                            min={2}
                            max={5}
                            step={0.05}
                            value={settings.vietnamese_speaking_rate_wps ?? 3.2}
                            onChange={(e) => setSettings({ ...settings, vietnamese_speaking_rate_wps: Number(e.target.value) })}
                          />
                        </label>
                        <label className="settings-label">
                          <SettingsFieldLabel
                            label="Ngưỡng kéo dài TTS ngắn (giây)"
                            hint="Chênh lệch tối thiểu giữa slot gốc và TTS hiện tại mới kích hoạt kéo dài/thêm từ. Tăng nếu repair làm audio dài hơn thực tế."
                          />
                          <input
                            className="settings-input"
                            type="number"
                            min={0.2}
                            max={5}
                            step={0.1}
                            value={settings.short_tts_lengthen_min_gap_sec ?? 1.5}
                            onChange={(e) => setSettings({ ...settings, short_tts_lengthen_min_gap_sec: Number(e.target.value) })}
                            disabled={!settings.exact_timing_enabled}
                          />
                        </label>
                        <label className="settings-label">
                          <SettingsFieldLabel
                            label={`Tỷ lệ kéo dài tối đa (${settings.short_tts_lengthen_max_ratio ?? 1.6}×)`}
                            hint="Giới hạn độ dài sau khi kéo dài TTS ngắn (vd. 1.6 = tối đa 160% độ dài hiện tại). Giảm nếu repair vẫn kéo dài quá mức."
                          />
                          <input
                            className="settings-input"
                            type="number"
                            min={1.05}
                            max={2}
                            step={0.05}
                            value={settings.short_tts_lengthen_max_ratio ?? 1.6}
                            onChange={(e) => setSettings({ ...settings, short_tts_lengthen_max_ratio: Number(e.target.value) })}
                            disabled={!settings.exact_timing_enabled}
                          />
                        </label>
                      </div>
                    </section>

                    <section className="settings-card settings-card--full">
                      <div className="card-header-accent">
                        <span className="accent-bar"></span>
                        <h3>Engine lồng tiếng</h3>
                      </div>
                      <p className="card-description card-description--compact">
                        Chọn engine phù hợp: clone offline (VoxCPM2), neural miễn phí (Edge / Google TTS), hoặc AI Studio (Gemini TTS).
                      </p>
                      <div className="settings-field-grid settings-backend-grid">
                        {TTS_BACKEND_OPTIONS.map((option) => (
                          <button
                            key={option.id}
                            type="button"
                            className={`settings-backend-card${activeTtsBackend === option.id ? " active" : ""}`}
                            onClick={() => setSettings({ ...settings, tts_backend: option.id })}
                          >
                            <strong>{option.label}</strong>
                            <span>{option.hint}</span>
                          </button>
                        ))}
                      </div>
                    </section>

                    {activeTtsBackend === "voxcpm" && (
                      <section className="settings-card">
                        <div className="card-header-accent">
                          <span className="accent-bar"></span>
                          <h3>VoxCPM2 — Clone giọng</h3>
                        </div>
                        <p className="card-description card-description--compact">
                          Upload giọng mẫu ở tab <strong>Clone Giọng</strong>, hoặc dùng voice design / auto voice.
                        </p>
                        <div className="settings-field-grid settings-field-grid--2">
                          <label className="settings-label">
                            <span>Audio tham chiếu (.wav)</span>
                            <select
                              className="settings-input"
                              value={settings.voxcpm_ref_audio ?? ""}
                              onChange={(e) => setSettings({ ...settings, voxcpm_ref_audio: e.target.value })}
                            >
                              <option value="">Không dùng / Auto voice</option>
                              {clonedVoices.map((voice) => (
                                <option key={voice.id} value={voice.wav_path}>
                                  {voice.name}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label className="settings-label">
                            <span>Chế độ clone</span>
                            <select
                              className="settings-input"
                              value={settings.voxcpm_clone_mode ?? "reference"}
                              onChange={(e) => setSettings({ ...settings, voxcpm_clone_mode: e.target.value })}
                            >
                              <option value="reference">Reference clone (ổn định)</option>
                              <option value="ultimate">Ultimate clone (dùng transcript)</option>
                            </select>
                          </label>
                          <label className="settings-label settings-field-grid__span-2">
                            <span>Voice design (tùy chọn)</span>
                            <input
                              className="settings-input"
                              placeholder="female, low pitch"
                              value={settings.voxcpm_instruct ?? ""}
                              onChange={(e) => setSettings({ ...settings, voxcpm_instruct: e.target.value })}
                            />
                          </label>
                          <label className="settings-label settings-label--inline settings-field-grid__span-2">
                            <input
                              type="checkbox"
                              checked={Boolean(settings.voxcpm_auto_voice ?? true)}
                              onChange={(e) => setSettings({ ...settings, voxcpm_auto_voice: e.target.checked })}
                            />
                            <span>Auto voice khi không có audio tham chiếu</span>
                          </label>
                        </div>
                        {settings.voxcpm_clone_mode === "ultimate" && !settings.voxcpm_ref_audio && (
                          <div className="alert-info-box warning">
                            <CircleAlert size={14} />
                            <span>Ultimate clone cần chọn audio tham chiếu đã upload.</span>
                          </div>
                        )}
                        {settings.voxcpm_clone_mode === "ultimate" && selectedClonedVoice && !selectedClonedVoice.transcribed && (
                          <div className="alert-info-box warning">
                            <CircleAlert size={14} />
                            <span>Giọng đã chọn chưa có transcript .txt; hãy upload lại file rõ tiếng hơn.</span>
                          </div>
                        )}
                      </section>
                    )}

                    {activeTtsBackend === "edge_tts" && (
                      <section className="settings-card">
                        <div className="card-header-accent">
                          <span className="accent-bar"></span>
                          <h3>Edge TTS — Giọng Microsoft</h3>
                        </div>
                        <p className="card-description card-description--compact">
                          Dùng giọng neural của Microsoft qua internet. Không cần GPU hay API key.
                        </p>
                        <label className="settings-label">
                          <span>Giọng tiếng Việt</span>
                          <select
                            className="settings-input"
                            value={settings.edge_tts_voice ?? "vi-VN-HoaiMyNeural"}
                            onChange={(e) => setSettings({ ...settings, edge_tts_voice: e.target.value })}
                          >
                            {(edgeTtsVoices.length > 0 ? edgeTtsVoices : [
                              { id: "vi-VN-HoaiMyNeural", name: "Hoài My (Nữ)" },
                              { id: "vi-VN-NamMinhNeural", name: "Nam Minh (Nam)" },
                            ]).map((voice) => (
                              <option key={voice.id} value={voice.id}>
                                {voice.name}
                              </option>
                            ))}
                          </select>
                        </label>
                      </section>
                    )}

                    {activeTtsBackend === "google_tts" && (
                      <section className="settings-card">
                        <div className="card-header-accent">
                          <span className="accent-bar"></span>
                          <h3>Google TTS — Cloud Text-to-Speech</h3>
                        </div>
                        <p className="card-description card-description--compact">
                          4 giọng Standard tiếng Việt (A/B/C/D). Cần API key từ Google Cloud Console — khác khóa Gemini AI Studio.
                        </p>
                        <div className="settings-field-grid settings-field-grid--2">
                          <label className="settings-label settings-field-grid__span-2">
                            <span>API key Google Cloud Text-to-Speech</span>
                            <input
                              className="settings-input"
                              type="password"
                              placeholder="AIza... (bật Cloud Text-to-Speech API)"
                              value={newGoogleTtsKey}
                              onChange={(e) => setNewGoogleTtsKey(e.target.value)}
                            />
                          </label>
                          {settings.google_tts_api_key_configured && (
                            <div className="alert-info-box info settings-field-grid__span-2">
                              <CheckCircle2 size={14} />
                              <span>Đã lưu khóa: {settings.google_tts_api_key_masked}</span>
                            </div>
                          )}
                          <label className="settings-label settings-field-grid__span-2">
                            <span>Giọng đọc tiếng Việt</span>
                            <select
                              className="settings-input"
                              value={settings.google_tts_voice ?? "vi-VN-Standard-A"}
                              onChange={(e) => setSettings({ ...settings, google_tts_voice: e.target.value })}
                            >
                              {GOOGLE_TTS_VOICES.map((voice) => (
                                <option key={voice.id} value={voice.id}>
                                  {voice.name}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label className="settings-label settings-field-grid__span-2">
                            <span>Tốc độ đọc ({settings.google_tts_speaking_rate ?? 1}x)</span>
                            <input
                              className="settings-input"
                              type="range"
                              min={0.75}
                              max={1.25}
                              step={0.05}
                              value={settings.google_tts_speaking_rate ?? 1}
                              onChange={(e) => setSettings({ ...settings, google_tts_speaking_rate: Number(e.target.value) })}
                            />
                          </label>
                        </div>
                        {!settings.google_tts_api_key_configured && !newGoogleTtsKey.trim() && (
                          <div className="alert-info-box warning">
                            <CircleAlert size={14} />
                            <span>Chưa có API key Cloud TTS. Lưu khóa trước khi chạy job hoặc nghe thử.</span>
                          </div>
                        )}
                      </section>
                    )}

                    {activeTtsBackend === "gemini_tts" && (
                      <section className="settings-card">
                        <div className="card-header-accent">
                          <span className="accent-bar"></span>
                          <h3>Gemini TTS — Google AI</h3>
                        </div>
                        <p className="card-description card-description--compact">
                          Tổng hợp giọng qua Google AI Studio. Cần ít nhất một khóa API ở tab Dịch thuật.
                        </p>
                        <div className="settings-field-grid settings-field-grid--2">
                          <label className="settings-label">
                            <span>Mô hình TTS</span>
                            <input
                              className="settings-input"
                              value={settings.gemini_tts_model ?? "gemini-2.5-flash-preview-tts"}
                              onChange={(e) => setSettings({ ...settings, gemini_tts_model: e.target.value })}
                            />
                          </label>
                          <label className="settings-label">
                            <span>Giọng đọc</span>
                            <select
                              className="settings-input"
                              value={settings.gemini_tts_voice ?? "Zephyr"}
                              onChange={(e) => setSettings({ ...settings, gemini_tts_voice: e.target.value })}
                            >
                              {GEMINI_TTS_VOICES.map((voice) => (
                                <option key={voice.id} value={voice.id}>
                                  {voice.name}
                                </option>
                              ))}
                            </select>
                          </label>
                        </div>
                        {(settings.gemini_api_keys ?? []).length === 0 && (
                          <div className="alert-info-box warning">
                            <CircleAlert size={14} />
                            <span>Chưa có khóa Gemini. Thêm khóa ở tab Dịch thuật trước khi chạy job.</span>
                          </div>
                        )}
                      </section>
                    )}

                    <section className="settings-card settings-card--full">
                      <div className="card-header-accent">
                        <span className="accent-bar"></span>
                        <h3>Nghe thử giọng đọc</h3>
                      </div>
                      <p className="card-description card-description--compact">
                        Tạo audio mẫu với cấu hình hiện tại (chưa cần lưu nếu chỉ đổi nội dung thử).
                      </p>
                      <div className="settings-field-grid settings-field-grid--2">
                        <label className="settings-label settings-field-grid__span-2">
                          <span>Nội dung nghe thử</span>
                          <textarea
                            className="settings-input"
                            rows={3}
                            value={ttsPreviewText}
                            onChange={(e) => setTtsPreviewText(e.target.value)}
                            placeholder="Nhập câu tiếng Việt để nghe thử..."
                          />
                        </label>
                        <div className="settings-field-grid__span-2" style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" }}>
                          <button
                            type="button"
                            className="gradient-button"
                            disabled={ttsPreviewLoading}
                            onClick={() => void handlePreviewTts()}
                            style={{ justifyContent: "center" }}
                          >
                            <Volume2 size={16} />
                            {ttsPreviewLoading ? "Đang tạo audio..." : "Nghe thử"}
                          </button>
                          {ttsPreviewUrl && (
                            <audio controls autoPlay src={ttsPreviewUrl} style={{ height: "36px", flex: "1 1 240px" }} />
                          )}
                        </div>
                      </div>
                    </section>
                  </div>
                )}

                {settingsTab === "subtitles" && (
                  <div className="settings-tab-panel">
                    <section className="settings-card settings-card--full">
                      <div className="card-header-accent">
                        <span className="accent-bar"></span>
                        <h3>Phụ đề trên video</h3>
                      </div>
                      <p className="card-description card-description--compact">
                        Chèn phụ đề tiếng Việt (bản dịch) trực tiếp vào video khi xuất thành phẩm.
                      </p>
                      <label className="settings-label settings-label--inline">
                        <input
                          type="checkbox"
                          checked={settings.subtitles_enabled ?? true}
                          onChange={(e) => setSettings({ ...settings, subtitles_enabled: e.target.checked })}
                        />
                        <span>Bật phụ đề trên video</span>
                      </label>
                      <div className="settings-field-grid settings-field-grid--2">
                        <label className="settings-label">
                          <span>Cỡ chữ</span>
                          <input
                            className="settings-input"
                            type="number"
                            min={16}
                            max={120}
                            value={settings.subtitle_font_size ?? 48}
                            onChange={(e) => setSettings({ ...settings, subtitle_font_size: Number(e.target.value) })}
                            disabled={!settings.subtitles_enabled}
                          />
                        </label>
                        <label className="settings-label">
                          <span>Vị trí phụ đề</span>
                          <select
                            className="settings-input"
                            value={settings.subtitle_position ?? "bottom"}
                            onChange={(e) => setSettings({ ...settings, subtitle_position: e.target.value })}
                            disabled={!settings.subtitles_enabled}
                          >
                            <option value="bottom">Dưới cùng</option>
                            <option value="center">Giữa màn hình</option>
                            <option value="top">Trên cùng</option>
                          </select>
                        </label>
                        <label className="settings-label">
                          <span>Màu chữ</span>
                          <input
                            className="settings-input"
                            type="color"
                            value={settings.subtitle_font_color ?? "#FFFFFF"}
                            onChange={(e) => setSettings({ ...settings, subtitle_font_color: e.target.value.toUpperCase() })}
                            disabled={!settings.subtitles_enabled}
                          />
                        </label>
                        <label className="settings-label">
                          <span>Màu nền chữ</span>
                          <input
                            className="settings-input"
                            type="color"
                            value={settings.subtitle_background_color ?? "#000000"}
                            onChange={(e) => setSettings({ ...settings, subtitle_background_color: e.target.value.toUpperCase() })}
                            disabled={!settings.subtitles_enabled}
                          />
                        </label>
                        <label className="settings-label settings-field-grid__span-2">
                          <span>Độ đặc nền ({settings.subtitle_background_opacity ?? 95}%)</span>
                          <input
                            className="settings-input"
                            type="range"
                            min={40}
                            max={100}
                            value={settings.subtitle_background_opacity ?? 95}
                            onChange={(e) => setSettings({ ...settings, subtitle_background_opacity: Number(e.target.value) })}
                            disabled={!settings.subtitles_enabled}
                          />
                        </label>
                        <label className="settings-label settings-field-grid__span-2">
                          <span>Đệm nền quanh chữ ({settings.subtitle_background_padding ?? 12}px)</span>
                          <input
                            className="settings-input"
                            type="range"
                            min={4}
                            max={40}
                            value={settings.subtitle_background_padding ?? 12}
                            onChange={(e) => setSettings({ ...settings, subtitle_background_padding: Number(e.target.value) })}
                            disabled={!settings.subtitles_enabled}
                          />
                        </label>
                        {(settings.subtitle_position === "bottom" || settings.subtitle_position === "top") && (
                          <label className="settings-label settings-field-grid__span-2">
                            <span>
                              {settings.subtitle_position === "bottom"
                                ? `Khoảng cách tới lề dưới (${settings.subtitle_edge_margin ?? 80}px)`
                                : `Khoảng cách tới lề trên (${settings.subtitle_edge_margin ?? 80}px)`}
                            </span>
                            <input
                              className="settings-input"
                              type="range"
                              min={0}
                              max={300}
                              value={settings.subtitle_edge_margin ?? 80}
                              onChange={(e) => setSettings({ ...settings, subtitle_edge_margin: Number(e.target.value) })}
                              disabled={!settings.subtitles_enabled}
                            />
                          </label>
                        )}
                      </div>
                      <div className="alert-info-box info">
                        <CircleAlert size={14} style={{ flexShrink: 0, marginTop: "2px" }} />
                        <span>
                          Phụ đề dùng bản dịch tiếng Việt theo từng phân đoạn. Chạy lại từ bước <strong>Xuất video thành phẩm</strong> để áp dụng thay đổi cho job đã hoàn thành.
                        </span>
                      </div>
                    </section>
                  </div>
                )}
              </div>
            </form>
          </>
        )}
      </main>
      </div>

      {/* Selected Job Drawer Panel */}
      {selectedJob && (
        <div className="overlay" onClick={closeSelectedJob}>
          <section className="runtime-panel" onClick={(event) => event.stopPropagation()} style={{ width: "min(760px, 100%)", display: "flex", flexDirection: "column" }}>
            <div className="runtime-head">
              <div>
                <p>Chi tiết tiến trình</p>
                <h2 style={{ fontSize: "20px", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap", maxWidth: "620px" }}>
                  {formatJobLabel(selectedJob)}
                </h2>
                {selectedJob.title_vi && selectedJob.title && selectedJob.title_vi !== selectedJob.title && (
                  <small style={{ color: "#747d90", display: "block", marginTop: "4px", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap", maxWidth: "620px" }}>
                    {selectedJob.title}
                  </small>
                )}
                <small style={{ fontFamily: "monospace" }}>ID: {selectedJob.id}</small>
              </div>
              <button aria-label="Close job details" onClick={closeSelectedJob}>
                <X />
              </button>
            </div>

            {/* Status actions */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
                gap: "10px",
                margin: "20px 0 10px",
              }}
            >
              <button
                className="smoke-button"
                style={{ minWidth: 0, background: "#20242e", whiteSpace: "nowrap" }}
                onClick={() => openJobFolder(selectedJob.id)}
              >
                <FolderOpen size={16} /> Mở thư mục
              </button>
              {(selectedJob.status === "queued" || selectedJob.status === "failed" || selectedJob.status === "interrupted") && (
                <button className="smoke-button" style={{ minWidth: 0, whiteSpace: "nowrap" }} onClick={() => startJob(selectedJob.id)}>
                  <RefreshCw size={16} /> {selectedJob.status === "queued" ? "Bắt đầu lồng tiếng" : "Tiếp tục lồng tiếng"}
                </button>
              )}
              {(selectedJob.status === "completed" || selectedJob.status === "failed" || selectedJob.status === "interrupted") && (
                <button
                  className="smoke-button"
                  style={{ minWidth: 0, background: "linear-gradient(135deg, #6244f7, #3d29a6)", color: "#fff", whiteSpace: "nowrap" }}
                  onClick={() => openRerunModal(selectedJob)}
                >
                  <RefreshCw size={16} /> Chạy lại
                </button>
              )}
              {selectedJob.status === "running" && (
                <button className="smoke-button" style={{ minWidth: 0, background: "#f16f7e", whiteSpace: "nowrap" }} onClick={() => cancelJob(selectedJob.id)}>
                  <X size={16} /> Hủy thực thi
                </button>
              )}
              {isDeletableJob(selectedJob) && (
                <button className="smoke-button" style={{ minWidth: 0, background: "#f16f7e", whiteSpace: "nowrap" }} onClick={() => deleteJob(selectedJob.id)}>
                  <Trash2 size={16} /> Xóa tiến trình
                </button>
              )}
            </div>

            {rerunModalOpen && (
              <div
                style={{
                  background: "#20242e",
                  border: "1px solid #343a48",
                  borderRadius: "12px",
                  padding: "16px",
                  margin: "12px 0",
                  display: "flex",
                  flexDirection: "column",
                  gap: "12px",
                }}
              >
                <h3 style={{ margin: 0, fontSize: "15px" }}>Chọn bước bắt đầu chạy lại</h3>
                <p style={{ margin: 0, fontSize: "13px", color: "#8f97a6", lineHeight: 1.5 }}>
                  Các bước phía trên điểm bạn chọn sẽ giữ cache. Từ bước được chọn trở xuống sẽ chạy lại
                  (vì mỗi bước dùng kết quả của bước trước — không thể giữ cache bước sau khi chạy lại bước giữa).
                  {rerunKeepSteps.length < PIPELINE_STEPS.length && (
                    <>
                      {" "}
                      Điểm bắt đầu: <strong>{translateStepName(PIPELINE_STEPS[rerunKeepSteps.length])}</strong>
                      {" · "}
                      Giữ {rerunKeepSteps.length} bước, chạy lại {PIPELINE_STEPS.length - rerunKeepSteps.length} bước
                    </>
                  )}
                </p>
                <div style={{ display: "grid", gap: "8px", maxHeight: "280px", overflowY: "auto", paddingRight: "4px" }}>
                  {PIPELINE_STEPS.map((stepName, index) => {
                    const step = selectedJob.steps.find((item) => item.name === stepName);
                    const rerunFromIndex = rerunKeepSteps.length;
                    const kept = index < rerunFromIndex;
                    const isBoundary = index === rerunFromIndex;
                    return (
                      <button
                        key={stepName}
                        type="button"
                        onClick={() => handleRerunBoundarySelect(index)}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "10px",
                          background: kept ? "#152218" : isBoundary ? "#2a2240" : "#12151c",
                          border: isBoundary
                            ? "1px solid #6244f7"
                            : kept
                              ? "1px solid #2f6b4a"
                              : "1px solid #292f3b",
                          borderRadius: "8px",
                          padding: "10px 12px",
                          cursor: "pointer",
                          textAlign: "left",
                          width: "100%",
                          color: "inherit",
                        }}
                      >
                        <span
                          aria-hidden
                          style={{
                            width: "18px",
                            height: "18px",
                            borderRadius: "50%",
                            flexShrink: 0,
                            border: isBoundary ? "5px solid #6244f7" : kept ? "none" : "2px solid #4a5160",
                            background: kept ? "#3ecf8e" : isBoundary ? "#12151c" : "transparent",
                          }}
                        />
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: "13px", color: "#fff" }}>{translateStepName(stepName)}</div>
                          <small style={{ color: "#747d90" }}>
                            {step ? translateStepStatus(step.status) : "Chưa có"} ·{" "}
                            {kept ? "Giữ cache" : isBoundary ? "Chạy lại từ đây" : "Sẽ chạy lại"}
                          </small>
                        </div>
                      </button>
                    );
                  })}
                </div>
                <div style={{ display: "flex", gap: "10px" }}>
                  <button className="smoke-button" style={{ flex: 1 }} onClick={closeRerunModal} disabled={rerunSubmitting}>
                    Hủy
                  </button>
                  <button
                    className="smoke-button"
                    style={{ flex: 1, background: "linear-gradient(135deg, #6244f7, #3d29a6)", color: "#fff" }}
                    onClick={handleRerunJob}
                    disabled={rerunSubmitting || rerunKeepSteps.length >= PIPELINE_STEPS.length}
                  >
                    {rerunSubmitting ? "Đang xử lý..." : "Xác nhận chạy lại"}
                  </button>
                </div>
              </div>
            )}

            {/* Error messaging */}
            {selectedJob.status === "waiting_for_selection" && resolveCp?.videos && (
              <div style={{ background: "#20242e", border: "1px solid #343a48", borderRadius: "12px", padding: "16px", margin: "12px 0" }}>
                <h3 style={{ margin: "0 0 12px", fontSize: "15px" }}>Chọn video trong playlist</h3>
                <div style={{ display: "grid", gap: "8px", maxHeight: "250px", overflowY: "auto" }}>
                  {resolveCp.videos.map((vid: any, idx: number) => (
                    <button
                      key={vid.id ?? idx}
                      type="button"
                      onClick={() => void handleSelectPlaylistVideo(selectedJob.id, idx)}
                      className="playlist-item"
                      style={{ display: "flex", gap: "10px", background: "#12151c", padding: "10px", borderRadius: "8px", cursor: "pointer", border: "1px solid #292f3b", textAlign: "left", color: "inherit" }}
                    >
                      <div style={{ flex: 1 }}>
                        <strong style={{ fontSize: "14px" }}>{vid.title || `Video ${idx + 1}`}</strong>
                        {vid.duration ? (
                          <small style={{ display: "block", color: "#747d90", marginTop: "4px" }}>
                            {Math.round(vid.duration)}s
                          </small>
                        ) : null}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {selectedJob.status === "failed" && selectedJob.last_error_code && (
              <div className="error" style={{ margin: "10px 0" }}>
                <CircleAlert size={22} />
                <div>
                  <strong>{selectedJob.last_error_code}</strong>
                  <p style={{ margin: "4px 0", fontSize: "13px" }}>{selectedJob.last_error_message}</p>
                  <small style={{ color: "#ffbcc9" }}>Gợi ý hành động: Hãy kiểm tra cài đặt của bạn và tiếp tục tiến trình.</small>
                </div>
              </div>
            )}

            {/* Tabs for Drawer */}
            <div style={{ display: "flex", borderBottom: "1px solid rgba(255,255,255,0.08)", marginBottom: "16px", flexShrink: 0 }}>
              <button
                type="button"
                className={`drawer-tab-btn ${drawerTab === "steps" ? "active" : ""}`}
                onClick={() => setDrawerTab("steps")}
                style={{
                  flex: 1,
                  padding: "10px",
                  background: "transparent",
                  color: drawerTab === "steps" ? "#a79aff" : "#8f97a6",
                  border: "none",
                  borderBottom: drawerTab === "steps" ? "2px solid #a79aff" : "none",
                  fontWeight: 600,
                  cursor: "pointer"
                }}
              >
                Trạng thái các bước
              </button>
              {durationRepairCp && durationRepairCp.segments && (
                <button
                  type="button"
                  className={`drawer-tab-btn ${drawerTab === "segments" ? "active" : ""}`}
                  onClick={() => setDrawerTab("segments")}
                  style={{
                    flex: 1,
                    padding: "10px",
                    background: "transparent",
                    color: drawerTab === "segments" ? "#a79aff" : "#8f97a6",
                    border: "none",
                    borderBottom: drawerTab === "segments" ? "2px solid #a79aff" : "none",
                    fontWeight: 600,
                    cursor: "pointer"
                  }}
                >
                  Phân đoạn ({durationRepairCp.segments.length})
                </button>
              )}
              {jobFiles.length > 0 && (
                <button
                  type="button"
                  className={`drawer-tab-btn ${drawerTab === "files" ? "active" : ""}`}
                  onClick={() => setDrawerTab("files")}
                  style={{
                    flex: 1,
                    padding: "10px",
                    background: "transparent",
                    color: drawerTab === "files" ? "#a79aff" : "#8f97a6",
                    border: "none",
                    borderBottom: drawerTab === "files" ? "2px solid #a79aff" : "none",
                    fontWeight: 600,
                    cursor: "pointer"
                  }}
                >
                  Tập tin ({jobFiles.length})
                </button>
              )}
            </div>

             {/* Vertical step checklist or Segments preview */}
             <div style={{ display: "grid", gap: "12px", margin: "10px 0 20px", overflowY: "auto", flex: 1 }}>
               {drawerTab === "steps" ? (
                 selectedJob.steps.map((step) => (
                   <div key={step.name} className="step-row" style={{ display: "flex", gap: "14px", alignItems: "center", padding: "10px 14px", background: "#1b1e26", borderRadius: "10px" }}>
                     <span className={`check-icon ${step.status}`}>
                       {step.status === "completed" ? (
                         <CheckCircle2 size={18} />
                       ) : step.status === "running" ? (
                         <RefreshCw size={18} className="spin" />
                       ) : step.status === "failed" ? (
                         <CircleAlert size={18} />
                       ) : (
                         <Clock3 size={18} style={{ color: "#626b7d" }} />
                       )}
                     </span>
                     <div style={{ flex: 1 }}>
                       <strong style={{ fontSize: "14px" }}>{translateStepName(step.name)}</strong>
                       <p style={{ margin: "2px 0 0", fontSize: "11px", color: "#747e90" }}>
                         Trạng thái: {translateStepStatus(step.status)} · Thời gian: {formatStepDuration(step.duration_ms, step.started_at, step.status)}
                       </p>
                     </div>
                   </div>
                 ))
               ) : drawerTab === "segments" ? (
                 durationRepairCp?.segments?.map((seg: any) => (
                   <div
                     key={seg.index}
                     style={{
                       padding: "12px",
                       background: "#12151c",
                       border: "1px solid #292f3b",
                       borderRadius: "10px",
                       display: "flex",
                       flexDirection: "column",
                       gap: "8px"
                     }}
                   >
                     <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                       <span style={{ fontSize: "12px", color: "#8170ff", fontWeight: 600 }}>
                         Phân đoạn #{seg.index + 1}
                       </span>
                       <small style={{ color: "#747d90" }}>Bắt đầu: {seg.start.toFixed(2)}s | Thời lượng: {seg.repaired_duration ? seg.repaired_duration.toFixed(2) : (seg.duration_budget ? seg.duration_budget.toFixed(2) : "--")}s</small>
                     </div>
                     {seg.text && (
                       <div style={{ fontSize: "12px", color: "#626b7d", fontStyle: "italic", background: "rgba(255,255,255,0.015)", padding: "6px 10px", borderRadius: "6px", border: "1px solid rgba(255,255,255,0.03)" }}>
                         Gốc (Trung): {seg.text}
                       </div>
                     )}
                     <div style={{ fontSize: "13.5px", color: "#fff", fontWeight: 500, lineHeight: 1.4 }}>
                       Dịch (Việt): {seg.translation}
                     </div>
                     {seg.speaker_id != null && (
                       <span style={{ fontSize: "11px", color: "#00d1b2" }}>
                         · Speaker {seg.speaker_id}
                       </span>
                     )}
                     <div style={{ display: "flex", alignItems: "center", gap: "10px", marginTop: "4px" }}>
                       <audio
                         controls
                         src={`http://127.0.0.1:8765/api/jobs/${selectedJob.id}/segments/${seg.index}/wav`}
                         style={{ height: "26px", width: "100%" }}
                       />
                     </div>
                   </div>
                 ))
               ) : (
                 jobFiles.map((file) => (
                   <div
                     key={file.key}
                     style={{
                       padding: "12px",
                       background: "#12151c",
                       border: "1px solid #292f3b",
                       borderRadius: "10px",
                       display: "flex",
                       flexDirection: "column",
                       gap: "10px"
                     }}
                   >
                     <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                       <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                         {file.media_type.startsWith("video") ? <FileVideo size={18} style={{ color: "#a79aff" }} /> : <FileAudio size={18} style={{ color: "#00d1b2" }} />}
                         <span style={{ fontSize: "13.5px", color: "#fff", fontWeight: 500 }}>{file.name}</span>
                       </div>
                       <small style={{ color: "#747d90" }}>{formatBytes(file.size)}</small>
                     </div>
                     {file.media_type.startsWith("video") ? (
                       <button
                         className="smoke-button"
                         style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "6px", fontSize: "12px", padding: "6px 12px" }}
                         onClick={() => setActiveVideoUrl(`http://127.0.0.1:8765${file.url}`)}
                       >
                         <Play size={12} /> Xem thử video
                       </button>
                     ) : (
                       <audio
                         controls
                         src={`http://127.0.0.1:8765${file.url}`}
                         style={{ height: "26px", width: "100%" }}
                       />
                     )}
                   </div>
                 ))
               )}
             </div>
          </section>
        </div>
      )}

      {/* Video Player Overlay */}
      {activeVideoUrl && (
        <div className="overlay" onClick={() => setActiveVideoUrl(null)} style={{ display: "grid", placeItems: "center", background: "#000000e0" }}>
          <div onClick={(e) => e.stopPropagation()} style={{ position: "relative", width: "90%", maxWidth: "800px", background: "#000", borderRadius: "14px", overflow: "hidden", boxShadow: "0 20px 50px rgba(0,0,0,0.5)" }}>
            <button
              onClick={() => setActiveVideoUrl(null)}
              style={{ position: "absolute", right: "12px", top: "12px", background: "#00000080", color: "#fff", border: "0", borderRadius: "50%", width: "32px", height: "32px", display: "grid", placeItems: "center", zIndex: 1 }}
            >
              <X size={18} />
            </button>
            <video controls autoPlay src={activeVideoUrl} style={{ width: "100%", height: "auto", display: "block" }} />
          </div>
        </div>
      )}

      {/* Runtime smoke test panel */}
      {runtimeOpen && (
        <div className="overlay" onClick={() => setRuntimeOpen(false)}>
          <section className="runtime-panel" onClick={(event) => event.stopPropagation()}>
            <div className="runtime-head">
              <div>
                <p>Môi trường thực thi</p>
                <h2>
                  Trạng thái: {runtime ? translateStatus(runtime.status) : runtimeLoading ? "đang kiểm tra" : "chưa tải được"}
                </h2>
                <small>
                  {runtime
                    ? `Kiểm tra lần cuối lúc ${new Date(runtime.checked_at).toLocaleString()}`
                    : runtimeError ?? "Đang tải chi tiết môi trường thực thi..."}
                </small>
              </div>
              <button aria-label="Close runtime panel" onClick={() => setRuntimeOpen(false)}>
                <X />
              </button>
            </div>
            {runtime ? (
              <>
            {runtime.gpu && (
              <div className="runtime-resource-card">
                <div className="runtime-resource-head">
                  <div>
                    <strong>VRAM hiện tại</strong>
                    <small>{runtime.gpu.device_name || (runtime.gpu.cuda_supported ? "CUDA GPU" : "Không phát hiện CUDA GPU")}</small>
                  </div>
                  <em className="runtime-status-pill optional">
                    {runtime.gpu.cuda_supported
                      ? `${formatVram(runtime.gpu.used_vram_mb)} / ${formatVram(runtime.gpu.total_vram_mb)}`
                      : "Không khả dụng"}
                  </em>
                </div>
                <div className="runtime-resource-grid">
                  <span>VRAM còn trống: {formatVram(runtime.gpu.free_vram_mb)}</span>
                  <span>Tiến trình VoxCPM đang giữ client: {runtime.gpu.active_voxcpm_clients}</span>
                  <span>PyTorch allocated: {formatVram(runtime.gpu.torch_allocated_mb)}</span>
                  <span>PyTorch peak: {formatVram(runtime.gpu.torch_peak_mb)}</span>
                </div>
                {runtime.gpu.resident_models.length > 0 && (
                  <div className="runtime-resource-list">
                    <strong>Model resident</strong>
                    <ul>
                      {runtime.gpu.resident_models.map((item) => <li key={item}>{item}</li>)}
                    </ul>
                  </div>
                )}
                {runtime.gpu.helper_processes.length > 0 && (
                  <div className="runtime-resource-list">
                    <strong>Tiến trình helper GPU</strong>
                    <ul>
                      {runtime.gpu.helper_processes.map((item) => <li key={item}>{item}</li>)}
                    </ul>
                  </div>
                )}
                <div className="runtime-resource-actions">
                  <button className="smoke-button" onClick={handleReleaseVram} disabled={releasingVram}>
                    <Server size={17} /> {releasingVram ? "Đang giải phóng VRAM..." : "Giải phóng VRAM"}
                  </button>
                  <small>
                    Thao tác này sẽ dừng các tiến trình ASR/TTS đang giữ VRAM và có thể làm job đang chạy thất bại.
                  </small>
                  {vramNotice && <div className="runtime-resource-note">{vramNotice}</div>}
                </div>
              </div>
            )}
            <div className="runtime-checks">
              {runtime.checks.map((check) => {
                const tone = runtimeCheckTone(check);
                return (
                  <div className={`runtime-check ${tone}`} key={check.id}>
                    <span className={`check-icon ${tone}`}>
                      <RuntimeCheckIcon check={check} />
                    </span>
                    <div>
                      <div className="runtime-check-title">
                        <strong>{check.display_name}</strong>
                        <span className={`runtime-kind ${check.required ? "required" : "optional"}`}>{runtimeCheckKindLabel(check)}</span>
                      </div>
                      <p>{check.message}</p>
                      <small>{check.action}</small>
                    </div>
                    <em className={`runtime-status-pill ${tone}`}>{runtimeCheckStatusLabel(check)}</em>
                  </div>
                );
              })}
            </div>
            <button className="smoke-button" onClick={runSmokeTest} disabled={testingRuntime}>
              <RefreshCw size={17} /> {testingRuntime ? "Đang kiểm tra..." : "Chạy thử nghiệm hệ thống"}
            </button>
              </>
            ) : (
              <div className="runtime-empty-state">
                <p>{runtimeLoading ? "Đang tải chi tiết môi trường thực thi..." : "Chưa lấy được trạng thái môi trường thực thi."}</p>
                {runtimeError && <small>{runtimeError}</small>}
                <button className="smoke-button" onClick={() => { void refreshRuntime(); }} disabled={runtimeLoading}>
                  <RefreshCw size={17} /> {runtimeLoading ? "Đang kiểm tra..." : "Tải lại trạng thái môi trường"}
                </button>
              </div>
            )}
          </section>
        </div>
      )}
    </>
  );
}
