import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import {
  Activity,
  AudioLines,
  CircleAlert,
  Clock3,
  Download,
  Info,
  Languages,
  Mic2,
  Plus,
  Radio,
  RefreshCw,
  Settings2,
  Subtitles,
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

import type { ClonedVoice, Job, JobsApi, OutputItem, ReleaseVramResult, RuntimeCheck, RuntimeReport, VoiceCalibrationStatus } from "../shared/contracts";
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
  "align_final_dub",
  "mix",
  "render",
  "qc",
] as const;

const PRESET_VOICES = [] as const;

const SETTINGS_NAV = [
  { id: "download", label: "Tải video", hint: "Cookies & yt-dlp", Icon: Download },
  { id: "translation", label: "Dịch thuật", hint: "Google / Gemini / OpenAI", Icon: Languages },
  { id: "audio", label: "Âm thanh", hint: "VAD & nhận dạng", Icon: AudioLines },
  { id: "tts", label: "Lồng tiếng", hint: "Engine & giọng đọc", Icon: Mic2 },
  { id: "subtitles", label: "Phụ đề", hint: "Chữ trên video", Icon: Subtitles },
] as const;

type SettingsTabId = (typeof SETTINGS_NAV)[number]["id"];
type SettingsHealth = "ready" | "attention" | "neutral";

function evaluateSettingsTabHealth(
  tabId: SettingsTabId,
  settings: Record<string, any>,
  activeTtsBackend: string,
): SettingsHealth {
  switch (tabId) {
    case "download":
      return "neutral";
    case "translation": {
      const backend = settings.translation_backend ?? "google_free";
      if (backend === "gemini" && (settings.gemini_api_keys ?? []).length === 0) {
        return "attention";
      }
      if (backend === "openai" && !settings.openai_api_key_configured) {
        return "attention";
      }
      return "ready";
    }
    case "audio":
      return "neutral";
    case "tts": {
      if (activeTtsBackend === "google_tts" && !settings.google_tts_api_key_configured) {
        return "attention";
      }
      if (activeTtsBackend === "gemini_tts" && (settings.gemini_api_keys ?? []).length === 0) {
        return "attention";
      }
      if (activeTtsBackend === "omnivoice" && !settings.omnivoice_ref_audio && !settings.omnivoice_instruct) {
        return "attention";
      }
      return "ready";
    }
    case "subtitles":
      return settings.subtitles_enabled === false ? "neutral" : "ready";
    default:
      return "neutral";
  }
}

function SettingsSectionHead({ title, description }: { title: string; description?: string }) {
  return (
    <div className="settings-section-head">
      <h3>{title}</h3>
      {description ? <p>{description}</p> : null}
    </div>
  );
}

function SettingsTabBar({
  activeTab,
  onSelect,
  settings,
  activeTtsBackend,
}: {
  activeTab: SettingsTabId;
  onSelect: (id: SettingsTabId) => void;
  settings: Record<string, any>;
  activeTtsBackend: string;
}) {
  return (
    <div className="settings-tabbar" role="tablist" aria-label="Nhóm cài đặt">
      {SETTINGS_NAV.map((tab) => {
        const health = evaluateSettingsTabHealth(tab.id, settings, activeTtsBackend);
        const Icon = tab.Icon;
        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-label={tab.label}
            aria-selected={activeTab === tab.id}
            className={`settings-tabbar__item${activeTab === tab.id ? " active" : ""}`}
            onClick={(event) => {
              onSelect(tab.id);
              event.currentTarget.scrollIntoView?.({ inline: "nearest", block: "nearest" });
            }}
          >
            <Icon size={16} aria-hidden="true" className="settings-tabbar__icon" />
            <span className="settings-tabbar__label">{tab.label}</span>
            <span
              className={`settings-tabbar__dot${health === "ready" ? " ready" : health === "attention" ? " attention" : ""}`}
              title={health === "attention" ? "Cần cấu hình thêm" : health === "ready" ? "Sẵn sàng" : "Tùy chọn"}
            />
          </button>
        );
      })}
    </div>
  );
}

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

const DUB_LANGUAGE_OPTIONS = [
  {
    id: "vi",
    label: "Tiếng Việt",
    defaultEdgeVoice: "vi-VN-HoaiMyNeural",
    defaultGoogleVoice: "vi-VN-Standard-A",
    omnivoiceLanguageId: "vi",
    speakingRate: 3.2,
    edgeLocale: "vi",
    googleLocale: "vi-VN",
  },
  {
    id: "th",
    label: "Tiếng Thái",
    defaultEdgeVoice: "th-TH-PremwadeeNeural",
    defaultGoogleVoice: "th-TH-Standard-A",
    omnivoiceLanguageId: "th",
    speakingRate: 2.8,
    edgeLocale: "th",
    googleLocale: "th-TH",
  },
] as const;

function getDubLanguageOption(language?: string) {
  const normalized = (language ?? "vi").trim().toLowerCase();
  return DUB_LANGUAGE_OPTIONS.find((option) => option.id === normalized) ?? DUB_LANGUAGE_OPTIONS[0];
}

const TTS_BACKEND_OPTIONS = [
  {
    id: "omnivoice",
    label: "OmniVoice",
    hint: "600+ ngôn ngữ, clone/design offline, cần GPU",
  },
  {
    id: "edge_tts",
    label: "Edge TTS",
    hint: "Microsoft neural, miễn phí, cần mạng",
  },
  {
    id: "google_tts",
    label: "Google TTS",
    hint: "Google Cloud Standard, hỗ trợ vi-VN và th-TH",
  },
  {
    id: "gemini_tts",
    label: "Gemini TTS",
    hint: "Google AI Studio, cần khóa API",
  },
] as const;

const GOOGLE_TTS_VOICES = [
  { id: "vi-VN-Standard-A", name: "Standard A — Nữ (vi)" },
  { id: "vi-VN-Standard-B", name: "Standard B — Nam (vi)" },
  { id: "vi-VN-Standard-C", name: "Standard C — Nữ (vi)" },
  { id: "vi-VN-Standard-D", name: "Standard D — Nam (vi)" },
  { id: "th-TH-Standard-A", name: "Standard A — Nữ (th)" },
  { id: "th-TH-Neural2-C", name: "Neural2 C — Nữ (th)" },
  { id: "th-TH-Neural2-D", name: "Neural2 D — Nam (th)" },
] as const;

function googleTtsVoicesForLanguage(language?: string) {
  const locale = getDubLanguageOption(language).googleLocale.toLowerCase();
  return GOOGLE_TTS_VOICES.filter((voice) => voice.id.toLowerCase().startsWith(locale));
}

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

const translateStepName = (name: string, dubLanguage?: string) => {
  const dubLang = getDubLanguageOption(dubLanguage);
  switch (name.toLowerCase()) {
    case "resolve": return "Phân tích liên kết";
    case "download": return "Tải video";
    case "extract_audio": return "Tách âm thanh";
    case "vad": return "Nhận diện giọng nói (VAD)";
    case "asr": return "Nhận dạng tiếng Trung (ASR)";
    case "normalize_segments": return "Chuẩn hóa phân đoạn";
    case "translate": return "Dịch thuật";
    case "tts": return `Lồng tiếng ${dubLang.label} (TTS)`;
    case "duration_repair": return "Khớp độ dài âm thanh";
    case "align_final_dub": return "Căn timing lồng tiếng";
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

type AppNoticeSeverity = "error" | "warning";
type CloneBackend = "omnivoice";

type AppNotice = {
  id: string;
  title: string;
  message: string;
  source: string;
  severity: AppNoticeSeverity;
  dismiss?: () => void;
  action?: { label: string; onClick: () => void };
};

function buildAppNotices(input: {
  error: string | null;
  voiceError: string | null;
  ttsPreviewError: string | null;
  runtimeError: string | null;
  runtime: RuntimeReport | null;
  backendConnection: BackendConnectionState;
  backendNotice: string | null;
  backendCheckedAt: number | null;
  jobs: Job[];
  onDismissError: () => void;
  onDismissVoiceError: () => void;
  onDismissTtsPreviewError: () => void;
  onDismissRuntimeError: () => void;
  onRecoverBackend: (reason: string) => void;
  formatBackendClock: (timestamp: number | null) => string;
}): AppNotice[] {
  const notices: AppNotice[] = [];

  if (input.error) {
    notices.push({
      id: "app-error",
      title: "Lỗi thao tác",
      message: input.error,
      source: "Ứng dụng",
      severity: "error",
      dismiss: input.onDismissError,
    });
  }

  if (input.voiceError) {
    notices.push({
      id: "voice-error",
      title: "Quản lý giọng đọc",
      message: input.voiceError,
      source: "Clone giọng",
      severity: "error",
      dismiss: input.onDismissVoiceError,
    });
  }

  if (input.ttsPreviewError && input.ttsPreviewError !== input.error) {
    notices.push({
      id: "tts-preview-error",
      title: "Nghe thử lồng tiếng",
      message: input.ttsPreviewError,
      source: "Cài đặt · Lồng tiếng",
      severity: "error",
      dismiss: input.onDismissTtsPreviewError,
    });
  }

  if (input.runtimeError) {
    notices.push({
      id: "runtime-error",
      title: "Môi trường thực thi",
      message: input.runtimeError,
      source: "Hệ thống",
      severity: "error",
      dismiss: input.onDismissRuntimeError,
    });
  }

  if (input.backendConnection === "offline") {
    notices.push({
      id: "backend-offline",
      title: "Backend không phản hồi",
      message:
        input.backendNotice
        ?? `Job có thể bị treo nếu backend đã dừng. Kiểm tra lần cuối lúc ${input.formatBackendClock(input.backendCheckedAt)}.`,
      source: "Backend",
      severity: "error",
      action: { label: "Khởi động lại", onClick: () => input.onRecoverBackend("Đang khởi động lại backend…") },
    });
  }

  if (input.backendConnection === "restarting") {
    notices.push({
      id: "backend-restarting",
      title: "Backend đang khởi động lại",
      message: input.backendNotice ?? "Tiến trình Python có thể đã dừng giữa chừng. Đang thử kết nối lại…",
      source: "Backend",
      severity: "warning",
    });
  }

  for (const job of input.jobs) {
    if ((job.status === "failed" || job.status === "interrupted") && job.last_error_message) {
      notices.push({
        id: `job-error-${job.id}-${job.updated_at}`,
        title: job.last_error_code ?? "Tiến trình thất bại",
        message: job.last_error_message,
        source: `Tiến trình · ${formatJobLabel(job)}`,
        severity: "error",
      });
    }
  }

  if (input.runtime) {
    for (const check of input.runtime.checks) {
      const status = check.status.toLowerCase();
      if (check.required && (status === "fail" || status === "blocked")) {
        notices.push({
          id: `runtime-check-${check.id}`,
          title: check.display_name,
          message: check.action ? `${check.message}\n\n${check.action}` : check.message,
          source: "Môi trường",
          severity: "error",
        });
      }
    }
  }

  return notices;
}

export function App({ api = defaultApi }: { api?: JobsApi }) {
  const [activeTab, setActiveTab] = useState<"jobs" | "outputs" | "settings" | "cloning">("jobs");
  const [jobs, setJobs] = useState<Job[]>([]);
  const jobsRef = useRef<Job[]>([]);
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
  const [errorsModalOpen, setErrorsModalOpen] = useState(false);
  const [dismissedNoticeIds, setDismissedNoticeIds] = useState<string[]>([]);
  const [toastNotice, setToastNotice] = useState<AppNotice | null>(null);
  const [toastVisible, setToastVisible] = useState(false);
  const [noticesReady, setNoticesReady] = useState(false);
  const prevNoticeIdsRef = useRef<Set<string>>(new Set());
  const noticesInitializedRef = useRef(false);
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
  const [backendChecking, setBackendChecking] = useState(false);
  const recoveringBackendRef = useRef(false);
  const lastBackendRecoverAtRef = useRef(0);
  const backendPollFailuresRef = useRef(0);
  const lastJobsPollAtRef = useRef(0);
  const lastJobDetailsRefreshAtRef = useRef(0);
  const lastOutputsRefreshAtRef = useRef(0);
  const lastClonedVoicesRefreshAtRef = useRef(0);

  useEffect(() => {
    jobsRef.current = jobs;
  }, [jobs]);

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
    subtitle_max_chars_per_line: 40,
    subtitle_max_lines_per_cue: 2,
    subtitle_min_cue_duration_ms: 700,
    subtitle_max_cue_duration_ms: 5500,
    subtitle_inter_cue_gap_ms: 50,
    gemini_api_keys: [],
    gemini_translation_model: "gemini-2.5-flash",
    openai_api_base: "https://api.openai.com/v1",
    openai_translation_model: "",
    gemini_tts_model: "gemini-2.5-flash-preview-tts",
    gemini_tts_voice: "Zephyr",
    tts_backend: "omnivoice",
    edge_tts_voice: "vi-VN-HoaiMyNeural",
    google_tts_voice: "vi-VN-Standard-A",
    google_tts_speaking_rate: 1,
    omnivoice_num_steps: 32,
    omnivoice_language_id: "",
    omnivoice_ref_text: "",
    omnivoice_instruct: "",
    omnivoice_auto_voice: true,
  });
  const [newGeminiKey, setNewGeminiKey] = useState("");
  const [newGoogleTtsKey, setNewGoogleTtsKey] = useState("");
  const [newOpenAiKey, setNewOpenAiKey] = useState("");
  const [, setSettingsSuccess] = useState(false);
  const settingsSaveInFlightRef = useRef(false);
  const settingsSaveQueuedRef = useRef(false);
  const lastSettingsSnapshotRef = useRef<string>("");
  const settingsAutoSaveReadyRef = useRef(false);
  const [settingsTab, setSettingsTab] = useState<SettingsTabId>("tts");
  const [ttsPreviewText, setTtsPreviewText] = useState("Xin chào, đây là bản nghe thử giọng đọc tiếng Việt.");
  const [ttsPreviewLoading, setTtsPreviewLoading] = useState(false);
  const [ttsPreviewUrl, setTtsPreviewUrl] = useState<string | null>(null);
  const [ttsPreviewError, setTtsPreviewError] = useState<string | null>(null);
  const [edgeTtsVoices, setEdgeTtsVoices] = useState<Array<{ id: string; name: string }>>([]);
  const [openAiModels, setOpenAiModels] = useState<Array<{ id: string; name: string }>>([]);
  const [openAiModelsLoading, setOpenAiModelsLoading] = useState(false);
  const ttsPreviewUrlRef = useRef<string | null>(null);

  // Cloned voices states
  const [clonedVoicesByBackend, setClonedVoicesByBackend] = useState<Record<CloneBackend, ClonedVoice[]>>({
    omnivoice: [],
  });
  const [cloningBackend, setCloningBackend] = useState<CloneBackend>("omnivoice");
  const [voiceName, setVoiceName] = useState("");
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [voiceRefText, setVoiceRefText] = useState("");
  const [voiceUploading, setVoiceUploading] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [voiceNotice, setVoiceNotice] = useState<string | null>(null);
  const [testText, setTestText] = useState<Record<string, string>>({});
  const [testSynthesizing, setTestSynthesizing] = useState<Record<string, boolean>>({});
  const [testAudioUrls, setTestAudioUrls] = useState<Record<string, string>>({});
  const [calibrationStatusByVoice, setCalibrationStatusByVoice] = useState<Record<string, VoiceCalibrationStatus>>({});
  const [calibrationBusy, setCalibrationBusy] = useState<Record<string, boolean>>({});
  const [calibrationDialogVoiceId, setCalibrationDialogVoiceId] = useState<string | null>(null);
  const activeTtsBackend = settings.tts_backend ?? "omnivoice";
  const activeCloneBackend: CloneBackend = "omnivoice";
  const clonedVoices = clonedVoicesByBackend[activeCloneBackend] ?? [];
  const cloningVoices = clonedVoicesByBackend[cloningBackend] ?? [];
  const selectedOmniVoiceClone = clonedVoicesByBackend.omnivoice.find(
    (voice) => voice.wav_path === settings.omnivoice_ref_audio,
  );

  useEffect(() => {
    if (activeTab !== "cloning") return;
    setCloningBackend("omnivoice");
  }, [activeTab]);
  const loadEdgeTtsVoices = useCallback(async () => {
    const language = settings.translation_target_language ?? "vi";
    try {
      const voices = await api.listTtsVoices("edge_tts", language);
      setEdgeTtsVoices(voices);
    } catch {
      const lang = getDubLanguageOption(language);
      setEdgeTtsVoices(
        lang.id === "th"
          ? [
              { id: "th-TH-PremwadeeNeural", name: "Premwadee (Nữ)" },
              { id: "th-TH-NiwatNeural", name: "Niwat (Nam)" },
            ]
          : [
              { id: "vi-VN-HoaiMyNeural", name: "Hoài My (Nữ)" },
              { id: "vi-VN-NamMinhNeural", name: "Nam Minh (Nam)" },
            ],
      );
    }
  }, [api, settings.translation_target_language]);

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
  }, [activeTab, settingsTab, activeTtsBackend, loadEdgeTtsVoices, settings.translation_target_language]);

  const activeDubLanguage = getDubLanguageOption(settings.translation_target_language);
  const activeGoogleTtsVoices = useMemo(
    () => googleTtsVoicesForLanguage(settings.translation_target_language),
    [settings.translation_target_language],
  );

  const applyDubLanguageChange = useCallback(async (languageId: string) => {
    const lang = getDubLanguageOption(languageId);
    const payload = {
      translation_target_language: lang.id,
      edge_tts_voice: lang.defaultEdgeVoice,
      google_tts_voice: lang.defaultGoogleVoice,
      omnivoice_language_id: lang.omnivoiceLanguageId,
      vietnamese_speaking_rate_wps: lang.speakingRate,
    };
    setSettings((current) => ({ ...current, ...payload }));
    setSettingsSuccess(false);
    setError(null);
    try {
      const updated = await api.updateSettings(payload);
      lastSettingsSnapshotRef.current = "";
      setSettings((current) => ({ ...current, ...updated }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể đổi ngôn ngữ lồng tiếng");
    }
  }, [api]);

  useEffect(() => {
    if (activeTab === "settings" && settingsTab === "translation" && settings.translation_backend === "openai") {
      void loadOpenAiModels();
    }
  }, [activeTab, settingsTab, settings.translation_backend, loadOpenAiModels]);

  const recoverBackend = useCallback(async (reason: string) => {
    const now = Date.now();
    if (recoveringBackendRef.current || now - lastBackendRecoverAtRef.current < 8_000) {
      return;
    }
    lastBackendRecoverAtRef.current = now;
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
    let cancelled = false;

    void (async () => {
      try {
        await refreshJobs();
        await refreshOutputs();
        await loadSettings();
        if (activeTab === "cloning" || activeTab === "settings") {
          await refreshClonedVoices();
        }
      } finally {
        if (!cancelled) {
          setNoticesReady(true);
        }
      }
    })();

    // Poll every 2s for responsiveness, but throttle expensive fetches while idle.
    const interval = setInterval(() => {
      const now = Date.now();
      const hasActiveJobs = jobsRef.current.some((job) =>
        !["done", "failed", "cancelled"].includes(String(job.status || "").toLowerCase()),
      );
      if (!hasActiveJobs && now - lastJobsPollAtRef.current < 8_000) {
        return;
      }
      lastJobsPollAtRef.current = now;
      api.listJobs().then((newJobs) => {
        setJobs(newJobs);
        if (selectedJobId) {
          const updated = newJobs.find((j) => j.id === selectedJobId);
          if (updated && now - lastJobDetailsRefreshAtRef.current >= 4_000) {
            lastJobDetailsRefreshAtRef.current = now;
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
      if (activeTab === "outputs" && now - lastOutputsRefreshAtRef.current >= 6_000) {
        lastOutputsRefreshAtRef.current = now;
        refreshOutputs();
      }
      if (activeTab === "cloning" && now - lastClonedVoicesRefreshAtRef.current >= 8_000) {
        lastClonedVoicesRefreshAtRef.current = now;
        refreshClonedVoices();
      }
    }, 2_000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
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

  const checkBackendHealth = useCallback(async () => {
    if (recoveringBackendRef.current || backendChecking) {
      return;
    }
    setBackendChecking(true);
    setBackendConnection((current) => (current === "restarting" ? "restarting" : "checking"));
    const baseUrl = backend?.kind === "ready" ? backend.base_url : BACKEND_BASE;
    try {
      const ok = await probeBackendHealth(baseUrl);
      const now = Date.now();
      setBackendCheckedAt(now);
      if (ok) {
        backendPollFailuresRef.current = 0;
        setBackendConnection("online");
        setBackendLastOkAt(now);
        setBackendNotice(null);
        setBackend({ kind: "ready", base_url: baseUrl });
        return;
      }
      backendPollFailuresRef.current += 1;
      setBackendConnection((current) => (current === "restarting" ? "restarting" : "offline"));
      setBackendNotice("Backend không phản hồi. Ứng dụng sẽ không tự tắt backend; hãy dùng nút khởi động lại nếu cần.");
    } finally {
      setBackendChecking(false);
    }
  }, [backend, backendChecking]);

  async function refreshClonedVoices(backend?: CloneBackend) {
    try {
      if (backend) {
        const voices = await api.listClonedVoices(backend);
        setClonedVoicesByBackend((prev) => ({ ...prev, [backend]: voices }));
        return;
      }
      const omnivoiceVoices = await api.listClonedVoices("omnivoice");
      setClonedVoicesByBackend({
        omnivoice: omnivoiceVoices,
      });
    } catch (cause) {
      setVoiceError(cause instanceof Error ? cause.message : "Không thể tải danh sách giọng clone");
    }
  }

  function formatDurationProfileStatus(voice: ClonedVoice, live?: VoiceCalibrationStatus): string {
    const status = live?.status || voice.duration_profile_status || "not_started";
    if (status === "running" || status === "queued") {
      const completed = live?.completed ?? 0;
      const total = live?.total ?? 0;
      return total > 0 ? `Đang hiệu chỉnh ${completed}/${total}` : "Đang hiệu chỉnh";
    }
    if (status === "ready") {
      const count = voice.duration_profile_sample_count || live?.accepted || 0;
      return `Sẵn sàng — ${count} mẫu`;
    }
    if (status === "partial") {
      const count = voice.duration_profile_sample_count || live?.accepted || 0;
      return `Một phần — ${count} mẫu`;
    }
    if (status === "failed") return "Lỗi hiệu chỉnh";
    if (status === "stale") return "Cần chạy lại";
    if (status === "cancelled") return "Đã hủy hiệu chỉnh";
    return "Chưa hiệu chỉnh";
  }

  async function refreshCalibrationStatus(voiceId: string) {
    try {
      const status = await api.getVoiceCalibration(voiceId);
      setCalibrationStatusByVoice((prev) => ({ ...prev, [voiceId]: status }));
      return status;
    } catch {
      return null;
    }
  }

  async function handleStartCalibration(voiceId: string, mode: "quick" | "standard" | "full") {
    setCalibrationBusy((prev) => ({ ...prev, [voiceId]: true }));
    setVoiceError(null);
    try {
      const status = await api.startVoiceCalibration(voiceId, mode);
      setCalibrationStatusByVoice((prev) => ({ ...prev, [voiceId]: status }));
      setCalibrationDialogVoiceId(null);
      await refreshClonedVoices(cloningBackend);
    } catch (cause) {
      setVoiceError(cause instanceof Error ? cause.message : "Không thể bắt đầu hiệu chỉnh");
    } finally {
      setCalibrationBusy((prev) => ({ ...prev, [voiceId]: false }));
    }
  }

  async function handleCancelCalibration(voiceId: string) {
    setCalibrationBusy((prev) => ({ ...prev, [voiceId]: true }));
    try {
      await api.cancelVoiceCalibration(voiceId);
      await refreshCalibrationStatus(voiceId);
      await refreshClonedVoices(cloningBackend);
    } catch (cause) {
      setVoiceError(cause instanceof Error ? cause.message : "Không thể hủy hiệu chỉnh");
    } finally {
      setCalibrationBusy((prev) => ({ ...prev, [voiceId]: false }));
    }
  }

  async function handleResumeCalibration(voiceId: string) {
    setCalibrationBusy((prev) => ({ ...prev, [voiceId]: true }));
    try {
      const status = await api.resumeVoiceCalibration(voiceId);
      setCalibrationStatusByVoice((prev) => ({ ...prev, [voiceId]: status }));
      await refreshClonedVoices(cloningBackend);
    } catch (cause) {
      setVoiceError(cause instanceof Error ? cause.message : "Không thể tiếp tục hiệu chỉnh");
    } finally {
      setCalibrationBusy((prev) => ({ ...prev, [voiceId]: false }));
    }
  }

  async function handleResetDurationProfile(voiceId: string) {
    if (!confirm("Đặt lại profile tốc độ đọc? Giọng clone vẫn giữ nguyên.")) return;
    try {
      await api.resetVoiceDurationProfile(voiceId);
      await refreshClonedVoices(cloningBackend);
      setCalibrationStatusByVoice((prev) => {
        const next = { ...prev };
        delete next[voiceId];
        return next;
      });
    } catch (cause) {
      setVoiceError(cause instanceof Error ? cause.message : "Không thể đặt lại profile");
    }
  }

  useEffect(() => {
    if (activeTab !== "cloning") return;
    const running = cloningVoices.filter((voice) =>
      ["running", "queued"].includes(voice.duration_profile_status || calibrationStatusByVoice[voice.id]?.status || "")
    );
    if (running.length === 0) return;
    const timer = window.setInterval(() => {
      running.forEach((voice) => {
        void refreshCalibrationStatus(voice.id).then((status) => {
          if (status && ["ready", "partial", "failed", "cancelled"].includes(status.status)) {
            void refreshClonedVoices(cloningBackend);
          }
        });
      });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [activeTab, cloningBackend, cloningVoices, calibrationStatusByVoice]);

  async function handleUploadVoice(e: FormEvent) {
    e.preventDefault();
    if (!voiceName || !voiceFile) {
      setVoiceError("Vui lòng nhập tên giọng và chọn tệp âm thanh mẫu (.wav hoặc .mp3).");
      return;
    }
    if (!voiceRefText.trim()) {
      setVoiceError("OmniVoice clone cần ref_text — hãy dán nguyên văn nội dung audio mẫu.");
      return;
    }
    setVoiceUploading(true);
    setVoiceError(null);
    setVoiceNotice(null);
    try {
      const created = await api.createClonedVoice(
        voiceName,
        voiceFile,
        cloningBackend,
        voiceRefText,
      );
      setVoiceNotice(
        `Đã lưu giọng OmniVoice với ref_text ${(created.transcript ?? "").length} ký tự. Sẵn sàng clone.`,
      );
      setCalibrationDialogVoiceId(created.id);
      setVoiceName("");
      setVoiceFile(null);
      setVoiceRefText("");
      const fileInput = document.getElementById("voice-file-input") as HTMLInputElement;
      if (fileInput) fileInput.value = "";
      await refreshClonedVoices(cloningBackend);
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
      await api.deleteClonedVoice(id, cloningBackend);
      await refreshClonedVoices(cloningBackend);
    } catch (cause) {
      setVoiceError(cause instanceof Error ? cause.message : "Xóa giọng nói thất bại");
    }
  }

  async function handleTestVoice(id: string) {
    const text = testText[id] || "Chào bạn, đây là thử nghiệm giọng nói clone offline của tôi.";
    const selectedVoice = cloningVoices.find((voice) => voice.id === id);
    if (cloningBackend === "omnivoice" && !selectedVoice?.transcribed) {
      setVoiceError("Giọng OmniVoice này chưa có ref_text. Hãy upload lại và dán transcript khớp audio mẫu.");
      return;
    }
    setTestSynthesizing(prev => ({ ...prev, [id]: true }));
    setVoiceError(null);
    try {
      const mode = "reference";
      const blob = await api.testClonedVoice(id, text, mode, cloningBackend);
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

  function buildSettingsSavePayload() {
    const {
      gemini_api_keys,
      gemini_api_key_add,
      gemini_api_key_remove,
      gemini_api_key_update,
      google_tts_api_key_configured,
      google_tts_api_key_masked,
      openai_api_key_configured,
      openai_api_key_masked,
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
    delete savePayload.omnivoice_speed;
    return { savePayload, pendingGeminiKey, pendingGoogleTtsKey, pendingOpenAiKey };
  }

  async function persistSettings({ silent = false }: { silent?: boolean } = {}) {
    if (settingsSaveInFlightRef.current) {
      settingsSaveQueuedRef.current = true;
      return;
    }

    const { savePayload, pendingGeminiKey, pendingGoogleTtsKey, pendingOpenAiKey } =
      buildSettingsSavePayload();
    const snapshot = JSON.stringify({
      savePayload,
      pendingGeminiKey,
      pendingGoogleTtsKey,
      pendingOpenAiKey,
    });
    if (snapshot === lastSettingsSnapshotRef.current) {
      return;
    }

    setSettingsSuccess(false);
    settingsSaveInFlightRef.current = true;
    try {
      const updated = await api.updateSettings(savePayload);
      lastSettingsSnapshotRef.current = snapshot;
      setSettings((current) => ({
        ...current,
        ...updated,
      }));
      if (pendingGeminiKey) {
        setNewGeminiKey("");
      }
      if (pendingGoogleTtsKey) {
        setNewGoogleTtsKey("");
      }
      if (pendingOpenAiKey) {
        setNewOpenAiKey("");
      }
      if (!silent) {
        setSettingsSuccess(true);
        setTimeout(() => setSettingsSuccess(false), 3000);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể lưu cài đặt");
      lastSettingsSnapshotRef.current = "";
    } finally {
      settingsSaveInFlightRef.current = false;
      if (settingsSaveQueuedRef.current) {
        settingsSaveQueuedRef.current = false;
        void persistSettings({ silent: true });
      }
    }
  }

  useEffect(() => {
    if (activeTab !== "settings") {
      settingsAutoSaveReadyRef.current = false;
      return;
    }
    if (!settingsAutoSaveReadyRef.current) {
      settingsAutoSaveReadyRef.current = true;
      return;
    }
    const timer = window.setTimeout(() => {
      void persistSettings({ silent: true });
    }, 150);
    return () => window.clearTimeout(timer);
  }, [activeTab, settings, newGeminiKey, newGoogleTtsKey, newOpenAiKey]);

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
    const backend = activeTtsBackend;
    if (
      backend === "omnivoice"
      && settings.omnivoice_ref_audio?.trim()
      && !settings.omnivoice_ref_text?.trim()
    ) {
      const message = "OmniVoice clone cần ref_text khớp audio mẫu. Chọn giọng từ tab Clone hoặc dán transcript vào ô ref_text.";
      setTtsPreviewError(message);
      setError(message);
      return;
    }
    setTtsPreviewLoading(true);
    setError(null);
    setTtsPreviewError(null);
    try {
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
          omnivoice_ref_audio: settings.omnivoice_ref_audio,
          omnivoice_ref_text: settings.omnivoice_ref_text,
          omnivoice_instruct: settings.omnivoice_instruct,
          omnivoice_auto_voice: settings.omnivoice_auto_voice,
          omnivoice_num_steps: settings.omnivoice_num_steps,
          omnivoice_language_id: settings.omnivoice_language_id,
        },
      });
      if (ttsPreviewUrlRef.current) {
        URL.revokeObjectURL(ttsPreviewUrlRef.current);
      }
      const url = URL.createObjectURL(blob);
      ttsPreviewUrlRef.current = url;
      setTtsPreviewUrl(url);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "Không thể nghe thử giọng đọc";
      setTtsPreviewError(message);
      setError(message);
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

  const appNotices = useMemo(
    () => buildAppNotices({
      error,
      voiceError,
      ttsPreviewError,
      runtimeError,
      runtime,
      backendConnection,
      backendNotice,
      backendCheckedAt,
      jobs,
      onDismissError: () => setError(null),
      onDismissVoiceError: () => setVoiceError(null),
      onDismissTtsPreviewError: () => setTtsPreviewError(null),
      onDismissRuntimeError: () => setRuntimeError(null),
      onRecoverBackend: (reason) => void recoverBackend(reason),
      formatBackendClock,
    }),
    [
      error,
      voiceError,
      ttsPreviewError,
      runtimeError,
      runtime,
      backendConnection,
      backendNotice,
      backendCheckedAt,
      jobs,
      recoverBackend,
    ],
  );

  const visibleNotices = useMemo(
    () => appNotices.filter((notice) => !dismissedNoticeIds.includes(notice.id)),
    [appNotices, dismissedNoticeIds],
  );

  const dismissNotice = useCallback((notice: AppNotice) => {
    notice.dismiss?.();
    setDismissedNoticeIds((prev) => (prev.includes(notice.id) ? prev : [...prev, notice.id]));
  }, []);

  const dismissAllNotices = useCallback(() => {
    for (const notice of visibleNotices) {
      notice.dismiss?.();
    }
    setDismissedNoticeIds((prev) => [...new Set([...prev, ...visibleNotices.map((notice) => notice.id)])]);
  }, [visibleNotices]);

  useEffect(() => {
    if (!noticesReady) {
      return;
    }

    const currentIds = new Set(visibleNotices.map((notice) => notice.id));

    if (!noticesInitializedRef.current) {
      prevNoticeIdsRef.current = currentIds;
      noticesInitializedRef.current = true;
      return;
    }

    const brandNew = visibleNotices.filter((notice) => !prevNoticeIdsRef.current.has(notice.id));
    prevNoticeIdsRef.current = currentIds;

    if (brandNew.length === 0) {
      return;
    }

    const latest = brandNew[brandNew.length - 1];
    setToastNotice(latest);
    setToastVisible(true);

    const fadeOutTimer = window.setTimeout(() => setToastVisible(false), 5000);
    const clearTimer = window.setTimeout(() => setToastNotice(null), 5600);

    return () => {
      window.clearTimeout(fadeOutTimer);
      window.clearTimeout(clearTimer);
    };
  }, [visibleNotices, noticesReady]);

  if (backend?.kind === "environment_missing") {
    return (
      <div className="p-6 max-w-3xl mx-auto text-zinc-100">
        <h1 className="text-xl font-semibold text-red-400">Môi trường dev chưa sẵn sàng</h1>
        <p className="mt-2 text-zinc-300">App không tìm thấy đủ file cần thiết trong repo. Chạy <code>pnpm run setup</code> rồi thử lại.</p>
        <div className="mt-4 rounded bg-zinc-900 p-3">
          <strong>Repo root</strong>
          <code className="block mt-1 text-sm text-zinc-300">{backend.root}</code>
        </div>
        <ul className="mt-4 list-disc pl-6 text-sm text-red-200">
          {backend.missing_items.map((item) => <li key={item}>{item}</li>)}
        </ul>
        <button onClick={() => location.reload()} className="mt-4 underline">Thử lại</button>
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
      {toastNotice && (
        <div
          className={`notice-toast notice-toast--${toastNotice.severity}${toastVisible ? " notice-toast--visible" : ""}`}
          role="status"
          onClick={() => setErrorsModalOpen(true)}
        >
          <span className="notice-toast__icon" aria-hidden="true">
            {toastNotice.severity === "warning" ? <AlertTriangle size={18} /> : <CircleAlert size={18} />}
          </span>
          <div className="notice-toast__body">
            <strong>{toastNotice.title}</strong>
            <span>{toastNotice.message}</span>
          </div>
          <button
            type="button"
            className="notice-toast__close"
            aria-label="Đóng thông báo"
            onClick={(event) => {
              event.stopPropagation();
              setToastVisible(false);
              window.setTimeout(() => setToastNotice(null), 350);
            }}
          >
            <X size={16} />
          </button>
        </div>
      )}
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
          <button
            type="button"
            className={`errors-foot-btn${visibleNotices.length > 0 ? " has-notices" : ""}`}
            onClick={() => setErrorsModalOpen(true)}
            aria-label={visibleNotices.length > 0 ? `Lỗi và thông báo (${visibleNotices.length})` : "Lỗi và thông báo"}
          >
            <CircleAlert size={14} />
            <div>
              <strong>Lỗi{visibleNotices.length > 0 ? ` (${visibleNotices.length})` : ""}</strong>
              <small>{visibleNotices.length > 0 ? "Có thông báo cần xem" : "Không có thông báo"}</small>
            </div>
          </button>
          <button
            type="button"
            className={`backend-status backend-status--${backendConnection}`}
            onClick={() => void checkBackendHealth()}
            disabled={backendChecking || recoveringBackendRef.current}
            title={
              backendConnection === "online"
                ? `Backend phản hồi ổn định. Lần kiểm tra gần nhất: ${formatBackendClock(backendCheckedAt)}`
                : `Backend ${backendConnectionLabel(backendConnection).toLowerCase()} — bấm để kiểm tra lại`
            }
            aria-label="Kiểm tra backend"
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
                    : backendCheckedAt
                      ? `Mất kết nối · kiểm tra ${formatBackendClock(backendCheckedAt)}`
                      : "Chưa kiểm tra"}
              </small>
            </div>
            <RefreshCw size={14} className={backendChecking ? "spin" : undefined} />
          </button>
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

      <main className={activeTab === "settings" ? "main--settings" : undefined}>
        {activeTab === "jobs" && (
          <>
            <header>
              <div>
                <h1>Bảng điều khiển tiến trình</h1>
              </div>
              <span className="phase">Tiến trình hoạt động</span>
            </header>

            <section className="new-job">
              <div>
                <h2>Tạo tiến trình lồng tiếng mới</h2>
                <p>
                  Tải từ liên kết Douyin/Bilibili hoặc chọn video local — nhiều file local sẽ được xếp hàng.
                  Ngôn ngữ lồng tiếng hiện tại: <strong>{activeDubLanguage.label}</strong> (đổi trong Cài đặt → Dịch thuật).
                </p>
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
                          <div key={step.name} title={`${translateStepName(step.name, settings.translation_target_language)}: ${translateStepStatus(step.status)}`} className={step.status} />
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
            <header className="settings-header settings-header--cloning">
              <div>
                <h1>Quản lý giọng nói nhân bản (Voice Cloning)</h1>
                <p className="settings-subtitle">
                  OmniVoice: upload audio mẫu 3–10 giây và dán ref_text khớp nguyên văn với nội dung audio.
                </p>
              </div>
            </header>

            {voiceNotice && (
              <div className="success" style={{ marginBottom: "20px" }}>
                <span className="voice-notice-check" aria-hidden="true">
                  <CheckCircle2 size={14} />
                </span>
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
                    Tải lên audio mẫu (.wav hoặc .mp3, 3–10 giây) và dán ref_text — transcript chính xác của đoạn audio đó.
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

                    <label className="settings-label">
                      <span>ref_text (bắt buộc) — nội dung audio mẫu</span>
                      <textarea
                        required
                        className="settings-input"
                        rows={4}
                        placeholder="Dán nguyên văn những gì được nói trong file audio mẫu..."
                        value={voiceRefText}
                        onChange={(e) => setVoiceRefText(e.target.value)}
                        style={{ minHeight: "96px", resize: "vertical" }}
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
                    {cloningVoices.length === 0 ? (
                      <div className="empty-keys-placeholder">Chưa có giọng nhân bản nào. Hãy tải lên một tệp mẫu ở cột bên trái.</div>
                    ) : (
                      cloningVoices.map((voice) => (
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
                                {cloningBackend === "omnivoice"
                                  ? voice.transcribed
                                    ? `ref_text (${(voice.transcript ?? "").length} ký tự): ${(voice.transcript ?? "").slice(0, 120)}${(voice.transcript ?? "").length > 120 ? "…" : ""}`
                                    : "Thiếu ref_text — upload lại và dán transcript khớp audio mẫu."
                                  : voice.transcribed
                                    ? `Transcript sẵn sàng (${(voice.transcript ?? "").length} ký tự) cho ultimate clone.`
                                    : "Chưa có transcript .txt; ultimate clone sẽ cần transcript chính xác."}
                              </span>
                            </div>

                            <div style={{ display: "flex", flexDirection: "column", gap: "8px", background: "rgba(90, 120, 255, 0.05)", border: "1px solid rgba(120, 140, 255, 0.15)", borderRadius: "8px", padding: "10px" }}>
                              <span style={{ fontSize: "11px", color: "#a79aff", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.03em" }}>Hiệu chỉnh tốc độ đọc</span>
                              <span style={{ fontSize: "12px", color: "#b6c0d0" }}>
                                Tạo profile tốc độ riêng cho giọng này để dự đoán thời lượng lồng tiếng chính xác hơn.
                              </span>
                              <div style={{ fontSize: "12px", color: "#dbe7ff" }}>
                                Trạng thái: {formatDurationProfileStatus(voice, calibrationStatusByVoice[voice.id])}
                              </div>
                              {(["running", "queued"].includes(voice.duration_profile_status || calibrationStatusByVoice[voice.id]?.status || "")) && (
                                <div style={{ fontSize: "12px", color: "#cbd5e1" }}>
                                  Đang hiệu chỉnh: {calibrationStatusByVoice[voice.id]?.completed ?? 0} / {calibrationStatusByVoice[voice.id]?.total ?? "?"}
                                  {" · "}
                                  Mẫu hợp lệ: {calibrationStatusByVoice[voice.id]?.accepted ?? 0}
                                  {" · "}
                                  Mẫu bỏ qua: {calibrationStatusByVoice[voice.id]?.rejected ?? 0}
                                </div>
                              )}
                              {(voice.duration_profile_status === "ready" || voice.duration_profile_status === "partial") && (
                                <div style={{ fontSize: "12px", color: "#9be7b1" }}>
                                  Sai số dự đoán trung vị: {calibrationStatusByVoice[voice.id]?.validation_median_error_ms ?? "—"} ms
                                </div>
                              )}
                              <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                                {!["running", "queued"].includes(voice.duration_profile_status || "") && (
                                  <>
                                    <button type="button" className="key-action-btn save-btn" disabled={calibrationBusy[voice.id]} onClick={() => handleStartCalibration(voice.id, "quick")}>Nhanh (~20)</button>
                                    <button type="button" className="key-action-btn save-btn" disabled={calibrationBusy[voice.id]} onClick={() => handleStartCalibration(voice.id, "standard")}>Chuẩn (~50)</button>
                                    <button type="button" className="key-action-btn save-btn" disabled={calibrationBusy[voice.id]} onClick={() => handleStartCalibration(voice.id, "full")}>Đầy đủ (~100)</button>
                                  </>
                                )}
                                {["running", "queued"].includes(voice.duration_profile_status || calibrationStatusByVoice[voice.id]?.status || "") && (
                                  <button type="button" className="key-action-btn delete-btn" disabled={calibrationBusy[voice.id]} onClick={() => handleCancelCalibration(voice.id)}>Hủy</button>
                                )}
                                {["cancelled", "failed"].includes(voice.duration_profile_status || "") && (
                                  <button type="button" className="key-action-btn save-btn" disabled={calibrationBusy[voice.id]} onClick={() => handleResumeCalibration(voice.id)}>Tiếp tục</button>
                                )}
                                {(voice.duration_profile_status === "ready" || voice.duration_profile_status === "partial") && (
                                  <>
                                    <button type="button" className="key-action-btn save-btn" disabled={calibrationBusy[voice.id]} onClick={() => handleStartCalibration(voice.id, "full")}>Cải thiện profile</button>
                                    <button type="button" className="key-action-btn delete-btn" onClick={() => handleResetDurationProfile(voice.id)}>Đặt lại profile</button>
                                  </>
                                )}
                              </div>
                            </div>

                            {/* Play reference audio */}
                            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                              <span style={{ fontSize: "12px", color: "#8b949e" }}>Âm thanh mẫu:</span>
                              <audio
                                controls
                                src={`http://127.0.0.1:8765/api/cloned-voices/${voice.id}/wav?backend=${encodeURIComponent(cloningBackend)}`}
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

            {calibrationDialogVoiceId && (
              <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.55)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 50 }}>
                <div className="settings-card" style={{ width: "min(520px, 92vw)", padding: "20px" }}>
                  <h3 style={{ marginTop: 0 }}>Voice clone thành công</h3>
                  <p className="card-description">
                    Bạn có muốn chạy hiệu chỉnh tốc độ đọc (Standard ~50 câu) để dự đoán thời lượng lồng tiếng tốt hơn ngay từ job đầu tiên?
                  </p>
                  <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
                    <button type="button" className="gradient-button" onClick={() => handleStartCalibration(calibrationDialogVoiceId, "standard")}>Bắt đầu Chuẩn</button>
                    <button type="button" className="key-action-btn save-btn" onClick={() => handleStartCalibration(calibrationDialogVoiceId, "quick")}>Nhanh</button>
                    <button type="button" className="key-action-btn delete-btn" onClick={() => setCalibrationDialogVoiceId(null)}>Bỏ qua</button>
                  </div>
                </div>
              </div>
            )}
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
              <div className="settings-header__row">
                <div>
                  <h1>Cài đặt</h1>
                  <p className="settings-subtitle">
                    Tự lưu khi thay đổi. Chọn nhóm bên dưới để cấu hình pipeline.
                  </p>
                </div>
                <div className="settings-readiness" aria-label="Trạng thái pipeline">
                  {([
                    { id: "translation" as const, label: "Dịch" },
                    { id: "tts" as const, label: "Lồng tiếng" },
                  ]).map((item) => {
                    const health = evaluateSettingsTabHealth(item.id, settings, activeTtsBackend);
                    return (
                      <button
                        key={item.id}
                        type="button"
                        className={`settings-readiness__chip${health === "ready" ? " ready" : health === "attention" ? " attention" : ""}`}
                        onClick={() => setSettingsTab(item.id)}
                      >
                        {health === "ready" ? <CheckCircle2 size={13} /> : health === "attention" ? <CircleAlert size={13} /> : null}
                        <span>{item.label}</span>
                        <em>{health === "ready" ? "Sẵn sàng" : health === "attention" ? "Cần cấu hình" : "Tùy chọn"}</em>
                      </button>
                    );
                  })}
                  <span className="settings-readiness__engine">
                    Engine: {TTS_BACKEND_OPTIONS.find((o) => o.id === activeTtsBackend)?.label ?? activeTtsBackend}
                  </span>
                </div>
              </div>
            </header>
            <div className="settings-page">
              <SettingsTabBar
                activeTab={settingsTab}
                onSelect={setSettingsTab}
                settings={settings}
                activeTtsBackend={activeTtsBackend}
              />
              <div className="settings-content-scroll">
              <form className="settings-panel">
                {settingsTab === "download" && (
                  <div className="settings-tab-panel" role="tabpanel">
                    <SettingsSectionHead
                      title="Tải video (Douyin / Bilibili)"
                      description="Ưu tiên cookies.txt; nếu thiếu app tự lấy cookie Firefox → Chrome → Edge → Brave."
                    />
                    <div className="settings-form-fields">
                      <label className="settings-label">
                        <span>Đường dẫn cookies.txt (Douyin)</span>
                        <input
                          className="settings-input"
                          placeholder="C:\Users\...\AppData\Local\DouyinVietnamizer\cookies\douyin_cookies.txt"
                          value={settings.cookies_file ?? ""}
                          onChange={(e) => setSettings({ ...settings, cookies_file: e.target.value })}
                        />
                      </label>
                      <p className="card-description card-description--compact">
                        Export từ Firefox (extension cookies.txt) khi đã đăng nhập Douyin. Để trống để chỉ dùng cookie trình duyệt.
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
                        <p className="settings-notice settings-notice--ok">{ytDlpNotice}</p>
                      )}
                    </div>
                  </div>
                )}

                {settingsTab === "translation" && (
                  <div className="settings-tab-panel" role="tabpanel">
                    <div className="settings-translation-layout">
                      <div className="settings-translation-main">
                        <SettingsSectionHead
                          title="Dịch thuật"
                          description="Chọn ngôn ngữ lồng tiếng và bộ dịch cho phụ đề, nội dung TTS."
                        />
                      <label className="settings-label">
                        <SettingsFieldLabel
                          label="Ngôn ngữ lồng tiếng"
                          hint="Video sẽ được dịch và lồng tiếng sang ngôn ngữ này. Đổi ngôn ngữ sẽ tự cập nhật giọng TTS mặc định."
                        />
                        <select
                          className="settings-input"
                          value={settings.translation_target_language ?? "vi"}
                          onChange={(e) => applyDubLanguageChange(e.target.value)}
                        >
                          {DUB_LANGUAGE_OPTIONS.map((option) => (
                            <option key={option.id} value={option.id}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      </label>
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
                      </div>

                      <div className="settings-translation-side">
                        <section className="settings-card">
                          <SettingsSectionHead
                            title="Khóa API Gemini"
                            description="Dùng cho dịch Gemini và Gemini TTS — thêm nhiều khóa để luân phiên quota."
                          />
                        <div className="settings-section settings-section--nested">
                          <SettingsSectionHead title="Quản lý khóa Gemini" />
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
                        </div>
                        </section>
                      </div>
                    </div>
                  </div>
                )}

                {settingsTab === "audio" && (
                  <div className="settings-tab-panel" role="tabpanel">
                    <div className="settings-audio-layout">
                      <div className="settings-audio-main">
                        <SettingsSectionHead
                          title="Nhận diện giọng nói (VAD)"
                          description="Xác định vùng có lời nói trước khi nhận dạng tiếng Trung. Chạy lại job từ VAD hoặc ASR để áp dụng."
                        />
                        <div className="settings-form-fields settings-form-fields--fluid">
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
                        </div>
                      </div>

                      <div className="settings-audio-side">
                        <SettingsSectionHead
                          title="Tinh chỉnh VAD & ASR"
                          description="Chỉ mở khi cần chỉnh chi tiết Silero, FFmpeg hoặc ASR thưa."
                        />
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
                    </div>
                  </div>
                )}

                {settingsTab === "tts" && (
                  <div className="settings-tab-panel settings-tts-panel" role="tabpanel">
                    <div className="settings-engine-block">
                      <SettingsSectionHead
                        title="Engine lồng tiếng"
                        description="Chọn engine — cấu hình bên dưới chỉ hiện đúng engine đang dùng."
                      />
                      <div className="settings-engine-rail" role="radiogroup" aria-label="Engine lồng tiếng">
                        {TTS_BACKEND_OPTIONS.map((option) => (
                          <button
                            key={option.id}
                            type="button"
                            role="radio"
                            aria-checked={activeTtsBackend === option.id}
                            className={`settings-engine-card${activeTtsBackend === option.id ? " active" : ""}`}
                            onClick={() => setSettings({ ...settings, tts_backend: option.id })}
                          >
                            <span className="settings-engine-card__mark" aria-hidden="true" />
                            <strong>{option.label}</strong>
                            <span>{option.hint}</span>
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="settings-tts-layout">
                      <div className="settings-tts-main">
                    {activeTtsBackend === "omnivoice" && (
                      <section className="settings-card">
                        <div className="card-header-accent">
                          <span className="accent-bar"></span>
                          <h3>OmniVoice — Đa ngôn ngữ</h3>
                        </div>
                        <p className="card-description card-description--compact">
                          Model TTS mới nhất hỗ trợ 600+ ngôn ngữ, clone giọng và voice design. Cần cài môi trường qua{" "}
                          <code>python scripts/setup_omnivoice.py</code>.
                        </p>
                        <div className="settings-field-grid settings-field-grid--2">
                          <label className="settings-label">
                            <span>Audio tham chiếu (.wav)</span>
                            <select
                              className="settings-input"
                              value={settings.omnivoice_ref_audio ?? ""}
                              onChange={(e) => {
                                const wavPath = e.target.value;
                                const voice = clonedVoicesByBackend.omnivoice.find((item) => item.wav_path === wavPath);
                                setSettings({
                                  ...settings,
                                  omnivoice_ref_audio: wavPath,
                                  omnivoice_ref_text: voice?.transcript?.trim() || "",
                                });
                              }}
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
                            <span>Ngôn ngữ đích OmniVoice</span>
                            <input
                              className="settings-input"
                              placeholder={activeDubLanguage.omnivoiceLanguageId}
                              value={settings.omnivoice_language_id ?? ""}
                              onChange={(e) => setSettings({ ...settings, omnivoice_language_id: e.target.value })}
                            />
                          </label>
                          <label className="settings-label settings-field-grid__span-2">
                            <span>ref_text (bắt buộc khi clone) — khớp audio mẫu</span>
                            <textarea
                              className="settings-input"
                              rows={3}
                              placeholder="Dán nguyên văn nội dung audio tham chiếu..."
                              value={settings.omnivoice_ref_text ?? ""}
                              onChange={(e) => setSettings({ ...settings, omnivoice_ref_text: e.target.value })}
                              style={{ minHeight: "84px", resize: "vertical" }}
                            />
                          </label>
                          <label className="settings-label settings-field-grid__span-2">
                            <span>Voice design (tùy chọn)</span>
                            <input
                              className="settings-input"
                              placeholder="female, low pitch, british accent"
                              value={settings.omnivoice_instruct ?? ""}
                              onChange={(e) => setSettings({ ...settings, omnivoice_instruct: e.target.value })}
                            />
                          </label>
                          <label className="settings-label">
                            <span>Diffusion steps ({settings.omnivoice_num_steps ?? 32})</span>
                            <input
                              className="settings-input"
                              type="range"
                              min={8}
                              max={64}
                              step={4}
                              value={settings.omnivoice_num_steps ?? 32}
                              onChange={(e) =>
                                setSettings({ ...settings, omnivoice_num_steps: Number(e.target.value) })
                              }
                            />
                          </label>
                          <label className="settings-label settings-label--inline settings-field-grid__span-2">
                            <input
                              type="checkbox"
                              checked={Boolean(settings.omnivoice_auto_voice ?? true)}
                              onChange={(e) => setSettings({ ...settings, omnivoice_auto_voice: e.target.checked })}
                            />
                            <span>Auto voice khi không có audio tham chiếu</span>
                          </label>
                        </div>
                        {settings.omnivoice_ref_audio && !(settings.omnivoice_ref_text ?? "").trim() && (
                          <div className="alert-info-box warning">
                            <CircleAlert size={14} />
                            <span>OmniVoice clone cần ref_text khớp với audio tham chiếu. Chọn giọng từ tab Clone hoặc dán thủ công.</span>
                          </div>
                        )}
                        {settings.omnivoice_ref_audio && selectedOmniVoiceClone && !selectedOmniVoiceClone.transcribed && (
                          <div className="alert-info-box warning">
                            <CircleAlert size={14} />
                            <span>Giọng đã chọn thiếu ref_text — hãy upload lại ở tab Clone giọng (OmniVoice).</span>
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
                          <span>Giọng {activeDubLanguage.label}</span>
                          <select
                            className="settings-input"
                            value={settings.edge_tts_voice ?? activeDubLanguage.defaultEdgeVoice}
                            onChange={(e) => setSettings({ ...settings, edge_tts_voice: e.target.value })}
                          >
                            {(edgeTtsVoices.length > 0 ? edgeTtsVoices : [
                              { id: activeDubLanguage.defaultEdgeVoice, name: activeDubLanguage.label },
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
                          Giọng Standard/Neural2 cho {activeDubLanguage.label}. Cần API key từ Google Cloud Console — khác khóa Gemini AI Studio.
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
                            <span>Giọng đọc {activeDubLanguage.label}</span>
                            <select
                              className="settings-input"
                              value={settings.google_tts_voice ?? activeDubLanguage.defaultGoogleVoice}
                              onChange={(e) => setSettings({ ...settings, google_tts_voice: e.target.value })}
                            >
                              {activeGoogleTtsVoices.map((voice) => (
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

                      </div>

                      <aside className="settings-tts-side">
                        <div className="settings-preview-card">
                          <SettingsSectionHead
                            title="Nghe thử"
                            description="Kiểm tra giọng trước khi chạy job."
                          />
                          <label className="settings-label">
                            <span>Nội dung</span>
                            <textarea
                              className="settings-input settings-textarea"
                              rows={6}
                              value={ttsPreviewText}
                              onChange={(e) => setTtsPreviewText(e.target.value)}
                              placeholder="Nhập câu tiếng Việt để nghe thử..."
                            />
                          </label>
                          <button
                            type="button"
                            className="gradient-button settings-preview-btn"
                            disabled={ttsPreviewLoading}
                            onClick={() => void handlePreviewTts()}
                          >
                            <Volume2 size={16} />
                            {ttsPreviewLoading
                              ? (activeTtsBackend === "omnivoice"
                                ? "Đang tạo audio (lần đầu có thể chờ tải model)..."
                                : "Đang tạo audio...")
                              : "Nghe thử"}
                          </button>
                          {ttsPreviewUrl && (
                            <div className="settings-preview-player">
                              <audio controls autoPlay src={ttsPreviewUrl} />
                            </div>
                          )}
                        </div>
                      </aside>
                      <section className="settings-duration-advanced settings-tts-advanced">
                        <SettingsSectionHead
                          title="Khớp thời lượng nâng cao"
                          description="Chỉ chỉnh khi lồng tiếng lệch timeline — mặc định đã bật khớp chính xác."
                        />
                        <p className="card-description card-description--compact">
                          Áp dụng ở bước <strong>duration_repair</strong>. Chạy lại job từ bước đó hoặc TTS để cập nhật.
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
                              hint="Nhân tốc độ đọc audio TTS sau khi chỉnh nội dung (kéo dài/rút gọn). Cần chạy lại job từ bước Sửa độ dài TTS để có hiệu lực. Tăng (vd. 1.5–2.5×) nếu lồng tiếng vẫn chậm hơn giọng gốc."
                            />
                            <input
                              className="settings-input"
                              type="range"
                              min={1}
                              max={2.5}
                              step={0.01}
                              value={settings.tts_global_speed ?? 1}
                              onChange={(e) => setSettings({ ...settings, tts_global_speed: Number(e.target.value) })}
                              disabled={!settings.exact_timing_enabled}
                            />
                          </label>
                          <label className="settings-label settings-field-grid__span-2">
                            <SettingsFieldLabel
                              label={`Tốc độ nói ${activeDubLanguage.label} ước lượng (${(settings.vietnamese_speaking_rate_wps ?? activeDubLanguage.speakingRate).toFixed(2)} từ/giây)`}
                              hint="Dùng khi dịch để ước lượng độ dài câu. Tự hiệu chỉnh sau mỗi job TTS; chỉnh tay nếu giọng đọc nhanh/chậm hơn mặc định."
                            />
                            <input
                              className="settings-input"
                              type="range"
                              min={2}
                              max={5}
                              step={0.05}
                              value={settings.vietnamese_speaking_rate_wps ?? activeDubLanguage.speakingRate}
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
                    </div>
                  </div>
                )}

                {settingsTab === "subtitles" && (
                  <div className="settings-tab-panel" role="tabpanel">
                    <SettingsSectionHead
                      title="Phụ đề trên video"
                      description="Chèn phụ đề tiếng Việt (bản dịch) trực tiếp vào video khi xuất thành phẩm."
                    />
                    <div className="settings-form-fields">
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
                      <SettingsSectionHead
                        title="Chia dòng phụ đề"
                        description="Cắt phụ đề theo câu/cụm và đồng bộ với nhịp nói của audio lồng tiếng, tránh hiển thị quá nhiều chữ một lúc."
                      />
                      <div className="settings-field-grid settings-field-grid--2">
                        <label className="settings-label">
                          <span>Số ký tự tối đa mỗi dòng</span>
                          <input
                            className="settings-input"
                            type="number"
                            min={12}
                            max={120}
                            value={settings.subtitle_max_chars_per_line ?? 40}
                            onChange={(e) => setSettings({ ...settings, subtitle_max_chars_per_line: Number(e.target.value) })}
                            disabled={!settings.subtitles_enabled}
                          />
                        </label>
                        <label className="settings-label">
                          <span>Số dòng tối đa mỗi lần hiển thị</span>
                          <input
                            className="settings-input"
                            type="number"
                            min={1}
                            max={4}
                            value={settings.subtitle_max_lines_per_cue ?? 2}
                            onChange={(e) => setSettings({ ...settings, subtitle_max_lines_per_cue: Number(e.target.value) })}
                            disabled={!settings.subtitles_enabled}
                          />
                        </label>
                        <label className="settings-label">
                          <span>Thời lượng tối thiểu mỗi cue (ms)</span>
                          <input
                            className="settings-input"
                            type="number"
                            min={200}
                            max={4000}
                            step={50}
                            value={settings.subtitle_min_cue_duration_ms ?? 700}
                            onChange={(e) => setSettings({ ...settings, subtitle_min_cue_duration_ms: Number(e.target.value) })}
                            disabled={!settings.subtitles_enabled}
                          />
                        </label>
                        <label className="settings-label">
                          <span>Thời lượng tối đa mỗi cue (ms)</span>
                          <input
                            className="settings-input"
                            type="number"
                            min={1500}
                            max={15000}
                            step={100}
                            value={settings.subtitle_max_cue_duration_ms ?? 5500}
                            onChange={(e) => setSettings({ ...settings, subtitle_max_cue_duration_ms: Number(e.target.value) })}
                            disabled={!settings.subtitles_enabled}
                          />
                        </label>
                        <label className="settings-label settings-field-grid__span-2">
                          <span>Khoảng cách tối thiểu giữa 2 cue (ms)</span>
                          <input
                            className="settings-input"
                            type="number"
                            min={0}
                            max={1000}
                            step={10}
                            value={settings.subtitle_inter_cue_gap_ms ?? 50}
                            onChange={(e) => setSettings({ ...settings, subtitle_inter_cue_gap_ms: Number(e.target.value) })}
                            disabled={!settings.subtitles_enabled}
                          />
                        </label>
                      </div>
                      <div className="alert-info-box info">
                        <CircleAlert size={14} style={{ flexShrink: 0, marginTop: "2px" }} />
                        <span>
                          Phụ đề dùng bản dịch tiếng Việt theo từng phân đoạn. Chạy lại từ bước <strong>Xuất video thành phẩm</strong> để áp dụng thay đổi cho job đã hoàn thành.
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </form>
              </div>
            </div>
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
                      Điểm bắt đầu: <strong>{translateStepName(PIPELINE_STEPS[rerunKeepSteps.length], settings.translation_target_language)}</strong>
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
                          <div style={{ fontSize: "13px", color: "#fff" }}>{translateStepName(stepName, settings.translation_target_language)}</div>
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
                       <strong style={{ fontSize: "14px" }}>{translateStepName(step.name, settings.translation_target_language)}</strong>
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
                       Dịch (Việt): {seg.tts_spoken_text || seg.translation}
                     </div>
                     {(seg.timing_status && seg.timing_status !== "OK") && (
                       <div style={{ fontSize: "11px", color: seg.timing_status === "OVERFLOW" ? "#ff8f8f" : "#ffb347" }}>
                         Timing: {seg.timing_status}
                         {seg.timing_overflow_sec > 0.05 ? ` · overflow ${Number(seg.timing_overflow_sec).toFixed(2)}s` : ""}
                         {seg.placement_drift_sec ? ` · drift ${Number(seg.placement_drift_sec).toFixed(2)}s` : ""}
                         {seg.soft_speed_factor ? ` · speed ${Number(seg.soft_speed_factor).toFixed(2)}x` : ""}
                       </div>
                     )}
                     {(seg.tts_chunk_count != null && seg.tts_chunk_count > 0) && (
                       <div style={{ fontSize: "11px", color: "#9aa3b5", display: "flex", flexWrap: "wrap", gap: "8px" }}>
                         <span>TTS: {seg.tts_chunk_count} chunk{seg.tts_chunk_count > 1 ? "s" : ""}</span>
                         {seg.tts_text_similarity != null && (
                           <span>Fidelity: {Math.round(Number(seg.tts_text_similarity) * 100)}%</span>
                         )}
                         {seg.tts_fidelity_status && seg.tts_fidelity_status !== "not_checked" && (
                           <span>
                             Trạng thái: {
                               seg.tts_fidelity_status === "good" ? "Tốt"
                               : seg.tts_fidelity_status === "review" ? "Cần kiểm tra"
                               : seg.tts_fidelity_status === "poor" ? "Cần kiểm tra"
                               : "Lỗi"
                             }
                           </span>
                         )}
                         {(seg.tts_fidelity_status === "poor" || seg.tts_fidelity_status === "failed" || seg.tts_fidelity_status === "review") && (
                           <span style={{ color: "#ffb347" }}>Audio có thể chưa đọc đầy đủ bản dịch.</span>
                         )}
                       </div>
                     )}
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
        <div className="overlay video-overlay" onClick={() => setActiveVideoUrl(null)}>
          <div className="video-overlay__panel" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setActiveVideoUrl(null)}
              style={{ position: "absolute", right: "12px", top: "12px", background: "#00000080", color: "#fff", border: "0", borderRadius: "50%", width: "32px", height: "32px", display: "grid", placeItems: "center", zIndex: 1 }}
            >
              <X size={18} />
            </button>
            <video className="video-overlay__player" controls autoPlay src={activeVideoUrl} />
          </div>
        </div>
      )}

      {/* Errors modal */}
      {errorsModalOpen && (
        <div className="overlay errors-overlay" onClick={() => setErrorsModalOpen(false)}>
          <section className="errors-panel" onClick={(event) => event.stopPropagation()}>
            <div className="errors-head">
              <div>
                <p>Lỗi & thông báo</p>
                <h2>{visibleNotices.length > 0 ? `${visibleNotices.length} mục cần xem` : "Không có lỗi"}</h2>
                <small>
                  {visibleNotices.length > 0
                    ? "Các lỗi từ thao tác, backend, tiến trình và môi trường được gom tại đây."
                    : "Ứng dụng đang hoạt động bình thường."}
                </small>
              </div>
              <button type="button" aria-label="Đóng danh sách lỗi" onClick={() => setErrorsModalOpen(false)}>
                <X />
              </button>
            </div>

            {visibleNotices.length > 0 ? (
              <>
                <div className="errors-toolbar">
                  <button type="button" className="errors-dismiss-all" onClick={dismissAllNotices}>
                    Xóa tất cả
                  </button>
                </div>
                <div className="errors-list">
                  {visibleNotices.map((notice) => (
                    <article key={notice.id} className={`errors-item errors-item--${notice.severity}`}>
                      <div className="errors-item__icon" aria-hidden="true">
                        {notice.severity === "warning" ? <AlertTriangle size={18} /> : <CircleAlert size={18} />}
                      </div>
                      <div className="errors-item__body">
                        <div className="errors-item__meta">
                          <strong>{notice.title}</strong>
                          <span>{notice.source}</span>
                        </div>
                        <p>{notice.message}</p>
                        {notice.action && (
                          <button type="button" className="errors-item__action" onClick={() => void notice.action?.onClick()}>
                            {notice.action.label}
                          </button>
                        )}
                      </div>
                      <button
                        type="button"
                        className="errors-item__dismiss"
                        aria-label={`Xóa: ${notice.title}`}
                        onClick={() => dismissNotice(notice)}
                      >
                        <Trash2 size={16} />
                      </button>
                    </article>
                  ))}
                </div>
              </>
            ) : (
              <div className="errors-empty">
                <CheckCircle2 size={28} />
                <p>Không có lỗi hoặc cảnh báo nào.</p>
              </div>
            )}
          </section>
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
                  <span>Tiến trình OmniVoice đang giữ client: {runtime.gpu.active_omnivoice_clients}</span>
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
