import { useEffect, useState, type FormEvent } from "react";
import {
  Activity,
  CircleAlert,
  Clock3,
  Link2,
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
  Upload
} from "lucide-react";

import type { Job, JobsApi, OutputItem, RuntimeCheck, RuntimeReport, ClonedVoice } from "../shared/contracts";
import { api as defaultApi } from "../shared/api";
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
    case "completed": return "đã hoàn thành";
    case "failed": return "thất bại";
    case "interrupted": return "bị gián đoạn";
    case "waiting_for_selection": return "chờ chọn video";
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

export function App({ api = defaultApi }: { api?: JobsApi }) {
  const [activeTab, setActiveTab] = useState<"jobs" | "outputs" | "settings" | "cloning">("jobs");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [sourceUrl, setSourceUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<RuntimeReport | null>(null);
  const [runtimeOpen, setRuntimeOpen] = useState(false);
  const [testingRuntime, setTestingRuntime] = useState(false);

  // Setup Wizard States
  const [hardware, setHardware] = useState<{
    cuda_supported: boolean;
    vulkan_supported: boolean;
    avx2_supported: boolean;
    espeak_installed: boolean;
    recommendation: string;
  } | null>(null);
  const [wizardStep, setWizardStep] = useState<1 | 2 | 3>(1);
  const [wizardProfile, setWizardProfile] = useState<string>("gpu_cuda");
  const [bootstrapProgress, setBootstrapProgress] = useState<any>(null);
  const [bootstrapRunning, setBootstrapRunning] = useState<boolean>(false);

  // Selected job details modal state
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [resolveCp, setResolveCp] = useState<any | null>(null);
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
    cookies_browser: "none",
    translation_backend: "google_free",
    translation_source_language: "zh-CN",
    translation_target_language: "vi",
    mix_mode: "duck",
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
  });
  const [newGeminiKey, setNewGeminiKey] = useState("");
  const [settingsSuccess, setSettingsSuccess] = useState(false);

  // Cloned voices states
  const [clonedVoices, setClonedVoices] = useState<ClonedVoice[]>([]);
  const [voiceName, setVoiceName] = useState("");
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [voiceUploading, setVoiceUploading] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [testText, setTestText] = useState<Record<string, string>>({});
  const [testSynthesizing, setTestSynthesizing] = useState<Record<string, boolean>>({});
  const [testAudioUrls, setTestAudioUrls] = useState<Record<string, string>>({});

  // Detect hardware when runtime is blocked
  useEffect(() => {
    if (runtime?.status === "blocked" && !hardware) {
      api.detectHardware().then((rep) => {
        setHardware(rep);
        setWizardProfile(rep.recommendation === "gpu_cuda" ? "gpu_cuda" : "cpu_avx2");
      }).catch(console.error);
    }
  }, [runtime, hardware, api]);

  // Poll bootstrap progress
  useEffect(() => {
    let interval: any;
    if (bootstrapRunning) {
      interval = setInterval(async () => {
        try {
          const progress = await api.bootstrapProgress();
          setBootstrapProgress(progress);
          
          if (progress.status === "completed") {
            setBootstrapRunning(false);
            setWizardStep(3);
            // Re-run smoke test to clear blocked status
            const report = await api.runSmokeTest();
            setRuntime(report);
          } else if (progress.status === "failed") {
            setBootstrapRunning(false);
          }
        } catch (e) {
          console.error(e);
        }
      }, 500);
    }
    return () => clearInterval(interval);
  }, [bootstrapRunning, api]);

  async function handleStartBootstrap() {
    setBootstrapRunning(true);
    setWizardStep(2);
    try {
      await api.bootstrapVendor(wizardProfile);
    } catch (cause) {
      setBootstrapRunning(false);
      setError(cause instanceof Error ? cause.message : "Tải môi trường thất bại");
    }
  }

  // Fetch initial data
  useEffect(() => {
    refreshJobs();
    refreshRuntime();
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
            if (updated.current_step === "download" || updated.status === "waiting_for_selection") {
              fetchResolveCheckpoint(updated.id);
            }
            fetchJobCheckpoints(updated.id);
            fetchJobFiles(updated.id);
          }
        }
      });
      if (activeTab === "outputs") {
        refreshOutputs();
      }
      if (activeTab === "cloning" || activeTab === "settings") {
        refreshClonedVoices();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [api, selectedJobId, activeTab]);

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
      setVoiceError("Vui lòng nhập tên giọng và chọn tệp âm thanh mẫu (.wav).");
      return;
    }
    setVoiceUploading(true);
    setVoiceError(null);
    try {
      await api.createClonedVoice(voiceName, voiceFile);
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
      const blob = await api.testClonedVoice(id, text);
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
    try {
      setRuntime(await api.runtimeStatus());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể tải trạng thái môi trường thực thi");
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

  async function fetchResolveCheckpoint(jobId: string) {
    try {
      const checkpoint = await api.getCheckpoint(jobId, "resolve");
      setResolveCp(checkpoint);
    } catch {
      setResolveCp(null);
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

  async function createJob(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      const job = await api.createJob(sourceUrl);
      setJobs((current) => [job, ...current]);
      setSourceUrl("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể tạo tiến trình");
    }
  }

  async function runSmokeTest() {
    setTestingRuntime(true);
    try {
      setRuntime(await api.runSmokeTest());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Thử nghiệm hệ thống thất bại");
    } finally {
      setTestingRuntime(false);
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

  async function handleSelectPlaylistVideo(jobId: string, index: number) {
    try {
      await api.selectVideo(jobId, index);
      refreshJobs();
      setResolveCp(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Không thể chọn video");
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
      savePayload.tts_backend = "voxcpm";
      const pendingGeminiKey = newGeminiKey.trim();
      if (pendingGeminiKey) {
        savePayload.gemini_api_key_add = pendingGeminiKey;
      }
      const updated = await api.updateSettings(savePayload);
      setSettings(updated);
      if (pendingGeminiKey) {
        setNewGeminiKey("");
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

  function formatDuration(sec?: number): string {
    if (sec === undefined || sec === null) return "--:--";
    const minutes = Math.floor(sec / 60);
    const seconds = Math.floor(sec % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  }

  function formatBytes(bytes?: number): string {
    if (!bytes) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  if (runtime?.status === "blocked") {
    return (
      <div className="wizard-container">
        <div className="wizard-card">
          <div className="wizard-header">
            <h1>Thiết lập môi trường ứng dụng</h1>
            <p>Ứng dụng cần tải và cấu hình một số thư viện hệ thống để bắt đầu hoạt động.</p>
          </div>

          {error && (
            <div className="error" style={{ margin: 0 }}>
              <CircleAlert size={22} />
              <div>
                <strong>Lỗi thiết lập</strong>
                <span>{error}</span>
              </div>
              <button onClick={() => setError(null)} style={{ background: "transparent", color: "inherit", marginLeft: "auto" }}>
                <X size={18} />
              </button>
            </div>
          )}

          {wizardStep === 1 && (
            <>
              <div className="specs-panel">
                <h4 style={{ margin: "0 0 10px", fontSize: "14px", color: "#a79aff", textTransform: "uppercase", letterSpacing: "0.05em" }}>Thông số phần cứng phát hiện</h4>
                {hardware ? (
                  <>
                    <div className="spec-item">
                      <span className="label">Hỗ trợ CUDA GPU (Qwen3-ASR 1.7B):</span>
                      <span className={`value ${hardware.cuda_supported ? "supported" : "unsupported"}`}>
                        {hardware.cuda_supported ? "Hỗ trợ" : "Không hỗ trợ"}
                      </span>
                    </div>
                    <div className="spec-item">
                      <span className="label">Hỗ trợ GPU Vulkan:</span>
                      <span className={`value ${hardware.vulkan_supported ? "supported" : "unsupported"}`}>
                        {hardware.vulkan_supported ? "Hỗ trợ" : "Không hỗ trợ"}
                      </span>
                    </div>
                    <div className="spec-item">
                      <span className="label">Hỗ trợ tập lệnh CPU AVX2:</span>
                      <span className={`value ${hardware.avx2_supported ? "supported" : "unsupported"}`}>
                        {hardware.avx2_supported ? "Hỗ trợ" : "Không hỗ trợ"}
                      </span>
                    </div>
                    <div className="spec-item">
                      <span className="label">eSpeak NG (Cần cho Offline Voice Clone):</span>
                      <span className={`value ${hardware.espeak_installed ? "supported" : "unsupported"}`}>
                        {hardware.espeak_installed ? "Đã cài đặt" : "Chưa cài đặt"}
                      </span>
                    </div>
                  </>
                ) : (
                  <div style={{ color: "#8f97a6", fontStyle: "italic", fontSize: "13px" }}>Đang quét phần cứng...</div>
                )}
              </div>

              <div>
                <h4 style={{ margin: "0 0 12px", fontSize: "14px", color: "#a79aff", textTransform: "uppercase", letterSpacing: "0.05em" }}>Chọn cấu hình ASR</h4>
                <div className="profile-selection">
                  <div
                    className={`profile-card ${wizardProfile === "gpu_cuda" ? "selected" : ""}`}
                    onClick={() => {
                      if (hardware?.cuda_supported) {
                        setWizardProfile("gpu_cuda");
                      }
                    }}
                    style={{ opacity: hardware?.cuda_supported ? 1 : 0.5, cursor: hardware?.cuda_supported ? "pointer" : "not-allowed" }}
                  >
                    <span className="profile-badge">Khuyên dùng</span>
                    <h3>Qwen3-ASR 1.7B trên GPU (CUDA)</h3>
                    <p>Nhận dạng tiếng Trung chính xác nhất với mô hình Qwen3-ASR-1.7B trên card NVIDIA. Yêu cầu VRAM khoảng 6–8 GB.</p>
                    {!hardware?.cuda_supported && <small style={{ color: "#f16f7e", fontSize: "11px", marginTop: "auto" }}>Không khả dụng: Không tìm thấy CUDA GPU</small>}
                  </div>

                  <div
                    className={`profile-card ${wizardProfile === "cpu_avx2" ? "selected" : ""}`}
                    onClick={() => setWizardProfile("cpu_avx2")}
                    style={{ opacity: 0.5, cursor: "not-allowed" }}
                  >
                    <span className="profile-badge" style={{ background: "rgba(91, 221, 154, 0.15)", color: "#5bdd9a" }}>Không hỗ trợ</span>
                    <h3>Chế độ CPU</h3>
                    <p>Qwen3-ASR 1.7B yêu cầu GPU CUDA. Ứng dụng không còn dùng whisper.cpp CPU làm ASR mặc định.</p>
                  </div>
                </div>
              </div>

              <button
                type="button"
                className="smoke-button"
                onClick={handleStartBootstrap}
                disabled={!hardware || !hardware.cuda_supported}
                style={{ marginTop: "10px" }}
              >
                Bắt đầu tải và thiết lập môi trường
              </button>
            </>
          )}

          {wizardStep === 2 && (
            <>
              <div className="progress-container">
                <div className="progress-meta">
                  <strong>{bootstrapProgress?.current_task || "Đang tải xuống..."}</strong>
                  <span>{bootstrapProgress?.download_percent || 0}%</span>
                </div>
                <div className="progress-bar-bg">
                  <div
                    className="progress-bar-fill"
                    style={{ width: `${bootstrapProgress?.download_percent || 0}%` }}
                  />
                </div>
                {bootstrapProgress && bootstrapProgress.download_speed_kb > 0 && (
                  <div style={{ textAlign: "right", fontSize: "12px", color: "#8f97a6", marginTop: "2px" }}>
                    Tốc độ: {(bootstrapProgress.download_speed_kb / 1024).toFixed(2)} MB/s | Đã tải: {(bootstrapProgress.downloaded_bytes / 1024 / 1024).toFixed(1)} MB / {(bootstrapProgress.total_bytes / 1024 / 1024).toFixed(1)} MB
                  </div>
                )}
              </div>

              <div>
                <h4 style={{ margin: "0 0 10px", fontSize: "13px", color: "#8f97a6", textTransform: "uppercase" }}>Nhật ký thiết lập</h4>
                <div className="logs-panel">
                  {bootstrapProgress?.logs?.map((line: string, i: number) => (
                    <div key={i} className="log-line">{line}</div>
                  )) || <div style={{ color: "#626b7d", fontStyle: "italic" }}>Đang khởi tạo kết nối...</div>}
                </div>
              </div>

              {bootstrapProgress?.status === "failed" && (
                <button
                  type="button"
                  className="smoke-button"
                  onClick={handleStartBootstrap}
                  style={{ background: "#f16f7e", marginTop: "10px" }}
                >
                  Tải lại môi trường (Thử lại)
                </button>
              )}
            </>
          )}

          {wizardStep === 3 && (
            <div className="wizard-success">
              <CheckCircle2 size={64} className="icon" />
              <h2>Thiết lập môi trường thành công!</h2>
              <p style={{ color: "#8f97a6", fontSize: "14px", margin: "0 0 10px" }}>
                Tất cả các thư viện ffmpeg, yt-dlp, Qwen3-ASR 1.7B và mô hình AI đã được cấu hình chính xác. Ứng dụng đã sẵn sàng hoạt động.
              </p>
              <button
                type="button"
                className="smoke-button"
                onClick={() => {
                  refreshRuntime();
                }}
              >
                Bắt đầu sử dụng ứng dụng
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
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
        <button className={`runtime ${runtime?.status ?? "loading"}`} onClick={() => setRuntimeOpen(true)}>
          <i />
          <div>
            <strong>Môi trường: {translateStatus(runtime?.status)}</strong>
            <small>Môi trường thực thi</small>
          </div>
        </button>
      </aside>

      <main>
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
                <p>Tải, dịch và lồng tiếng tự động từ liên kết Douyin.</p>
              </div>
              <form onSubmit={createJob}>
                <label>
                  <Link2 size={18} />
                  <input
                    required
                    value={sourceUrl}
                    onChange={(event) => setSourceUrl(event.target.value)}
                    placeholder="Dán liên kết video hoặc kênh Douyin (ví dụ: https://www.douyin.com/video/...)"
                  />
                </label>
                <button type="submit" disabled={runtime?.status === "blocked"}>
                  <Plus size={18} /> Tạo tiến trình
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
                  <p>Dán liên kết Douyin ở trên để tạo tiến trình lồng tiếng đầu tiên.</p>
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
                            {job.title_vi || job.title || job.source_url}
                          </h3>
                          {job.title_vi && job.title && job.title_vi !== job.title && (
                            <small style={{ color: "#747d90", display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              {job.title}
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
                <p className="settings-subtitle">Thêm mới các giọng nói nhân bản từ tệp mẫu .wav và thực hiện các tổng hợp âm thanh thử nghiệm.</p>
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

            <div className="settings-page-layout">
              {/* Cột Trái: Tải lên giọng đọc mẫu */}
              <div className="settings-column">
                <section className="settings-card">
                  <div className="card-header-accent">
                    <span className="accent-bar"></span>
                    <h3>Thêm Giọng Nhân Bản Mới</h3>
                  </div>
                  <p className="card-description">
                    Tải lên một tệp âm thanh mẫu (tần số tối ưu là 16kHz - 48kHz, định dạng .wav, dài từ 3-10 giây) để sử dụng với VoxCPM2.
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
                      <span>Chọn tệp mẫu .wav</span>
                      <input
                        id="voice-file-input"
                        required
                        type="file"
                        accept=".wav"
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
                        Nguồn: <a href={out.source_url} target="_blank" rel="noreferrer" style={{ color: "#8170ff" }}>{out.source_url}</a>
                      </p>
                      <button
                        className="smoke-button"
                        style={{ marginTop: "auto", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}
                        onClick={() => setActiveVideoUrl(`http://127.0.0.1:8765/api/jobs/${out.job_id}/output`)}
                      >
                        <Play size={16} /> Phát Video Lồng Tiếng
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
            <header className="settings-header">
              <div>
                <h1>Cài đặt ứng dụng</h1>
                <p className="settings-subtitle">Cấu hình dịch thuật, VoxCPM2 và quản lý khóa API Gemini.</p>
              </div>
            </header>
            <form onSubmit={handleSaveSettings} className="settings-page-layout">
              <div className="settings-column">
                {/* Cấu hình Dịch thuật và Cookie */}
                <section className="settings-card">
                  <div className="card-header-accent">
                    <span className="accent-bar"></span>
                    <h3>Google Dịch Miễn Phí</h3>
                  </div>
                  <p className="card-description">
                    Dịch miễn phí qua Google Translate. Tần suất yêu cầu có thể bị giới hạn.
                  </p>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
                    <label className="settings-label">
                      <span>Cookie trình duyệt cho Douyin</span>
                      <select
                        className="settings-input"
                        value={settings.cookies_browser ?? "none"}
                        onChange={(e) => setSettings({ ...settings, cookies_browser: e.target.value })}
                      >
                        <option value="none">Không sử dụng cookie trình duyệt</option>
                        <option value="edge">Microsoft Edge</option>
                        <option value="chrome">Google Chrome</option>
                        <option value="firefox">Mozilla Firefox</option>
                        <option value="brave">Brave</option>
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
                      </select>
                    </label>
                  </div>
                  <div className="alert-info-box warning">
                    <CircleAlert size={14} style={{ flexShrink: 0, marginTop: "2px" }} />
                    <span>Cookie dùng cho yt-dlp để tải video, không lưu trữ trên hệ thống.</span>
                  </div>
                </section>

                {/* Google AI Studio / Khóa API Gemini */}
                <section className="settings-card">
                  <div className="card-header-accent">
                    <span className="accent-bar"></span>
                    <h3>Google AI Studio / Khóa API Gemini</h3>
                  </div>
                  <p className="card-description">
                    Quản lý khóa API Gemini dùng cho dịch thuật.
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
                    <button
                      type="button"
                      className="gradient-button"
                      onClick={handleAddGeminiKey}
                    >
                      <Plus size={16} /> Thêm khóa Gemini
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

              <div className="settings-column">
                {settings.translation_backend === "gemini" && (
                  <section className="settings-card">
                    <div className="card-header-accent">
                      <span className="accent-bar"></span>
                      <h3>Cấu hình mô hình Gemini</h3>
                    </div>
                    <p className="card-description">Chọn mô hình Gemini dùng cho bước dịch thuật.</p>
                    <label className="settings-label">
                      <span>Mô hình dịch thuật Gemini</span>
                      <input
                        className="settings-input"
                        value={settings.gemini_translation_model ?? "gemini-2.5-flash"}
                        onChange={(e) => setSettings({ ...settings, gemini_translation_model: e.target.value })}
                      />
                    </label>
                  </section>
                )}

                <section className="settings-card">
                  <div className="card-header-accent">
                    <span className="accent-bar"></span>
                    <h3>Lồng tiếng VoxCPM2</h3>
                  </div>
                  <p className="card-description">
                    VoxCPM2 là engine TTS duy nhất. Chọn audio tham chiếu .wav, nhập voice design, hoặc để auto voice.
                  </p>
                  <div className="inputs-vertical-stack">
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
                      <span>Voice design (tùy chọn)</span>
                      <input
                        className="settings-input"
                        placeholder="female, low pitch"
                        value={settings.voxcpm_instruct ?? ""}
                        onChange={(e) => setSettings({ ...settings, voxcpm_instruct: e.target.value })}
                      />
                    </label>
                    <label className="settings-label" style={{ flexDirection: "row", alignItems: "center", gap: "10px" }}>
                      <input
                        type="checkbox"
                        checked={Boolean(settings.voxcpm_auto_voice ?? true)}
                        onChange={(e) => setSettings({ ...settings, voxcpm_auto_voice: e.target.checked })}
                      />
                      <span>Auto voice khi không có audio tham chiếu</span>
                    </label>
                  </div>
                </section>

                <section className="settings-card">
                  <div className="card-header-accent">
                    <span className="accent-bar"></span>
                    <h3>Trộn âm thanh</h3>
                  </div>
                  <p className="card-description">
                    Chọn cách ghép giọng lồng tiếng Việt với âm thanh gốc của video.
                  </p>
                  <label className="settings-label">
                    <span>Chế độ trộn</span>
                    <select
                      className="settings-input"
                      value={settings.mix_mode ?? "duck"}
                      onChange={(e) => setSettings({ ...settings, mix_mode: e.target.value })}
                    >
                      <option value="duck">Giảm âm gốc (ducking) — mặc định</option>
                      <option value="separate">Tách giọng nói, giữ nhạc nền (Demucs)</option>
                    </select>
                  </label>
                  <div className="alert-info-box info">
                    <CircleAlert size={14} style={{ flexShrink: 0, marginTop: "2px" }} />
                    <span>
                      Chế độ <strong>Tách giọng nói</strong> dùng Demucs trên GPU để loại giọng Trung gốc và giữ nhạc nền.
                      Xử lý lâu hơn nhưng phù hợp video có nhạc nền mạnh.
                    </span>
                  </div>
                </section>

                <section className="settings-card">
                  <div className="card-header-accent">
                    <span className="accent-bar"></span>
                    <h3>Phụ đề trên video</h3>
                  </div>
                  <p className="card-description">
                    Chèn phụ đề tiếng Việt (bản dịch) trực tiếp vào video khi xuất thành phẩm.
                  </p>
                  <label className="settings-label" style={{ flexDirection: "row", alignItems: "center", gap: "10px" }}>
                    <input
                      type="checkbox"
                      checked={settings.subtitles_enabled ?? true}
                      onChange={(e) => setSettings({ ...settings, subtitles_enabled: e.target.checked })}
                    />
                    <span>Bật phụ đề trên video</span>
                  </label>
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
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
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
                      <span>Màu nền chữ (vùng bao quanh)</span>
                      <input
                        className="settings-input"
                        type="color"
                        value={settings.subtitle_background_color ?? "#000000"}
                        onChange={(e) => setSettings({ ...settings, subtitle_background_color: e.target.value.toUpperCase() })}
                        disabled={!settings.subtitles_enabled}
                      />
                    </label>
                  </div>
                  <label className="settings-label">
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
                  <label className="settings-label">
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
                  {(settings.subtitle_position === "bottom" || settings.subtitle_position === "top") && (
                    <label className="settings-label">
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
                  <div className="alert-info-box info">
                    <CircleAlert size={14} style={{ flexShrink: 0, marginTop: "2px" }} />
                    <span>
                      Phụ đề dùng bản dịch tiếng Việt theo từng phân đoạn. Chạy lại từ bước <strong>Xuất video thành phẩm</strong> để áp dụng thay đổi cho job đã hoàn thành.
                    </span>
                  </div>
                </section>
              </div>

              <div className="settings-actions">
                <button type="submit" className="save-settings-button">
                  <Save size={18} /> Lưu Cài Đặt
                </button>
                {settingsSuccess && (
                  <span className="save-success-badge">
                    <CheckCircle2 size={18} /> Đã lưu cài đặt thành công!
                  </span>
                )}
              </div>
            </form>
          </>
        )}
      </main>

      {/* Selected Job Drawer Panel */}
      {selectedJob && (
        <div className="overlay" onClick={closeSelectedJob}>
          <section className="runtime-panel" onClick={(event) => event.stopPropagation()} style={{ width: "min(600px, 100%)", display: "flex", flexDirection: "column" }}>
            <div className="runtime-head">
              <div>
                <p>Chi tiết tiến trình</p>
                <h2 style={{ fontSize: "20px", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap", maxWidth: "450px" }}>
                  {selectedJob.title_vi || selectedJob.title || selectedJob.source_url}
                </h2>
                {selectedJob.title_vi && selectedJob.title && selectedJob.title_vi !== selectedJob.title && (
                  <small style={{ color: "#747d90", display: "block", marginTop: "4px", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap", maxWidth: "450px" }}>
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
            <div style={{ display: "flex", gap: "10px", margin: "20px 0 10px" }}>
              {(selectedJob.status === "failed" || selectedJob.status === "interrupted") && (
                <button className="smoke-button" style={{ flex: 1 }} onClick={() => startJob(selectedJob.id)}>
                  <RefreshCw size={16} /> Tiếp tục lồng tiếng
                </button>
              )}
              {(selectedJob.status === "completed" || selectedJob.status === "failed" || selectedJob.status === "interrupted") && (
                <button
                  className="smoke-button"
                  style={{ flex: 1, background: "linear-gradient(135deg, #6244f7, #3d29a6)", color: "#fff" }}
                  onClick={() => openRerunModal(selectedJob)}
                >
                  <RefreshCw size={16} /> Chạy lại
                </button>
              )}
              {selectedJob.status === "running" && (
                <button className="smoke-button" style={{ flex: 1, background: "#f16f7e" }} onClick={() => cancelJob(selectedJob.id)}>
                  <X size={16} /> Hủy thực thi
                </button>
              )}
              {isDeletableJob(selectedJob) && (
                <button className="smoke-button" style={{ flex: 1, background: "#f16f7e" }} onClick={() => deleteJob(selectedJob.id)}>
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

            {/* Playlist videos resolution selector */}
            {selectedJob.status === "waiting_for_selection" && resolveCp && resolveCp.videos && (
              <div style={{ background: "#20242e", border: "1px solid #343a48", borderRadius: "12px", padding: "16px", margin: "12px 0" }}>
                <h3 style={{ margin: "0 0 12px", fontSize: "15px", display: "flex", alignItems: "center", gap: "8px" }}>
                  <AlertTriangle size={18} style={{ color: "#f2ba5b" }} /> Chọn video để lồng tiếng
                </h3>
                <div style={{ display: "grid", gap: "8px", maxHeight: "250px", overflowY: "auto", paddingRight: "4px" }}>
                  {resolveCp.videos.map((vid: any, idx: number) => (
                    <div
                      key={vid.id}
                      onClick={() => handleSelectPlaylistVideo(selectedJob.id, idx)}
                      className="playlist-item"
                      style={{ display: "flex", gap: "10px", background: "#12151c", padding: "10px", borderRadius: "8px", cursor: "pointer", border: "1px solid #292f3b" }}
                    >
                      {vid.thumbnail && <img src={vid.thumbnail} alt="" style={{ width: "60px", height: "45px", objectFit: "cover", borderRadius: "4px" }} />}
                      <div style={{ flex: 1, overflow: "hidden" }}>
                        <h4 style={{ margin: 0, fontSize: "13px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{vid.title}</h4>
                        <small style={{ color: "#747d90" }}>Thời lượng: {formatDuration(vid.duration)}</small>
                      </div>
                      <ChevronRight size={18} style={{ alignSelf: "center", color: "#626b7d" }} />
                    </div>
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
                       <strong style={{ fontSize: "14px" }}>{translateStepName(step.name)}</strong>
                       <p style={{ margin: "2px 0 0", fontSize: "11px", color: "#747e90" }}>Trạng thái: {translateStepStatus(step.status)}</p>
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
                         {seg.speaker_id != null && (
                           <span style={{ marginLeft: "8px", color: "#00d1b2" }}>
                             · Speaker {seg.speaker_id}
                           </span>
                         )}
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
      {runtimeOpen && runtime && (
        <div className="overlay" onClick={() => setRuntimeOpen(false)}>
          <section className="runtime-panel" onClick={(event) => event.stopPropagation()}>
            <div className="runtime-head">
              <div>
                <p>Môi trường thực thi</p>
                <h2>Trạng thái: {translateStatus(runtime.status)}</h2>
                <small>Kiểm tra lần cuối lúc {new Date(runtime.checked_at).toLocaleString()}</small>
              </div>
              <button aria-label="Close runtime panel" onClick={() => setRuntimeOpen(false)}>
                <X />
              </button>
            </div>
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
          </section>
        </div>
      )}
    </div>
  );
}
