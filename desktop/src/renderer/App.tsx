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
  ShieldCheck,
  X,
  Play,
  FileVideo,
  AlertTriangle,
  CheckCircle2,
  Save,
  ChevronRight
} from "lucide-react";

import type { Job, JobsApi, RuntimeReport } from "../shared/contracts";
import { api as defaultApi } from "../shared/api";
import "./styles.css";
import "./runtime.css";

export function App({ api = defaultApi }: { api?: JobsApi }) {
  const [activeTab, setActiveTab] = useState<"jobs" | "outputs" | "settings">("jobs");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [sourceUrl, setSourceUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<RuntimeReport | null>(null);
  const [runtimeOpen, setRuntimeOpen] = useState(false);
  const [testingRuntime, setTestingRuntime] = useState(false);

  // Selected job details modal state
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [resolveCp, setResolveCp] = useState<any | null>(null);

  // Outputs gallery state
  const [outputs, setOutputs] = useState<any[]>([]);
  const [activeVideoUrl, setActiveVideoUrl] = useState<string | null>(null);

  // Settings form state
  const [settings, setSettings] = useState<Record<string, any>>({
    cookies_browser: "none",
    translation_backend: "google_free",
    translation_source_language: "zh-CN",
    translation_target_language: "vi",
    asr_backend: "whisper_cpu",
    whisper_model_path: "",
    tts_backend: "edge",
    edge_tts_voice: "vi-VN-HoaiMyNeural",
    edge_tts_rate: "+0%",
    gemini_api_keys: [],
    gemini_translation_model: "gemini-2.5-flash",
    gemini_tts_model: "gemini-2.5-flash-preview-tts",
    gemini_tts_voice: "Zephyr"
  });
  const [newGeminiKey, setNewGeminiKey] = useState("");
  const [settingsSuccess, setSettingsSuccess] = useState(false);

  // Fetch initial data
  useEffect(() => {
    refreshJobs();
    refreshRuntime();
    refreshOutputs();
    loadSettings();

    // Auto-refresh running jobs every 2 seconds
    const interval = setInterval(() => {
      api.listJobs().then((newJobs) => {
        setJobs(newJobs);
        // If the selected job is running, refresh its details
        if (selectedJobId) {
          const updated = newJobs.find((j) => j.id === selectedJobId);
          if (updated) {
            setSelectedJob(updated);
            if (updated.current_step === "download" || updated.status === "waiting_for_selection") {
              fetchResolveCheckpoint(updated.id);
            }
          }
        }
      });
      if (activeTab === "outputs") {
        refreshOutputs();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [api, selectedJobId, activeTab]);

  async function refreshJobs() {
    try {
      setJobs(await api.listJobs());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to load jobs");
    }
  }

  async function refreshRuntime() {
    try {
      setRuntime(await api.runtimeStatus());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to load runtime status");
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

  async function createJob(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      const job = await api.createJob(sourceUrl);
      setJobs((current) => [job, ...current]);
      setSourceUrl("");
      handleSelectJob(job);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to create job");
    }
  }

  async function runSmokeTest() {
    setTestingRuntime(true);
    try {
      setRuntime(await api.runSmokeTest());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Runtime smoke test failed");
    } finally {
      setTestingRuntime(false);
    }
  }

  async function handleSelectJob(job: Job) {
    setSelectedJobId(job.id);
    setSelectedJob(job);
    setResolveCp(null);
    fetchResolveCheckpoint(job.id);
  }

  async function startJob(jobId: string) {
    try {
      await api.startJob(jobId);
      refreshJobs();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to start job");
    }
  }

  async function cancelJob(jobId: string) {
    try {
      await api.cancelJob(jobId);
      refreshJobs();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to cancel job");
    }
  }

  async function handleSelectPlaylistVideo(jobId: string, index: number) {
    try {
      await api.selectVideo(jobId, index);
      refreshJobs();
      setResolveCp(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to select video");
    }
  }

  async function handleSaveSettings(event: FormEvent) {
    event.preventDefault();
    setSettingsSuccess(false);
    try {
      const { gemini_api_keys, ...savePayload } = settings;
      await api.updateSettings(savePayload);
      setSettingsSuccess(true);
      setTimeout(() => setSettingsSuccess(false), 3000);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to save settings");
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
      setError(cause instanceof Error ? cause.message : "Unable to add Gemini API key");
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
      setError(cause instanceof Error ? cause.message : "Unable to remove Gemini API key");
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
            <Activity size={18} /> Jobs
          </button>
          <button className={activeTab === "outputs" ? "active" : ""} onClick={() => setActiveTab("outputs")}>
            <Radio size={18} /> Outputs
          </button>
          <button className={activeTab === "settings" ? "active" : ""} onClick={() => setActiveTab("settings")}>
            <Settings2 size={18} /> Settings
          </button>
        </nav>
        <button className={`runtime ${runtime?.status ?? "loading"}`} onClick={() => setRuntimeOpen(true)}>
          <i />
          <div>
            <strong>Runtime {runtime?.status ?? "checking"}</strong>
            <small>Portable runtime</small>
          </div>
        </button>
      </aside>

      <main>
        {activeTab === "jobs" && (
          <>
            <header>
              <div>
                <p>Portable Edition</p>
                <h1>Jobs Dashboard</h1>
              </div>
              <span className="phase">Pipeline Active</span>
            </header>

            {error && (
              <div className="error">
                <CircleAlert size={22} />
                <div>
                  <strong>{error}</strong>
                  <span>Check logs or configuration, then retry.</span>
                </div>
                <button onClick={() => setError(null)} style={{ background: "transparent", color: "inherit", marginLeft: "auto" }}>
                  <X size={18} />
                </button>
              </div>
            )}

            <section className="new-job">
              <div>
                <h2>Create new dubbing job</h2>
                <p>Paste a Douyin video or channel link to download, transcribe, and translate automatically.</p>
              </div>
              <form onSubmit={createJob}>
                <label>
                  <Link2 size={18} />
                  <input
                    required
                    value={sourceUrl}
                    onChange={(event) => setSourceUrl(event.target.value)}
                    placeholder="Paste a Douyin video or channel URL (e.g. https://www.douyin.com/video/...)"
                  />
                </label>
                <button type="submit" disabled={runtime?.status === "blocked"}>
                  <Plus size={18} /> Create job
                </button>
              </form>
            </section>

            <section className="jobs">
              <div className="section-title">
                <h2>Recent jobs</h2>
                <span>{jobs.length} total</span>
              </div>
              {jobs.length === 0 && !error && (
                <div className="empty">
                  <Clock3 size={32} />
                  <h3>No jobs yet</h3>
                  <p>Paste a Douyin link above to create the first checkpointed job.</p>
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
                          <span className={`status ${job.status}`}>{job.status.replaceAll("_", " ")}</span>
                          <h3 style={{ margin: "8px 0 4px", fontSize: "16px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {job.title || job.source_url}
                          </h3>
                          <small style={{ fontFamily: "monospace", color: "#626b7d" }}>{job.id}</small>
                        </div>
                        <div style={{ textAlign: "right" }}>
                          <b>{completedSteps} / {job.steps.length} steps</b>
                          <br />
                          <small style={{ color: "#747d90" }}>{new Date(job.created_at).toLocaleTimeString()}</small>
                        </div>
                      </div>
                      <div className="timeline">
                        {job.steps.map((step) => (
                          <div key={step.name} title={`${step.name.replaceAll("_", " ")}: ${step.status}`} className={step.status} />
                        ))}
                      </div>
                    </article>
                  );
                })}
              </div>
            </section>
          </>
        )}

        {activeTab === "outputs" && (
          <>
            <header>
              <div>
                <p>Portable Edition</p>
                <h1>QC Outputs Gallery</h1>
              </div>
            </header>
            <section style={{ marginTop: "24px" }}>
              {outputs.length === 0 ? (
                <div className="empty">
                  <FileVideo size={40} />
                  <h3>No completed outputs yet</h3>
                  <p>Dubbed videos will appear here once the pipeline is completed.</p>
                </div>
              ) : (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: "20px" }}>
                  {outputs.map((out) => (
                    <div key={out.job_id} className="output-card" style={{ background: "#12151c", border: "1px solid #292f3b", borderRadius: "14px", padding: "18px", display: "flex", flexDirection: "column" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
                        <FileVideo size={28} style={{ color: "#8170ff" }} />
                        <div style={{ overflow: "hidden" }}>
                          <h3 style={{ margin: 0, fontSize: "15px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{out.title}</h3>
                          <small style={{ color: "#626b7d" }}>{formatBytes(out.file_size)}</small>
                        </div>
                      </div>
                      <p style={{ color: "#8f97a6", fontSize: "13px", margin: "0 0 16px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        Source: <a href={out.source_url} target="_blank" rel="noreferrer" style={{ color: "#8170ff" }}>{out.source_url}</a>
                      </p>
                      <button
                        className="smoke-button"
                        style={{ marginTop: "auto", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}
                        onClick={() => setActiveVideoUrl(`http://127.0.0.1:8765/api/jobs/${out.job_id}/output`)}
                      >
                        <Play size={16} /> Play Dubbed Video
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
            <header>
              <div>
                <p>Portable Edition</p>
                <h1>Application Settings</h1>
              </div>
            </header>
            <form onSubmit={handleSaveSettings} style={{ marginTop: "28px", display: "grid", gap: "20px", background: "#12151c", border: "1px solid #292f3b", borderRadius: "18px", padding: "28px", maxWidth: "800px" }}>
              <h3 style={{ margin: 0, color: "#8170ff" }}>Google Translate Free</h3>
              <p style={{ margin: 0, color: "#9ca3af" }}>
                Uses Google's free web translation service. It may be rate-limited or temporarily unavailable.
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
                <label style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <span>Browser cookies for Douyin</span>
                  <select
                    className="settings-input"
                    value={settings.cookies_browser ?? "none"}
                    onChange={(e) => setSettings({ ...settings, cookies_browser: e.target.value })}
                  >
                    <option value="none">Do not use browser cookies</option>
                    <option value="edge">Microsoft Edge</option>
                    <option value="chrome">Google Chrome</option>
                    <option value="firefox">Mozilla Firefox</option>
                    <option value="brave">Brave</option>
                  </select>
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <span>Translation backend</span>
                  <select
                    className="settings-input"
                    value={settings.translation_backend ?? "google_free"}
                    onChange={(e) => setSettings({ ...settings, translation_backend: e.target.value })}
                  >
                    <option value="google_free">Google Translate Free</option>
                    <option value="gemini">Gemini</option>
                  </select>
                </label>
              </div>
              <small style={{ color: "#f6c177" }}>
                Browser cookies may contain sensitive session data. They are passed only to yt-dlp and are not stored by the application.
              </small>

              <hr style={{ border: "0", borderTop: "1px solid #292f3b", margin: "10px 0" }} />

              <h3 style={{ margin: 0, color: "#8170ff" }}>Google AI Studio / Gemini API Keys</h3>
              <p style={{ margin: 0, color: "#9ca3af" }}>
                Add multiple Gemini API keys for translation and TTS. Keys are stored locally and shown only in masked form.
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "10px", alignItems: "end" }}>
                <label style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <span>New Gemini API key</span>
                  <input
                    className="settings-input"
                    type="password"
                    placeholder="Paste Google AI Studio API key"
                    value={newGeminiKey}
                    onChange={(e) => setNewGeminiKey(e.target.value)}
                  />
                </label>
                <button
                  type="button"
                  className="smoke-button"
                  style={{ width: "auto", padding: "13px 18px" }}
                  onClick={handleAddGeminiKey}
                >
                  <Plus size={16} /> Add Gemini key
                </button>
              </div>
              <div style={{ display: "grid", gap: "8px" }}>
                {(settings.gemini_api_keys ?? []).length === 0 ? (
                  <small style={{ color: "#9ca3af" }}>No Gemini API keys added yet.</small>
                ) : (
                  (settings.gemini_api_keys ?? []).map((item: any) => (
                    <div
                      key={item.id}
                      style={{ display: "flex", justifyContent: "space-between", alignItems: "center", background: "#0e1117", border: "1px solid #292f3b", borderRadius: "10px", padding: "10px 12px" }}
                    >
                      <span style={{ fontFamily: "monospace", color: "#d7ddf0" }}>{item.masked ?? item.label}</span>
                      <button
                        type="button"
                        aria-label={`Remove Gemini key ${item.masked ?? item.label}`}
                        onClick={() => handleRemoveGeminiKey(item.id)}
                        style={{ background: "transparent", color: "#ffbcc9", display: "flex", alignItems: "center", gap: "6px" }}
                      >
                        <X size={16} /> Remove
                      </button>
                    </div>
                  ))
                )}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "14px" }}>
                <label style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <span>Gemini translation model</span>
                  <input
                    className="settings-input"
                    value={settings.gemini_translation_model ?? "gemini-2.5-flash"}
                    onChange={(e) => setSettings({ ...settings, gemini_translation_model: e.target.value })}
                  />
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <span>Gemini TTS model</span>
                  <input
                    className="settings-input"
                    value={settings.gemini_tts_model ?? "gemini-2.5-flash-preview-tts"}
                    onChange={(e) => setSettings({ ...settings, gemini_tts_model: e.target.value })}
                  />
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <span>Gemini TTS voice</span>
                  <input
                    className="settings-input"
                    value={settings.gemini_tts_voice ?? "Zephyr"}
                    onChange={(e) => setSettings({ ...settings, gemini_tts_voice: e.target.value })}
                  />
                </label>
              </div>

              <hr style={{ border: "0", borderTop: "1px solid #292f3b", margin: "10px 0" }} />

              <h3 style={{ margin: 0, color: "#8170ff" }}>ASR (Speech-To-Text)</h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
                <label style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <span>ASR Backend</span>
                  <select
                    className="settings-input"
                    value={settings.asr_backend ?? "whisper_cpu"}
                    onChange={(e) => setSettings({ ...settings, asr_backend: e.target.value })}
                  >
                    <option value="whisper_cpu">whisper.cpp CPU</option>
                    <option value="whisper_vulkan">whisper.cpp Vulkan (AMD RX6600 optimized)</option>
                    <option value="qwen3_asr">Qwen3-ASR CPU High Accuracy</option>
                  </select>
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <span>whisper.cpp GGML Model Path</span>
                  <input
                    className="settings-input"
                    placeholder="C:\path\to\ggml-base.bin"
                    value={settings.whisper_model_path ?? ""}
                    onChange={(e) => setSettings({ ...settings, whisper_model_path: e.target.value })}
                  />
                </label>
              </div>

              <hr style={{ border: "0", borderTop: "1px solid #292f3b", margin: "10px 0" }} />

              <h3 style={{ margin: 0, color: "#8170ff" }}>Microsoft Edge TTS</h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
                <label style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <span>TTS Backend</span>
                  <select
                    className="settings-input"
                    value={settings.tts_backend ?? "edge"}
                    onChange={(e) => setSettings({ ...settings, tts_backend: e.target.value })}
                  >
                    <option value="edge">Microsoft Edge TTS</option>
                    <option value="gemini">Gemini TTS</option>
                  </select>
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <span>Vietnamese voice</span>
                  <input
                    className="settings-input"
                    value={settings.edge_tts_voice ?? "vi-VN-HoaiMyNeural"}
                    onChange={(e) => setSettings({ ...settings, edge_tts_voice: e.target.value })}
                  />
                </label>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "10px" }}>
                <button type="submit" className="smoke-button" style={{ width: "auto", padding: "13px 28px" }}>
                  <Save size={18} /> Save Settings
                </button>
                {settingsSuccess && (
                  <span style={{ color: "#5bdd9a", display: "flex", alignItems: "center", gap: "6px" }}>
                    <CheckCircle2 size={18} /> Settings saved successfully!
                  </span>
                )}
              </div>
            </form>
          </>
        )}
      </main>

      {/* Selected Job Drawer Panel */}
      {selectedJob && (
        <div className="overlay" onClick={() => setSelectedJob(null)}>
          <section className="runtime-panel" onClick={(event) => event.stopPropagation()} style={{ width: "min(600px, 100%)", display: "flex", flexDirection: "column" }}>
            <div className="runtime-head">
              <div>
                <p>Job Details</p>
                <h2 style={{ fontSize: "20px", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap", maxWidth: "450px" }}>
                  {selectedJob.title || selectedJob.source_url}
                </h2>
                <small style={{ fontFamily: "monospace" }}>ID: {selectedJob.id}</small>
              </div>
              <button aria-label="Close job details" onClick={() => setSelectedJob(null)}>
                <X />
              </button>
            </div>

            {/* Status actions */}
            <div style={{ display: "flex", gap: "10px", margin: "20px 0 10px" }}>
              {(selectedJob.status === "failed" || selectedJob.status === "interrupted") && (
                <button className="smoke-button" style={{ flex: 1 }} onClick={() => startJob(selectedJob.id)}>
                  <RefreshCw size={16} /> Resume Dubbing
                </button>
              )}
              {selectedJob.status === "running" && (
                <button className="smoke-button" style={{ flex: 1, background: "#f16f7e" }} onClick={() => cancelJob(selectedJob.id)}>
                  <X size={16} /> Cancel Execution
                </button>
              )}
            </div>

            {/* Error messaging */}
            {selectedJob.status === "failed" && selectedJob.last_error_code && (
              <div className="error" style={{ margin: "10px 0" }}>
                <CircleAlert size={22} />
                <div>
                  <strong>{selectedJob.last_error_code}</strong>
                  <p style={{ margin: "4px 0", fontSize: "13px" }}>{selectedJob.last_error_message}</p>
                  <small style={{ color: "#ffbcc9" }}>Suggested Action: Check your settings and resume the job.</small>
                </div>
              </div>
            )}

            {/* Playlist videos resolution selector */}
            {selectedJob.status === "waiting_for_selection" && resolveCp && resolveCp.videos && (
              <div style={{ background: "#20242e", border: "1px solid #343a48", borderRadius: "12px", padding: "16px", margin: "12px 0" }}>
                <h3 style={{ margin: "0 0 12px", fontSize: "15px", display: "flex", alignItems: "center", gap: "8px" }}>
                  <AlertTriangle size={18} style={{ color: "#f2ba5b" }} /> Choose a Video to Dub
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
                        <small style={{ color: "#747d90" }}>Duration: {formatDuration(vid.duration)}</small>
                      </div>
                      <ChevronRight size={18} style={{ alignSelf: "center", color: "#626b7d" }} />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Vertical step checklist */}
            <div style={{ display: "grid", gap: "12px", margin: "20px 0", overflowY: "auto", flex: 1 }}>
              {selectedJob.steps.map((step) => (
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
                    <strong style={{ fontSize: "14px", textTransform: "capitalize" }}>{step.name.replaceAll("_", " ")}</strong>
                    <p style={{ margin: "2px 0 0", fontSize: "11px", color: "#747e90" }}>Status: {step.status}</p>
                  </div>
                </div>
              ))}
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
                <p>Portable runtime</p>
                <h2>Runtime {runtime.status}</h2>
                <small>Last checked {new Date(runtime.checked_at).toLocaleString()}</small>
              </div>
              <button aria-label="Close runtime panel" onClick={() => setRuntimeOpen(false)}>
                <X />
              </button>
            </div>
            <div className="runtime-checks">
              {runtime.checks.map((check) => (
                <div className="runtime-check" key={check.id}>
                  <span className={`check-icon ${check.status}`}>
                    <ShieldCheck size={18} />
                  </span>
                  <div>
                    <strong>{check.display_name}</strong>
                    <p>{check.message}</p>
                    <small>{check.action}</small>
                  </div>
                  <em>{check.status}</em>
                </div>
              ))}
            </div>
            <button className="smoke-button" onClick={runSmokeTest} disabled={testingRuntime}>
              <RefreshCw size={17} /> {testingRuntime ? "Testing..." : "Run smoke test"}
            </button>
          </section>
        </div>
      )}
    </div>
  );
}
