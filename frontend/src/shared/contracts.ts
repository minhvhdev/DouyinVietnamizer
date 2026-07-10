export type JobStep = {
  name: string;
  position: number;
  status: string;
  checkpoint_path?: string;
  started_at?: string | null;
  completed_at?: string | null;
  duration_ms?: number | null;
};

export type Job = {
  id: string;
  source_url: string;
  title: string | null;
  title_vi: string | null;
  status: string;
  current_step: string | null;
  last_error_code: string | null;
  last_error_message: string | null;
  created_at: string;
  updated_at: string;
  steps: JobStep[];
};

export type ClonedVoice = {
  id: string;
  backend?: "omnivoice";
  name: string;
  wav_filename: string;
  wav_path: string;
  transcript: string | null;
  transcript_error?: string | null;
  transcribed: boolean;
  created_at: string;
};

export type OutputItem = {
  job_id: string;
  title: string;
  title_vi: string | null;
  source_url: string;
  completed_at: string;
  file_size: number;
};

export type JobFolder = {
  path: string;
  exists: boolean;
};

export type JobsApi = {
  listJobs(): Promise<Job[]>;
  createJob(sourceUrl: string): Promise<Job>;
  selectVideo(jobId: string, index: number): Promise<{ status: string; video: Record<string, unknown> }>;
  updateYtDlp(): Promise<{ status: string; version: string; previous_version: string; method: string }>;
  importJob(file: File, title?: string): Promise<Job>;
  runtimeStatus(): Promise<RuntimeReport>;
  runSmokeTest(): Promise<RuntimeReport>;
  releaseVram(): Promise<ReleaseVramResult>;
  startJob(jobId: string): Promise<{ status: string }>;
  cancelJob(jobId: string): Promise<{ status: string }>;
  deleteJob(jobId: string): Promise<{ status: string }>;
  getCheckpoint(jobId: string, stepName: string): Promise<any>;
  getSettings(): Promise<Record<string, any>>;
  updateSettings(payload: Record<string, any>): Promise<Record<string, any>>;
  getEvents(): Promise<any[]>;
  listOutputs(): Promise<OutputItem[]>;
  listClonedVoices(backend?: "omnivoice"): Promise<ClonedVoice[]>;
  createClonedVoice(name: string, file: File, backend?: "omnivoice", refText?: string): Promise<ClonedVoice>;
  deleteClonedVoice(voiceId: string, backend?: "omnivoice"): Promise<{ status: string }>;
  testClonedVoice(voiceId: string, text: string, mode?: "reference" | "ultimate", backend?: "omnivoice"): Promise<Blob>;
  previewPresetVoice(voice: string, text: string): Promise<Blob>;
  listTtsVoices(backend: string, locale?: string): Promise<Array<{ id: string; name: string; gender?: string; kind?: string }>>;
  listOpenAiModels(options?: { baseUrl?: string; apiKey?: string }): Promise<Array<{ id: string; name: string }>>;
  previewTts(text: string, options?: { backend?: string; voice?: string; settings?: Record<string, unknown> }): Promise<Blob>;
  rerunJob(jobId: string, keepSteps: string[]): Promise<{ status: string; job: Job }>;
  redubJob(jobId: string): Promise<{ status: string; job: Job }>;
  getJobFiles(jobId: string): Promise<any[]>;
  getJobFolder(jobId: string): Promise<JobFolder>;
  detectHardware(): Promise<{ cuda_supported: boolean; vulkan_supported: boolean; avx2_supported: boolean; espeak_installed: boolean; recommendation: string }>;
  bootstrapVendor(profile: string): Promise<{ status: string }>;
  bootstrapProgress(): Promise<{
    status: string;
    current_task: string;
    download_percent: number;
    download_speed_kb: number;
    downloaded_bytes: number;
    total_bytes: number;
    error_message: string;
    logs: string[];
  }>;
};

export type RuntimeCheck = { id: string; display_name: string; status: string; required: boolean; message: string; action: string; source?: string; version?: string; resolved_path?: string };
export type RuntimeGpuStatus = {
  cuda_supported: boolean;
  device_name?: string | null;
  total_vram_mb?: number | null;
  used_vram_mb?: number | null;
  free_vram_mb?: number | null;
  torch_allocated_mb?: number | null;
  torch_reserved_mb?: number | null;
  torch_peak_mb?: number | null;
  active_omnivoice_clients: number;
  resident_models: string[];
  helper_processes: string[];
};
export type RuntimeReport = { status: string; checked_at: string; checks: RuntimeCheck[]; gpu?: RuntimeGpuStatus | null };
export type ReleaseVramResult = {
  status: string;
  released: string[];
  terminated_processes: string[];
  errors: string[];
  gpu: RuntimeGpuStatus;
};
