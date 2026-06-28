export type JobStep = { name: string; position: number; status: string; checkpoint_path?: string };

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
  name: string;
  wav_filename: string;
  wav_path: string;
  transcript: string | null;
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

export type JobsApi = {
  listJobs(): Promise<Job[]>;
  createJob(sourceUrl: string): Promise<Job>;
  importJob(file: File, title?: string): Promise<Job>;
  runtimeStatus(): Promise<RuntimeReport>;
  runSmokeTest(): Promise<RuntimeReport>;
  startJob(jobId: string): Promise<{ status: string }>;
  cancelJob(jobId: string): Promise<{ status: string }>;
  deleteJob(jobId: string): Promise<{ status: string }>;
  selectVideo(jobId: string, index: number): Promise<{ status: string }>;
  getCheckpoint(jobId: string, stepName: string): Promise<any>;
  getSettings(): Promise<Record<string, any>>;
  updateSettings(payload: Record<string, any>): Promise<Record<string, any>>;
  getEvents(): Promise<any[]>;
  listOutputs(): Promise<OutputItem[]>;
  listClonedVoices(): Promise<ClonedVoice[]>;
  createClonedVoice(name: string, file: File): Promise<ClonedVoice>;
  deleteClonedVoice(voiceId: string): Promise<{ status: string }>;
  testClonedVoice(voiceId: string, text: string, mode?: "reference" | "ultimate"): Promise<Blob>;
  previewPresetVoice(voice: string, text: string): Promise<Blob>;
  rerunJob(jobId: string, keepSteps: string[]): Promise<{ status: string; job: Job }>;
  redubJob(jobId: string): Promise<{ status: string; job: Job }>;
  getJobFiles(jobId: string): Promise<any[]>;
  detectHardware(): Promise<{ cuda_supported: boolean; vulkan_supported: boolean; avx2_supported: boolean; espeak_installed: boolean; recommendation: string }>;
  bootstrapVendor(profile: string): Promise<{ status: string }>;
  bootstrapPyannote(): Promise<{ status: string }>;
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
export type RuntimeReport = { status: string; checked_at: string; checks: RuntimeCheck[] };
