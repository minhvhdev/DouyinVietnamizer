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
  voice_status?: string;
  duration_profile_status?: string;
  duration_profile_key?: string | null;
  duration_profile_quality?: string | null;
  duration_profile_sample_count?: number;
  last_calibrated_at?: string | null;
  active_calibration_job_id?: string | null;
};

export type VoiceCalibrationStatus = {
  voice_id: string;
  job_id?: string;
  status: string;
  mode?: string;
  completed?: number;
  total?: number;
  accepted?: number;
  rejected?: number;
  estimated_profile_quality?: string;
  validation_median_error_ms?: number | null;
  syllables_per_second?: number | null;
  duration_profile_quality?: string | null;
  duration_profile_sample_count?: number;
};

export type VoiceWpsEntry = {
  catalog_key: string;
  provider: string;
  provider_label: string;
  kind: string;
  voice_id: string;
  voice_name: string;
  language: string;
  words_per_second: number | null;
  effective_words_per_second: number;
  default_words_per_second: number;
  profile_source: string | null;
  profile_key: string | null;
  cloned_voice_id: string | null;
  duration_profile_status?: string | null;
  measure_supported: boolean;
};

export type VoiceWpsMeasureResult = {
  catalog_key: string;
  words_per_second: number;
  syllables_per_second?: number;
  sample_count_total?: number;
  sample_count_accepted?: number;
  sample_count_rejected?: number;
  prediction_mae_ms?: number;
  profile_source?: string;
  profile_key?: string;
};

export type JobFolder = {
  path: string;
  exists: boolean;
};

export type TimingReviewSegment = {
  index: number;
  start?: number;
  end?: number;
  source_start?: number | null;
  source_end?: number | null;
  placement_start?: number | null;
  placement_end?: number | null;
  effective_start?: number;
  effective_end?: number;
  timing_stage?: string;
  source_text?: string;
  spoken_text: string;
  plan_version?: number;
  timing_status?: string;
  timing_review_reason?: string;
  required_speed?: number;
  max_allowed_speed?: number;
  overflow_seconds?: number;
  estimated_words_to_remove?: number;
  estimated_words_to_remove_min?: number;
  estimated_words_to_remove_max?: number;
  timing_available_duration?: number;
  repaired_duration?: number;
  release_blocking?: boolean;
};

export type TimingReviewPayload = {
  job_id: string;
  source_step: string;
  timing_stage?: string;
  segments: TimingReviewSegment[];
  remaining_count: number;
  release_eligible: boolean;
  max_speed: number;
  pace_policy?: string;
};

export type TimingReviewSubmitResult = {
  status: string;
  edited_indices: number[];
  remaining_count: number;
  overlap_count: number;
  release_eligible: boolean;
  segments: TimingReviewSegment[];
  detail?: string | null;
};

export type EditableSegmentOrigin = "pipeline" | "user";

export type EditableSegmentDto = {
  segment_id: string;
  start_ms: number;
  end_ms: number;
  spoken_text: string;
  source_text: string | null;
  origin: EditableSegmentOrigin;
  source_segment_index: number | null;
};

export type SegmentEditDelta = {
  segment_id: string;
  added?: boolean;
  deleted?: boolean;
  text_changed?: boolean;
  timing_changed?: boolean;
  order_changed?: boolean;
};

export type SegmentEditDiff = {
  has_changes: boolean;
  structural_changed: boolean;
  deltas: SegmentEditDelta[];
  requires_tts_segment_ids: string[];
  requires_duration_check_segment_ids: string[];
  reusable_tts_segment_ids: string[];
  deleted_segment_ids: string[];
};

export type SegmentEditPlanResponse = {
  schema_version: number;
  plan_version: number;
  applied_plan_version: number;
  draft_segments: EditableSegmentDto[];
  diff: SegmentEditDiff;
};

export type SegmentEditSaveRequest = {
  expected_plan_version: number;
  segments: Array<{
    segment_id: string | null;
    start_ms: number;
    end_ms: number;
    spoken_text: string;
  }>;
};

export type SegmentExportRequest = {
  expected_plan_version: number;
};

export type SegmentExportResponse = {
  status: "unchanged" | "queued" | string;
  captured_plan_version?: number;
  [key: string]: unknown;
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
  listClonedVoices(backend?: "omnivoice"): Promise<ClonedVoice[]>;
  createClonedVoice(name: string, file: File, backend?: "omnivoice", refText?: string): Promise<ClonedVoice>;
  deleteClonedVoice(voiceId: string, backend?: "omnivoice"): Promise<{ status: string }>;
  testClonedVoice(voiceId: string, text: string, mode?: "reference" | "ultimate", backend?: "omnivoice"): Promise<Blob>;
  startVoiceCalibration(voiceId: string, mode?: "full"): Promise<VoiceCalibrationStatus>;
  getVoiceCalibration(voiceId: string): Promise<VoiceCalibrationStatus>;
  cancelVoiceCalibration(voiceId: string): Promise<{ status: string; job_id?: string; voice_id: string }>;
  resumeVoiceCalibration(voiceId: string): Promise<VoiceCalibrationStatus>;
  resetVoiceDurationProfile(voiceId: string): Promise<{ status: string; voice_id: string }>;
  getVoiceWpsCatalog(language?: string): Promise<VoiceWpsEntry[]>;
  updateVoiceWps(catalogKey: string, wordsPerSecond: number, language?: string): Promise<{ catalog_key: string; words_per_second: number }>;
  measureVoiceWps(catalogKey: string, language?: string): Promise<VoiceWpsMeasureResult>;
  previewPresetVoice(voice: string, text: string): Promise<Blob>;
  listTtsVoices(backend: string, locale?: string): Promise<Array<{ id: string; name: string; gender?: string; kind?: string }>>;
  listOpenAiModels(options?: { baseUrl?: string; apiKey?: string }): Promise<Array<{ id: string; name: string }>>;
  previewTts(text: string, options?: { backend?: string; voice?: string; settings?: Record<string, unknown> }): Promise<Blob>;
  rerunJob(jobId: string, keepSteps: string[]): Promise<{ status: string; job: Job }>;
  redubJob(jobId: string): Promise<{ status: string; job: Job }>;
  getJobFiles(jobId: string): Promise<any[]>;
  getJobFolder(jobId: string): Promise<JobFolder>;
  getTimingReview(jobId: string): Promise<TimingReviewPayload>;
  submitTimingReview(
    jobId: string,
    edits: Array<{ index: number; spoken_text: string; expected_plan_version?: number }>,
    resumePipeline?: boolean,
  ): Promise<TimingReviewSubmitResult>;
  getSegmentEditPlan(jobId: string): Promise<SegmentEditPlanResponse>;
  saveSegmentEditPlan(jobId: string, payload: SegmentEditSaveRequest): Promise<SegmentEditPlanResponse>;
  exportSegmentDraft(jobId: string, payload: SegmentExportRequest): Promise<SegmentExportResponse>;
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
