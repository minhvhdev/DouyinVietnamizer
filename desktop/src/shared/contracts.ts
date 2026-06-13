export type JobStep = { name: string; position: number; status: string; checkpoint_path?: string };
export type Job = {
  id: string;
  source_url: string;
  title: string | null;
  status: string;
  current_step: string | null;
  last_error_code: string | null;
  last_error_message: string | null;
  created_at: string;
  updated_at: string;
  steps: JobStep[];
};
export type JobsApi = {
  listJobs(): Promise<Job[]>;
  createJob(sourceUrl: string): Promise<Job>;
  runtimeStatus(): Promise<RuntimeReport>;
  runSmokeTest(): Promise<RuntimeReport>;
  startJob(jobId: string): Promise<{ status: string }>;
  cancelJob(jobId: string): Promise<{ status: string }>;
  selectVideo(jobId: string, index: number): Promise<{ status: string }>;
  getCheckpoint(jobId: string, stepName: string): Promise<any>;
  getSettings(): Promise<Record<string, any>>;
  updateSettings(payload: Record<string, any>): Promise<Record<string, any>>;
  getEvents(): Promise<any[]>;
  listOutputs(): Promise<any[]>;
};
export type RuntimeCheck = { id: string; display_name: string; status: string; required: boolean; message: string; action: string; source?: string; version?: string; resolved_path?: string };
export type RuntimeReport = { status: string; checked_at: string; checks: RuntimeCheck[] };

