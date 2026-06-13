export type JobStep = { name: string; position: number; status: string; checkpoint_path?: string };
export type Job = {
  id: string;
  source_url: string;
  status: string;
  current_step: string | null;
  created_at: string;
  updated_at: string;
  steps: JobStep[];
};
export type JobsApi = {
  listJobs(): Promise<Job[]>;
  createJob(sourceUrl: string): Promise<Job>;
  runtimeStatus(): Promise<RuntimeReport>;
  runSmokeTest(): Promise<RuntimeReport>;
};
export type RuntimeCheck = { id: string; display_name: string; status: string; required: boolean; message: string; action: string; source?: string; version?: string; resolved_path?: string };
export type RuntimeReport = { status: string; checked_at: string; checks: RuntimeCheck[] };

