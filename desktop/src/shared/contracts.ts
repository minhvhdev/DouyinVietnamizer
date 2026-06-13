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
};

