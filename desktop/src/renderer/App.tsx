import { useEffect, useState, type FormEvent } from "react";
import { Activity, CircleAlert, Clock3, Link2, Plus, Radio, RefreshCw, Settings2, ShieldCheck, X } from "lucide-react";

import type { Job, JobsApi, RuntimeReport } from "../shared/contracts";
import { api as defaultApi } from "../shared/api";
import "./styles.css";
import "./runtime.css";

export function App({ api = defaultApi }: { api?: JobsApi }) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [sourceUrl, setSourceUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<RuntimeReport | null>(null);
  const [runtimeOpen, setRuntimeOpen] = useState(false);
  const [testingRuntime, setTestingRuntime] = useState(false);

  useEffect(() => {
    api.listJobs().then(setJobs).catch((cause: Error) => setError(cause.message));
    api.runtimeStatus().then(setRuntime).catch((cause: Error) => setError(cause.message));
  }, [api]);

  async function createJob(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      const job = await api.createJob(sourceUrl);
      setJobs((current) => [job, ...current]);
      setSourceUrl("");
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

  return (
    <div className="shell">
      <aside>
        <div className="brand"><span>DV</span><strong>Douyin<br />Vietnamizer</strong></div>
        <nav><button className="active"><Activity size={18} /> Jobs</button><button><Radio size={18} /> Outputs</button><button><Settings2 size={18} /> Settings</button></nav>
        <button className={`runtime ${runtime?.status ?? "loading"}`} onClick={() => setRuntimeOpen(true)}><i /><div><strong>Runtime {runtime?.status ?? "checking"}</strong><small>Portable runtime</small></div></button>
      </aside>
      <main>
        <header><div><p>Portable Edition</p><h1>Jobs</h1></div><span className="phase">Phase 2 runtime</span></header>
        {error && <div className="error"><CircleAlert size={22} /><div><strong>{error}</strong><span>Start the local backend, then retry. Details remain available in backend logs.</span></div></div>}
        <section className="new-job">
          <div><h2>New dubbing job</h2><p>Create a durable workspace now. Processing steps become available incrementally.</p></div>
          <form onSubmit={createJob}><label><Link2 size={18} /><input required value={sourceUrl} onChange={(event) => setSourceUrl(event.target.value)} placeholder="Paste a Douyin video or channel URL" /></label><button type="submit" disabled={runtime?.status === "blocked"}><Plus size={18} /> Create job</button></form>
        </section>
        <section className="jobs">
          <div className="section-title"><h2>Recent jobs</h2><span>{jobs.length} total</span></div>
          {jobs.length === 0 && !error && <div className="empty"><Clock3 size={32} /><h3>No jobs yet</h3><p>Paste a Douyin link above to create the first checkpointed job.</p></div>}
          {jobs.map((job) => <article key={job.id}><div className="job-top"><div><span className={`status ${job.status}`}>{job.status}</span><h3>{job.source_url}</h3><small>{job.id}</small></div><b>0 / {job.steps.length}</b></div><div className="timeline">{job.steps.map((step) => <div key={step.name} title={step.name} className={step.status} />)}</div><div className="steps">{job.steps.map((step) => <span key={step.name}>{step.name.replaceAll("_", " ")}</span>)}</div></article>)}
        </section>
      </main>
      {runtimeOpen && runtime && <div className="overlay" onClick={() => setRuntimeOpen(false)}><section className="runtime-panel" onClick={(event) => event.stopPropagation()}>
        <div className="runtime-head"><div><p>Portable runtime</p><h2>Runtime {runtime.status}</h2><small>Last checked {new Date(runtime.checked_at).toLocaleString()}</small></div><button aria-label="Close runtime panel" onClick={() => setRuntimeOpen(false)}><X /></button></div>
        <div className="runtime-checks">{runtime.checks.map((check) => <div className="runtime-check" key={check.id}><span className={`check-icon ${check.status}`}><ShieldCheck size={18} /></span><div><strong>{check.display_name}</strong><p>{check.message}</p><small>{check.action}</small></div><em>{check.status}</em></div>)}</div>
        <button className="smoke-button" onClick={runSmokeTest} disabled={testingRuntime}><RefreshCw size={17} /> {testingRuntime ? "Testing..." : "Run smoke test"}</button>
      </section></div>}
    </div>
  );
}
