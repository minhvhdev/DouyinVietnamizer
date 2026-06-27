import { useEffect, useState } from "react";
import {
  invokeSetup,
  subscribeSetupProgress,
  waitForBackend,
  type BackendStatus,
} from "../lib/tauri-bridge";

interface Props {
  status: BackendStatus;
  onComplete: (baseUrl: string) => void;
  onOpenBackendFolder: () => void;
  onCopyError: (text: string) => void;
}

const STAGE_COPY: Record<string, { title: string; body: string }> = {
  missing_uv: {
    title: "uv is not installed",
    body: "Install uv from https://docs.astral.sh/uv/, then click Retry.",
  },
  missing_python: {
    title: "Python 3.12 is not on PATH",
    body: "Install Python 3.12 (https://www.python.org/) or let the wizard fetch it via uv.",
  },
  missing_venv: {
    title: "Python environment is not initialized",
    body: "The first-run setup will install Python 3.12 and create the venv.",
  },
};

export function SetupWizard({ status, onComplete, onOpenBackendFolder, onCopyError }: Props) {
  const [progress, setProgress] = useState<{ stage: string; pct: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let unlisten: (() => void) | null = null;
    subscribeSetupProgress((p) => setProgress(p)).then((u) => { unlisten = u; });
    return () => { unlisten?.(); };
  }, []);

  async function startSetup() {
    setError(null);
    setProgress({ stage: "starting", pct: 0 });
    setRunning(true);
    try {
      await invokeSetup();
      const baseUrl = await waitForBackend({ timeoutMs: 60_000 });
      onComplete(baseUrl);
    } catch (e) {
      const msg = (e && typeof e === "object" && "stderr" in e)
        ? String((e as { stderr: unknown }).stderr)
        : (e instanceof Error ? e.message : String(e));
      setError(msg);
    } finally {
      setRunning(false);
    }
  }

  if (status.kind === "crashed") {
    return (
      <div className="p-6 max-w-2xl mx-auto">
        <h1 className="text-xl font-semibold text-red-600">Backend crashed</h1>
        <pre className="mt-3 p-3 bg-zinc-900 text-zinc-100 text-sm overflow-auto rounded">
{status.stderr || "(no stderr captured)"}
        </pre>
        <div className="mt-4 flex gap-2">
          <button onClick={onOpenBackendFolder} className="px-3 py-1.5 rounded bg-zinc-200 hover:bg-zinc-300">
            Open backend folder
          </button>
          <button onClick={() => onCopyError(status.stderr)} className="px-3 py-1.5 rounded bg-zinc-200 hover:bg-zinc-300">
            Copy error
          </button>
          <button
            onClick={async () => {
              try {
                const baseUrl = await waitForBackend({ timeoutMs: 60_000 });
                onComplete(baseUrl);
              } catch (e) {
                setError(String(e));
              }
            }}
            className="px-3 py-1.5 rounded bg-emerald-500 text-white hover:bg-emerald-600"
          >
            Retry
          </button>
        </div>
        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
      </div>
    );
  }

  const stage = status.kind === "setup_required" ? status.stage : "missing_venv";
  const copy = STAGE_COPY[stage] ?? STAGE_COPY.missing_venv;

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h1 className="text-xl font-semibold">{copy.title}</h1>
      <p className="mt-2 text-zinc-700">{copy.body}</p>
      {progress && (
        <div className="mt-4">
          <div className="text-sm text-zinc-600">
            {progress.stage}: {progress.pct}%
          </div>
          <div className="mt-1 h-2 bg-zinc-200 rounded">
            <div className="h-2 bg-emerald-500 rounded transition-all" style={{ width: `${progress.pct}%` }} />
          </div>
        </div>
      )}
      {error && (
        <div className="mt-4 p-3 rounded bg-red-50 text-red-700 text-sm">
          {error}
        </div>
      )}
      <div className="mt-4">
        <button
          disabled={running}
          onClick={startSetup}
          className="px-4 py-2 rounded bg-emerald-500 text-white hover:bg-emerald-600 disabled:opacity-50"
        >
          {running ? "Setting up..." : "Setup now"}
        </button>
      </div>
    </div>
  );
}
