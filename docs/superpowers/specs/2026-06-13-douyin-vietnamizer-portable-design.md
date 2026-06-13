# Douyin Vietnamizer Portable Edition Design

## Scope

Douyin Vietnamizer Portable Edition is a Windows-first desktop application that downloads Douyin videos, creates Vietnamese narration, renders dubbed MP4 files, and produces quality-control reports. The customer runtime must not require Docker, ROCm, CUDA, WSL2, Redis, or a separately installed Python environment.

Development is divided into twelve independently testable phases. Phase 1 delivers the durable application foundation: Electron/React shell, supervised local Python backend, SQLite state, resumable jobs, settings, logs, and actionable UI errors. Later phases add pipeline capabilities without changing the foundation's public contracts.

## Recommended Architecture

The desktop process is Electron with a React renderer. Electron starts and supervises one packaged Python backend process, waits for its health endpoint, and terminates it during shutdown. Development uses the system Python environment; the customer build uses a PyInstaller executable.

The backend is a FastAPI application bound only to `127.0.0.1`. Electron supplies a random session token at startup and sends it with every API request. The backend owns SQLite, job state, pipeline orchestration, logs, checkpoints, tool execution, and adapters.

The backend remains operationally simple while its internals are modular:

- API routes expose health, capabilities, jobs, settings, logs, and outputs.
- A job service owns job state transitions and validation.
- A pipeline runner executes one declared step at a time.
- Pipeline steps communicate through typed artifact manifests.
- ASR and TTS implementations use adapter interfaces.
- Vendor tools are invoked through a common subprocess runner with timeout, cancellation, captured logs, and actionable errors.

## Windows Data Layout

Customer data is stored under `%LOCALAPPDATA%\DouyinVietnamizer` so application upgrades do not destroy jobs or outputs:

```text
DouyinVietnamizer/
  app.db
  logs/backend.log
  settings.json
  jobs/<job-id>/
    checkpoints/<step-name>.json
    artifacts/
    output/
```

Development may override the root with `DV_DATA_DIR`. Vendor binaries are read from an application-managed `vendor/` directory and never downloaded silently during a production job.

## Job and Checkpoint Model

A job progresses through the declared pipeline:

`resolve`, `download`, `extract_audio`, `vad`, `asr`, `normalize_segments`, `translate`, `tts`, `duration_repair`, `mix`, `render`, `qc`.

Each step has `pending`, `running`, `completed`, `failed`, or `skipped` status. Before a step starts, SQLite records `running`. On success, the step writes artifacts and an atomic JSON checkpoint, then SQLite records `completed`. A job resumes at the first incomplete or failed step whose dependencies are satisfied. Checkpoint files include schema version, job ID, step name, completion time, input fingerprints, artifact paths, and summary metadata.

SQLite is the queryable index and UI source of truth. Checkpoint files and artifacts are the durable recovery source. On startup, the backend reconciles jobs left in `running` state to `interrupted`, then exposes a resume action.

## Pipeline Behavior

1. Resolve a Douyin URL or channel and fetch a selectable video list.
2. Download selected media with bundled yt-dlp.
3. Use bundled FFmpeg to create original 48 kHz WAV and ASR 16 kHz mono WAV.
4. Detect speech regions with local VAD.
5. Transcribe through an ASR adapter. whisper.cpp CPU is always available. Vulkan is optional and automatically retries on CPU after initialization or execution failure. Qwen3-ASR CPU is an optional high-accuracy adapter.
6. Normalize ASR output into typed `DubSegment` records with duration budgets.
7. Translate through an OpenAI-compatible API using duration-aware prompts and validated JSON output.
8. Generate TTS per segment using an API provider by default, with Piper as the local fallback.
9. Measure generated speech and repair translations that exceed their budgets.
10. Mix narration with source audio using FFmpeg sidechain ducking.
11. Normalize loudness and render the final MP4.
12. Produce a machine-readable and human-readable QC report.

## Phase 1 User Experience

The main screen is a Jobs dashboard. It shows backend health, job status, progress, current step, last error, and output availability. A New Job form accepts a Douyin URL and creates a durable queued job. Pipeline steps not implemented yet are visibly marked unavailable rather than appearing to work.

The job detail screen shows a checkpoint timeline, job metadata, resume/cancel controls when applicable, and an error panel with a concise message, suggested action, technical detail, and log location.

Settings exposes storage location and future provider/tool settings. Logs shows recent structured backend events. All screens remain useful when the backend is unavailable by showing a reconnecting state and a concrete diagnostic message.

## Error Handling

Backend errors use a stable envelope containing `code`, `message`, `action`, `detail`, `retryable`, and optional `log_path`. Errors are persisted against the job and step and returned to the UI. Raw subprocess failures are translated into actionable domain errors while preserving command exit code and stderr in logs.

The UI never hides failures behind a generic toast. Job failures remain visible on the dashboard and job detail screen until retried or dismissed. GPU/Vulkan failures are warnings when CPU fallback succeeds and errors only when CPU also fails.

## Packaging and Runtime

The final Windows distribution bundles Electron, the PyInstaller backend, FFmpeg, yt-dlp, whisper.cpp CPU, optional whisper.cpp Vulkan files, Piper, default configuration, and license notices. The installer performs a first-run smoke test for backend startup, SQLite access, writable storage, and vendor binary execution.

CPU mode is the compatibility baseline. RX6600 Vulkan acceleration is an optional optimization and cannot prevent the application from completing a job in CPU mode.

## Testing Strategy

- Backend unit tests cover configuration, database migrations, job transitions, checkpoint atomicity, reconciliation, and error envelopes.
- Backend API tests cover health, capabilities, jobs, settings, logs, and invalid requests.
- Adapter contract tests use fake ASR, TTS, translation, and tool runners.
- Pipeline integration tests use tiny fixture media and verify resume behavior after every step.
- Frontend tests cover API error rendering and core job interactions.
- Electron smoke tests verify backend supervision and shutdown.
- Windows release smoke tests run on a clean machine profile with no Python, Docker, CUDA, ROCm, or WSL2 dependencies.

## Phase Boundaries

Each phase must leave the application runnable and add capability declarations used by the UI. Phase 1 establishes all long-lived contracts. Phases 2-10 add the default production pipeline. Phase 11 adds Qwen3-ASR without changing the ASR contract. Phase 12 packages and verifies the customer installer.

