# Phases 3-12 Portable Hybrid Design

## Scope

Phases 3-12 complete Douyin Vietnamizer as a Windows-first portable product that
accepts a real Douyin video or channel URL and produces a Vietnamese-dubbed MP4
plus machine-readable and human-readable quality-control reports.

The default pipeline requires no paid API key:

- yt-dlp resolves and downloads Douyin media.
- Browser cookies may be read from an explicitly selected installed browser.
- whisper.cpp CPU performs Chinese speech recognition.
- Google Translate's free web service performs Chinese-to-Vietnamese translation.
- Microsoft Edge's online TTS service produces Vietnamese narration.
- FFmpeg performs audio extraction, timing repair, mixing, loudness normalization,
  and final rendering.

Qwen3-ASR and whisper.cpp Vulkan remain optional. Their absence or failure cannot
block the default CPU pipeline.

Google Translate and Edge TTS are unofficial or free network-backed services.
They can rate-limit, change behavior, or become unavailable. The application
must expose those failures clearly and preserve completed checkpoints for retry.

## Completion Criteria

The remaining phases are complete only when all of the following are true:

1. A clean Windows user profile can install and start the application without a
   separate Python, Node.js, FFmpeg, yt-dlp, or whisper.cpp installation.
2. Runtime smoke tests verify bundled required tools and models before jobs run.
3. A real Douyin URL can complete from resolution through rendered output using
   the default CPU pipeline.
4. A channel or playlist URL pauses for user selection before downloading.
5. A failed or cancelled job resumes from its first incomplete checkpoint.
6. The final output includes `dubbed.mp4`, `qc_report.json`, and `qc_report.html`.
7. Automated backend, frontend, Electron, packaging, and fixture-media tests pass.
8. A documented manual release smoke test succeeds with a real Douyin URL.

## Architecture

The existing Electron renderer, FastAPI backend, SQLite database, vendor runtime,
job model, and checkpoint declarations remain the long-lived foundation.

The pipeline is split into focused step modules behind a `PipelineStep` contract.
The `JobRunner` owns orchestration, cancellation, status transitions, retries,
and resume behavior. Steps read typed checkpoint data from completed dependencies
and atomically write their own checkpoint only after all required artifacts exist.

Network-backed translation and TTS are adapters. This keeps Google Translate and
Edge TTS replaceable without changing pipeline orchestration or checkpoint shapes.
Tool execution remains behind one cancellable subprocess runner.

## Default Runtime Dependencies

The packaged customer build contains:

- Electron application and compiled renderer.
- PyInstaller backend executable.
- FFmpeg and ffprobe.
- yt-dlp.
- whisper.cpp CPU executable.
- A compatible whisper.cpp multilingual model.
- Python dependencies embedded in the backend: `edge-tts` for speech generation
  and `deep-translator`'s `GoogleTranslator` adapter for free translation.
- Vendor manifest, license notices, and model/tool provenance.

The installer does not silently download required runtime files. Release assembly
downloads pinned vendor artifacts into a staging cache, verifies checksums, and
packages them. Development setup may download the same pinned artifacts through
an explicit bootstrap command.

## Settings

Settings add the following user-visible values:

- `cookies_browser`: `none`, `edge`, `chrome`, `firefox`, or `brave`.
- `translation_backend`: defaults to `google_free`.
- `translation_source_language`: defaults to `zh-CN`.
- `translation_target_language`: defaults to `vi`.
- `edge_tts_voice`: defaults to a supported Vietnamese neural voice.
- `edge_tts_rate`, `edge_tts_pitch`, and `edge_tts_volume`.
- `asr_backend`: defaults to `whisper_cpu`.
- `whisper_model_path`: defaults to the bundled multilingual model.

Cookie access is disabled until the user selects a browser. The application shows
that browser cookies may contain sensitive session data and are passed only to
yt-dlp. Cookies are never copied into checkpoints, logs, or the SQLite database.

## Pipeline Phases

### Phase 3: Resolve and Download

yt-dlp resolves single videos and channel or playlist URLs. Commands include the
selected `--cookies-from-browser` value when enabled. A single video is selected
automatically. Multiple videos produce a durable selection checkpoint and place
the job in `waiting_for_selection`.

The download step stores the original media and metadata JSON. It validates that
the downloaded file exists and is readable by ffprobe before completing.

### Phase 4: Audio Extraction and VAD

FFmpeg creates:

- `source_48k.wav` for mixing.
- `asr_16k_mono.wav` for speech recognition.

The VAD baseline uses FFmpeg silence detection and converts silence intervals into
speech regions. Checkpoints include media duration and normalized region timing.

### Phase 5: Default ASR

whisper.cpp CPU transcribes the ASR WAV into timestamped Chinese segments. The
adapter validates output JSON, normalizes timestamps, and rejects empty results.

If whisper.cpp Vulkan is selected and fails to initialize or execute, the adapter
records a warning and retries once with whisper.cpp CPU.

### Phase 6: Segment Normalization

ASR segments are cleaned, ordered, clamped to media duration, and made
non-overlapping. Each `DubSegment` contains source text, start/end time, original
duration, and narration duration budget.

### Phase 7: Free Translation

The Google Translate adapter uses `deep-translator`'s `GoogleTranslator` to
translate segments from Chinese to Vietnamese in bounded batches. It applies
timeout, limited exponential-backoff retries, and rate limiting. It validates
that every source segment receives non-empty output and preserves
source/translation pairs in the checkpoint.

Translation failure is retryable and never destroys ASR or normalized-segment
checkpoints.

### Phase 8: Edge TTS

The Edge TTS adapter produces one narration file per translated segment using the
configured Vietnamese voice. It retries transient network failures and validates
that each generated file is non-empty and decodable by ffprobe.

The checkpoint records voice settings and measured narration durations.

### Phase 9: Duration Repair

Narration that exceeds its duration budget is repaired in this order:

1. Regenerate Edge TTS with a bounded faster speaking rate.
2. Apply bounded FFmpeg `atempo` adjustment.
3. Mark the segment as a QC warning if it still exceeds the budget.

The free Google Translate adapter is not asked to shorten or rewrite text because
it does not reliably follow instructions. No repair may silently truncate speech.

### Phase 10: Mix, Render, and QC

Narration segments are placed on a media-length timeline. FFmpeg ducks the source
audio while narration is active, normalizes loudness, and renders `dubbed.mp4`
while preserving the source video stream when compatible.

QC reports include:

- Input and output metadata.
- Segment counts and timing coverage.
- Empty or failed translations.
- TTS generation and duration-repair warnings.
- ASR fallback and network retry warnings.
- Output existence, duration delta, and decode probe result.

### Phase 11: Optional Qwen3-ASR

Qwen3-ASR is implemented behind the same ASR adapter contract and exposed only
when its executable and model pass runtime checks. It is never bundled as a
required dependency and never changes the default CPU completion path.

### Phase 12: Packaging and Release Verification

The release process:

1. Builds and tests the renderer, Electron main process, and backend.
2. Downloads or reuses pinned vendor artifacts and verifies checksums.
3. Builds the PyInstaller backend.
4. Packages Electron with `electron-builder`.
5. Includes vendor binaries, model, manifest, licenses, and backend executable.
6. Installs into a clean Windows test profile.
7. Runs startup, shutdown, runtime, fixture-media, and real-Douyin smoke tests.

The release command fails when a required vendor artifact, checksum, test, or
packaging output is missing.

## API and UI

The existing jobs API is extended with start, resume, cancel, video selection,
checkpoint inspection, output listing, and output download endpoints.

The renderer provides:

- New-job form with cookie-browser selection and disclosure.
- Jobs dashboard with progress, current step, warnings, and last actionable error.
- Selection UI for channel or playlist results.
- Job detail with resume/cancel actions and checkpoint timeline.
- Settings for ASR, Google Translate, Edge TTS, and cookie browser.
- Output playback/download and QC report access.
- Runtime panel covering required binaries, model, network-backed adapters, and
  optional ASR backends.

Network service availability is reported separately from required bundled runtime
readiness. Temporary Google Translate or Edge TTS outages warn the user but do not
make the installed application structurally invalid.

## Error Handling and Security

All failures use the stable actionable error envelope and are persisted as events.
Subprocess stderr is bounded and secrets are redacted. Commands displayed in logs
replace cookie browser profile details and sensitive paths with safe summaries.

The backend binds only to `127.0.0.1`. Electron generates a random session token,
passes it to the backend process, and sends it with API requests. The backend
rejects missing or invalid tokens in packaged mode.

Translation and TTS adapters use explicit connect/read timeouts, bounded retries,
and cancellation checks between requests. Cancellation terminates active vendor
processes and leaves completed checkpoints reusable.

## Testing Strategy

Backend unit tests cover each adapter and step using deterministic fake services
and tiny media fixtures. Integration tests execute the real bundled FFmpeg against
fixture media and exercise resume after every checkpoint boundary.

Frontend tests cover cookie disclosure, selection, progress, actionable failures,
resume/cancel, settings, outputs, and QC display. Electron smoke tests verify
backend token handoff, health waiting, supervision, and shutdown.

Packaging tests inspect installer contents and run the packaged backend/runtime
smoke test. The real-Douyin release smoke test is manual or opt-in because Douyin,
Google Translate, and Edge TTS are external services. Its URL and result are
recorded in a release verification report without recording cookies.

## Phase Documentation

Implementation plans are split into independently verifiable groups:

1. Pipeline foundation and real media ingestion.
2. ASR, segment normalization, free translation, and Edge TTS.
3. Duration repair, mixing, rendering, QC, and UI completion.
4. Optional Qwen3-ASR.
5. Packaging, installer, and release verification.

Each group must leave the application runnable, keep prior tests green, and add
fresh verification evidence before it is marked complete.
