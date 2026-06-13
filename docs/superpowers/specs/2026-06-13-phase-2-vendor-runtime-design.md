# Phase 2 Vendor Runtime and First-Run Smoke Test Design

## Scope

Phase 2 establishes how bundled vendor tools are discovered, validated, and reported before media jobs begin. It adds no downloading or media processing behavior. The goal is to make runtime failures visible and actionable on a Windows customer machine before they interrupt a job.

## Vendor Manifest

The application contains `vendor/manifest.json`, a versioned declaration of supported vendor tools. Each entry declares:

- Stable tool ID and display name.
- Relative bundled executable path.
- Development command used only when explicitly allowed.
- Version probe arguments.
- Whether the tool is required for the default CPU pipeline.
- Capability provided by the tool.
- Expected version text and optional release checksum metadata.

The initial manifest declares FFmpeg, yt-dlp, whisper.cpp CPU, whisper.cpp Vulkan, and Piper. FFmpeg, yt-dlp, and whisper.cpp CPU are required. Vulkan and Piper are optional during Phase 2.

Production and packaged execution only use bundled paths. Development execution may resolve a tool from `%PATH%` when `DV_ALLOW_PATH_TOOLS=1`. This exception is visible in smoke-test results and cannot silently become production behavior.

## Runtime Smoke Test

A focused backend service executes the smoke test. It checks:

1. The application data directory exists and accepts an atomic write/delete cycle.
2. SQLite accepts a read/write transaction.
3. The vendor manifest is valid.
4. Each declared executable exists or is resolved through the explicit development fallback.
5. Each resolved executable starts, exits within its timeout, and returns recognizable version output.

Each check produces a structured result with status, message, action, detail, resolved path, detected version, duration, and timestamp. Tool subprocess output is bounded so a faulty binary cannot exhaust memory.

Overall runtime status is:

- `ready`: all required checks pass.
- `warning`: required checks pass but one or more optional checks fail or development PATH fallback is active.
- `blocked`: storage, SQLite, manifest, or a required tool check fails.

whisper.cpp Vulkan failure never blocks CPU operation. A missing or failing whisper.cpp CPU binary blocks the default pipeline.

## Persistence and API

The latest full smoke-test report is stored in SQLite as JSON. A lightweight runtime status is also exposed through capabilities.

Endpoints:

- `GET /api/runtime/status`: returns the latest report or `not_run`.
- `POST /api/runtime/smoke-test`: runs all checks synchronously and persists the report.

The first automatic smoke test runs when no persisted report exists. Later app starts load the cached report immediately; the UI can request a fresh test. A later installer phase will invoke the same service during installation verification.

## User Experience

The existing sidebar runtime indicator becomes an interactive Runtime panel. It shows the overall status, last test time, and one row per check. Failed rows display a concise message and concrete action. Development PATH fallback is labeled clearly.

The jobs dashboard remains usable when optional tools fail. When runtime status is `blocked`, creating a new job is disabled and the Runtime panel becomes the primary action. Phase 2 does not claim pipeline steps are implemented.

## Error Handling

Manifest parsing errors, missing files, process launch failures, timeouts, non-zero exits, and unrecognized version output are represented as check results instead of uncaught exceptions. Unexpected service failures return the standard actionable API error envelope and are persisted as backend events.

No vendor executable is downloaded, repaired, or modified by the smoke test.

## Testing

- Manifest parser tests cover valid manifests, unsafe paths, and missing required fields.
- Tool resolver tests prove packaged mode never uses `%PATH%` and development fallback requires explicit opt-in.
- Subprocess probe tests use tiny fake executables/scripts for success, non-zero exit, and timeout behavior.
- Smoke-test service tests cover `ready`, `warning`, and `blocked` aggregation and report persistence.
- API tests cover runtime status and smoke-test execution.
- Frontend tests cover runtime rows, actionable failures, rerun behavior, and disabling job creation when blocked.

