# Phase 3-4 Real Media Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make real Douyin resolution, browser-cookie download, audio extraction, VAD, cancellation, and resume reliable.

**Architecture:** Keep `JobRunner` as the orchestrator, but move common subprocess and settings behavior into focused modules. Resolve/download steps receive cookie settings without persisting cookie data and validate media through ffprobe before completing.

**Tech Stack:** Python, FastAPI, SQLite, yt-dlp, FFmpeg/ffprobe, pytest

---

## File Structure

- Create `backend/dv_backend/processes.py`: bounded cancellable subprocess execution.
- Create `backend/dv_backend/settings.py`: typed defaults and settings access.
- Create `backend/dv_backend/steps/ingestion.py`: resolve, download, extract-audio, and VAD steps.
- Modify `backend/dv_backend/runner.py`: explicit step registry and reliable resume/cancel transitions.
- Modify `backend/dv_backend/api.py`: resume, selection, and validated settings endpoints.
- Test `backend/tests/test_ingestion.py`, `backend/tests/test_runner.py`, and `backend/tests/test_api.py`.

### Task 1: Typed Settings and Cookie Disclosure

- [ ] Write failing tests proving defaults use `cookies_browser=none` and only supported browser names are accepted.
- [ ] Run `python -m pytest tests/test_settings.py tests/test_api.py -v` and verify the new tests fail.
- [ ] Implement `SettingsService` with defaults and an allowlist of `none`, `edge`, `chrome`, `firefox`, and `brave`.
- [ ] Make resolve/download append `--cookies-from-browser <browser>` only when the setting is not `none`.
- [ ] Run the focused tests and commit with `feat: add browser cookie settings`.

### Task 2: Cancellable Process Runner

- [ ] Write failing tests for success, bounded stderr, timeout, cancellation, and secret-redacted command summaries.
- [ ] Run `python -m pytest tests/test_processes.py -v` and verify failure.
- [ ] Move process execution from `pipeline.py` into `processes.py`; register and unregister processes with `JobRunner`.
- [ ] Run focused tests and commit with `feat: add cancellable vendor process runner`.

### Task 3: Resolve and Download Real Media

- [ ] Write failing tests for single video, playlist selection, cookie arguments, yt-dlp errors, and ffprobe validation.
- [ ] Run `python -m pytest tests/test_ingestion.py -v` and verify failure.
- [ ] Implement ingestion steps with durable metadata, atomic checkpoints, and actionable errors.
- [ ] Ensure playlist selection resumes at download without rerunning resolve.
- [ ] Run focused tests and commit with `feat: ingest real Douyin media`.

### Task 4: Audio Extraction and VAD

- [ ] Write failing tests that execute real FFmpeg against a generated tiny fixture and assert 48 kHz, 16 kHz mono, duration, and speech-region checkpoints.
- [ ] Run `python -m pytest tests/test_ingestion_integration.py -v` and verify failure.
- [ ] Implement extraction and silence-based VAD with ffprobe validation.
- [ ] Run focused tests and commit with `feat: extract audio and detect speech`.

### Task 5: Resume, Cancel, and API Verification

- [ ] Write failing tests proving cancel terminates active processes and resume starts at the first incomplete or failed step.
- [ ] Run `python -m pytest tests/test_runner.py tests/test_api.py -v` and verify failure.
- [ ] Implement explicit resume endpoint and consistent job/step status transitions.
- [ ] Run `powershell -ExecutionPolicy Bypass -File scripts/test.ps1`.
- [ ] Commit with `feat: make jobs resumable and cancellable`.

