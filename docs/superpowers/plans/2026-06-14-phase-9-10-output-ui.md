# Phase 9-10 Output and UI Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair narration timing, render a usable dubbed MP4, generate QC reports, and complete the desktop workflow.

**Architecture:** Output steps consume typed segment checkpoints and produce validated artifacts. The renderer uses stable API contracts for job control, settings, outputs, and QC rather than embedding backend assumptions.

**Tech Stack:** Python, FFmpeg/ffprobe, FastAPI, React, TypeScript, Vitest, pytest

---

### Task 1: Duration Repair

- [ ] Write failing tests for Edge TTS faster-rate regeneration, bounded `atempo`, unchanged in-budget segments, and unresolved QC warnings.
- [ ] Run `python -m pytest tests/test_duration_repair.py -v` and verify failure.
- [ ] Implement bounded repair without truncation.
- [ ] Run focused tests and commit with `feat: repair narration duration`.

### Task 2: Timeline Mix and Render

- [ ] Write failing integration tests using tiny fixture video and narration WAV files.
- [ ] Run `python -m pytest tests/test_output_integration.py -v` and verify failure.
- [ ] Implement narration timeline placement, source ducking, loudness normalization, render fallback, and ffprobe validation.
- [ ] Run focused tests and commit with `feat: mix and render dubbed output`.

### Task 3: QC Reports and Output API

- [ ] Write failing tests for `qc_report.json`, `qc_report.html`, warning aggregation, output listing, and safe file responses.
- [ ] Run `python -m pytest tests/test_qc.py tests/test_api.py -v` and verify failure.
- [ ] Implement QC generation and report/output endpoints.
- [ ] Run focused tests and commit with `feat: generate output quality reports`.

### Task 4: Desktop Workflow

- [ ] Write failing renderer tests for cookie disclosure, playlist selection, progress, resume/cancel, settings, output playback/download, and QC warnings.
- [ ] Run `npm test --workspace desktop` and verify failure.
- [ ] Split the oversized renderer into focused Jobs, Settings, Runtime, Outputs, and Job Detail components while preserving behavior.
- [ ] Run renderer tests and production build; commit with `feat: complete desktop job workflow`.

### Task 5: Fixture End-to-End Gate

- [ ] Add a deterministic fixture E2E test from local input through `dubbed.mp4` and both QC reports.
- [ ] Run `powershell -ExecutionPolicy Bypass -File scripts/test.ps1`.
- [ ] Record artifact paths and probe results in test output.
- [ ] Commit with `test: add fixture end-to-end gate`.

