# Phase 12 Packaging and Release Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and verify a Windows installer containing the complete CPU baseline.

**Architecture:** A deterministic bootstrap script downloads pinned vendor artifacts and verifies checksums. PyInstaller packages the backend, electron-builder packages the desktop, and release verification inspects and smoke-tests the result.

**Tech Stack:** PowerShell, PyInstaller, Electron, electron-builder, Windows installer, pytest, Vitest

---

### Task 1: Session Token and Electron Supervision

- [ ] Write failing backend and Electron tests for token rejection, token handoff, health waiting, startup failure, and shutdown.
- [ ] Run backend and Electron tests and verify failure.
- [ ] Implement packaged-mode token middleware and a testable backend supervisor.
- [ ] Commit with `feat: secure and supervise packaged backend`.

### Task 2: Vendor Bootstrap

- [ ] Create a pinned vendor lock file with URL, version, checksum, license, and destination for FFmpeg, ffprobe, yt-dlp, whisper.cpp CPU, and multilingual model.
- [ ] Write script tests proving checksum mismatch and missing artifacts fail.
- [ ] Implement `scripts/bootstrap_vendor.ps1` with explicit download and cache behavior.
- [ ] Commit with `build: bootstrap pinned vendor runtime`.

### Task 3: Backend and Electron Packaging

- [ ] Fix and test `scripts/build_packaged.ps1`.
- [ ] Add PyInstaller configuration including `edge-tts` and `deep-translator`.
- [ ] Add Electron main-process build, electron-builder dependency/config, vendor resources, licenses, and installer metadata.
- [ ] Run package assembly and inspect required files.
- [ ] Commit with `build: package portable Windows application`.

### Task 4: Packaged Smoke Tests

- [ ] Add tests that launch the packaged backend, verify runtime status, start Electron, and confirm clean shutdown.
- [ ] Add installer-content and clean-profile verification scripts.
- [ ] Run the packaged smoke suite and commit with `test: verify packaged application`.

### Task 5: Real Douyin Release Gate

- [ ] Add `scripts/smoke_real_douyin.ps1` requiring an explicit URL and optional cookie browser.
- [ ] Ensure the script records statuses, output probes, and QC results without cookies or secrets.
- [ ] Run the real-Douyin flow from resolve through `dubbed.mp4`.
- [ ] Update README with installation, privacy disclosure, limitations, and release verification.
- [ ] Commit with `release: verify real Douyin end-to-end flow`.

