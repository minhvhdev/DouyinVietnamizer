# Phase 5-8 Speech Translation and Edge TTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide the no-key CPU path from Chinese audio to timestamped Vietnamese narration.

**Architecture:** ASR, translation, and TTS use adapter contracts injected into focused pipeline steps. whisper.cpp CPU is the baseline, `deep-translator` provides free translation, and `edge-tts` provides Vietnamese speech.

**Tech Stack:** Python, whisper.cpp, deep-translator, edge-tts, FFmpeg/ffprobe, pytest

---

## File Structure

- Create `backend/dv_backend/adapters/asr.py`.
- Create `backend/dv_backend/adapters/translation.py`.
- Create `backend/dv_backend/adapters/tts.py`.
- Create `backend/dv_backend/steps/language.py`.
- Modify `backend/pyproject.toml`, settings, runtime checks, and API capabilities.
- Test adapter contracts and language steps.

### Task 1: ASR Contract and whisper.cpp CPU

- [ ] Write failing contract tests for timestamp normalization, empty output, malformed JSON, and CPU fallback.
- [ ] Run `python -m pytest tests/test_asr.py -v` and verify failure.
- [ ] Implement `AsrAdapter` and `WhisperCppAdapter`; expose actionable errors and warnings.
- [ ] Run focused tests and commit with `feat: add whisper cpu transcription`.

### Task 2: Segment Normalization

- [ ] Write failing tests for ordering, overlap removal, duration clamping, blank text removal, and duration budgets.
- [ ] Run `python -m pytest tests/test_language_steps.py -v` and verify failure.
- [ ] Implement typed `DubSegment` normalization and checkpoint serialization.
- [ ] Run focused tests and commit with `feat: normalize dub segments`.

### Task 3: Google Free Translation Adapter

- [ ] Add `deep-translator` to backend dependencies.
- [ ] Write failing tests for batching, retryable failure, cancellation between batches, and non-empty one-to-one output.
- [ ] Run `python -m pytest tests/test_translation.py -v` and verify failure.
- [ ] Implement `GoogleFreeTranslator` with bounded retries, delay, and injectable client factory.
- [ ] Run focused tests and commit with `feat: add free Google translation`.

### Task 4: Edge TTS Adapter

- [ ] Add `edge-tts` to backend dependencies.
- [ ] Write failing tests for voice settings, transient retry, cancellation, empty output, and ffprobe validation.
- [ ] Run `python -m pytest tests/test_tts.py -v` and verify failure.
- [ ] Implement `EdgeTtsAdapter` and per-segment narration generation.
- [ ] Run focused tests and commit with `feat: add Edge Vietnamese narration`.

### Task 5: Runtime and Integrated Fixture Verification

- [ ] Add tests proving network adapters report warnings separately from required bundled runtime readiness.
- [ ] Add an opt-in network smoke test guarded by `DV_RUN_NETWORK_TESTS=1`.
- [ ] Run all backend tests and the fixture pipeline through TTS.
- [ ] Commit with `test: verify speech translation and tts pipeline`.

