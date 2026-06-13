# Phase 11 Optional Qwen3-ASR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Qwen3-ASR as an optional high-accuracy adapter without affecting the default CPU path.

**Architecture:** Qwen implements the existing ASR adapter contract. Runtime capabilities expose it only when executable and model checks pass.

**Tech Stack:** Python, Qwen3-ASR CLI/runtime, pytest, React

---

### Task 1: Adapter Contract

- [ ] Write failing tests proving Qwen output normalizes to the same segment contract as whisper.cpp.
- [ ] Run `python -m pytest tests/test_qwen_asr.py -v` and verify failure.
- [ ] Implement the adapter with bounded execution and actionable errors.
- [ ] Run focused tests and commit with `feat: add optional qwen asr adapter`.

### Task 2: Runtime and UI Capability

- [ ] Write failing tests proving missing Qwen files only warn and hide selection.
- [ ] Implement executable/model runtime checks and conditional UI selection.
- [ ] Run backend/frontend tests and commit with `feat: expose qwen asr capability`.

### Task 3: Optional Real Smoke Test

- [ ] Add a `DV_RUN_QWEN_TESTS=1` smoke test using a configured model path.
- [ ] Document setup, expected memory use, and fallback behavior.
- [ ] Run baseline tests without Qwen and commit with `docs: verify optional qwen asr`.

