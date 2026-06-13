# Phase 2 Vendor Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate bundled vendor tools and local storage through a persisted first-run smoke test with actionable runtime status in the UI.

**Architecture:** A typed manifest parser feeds a strict vendor resolver and bounded subprocess probe. A runtime smoke-test service aggregates storage, SQLite, manifest, and tool checks, persists the latest report, and exposes it through FastAPI for a dedicated React Runtime panel.

**Tech Stack:** Python, Pydantic, SQLite, FastAPI, pytest, React, TypeScript, Vitest

---

## File Structure

- `vendor/manifest.json`: versioned vendor tool declarations without bundled binaries.
- `backend/dv_backend/vendor.py`: manifest models, parser, and executable resolver.
- `backend/dv_backend/tool_probe.py`: bounded executable version probe.
- `backend/dv_backend/runtime.py`: smoke-test orchestration, aggregation, and persistence.
- `backend/dv_backend/database.py`: runtime report migration.
- `backend/dv_backend/api.py`: runtime status and smoke-test endpoints.
- `backend/tests/test_vendor.py`: parser and resolver tests.
- `backend/tests/test_tool_probe.py`: subprocess behavior tests.
- `backend/tests/test_runtime.py`: aggregation and persistence tests.
- `backend/tests/test_api.py`: runtime endpoint tests.
- `desktop/src/shared/contracts.ts`: runtime report contracts.
- `desktop/src/shared/api.ts`: runtime API calls.
- `desktop/src/renderer/App.tsx`: Runtime panel and blocked-job behavior.
- `desktop/tests/App.test.tsx`: Runtime panel behavior tests.

### Task 1: Vendor Manifest and Resolver

- [ ] Write failing tests for valid parsing, unsafe relative paths, packaged-mode strictness, and explicit development PATH fallback.
- [ ] Run `python -m pytest tests/test_vendor.py -v` and verify expected failures.
- [ ] Add `vendor/manifest.json` and implement manifest models, validation, and resolver.
- [ ] Run the focused tests and verify they pass.
- [ ] Commit with `feat: add vendor manifest and resolver`.

### Task 2: Bounded Tool Probe

- [ ] Write failing tests using fake commands for successful version output, non-zero exit, and timeout.
- [ ] Run `python -m pytest tests/test_tool_probe.py -v` and verify expected failures.
- [ ] Implement bounded subprocess execution and structured probe results.
- [ ] Run the focused tests and verify they pass.
- [ ] Commit with `feat: add vendor executable probes`.

### Task 3: Runtime Smoke-Test Service and API

- [ ] Write failing tests for storage and SQLite checks, ready/warning/blocked aggregation, persistence, and API endpoints.
- [ ] Run `python -m pytest tests/test_runtime.py tests/test_api.py -v` and verify expected failures.
- [ ] Add the runtime report table, smoke-test service, automatic first-run execution, API routes, and capability summary.
- [ ] Run all backend tests and verify they pass.
- [ ] Commit with `feat: add persisted runtime smoke tests`.

### Task 4: Runtime UI

- [ ] Write failing renderer tests for check rows, rerun action, warnings, and disabled job creation when blocked.
- [ ] Run `npm test --workspace desktop` and verify expected failures.
- [ ] Add runtime contracts/client calls and implement the interactive Runtime panel.
- [ ] Run renderer tests and production build.
- [ ] Commit with `feat: show runtime readiness in desktop app`.

### Task 5: Documentation and Verification

- [ ] Update `README.md` with vendor layout, packaged-mode rules, and development PATH fallback.
- [ ] Run `powershell -ExecutionPolicy Bypass -File scripts/test.ps1`.
- [ ] Run a live backend smoke test with fake vendor executables and verify persisted status.
- [ ] Review the implementation against the Phase 2 design requirements.
- [ ] Commit with `docs: document vendor runtime checks`.

