# Phase 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Windows-first Electron/React shell with a supervised FastAPI backend, SQLite jobs, durable checkpoint declarations, settings, logs, and actionable errors.

**Architecture:** A Vite React renderer calls a loopback FastAPI backend through Electron-provided connection settings. The backend uses focused service modules around SQLite and exposes stable API contracts that later pipeline phases extend.

**Tech Stack:** Electron, React, TypeScript, Vite, Vitest, Python 3.11+, FastAPI, Pydantic, SQLite, pytest

---

## File Structure

- `backend/dv_backend/config.py`: resolves Windows-compatible data paths and runtime configuration.
- `backend/dv_backend/database.py`: SQLite connection, schema migration, and transaction helper.
- `backend/dv_backend/models.py`: API and domain models.
- `backend/dv_backend/jobs.py`: job creation, listing, lookup, and interruption reconciliation.
- `backend/dv_backend/checkpoints.py`: pipeline step declarations and checkpoint path helpers.
- `backend/dv_backend/errors.py`: stable actionable error envelope.
- `backend/dv_backend/api.py`: FastAPI route composition.
- `backend/dv_backend/main.py`: packaged backend entrypoint.
- `backend/tests/`: backend unit and API tests.
- `desktop/src/main/`: Electron backend supervisor and secure preload bridge.
- `desktop/src/renderer/`: React jobs dashboard, new-job form, job detail, settings, and logs.
- `desktop/src/shared/`: frontend API contracts and client.
- `desktop/tests/`: renderer component tests.

### Task 1: Backend Configuration and Database

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/dv_backend/__init__.py`
- Create: `backend/dv_backend/config.py`
- Create: `backend/dv_backend/database.py`
- Test: `backend/tests/test_config.py`
- Test: `backend/tests/test_database.py`

- [ ] Write tests proving `DV_DATA_DIR` overrides the default and database migration creates `jobs`, `job_steps`, `settings`, and `events`.
- [ ] Run `python -m pytest backend/tests/test_config.py backend/tests/test_database.py -v` and verify failure.
- [ ] Implement configuration and idempotent SQLite migration.
- [ ] Run the tests and verify they pass.
- [ ] Commit with `feat: add backend configuration and database`.

### Task 2: Durable Job Domain and API

**Files:**
- Create: `backend/dv_backend/models.py`
- Create: `backend/dv_backend/errors.py`
- Create: `backend/dv_backend/checkpoints.py`
- Create: `backend/dv_backend/jobs.py`
- Create: `backend/dv_backend/api.py`
- Test: `backend/tests/test_jobs.py`
- Test: `backend/tests/test_api.py`

- [ ] Write tests proving job creation creates all twelve pending steps, invalid URLs return an actionable error envelope, jobs can be listed and fetched, and interrupted jobs are reconciled.
- [ ] Run `python -m pytest backend/tests/test_jobs.py backend/tests/test_api.py -v` and verify failure.
- [ ] Implement typed models, error handling, checkpoint declarations, job service, and FastAPI routes.
- [ ] Run the tests and verify they pass.
- [ ] Commit with `feat: add durable jobs API`.

### Task 3: Backend Entry Point and Logs

**Files:**
- Create: `backend/dv_backend/main.py`
- Modify: `backend/dv_backend/api.py`
- Test: `backend/tests/test_api.py`

- [ ] Add API tests for health, capabilities, settings, and recent events.
- [ ] Run the focused API tests and verify failure.
- [ ] Implement runtime entrypoint, structured event persistence, settings routes, and capability declarations.
- [ ] Run all backend tests and verify they pass.
- [ ] Commit with `feat: expose backend health settings and logs`.

### Task 4: Electron and React Shell

**Files:**
- Create: `package.json`
- Create: `desktop/package.json`
- Create: `desktop/tsconfig.json`
- Create: `desktop/vite.config.ts`
- Create: `desktop/index.html`
- Create: `desktop/src/main/main.ts`
- Create: `desktop/src/main/backendSupervisor.ts`
- Create: `desktop/src/main/preload.ts`
- Create: `desktop/src/shared/contracts.ts`
- Create: `desktop/src/shared/api.ts`
- Create: `desktop/src/renderer/main.tsx`
- Create: `desktop/src/renderer/App.tsx`
- Create: `desktop/src/renderer/styles.css`
- Test: `desktop/tests/App.test.tsx`

- [ ] Write renderer tests proving backend-unavailable errors are visible and creating a job updates the dashboard.
- [ ] Run `npm test --workspace desktop` and verify failure.
- [ ] Implement Electron supervision, preload bridge, API client, and React app shell.
- [ ] Run renderer tests and verify they pass.
- [ ] Commit with `feat: add Electron jobs dashboard`.

### Task 5: Verification and Developer Workflow

**Files:**
- Create: `README.md`
- Create: `scripts/dev.ps1`
- Create: `scripts/test.ps1`

- [ ] Document prerequisites, local development, data layout, and Phase 1 scope.
- [ ] Add PowerShell scripts that create the backend virtual environment, install dependencies, and run backend plus desktop development processes.
- [ ] Run `powershell -ExecutionPolicy Bypass -File scripts/test.ps1`.
- [ ] Run `npm run build --workspace desktop`.
- [ ] Start the backend and verify `/api/health`, job creation, job listing, and checkpoint step declarations.
- [ ] Commit with `chore: add development and verification workflow`.

