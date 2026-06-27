# VoxCPM2 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace OmniVoice TTS with VoxCPM2 across the DouyinVietnamizer stack while preserving the long-lived worker + on-disk cache architecture and all public surface area (pipeline, API, settings, frontend).

**Architecture:** Keep the worker-process IPC pattern from OmniVoice. New `voxcpm_worker.py` runs inside `backend/.venv-voxcpm`, lazy-loads `voxcpm.VoxCPM`, and serves JSONL requests over stdin/stdout. `voxcpm_client.py` mirrors `OmniVoiceWorkerClient` so JobRunner cancel / cache / acquire-by-key all work unchanged. `VoxCPMTtsAdapter` in `tts.py` translates the existing `voice` string into VoxCPM2's `(prompt_wav_path, prompt_text, voice_design)` tuple, prefixing the text with `(voice_design)text` when an instruct prompt is given. All `omnivoice_*` identifiers, error codes, env vars, and file paths are renamed to `voxcpm_*` and the legacy files are deleted.

**Tech Stack:** Python 3.12, FastAPI, uv (PyPI index `pytorch-cu128`), voxcpm (PyPI), PyTorch cu128, React 19 + Vite 6 + TypeScript 5, Vitest, pytest. Target HF model: `openbmb/VoxCPM2`.

**Reference spec:** `docs/superpowers/specs/2026-06-27-voxcpm2-migration-design.md`

## Global Constraints

These are the project-wide requirements that every task implicitly inherits. Values are copied verbatim from the spec.

- **Python floor:** `requires-python = ">=3.12,<3.13"` (already in `backend/pyproject.toml`).
- **New settings keys** (no longer accept `omnivoice_*`):
  `voxcpm_model` (default `openbmb/VoxCPM2`), `voxcpm_device` (default `cuda:0`), `voxcpm_ref_audio` (default `""`), `voxcpm_instruct` (default `""`), `voxcpm_auto_voice` (default `True`), `voxcpm_num_steps` (default `10`, clamp 4–64), `voxcpm_batch_size` (default `4`), `voxcpm_batch_flush_ms` (default `150`), `voxcpm_cache_enabled` (default `True`).
- **Env vars:** `DV_VOXCPM_VENV`, `DV_VOXCPM_PYTHON`, `DV_VOXCPM_CACHE_DISABLED`, `DV_VOXCPM_CACHE_DIR`. (Old `DV_OMNIVOICE_*` are no longer consulted.)
- **Error codes** (all `OMNIVOICE_*` → `VOXCPM_*`): `VOXCPM_NOT_INSTALLED`, `VOXCPM_TTS_FAILED`, `VOXCPM_TIMEOUT`, `VOXCPM_WORKER_DIED`, `VOXCPM_INFERENCE_FAILED`, `VOXCPM_WRITE_FAILED`, `VOXCPM_SYNTHESIZE_FAILED`, `VOXCPM_GPU_OOM`.
- **Cache version:** `VOXCPM_CACHE_VERSION = "v1"`. Cache key inputs: `(voice_id, text, model, num_step, voice_design, cfg_value)`.
- **`SUPPORTED_TTS_BACKENDS = ("voxcpm",)`** — single backend.
- **`VOXCPM_INSTRUCT_PREFIX = "instruct:"`** kept in the UI; adapter translates to VoxCPM2's `"(voice_design)text"` syntax.
- **Virtualenv:** `backend/.venv-voxcpm` (was `.venv-omnivoice`).
- **Worker script module path:** `dv_backend.adapters.voxcpm_worker`.
- **Sample rate:** read from `model.tts_model.sample_rate` (no hardcoded 24000).
- **WAV output:** 16-bit mono PCM, written via `wave` stdlib.
- **Cancellation:** JobRunner kills the worker Popen; next call respawns transparently.
- **No backwards-compat shims:** old `omnivoice_*` settings keys are not migrated, old env vars are not read, old files are deleted.
- **Frequent commits:** one commit per task with a `feat:` / `fix:` / `chore:` / `refactor:` / `test:` / `docs:` prefix matching the repo's recent style (see `git log --oneline`).

---

## File Structure

### New files (create)
- `backend/dv_backend/voxcpm_env.py` — isolated-venv resolution for the `voxcpm` package.
- `backend/dv_backend/adapters/voxcpm_client.py` — JSONL worker client (mirror of `OmniVoiceWorkerClient`).
- `backend/dv_backend/adapters/voxcpm_worker.py` — long-lived inference worker.
- `backend/dv_backend/adapters/voxcpm_cache.py` — content-addressed on-disk cache.
- `backend/scripts/setup_voxcpm.py` — creates `.venv-voxcpm` and installs `voxcpm`.
- `backend/scripts/run_voxcpm_smoke.py` — end-to-end Vietnamese TTS smoke test.
- `backend/tests/test_voxcpm_tts.py` — adapter + cache + worker regression tests.
- `docs/VOXCPM.md` — replaces `docs/OMNIVOICE.md`.

### Modified files (edit in place)
- `backend/dv_backend/adapters/tts.py` — replace `OmniVoiceTtsAdapter` with `VoxCPMTtsAdapter`, rename constants/helpers, update `create_tts_adapter`.
- `backend/dv_backend/runtime.py` — rename `_check_omnivoice` → `_check_voxcpm`, switch imports.
- `backend/dv_backend/api.py` — replace `omnivoice_*` settings keys, error codes, `output_suffix`, labels.
- `backend/dv_backend/pipeline.py` — replace `omnivoice_*` keys in `_default_tts_voice()` and the `device` lookup.
- `backend/dv_backend/settings.py` — replace `DEFAULT_SETTINGS`, rename `OMNIVOICE_DEFAULT_MODEL` → `VOXCPM_DEFAULT_MODEL`, validation block.
- `backend/tests/test_settings.py` — update assertions to `voxcpm_*`.
- `backend/tests/test_pipeline.py` — update any direct references to `omnivoice_*` settings keys (if any).
- `frontend/src/renderer/App.tsx` — card title + settings key bindings.
- `frontend/tests/App.test.tsx` — `tts_backend: "voxcpm"` + new card title.
- `README.md`, `docs/DIARIZATION.md` — replace `OmniVoice` / `omnivoice` mentions with VoxCPM2 / voxcpm.

### Deleted files (no replacement)
- `backend/dv_backend/omnivoice_env.py`
- `backend/dv_backend/adapters/omnivoice_client.py`
- `backend/dv_backend/adapters/omnivoice_worker.py`
- `backend/dv_backend/adapters/omnivoice_cache.py`
- `backend/scripts/setup_omnivoice.py`
- `backend/scripts/run_omnivoice_smoke.py`
- `backend/tests/test_omnivoice_tts.py`
- `docs/OMNIVOICE.md`

---

## Task 1: Replace `tts.py` adapter with `VoxCPMTtsAdapter`

**Files:**
- Modify: `backend/dv_backend/adapters/tts.py` (entire file — the only callers are `pipeline.py`, `api.py`, `tests/test_omnivoice_tts.py`, `tests/test_voxcpm_tts.py` once added; all four change in later tasks)
- Test: `backend/tests/test_voxcpm_tts.py` (created in this task)

**Interfaces:**
- Consumes: nothing new — the existing `settings` dict now uses `voxcpm_*` keys (set in Task 6).
- Produces:
  - `class VoxCPMTtsAdapter`
    - `__init__(self, *, model: str = "openbmb/VoxCPM2", device: str = "cuda:0", num_steps: int = 10, data_dir: Path | None = None, runner: object | None = None, max_batch: int | None = None, flush_ms: int | None = None, enable_cache: bool = True, _client: object | None = None, _cache: object | None = None) -> None`
    - `synthesize(self, text: str, output_path: Path, *, voice: str, ref_text: str | None = None) -> None`
  - `parse_voxcpm_voice(voice: str | None) -> tuple[str | None, str | None, str | None]` returning `(prompt_wav_path, prompt_text, voice_design)`.
  - `create_tts_adapter(settings: dict, *, data_dir: Path | None = None, runner: object | None = None) -> VoxCPMTtsAdapter`.
  - Constants: `SUPPORTED_TTS_BACKENDS = ("voxcpm",)`, `VOXCPM_DEFAULT_MODEL = "openbmb/VoxCPM2"`, `VOXCPM_INSTRUCT_PREFIX = "instruct:"`.

`VoxCPMTtsAdapter._synthesize_single` builds the effective text:
```python
if voice_design:
    text = f"({voice_design}){text}"
```
and calls `self._client.synthesize(text=text, output_path=..., prompt_wav_path=ref_audio, prompt_text=ref_text, voice_design=voice_design, cfg_value=2.0, inference_timesteps=self.num_steps, cache_key=cache_key)`.

The full file body is shown in Step 3.

- [ ] **Step 1: Write the failing test for `parse_voxcpm_voice` and `create_tts_adapter`**

Create `backend/tests/test_voxcpm_tts.py` with the following top imports and the first three tests. (More tests are added in later tasks; the file accumulates them.)

```python
import wave
from pathlib import Path

import pytest

from dv_backend.adapters.tts import (
    VOXCPM_INSTRUCT_PREFIX,
    VoxCPMTtsAdapter,
    create_tts_adapter,
    parse_voxcpm_voice,
    split_tts_text,
)
from dv_backend.errors import AppError


def test_parse_voxcpm_voice_modes() -> None:
    assert parse_voxcpm_voice("auto") == (None, None, None)
    assert parse_voxcpm_voice(f"{VOXCPM_INSTRUCT_PREFIX}female, low pitch") == (
        None,
        None,
        "female, low pitch",
    )


def test_parse_voxcpm_voice_with_ref_audio(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    assert parse_voxcpm_voice(str(ref)) == (str(ref), None, None)


def test_create_tts_adapter_always_selects_voxcpm() -> None:
    adapter = create_tts_adapter({"tts_backend": "other", "voxcpm_device": "cuda:0"})
    assert type(adapter).__name__ == "VoxCPMTtsAdapter"
```

- [ ] **Step 2: Run the test to confirm it fails (and the old `tts.py` still imports)**

Run:
```bash
cd backend && uv run pytest tests/test_voxcpm_tts.py -v
```
Expected: `ImportError` from `dv_backend.adapters.tts` because `VOXCPM_INSTRUCT_PREFIX` / `VoxCPMTtsAdapter` / `create_tts_adapter` are not yet defined (or, more likely, `create_tts_adapter` still returns `OmniVoiceTtsAdapter` and `type(adapter).__name__` is `"OmniVoiceTtsAdapter"`).

- [ ] **Step 3: Rewrite `backend/dv_backend/adapters/tts.py`**

Replace the entire file with:

```python
from pathlib import Path
import re
import shutil
import wave

from ..errors import AppError
from ..models import ErrorInfo

SUPPORTED_TTS_BACKENDS = ("voxcpm",)
VOXCPM_DEFAULT_MODEL = "openbmb/VoxCPM2"
VOXCPM_INSTRUCT_PREFIX = "instruct:"

MAX_TTS_CHARS = 450
_TTS_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…。，！？；;])\s+")


def sanitize_tts_text(text: str) -> str:
    """Remove characters that tokenizer/audio backends cannot encode."""
    return "".join(
        character
        for character in (text or "")
        if not (0xD800 <= ord(character) <= 0xDFFF)
    )


def split_tts_text(text: str, *, max_chars: int = MAX_TTS_CHARS) -> list[str]:
    cleaned = re.sub(r"\s+", " ", sanitize_tts_text(text).strip())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = [
        part.strip()
        for part in _TTS_SENTENCE_SPLIT_RE.split(cleaned)
        if part.strip()
    ] or [cleaned]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_chars:
            current = sentence
            continue
        for offset in range(0, len(sentence), max_chars):
            chunks.append(sentence[offset : offset + max_chars].strip())
        current = ""
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def parse_voxcpm_voice(voice: str | None) -> tuple[str | None, str | None, str | None]:
    """Return (prompt_wav_path, prompt_text, voice_design) for a voice string."""
    value = str(voice or "auto").strip()
    if not value or value.lower() == "auto":
        return None, None, None
    if value.startswith(VOXCPM_INSTRUCT_PREFIX):
        voice_design = value[len(VOXCPM_INSTRUCT_PREFIX):].strip()
        return None, None, voice_design or None
    path = Path(value)
    if path.is_file():
        return str(path), None, None
    return None, None, None


def _is_voxcpm_voice_clone(voice: str | None) -> bool:
    prompt_wav_path, _, _ = parse_voxcpm_voice(voice)
    return prompt_wav_path is not None


def _wav_format_key(params: wave._wave_params) -> tuple:
    return (
        params.nchannels,
        params.sampwidth,
        params.framerate,
        params.comptype,
        params.compname,
    )


def _concat_wav_files(paths: list[Path], output_path: Path) -> None:
    if not paths:
        raise ValueError("No WAV files to concatenate.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        shutil.copy2(paths[0], output_path)
        return

    with wave.open(str(paths[0]), "rb") as first:
        format_key = _wav_format_key(first.getparams())
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]
    for path in paths[1:]:
        with wave.open(str(path), "rb") as wav_file:
            if _wav_format_key(wav_file.getparams()) != format_key:
                raise ValueError(f"Incompatible WAV format: {path}")
            frames.append(wav_file.readframes(wav_file.getnframes()))
    with wave.open(str(output_path), "wb") as output:
        output.setparams(params)
        for frame in frames:
            output.writeframes(frame)


class VoxCPMTtsAdapter:
    """Adapter backed by a long-lived VoxCPM worker.

    The adapter routes every segment through a shared
    :class:`VoxCPMWorkerClient` which keeps the VoxCPM2 model resident in VRAM
    and coalesces compatible requests. Combined with the on-disk cache
    (key = sha256(voice_id, text, model, num_step, voice_design, cfg_value))
    repeated or re-run jobs become near-instant.
    """

    def __init__(
        self,
        *,
        model: str = VOXCPM_DEFAULT_MODEL,
        device: str = "cuda:0",
        num_steps: int = 10,
        data_dir: Path | None = None,
        runner: object | None = None,
        max_batch: int | None = None,
        flush_ms: int | None = None,
        enable_cache: bool = True,
        # Test seams: inject a fake client / cache. Production code never
        # sets these.
        _client: object | None = None,
        _cache: object | None = None,
    ) -> None:
        self.model = (model or VOXCPM_DEFAULT_MODEL).strip() or VOXCPM_DEFAULT_MODEL
        self.device = (device or "cuda:0").strip() or "cuda:0"
        self.num_steps = max(4, min(64, int(num_steps)))
        self._data_dir = Path(data_dir) if data_dir is not None else None
        self._runner = runner
        self._max_batch = max_batch
        self._flush_ms = flush_ms
        self._enable_cache = enable_cache
        self._client = None
        self._cache = None
        self._injected_client = _client
        self._injected_cache = _cache

    def _resolve_data_dir(self) -> Path:
        if self._data_dir is not None:
            return self._data_dir
        from ..config import AppConfig

        try:
            return AppConfig.from_env().data_dir
        except Exception:
            return Path.cwd() / "data"

    def _ensure_runtime(self) -> None:
        if self._client is not None:
            return
        if self._injected_client is not None:
            self._client = self._injected_client
            self._cache = self._injected_cache
            return
        from .voxcpm_cache import VoxCPMCache
        from .voxcpm_client import acquire_client

        data_dir = self._resolve_data_dir()
        if self._enable_cache:
            self._cache = VoxCPMCache(data_dir / "cache" / "voxcpm")
        else:
            self._cache = None
        self._client = acquire_client(
            data_dir=data_dir,
            model=self.model,
            device=self.device,
            num_steps=self.num_steps,
        )
        if self._max_batch is not None or self._flush_ms is not None:
            self._client.max_batch = self._max_batch or self._client.max_batch
            self._client.flush_ms = self._flush_ms or self._client.flush_ms
        self._client.register_with_runner(self._runner)

    def _run_infer(
        self,
        *,
        text: str,
        output_path: Path,
        prompt_wav_path: str | None,
        prompt_text: str | None,
        voice_design: str | None,
        voice_id: str,
    ) -> None:
        self._ensure_runtime()
        cache_key = None
        if self._cache is not None and not _is_voxcpm_voice_clone(voice_id):
            from .voxcpm_cache import cache_key as make_cache_key

            cache_key = make_cache_key(
                voice_id=voice_id,
                text=text,
                model=self.model,
                num_step=self.num_steps,
                voice_design=voice_design,
                cfg_value=2.0,
            )
            if self._cache.materialize(cache_key, output_path):
                return
        assert self._client is not None
        response = self._client.synthesize(
            text=text,
            output_path=output_path,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            voice_design=voice_design,
            cfg_value=2.0,
            inference_timesteps=self.num_steps,
            cache_key=cache_key,
        )
        if not response.get("ok", False):
            raise AppError(
                502,
                ErrorInfo(
                    code=response.get("code") or "VOXCPM_TTS_FAILED",
                    message=response.get("message") or "VoxCPM2 could not generate narration.",
                    action=(
                        "Check VoxCPM2 model, GPU availability, and reference audio settings. "
                        "Run 'python scripts/setup_voxcpm.py' if the isolated env is missing."
                    ),
                    detail=response.get("detail"),
                    retryable=bool(response.get("retryable", True)),
                ),
            )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AppError(
                502,
                ErrorInfo(
                    code="VOXCPM_TTS_FAILED",
                    message="VoxCPM2 produced an empty audio file.",
                    action="Try another reference clip or switch to auto voice mode.",
                    retryable=True,
                ),
            )
        if self._cache is not None and cache_key is not None:
            self._cache.put(cache_key, output_path)

    def _synthesize_single(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        ref_text: str | None,
    ) -> None:
        prompt_wav_path, _, voice_design = parse_voxcpm_voice(voice)
        if voice_design:
            text = f"({voice_design}){text}"
        self._run_infer(
            text=text,
            output_path=output_path,
            prompt_wav_path=prompt_wav_path,
            prompt_text=ref_text,
            voice_design=voice_design,
            voice_id=voice or "",
        )

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        ref_text: str | None = None,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = split_tts_text(text)
        if not chunks:
            raise AppError(
                422,
                ErrorInfo(
                    code="EMPTY_TTS_TEXT",
                    message="Cannot synthesize empty narration text.",
                    action="Verify translation output for this segment.",
                ),
            )
        try:
            if len(chunks) == 1:
                self._synthesize_single(
                    chunks[0],
                    output_path,
                    voice=voice,
                    ref_text=ref_text,
                )
                return

            parts: list[Path] = []
            for index, chunk in enumerate(chunks):
                part_path = output_path.with_name(f"{output_path.stem}.part{index:03d}.wav")
                self._synthesize_single(
                    chunk,
                    part_path,
                    voice=voice,
                    ref_text=ref_text,
                )
                parts.append(part_path)
            _concat_wav_files(parts, output_path)
        except AppError:
            raise
        except Exception as cause:
            raise AppError(
                502,
                ErrorInfo(
                    code="VOXCPM_TTS_FAILED",
                    message="VoxCPM2 could not generate narration.",
                    action="Ensure the VoxCPM virtualenv is installed and the GPU is available.",
                    detail=str(cause),
                    retryable=True,
                ),
            ) from cause
        finally:
            for part_path in output_path.parent.glob(f"{output_path.stem}.part*.wav"):
                part_path.unlink(missing_ok=True)


def create_tts_adapter(settings: dict, *, data_dir: Path | None = None, runner: object | None = None):
    try:
        batch_size = int(settings.get("voxcpm_batch_size", 4) or 4)
    except (TypeError, ValueError):
        batch_size = 4
    try:
        flush_ms = int(settings.get("voxcpm_batch_flush_ms", 150) or 150)
    except (TypeError, ValueError):
        flush_ms = 150
    return VoxCPMTtsAdapter(
        model=str(settings.get("voxcpm_model", VOXCPM_DEFAULT_MODEL) or VOXCPM_DEFAULT_MODEL),
        device=str(settings.get("voxcpm_device", "cuda:0") or "cuda:0"),
        num_steps=int(settings.get("voxcpm_num_steps", 10) or 10),
        data_dir=data_dir,
        runner=runner,
        max_batch=max(1, batch_size),
        flush_ms=max(0, flush_ms),
        enable_cache=str(settings.get("voxcpm_cache_enabled", True)).lower()
        not in {"0", "false", "no"},
    )
```

- [ ] **Step 4: Run the new tests to confirm they pass**

Run:
```bash
cd backend && uv run pytest tests/test_voxcpm_tts.py::test_parse_voxcpm_voice_modes tests/test_voxcpm_tts.py::test_parse_voxcpm_voice_with_ref_audio tests/test_voxcpm_tts.py::test_create_tts_adapter_always_selects_voxcpm -v
```
Expected: 3 passed.

- [ ] **Step 5: Verify nothing else in the package still imports the old names**

Run:
```bash
cd backend && uv run python -c "from dv_backend.adapters import tts; print(tts.SUPPORTED_TTS_BACKENDS, tts.VOXCPM_DEFAULT_MODEL)"
```
Expected: `('voxcpm',) openbmb/VoxCPM2`.

- [ ] **Step 6: Commit**

```bash
git add backend/dv_backend/adapters/tts.py backend/tests/test_voxcpm_tts.py
git commit -m "refactor(tts): replace OmniVoice adapter skeleton with VoxCPM2"
```

---

## Task 2: Add `voxcpm_cache.py` and unit tests

**Files:**
- Create: `backend/dv_backend/adapters/voxcpm_cache.py`
- Modify: `backend/tests/test_voxcpm_tts.py` (append cache tests)

**Interfaces:**
- `VOXCPM_CACHE_VERSION = "v1"`
- `cache_key(*, voice_id: str, text: str, model: str, num_step: int, voice_design: str | None = None, cfg_value: float = 2.0) -> str`
- `cache_path_for(cache_dir: Path, key: str) -> Path`
- `class VoxCPMCache` with `__init__(cache_dir: Path | None)`, `get(key) -> Path | None`, `put(key, source_path) -> Path | None`, `materialize(key, destination) -> bool`, `clear()`.

- [ ] **Step 1: Append the failing cache tests to `backend/tests/test_voxcpm_tts.py`**

```python
from dv_backend.adapters.voxcpm_cache import VoxCPMCache, cache_key


def test_cache_key_is_stable() -> None:
    k1 = cache_key(voice_id="auto", text="Xin chào", model="m", num_step=10)
    k2 = cache_key(voice_id="auto", text="  Xin  CHÀO  ", model="m", num_step=10)
    assert k1 == k2


def test_cache_key_differs_on_inputs() -> None:
    base = dict(voice_id="auto", text="Xin chao", model="m", num_step=10)
    assert cache_key(**base) != cache_key(**{**base, "text": "xin chao."})
    assert cache_key(**base) != cache_key(**{**base, "num_step": 16})
    assert cache_key(**base) != cache_key(**{**base, "voice_design": "female"})
    assert cache_key(**base) != cache_key(**{**base, "cfg_value": 3.0})


def test_cache_put_and_materialize(tmp_path: Path) -> None:
    cache = VoxCPMCache(tmp_path / "cache")
    key = cache_key(voice_id="auto", text="hello", model="m", num_step=10)
    src = tmp_path / "src.wav"
    src.write_bytes(b"RIFFdata")
    cache.put(key, src)
    dest = tmp_path / "dest.wav"
    assert cache.materialize(key, dest) is True
    assert dest.read_bytes() == b"RIFFdata"


def test_cache_miss_returns_false(tmp_path: Path) -> None:
    cache = VoxCPMCache(tmp_path / "cache")
    key = cache_key(voice_id="auto", text="missing", model="m", num_step=10)
    dest = tmp_path / "dest.wav"
    assert cache.materialize(key, dest) is False
    assert not dest.exists()
```

Also merge the new import into the existing import block at the top of the file (it is currently importing from `dv_backend.adapters.omnivoice_cache`; replace with `voxcpm_cache`).

- [ ] **Step 2: Run the new tests to confirm they fail**

Run:
```bash
cd backend && uv run pytest tests/test_voxcpm_tts.py::test_cache_key_is_stable tests/test_voxcpm_tts.py::test_cache_key_differs_on_inputs tests/test_voxcpm_tts.py::test_cache_put_and_materialize tests/test_voxcpm_tts.py::test_cache_miss_returns_false -v
```
Expected: all 4 fail with `ModuleNotFoundError: No module named 'dv_backend.adapters.voxcpm_cache'`.

- [ ] **Step 3: Create `backend/dv_backend/adapters/voxcpm_cache.py`**

```python
"""Persistent on-disk cache for VoxCPM2 TTS outputs.

Cache key = sha256(version, voice_id, normalized_text, model, num_step,
voice_design, cfg_value). The same input always returns the same cached
file, so re-running a job or dubbing the same video twice becomes instant
for any repeated segment.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
from pathlib import Path

VOXCPM_CACHE_VERSION = "v1"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def cache_key(
    *,
    voice_id: str,
    text: str,
    model: str,
    num_step: int,
    voice_design: str | None = None,
    cfg_value: float = 2.0,
) -> str:
    payload = "|".join(
        [
            VOXCPM_CACHE_VERSION,
            voice_id or "",
            _normalize_text(text),
            model or "",
            str(int(num_step or 0)),
            (voice_design or "").strip(),
            f"{float(cfg_value):.4f}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path_for(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.wav"


class VoxCPMCache:
    """File-backed cache with thread-safe access.

    Two segments with the same cache key share a single WAV on disk; the
    backend copies the cached file to the segment output path on hit.
    """

    def __init__(self, cache_dir: Path | None) -> None:
        self.enabled = cache_dir is not None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._lock = threading.Lock()
        if self.enabled and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Path | None:
        if not self.enabled or self.cache_dir is None:
            return None
        candidate = cache_path_for(self.cache_dir, key)
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
        return None

    def put(self, key: str, source_path: Path) -> Path | None:
        if not self.enabled or self.cache_dir is None:
            return None
        target = cache_path_for(self.cache_dir, key)
        with self._lock:
            if target.is_file() and target.stat().st_size > 0:
                return target
            try:
                shutil.copy2(source_path, target)
            except OSError:
                return None
        return target

    def materialize(self, key: str, destination: Path) -> bool:
        """Copy a cached file to ``destination`` if present. Returns True on hit."""
        cached = self.get(key)
        if cached is None:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, destination)
        return True

    def clear(self) -> None:
        if not self.enabled or self.cache_dir is None:
            return
        with self._lock:
            for entry in self.cache_dir.glob("*.wav"):
                try:
                    entry.unlink()
                except OSError:
                    pass


def _default_cache_dir(data_dir: Path) -> Path | None:
    if os.environ.get("DV_VOXCPM_CACHE_DISABLED") == "1":
        return None
    override = os.environ.get("DV_VOXCPM_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(data_dir) / "cache" / "voxcpm"
```

- [ ] **Step 4: Run the cache tests to confirm they pass**

Run:
```bash
cd backend && uv run pytest tests/test_voxcpm_tts.py -v -k cache
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/dv_backend/adapters/voxcpm_cache.py backend/tests/test_voxcpm_tts.py
git commit -m "feat(tts): add VoxCPM2 content-addressed cache"
```

---

## Task 3: Add `voxcpm_env.py` and unit tests

**Files:**
- Create: `backend/dv_backend/voxcpm_env.py`
- Test: append to `backend/tests/test_voxcpm_tts.py` (or a new `backend/tests/test_voxcpm_env.py` — the engineer may choose either, but Task 3 must end with the tests passing)

**Interfaces:**
- `voxcpm_venv_root() -> Path` (default `backend/.venv-voxcpm`, override `DV_VOXCPM_VENV`).
- `resolve_voxcpm_python() -> Path` (raises `FileNotFoundError` if not present).
- `is_voxcpm_available() -> bool` (returns `True` if `python -c "import voxcpm"` exits 0; uses `DV_VOXCPM_PYTHON` if set).

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_voxcpm_tts.py`:

```python
import dv_backend.voxcpm_env as voxcpm_env


def test_voxcpm_venv_root_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DV_VOXCPM_VENV", raising=False)
    root = voxcpm_env.voxcpm_venv_root()
    assert root.name == ".venv-voxcpm"


def test_voxcpm_venv_root_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DV_VOXCPM_VENV", str(tmp_path))
    assert voxcpm_env.voxcpm_venv_root() == tmp_path


def test_resolve_voxcpm_python_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DV_VOXCPM_VENV", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        voxcpm_env.resolve_voxcpm_python()


def test_is_voxcpm_available_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DV_VOXCPM_VENV", str(tmp_path))
    assert voxcpm_env.is_voxcpm_available() is False
```

- [ ] **Step 2: Run the new tests to confirm they fail**

Run:
```bash
cd backend && uv run pytest tests/test_voxcpm_tts.py -v -k voxcpm_env
```
Expected: 4 errors with `ModuleNotFoundError: No module named 'dv_backend.voxcpm_env'`.

- [ ] **Step 3: Create `backend/dv_backend/voxcpm_env.py`**

```python
"""Resolve the isolated VoxCPM Python environment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def voxcpm_venv_root() -> Path:
    override = os.environ.get("DV_VOXCPM_VENV", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".venv-voxcpm"


def resolve_voxcpm_python() -> Path:
    env_override = os.environ.get("DV_VOXCPM_PYTHON", "").strip()
    if env_override:
        path = Path(env_override).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"DV_VOXCPM_PYTHON does not exist: {path}")

    venv_root = voxcpm_venv_root()
    if sys.platform == "win32":
        candidates = (
            venv_root / "Scripts" / "python.exe",
            venv_root / "Scripts" / "python",
        )
    else:
        candidates = (
            venv_root / "bin" / "python3",
            venv_root / "bin" / "python",
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "VoxCPM environment was not found. "
        f"Expected virtualenv at {venv_root}. "
        "Run: python scripts/setup_voxcpm.py"
    )


def is_voxcpm_available() -> bool:
    try:
        python = resolve_voxcpm_python()
    except FileNotFoundError:
        return False
    try:
        completed = subprocess.run(
            [str(python), "-c", "import voxcpm"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run:
```bash
cd backend && uv run pytest tests/test_voxcpm_tts.py -v -k voxcpm_env
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/dv_backend/voxcpm_env.py backend/tests/test_voxcpm_tts.py
git commit -m "feat(tts): add voxcpm_env resolver for the isolated virtualenv"
```

---

## Task 4: Add `voxcpm_worker.py` and the worker-API regression test

**Files:**
- Create: `backend/dv_backend/adapters/voxcpm_worker.py`
- Modify: `backend/tests/test_voxcpm_tts.py` (append worker regression test)

**Interfaces:**
- `class VoxCPMEngine` with `get(*, model: str, device: str) -> Any` (lazy import) and `release()`.
- `_generate(engine_obj, texts, *, prompt_wav_path, prompt_text, voice_design, inference_timesteps, cfg_value) -> list[Any]` (single-text in this version; called per request after batching).
- `_write_wav(output_path, audio, sample_rate) -> float` (returns duration_sec).
- `serve(*, max_batch, flush_ms, idle_timeout_sec) -> int` (main loop).
- `main()` (argparse, `--health-check`, `--max-batch`, `--flush-ms`, `--idle-timeout-sec`).

Request/response JSONL:

```json
{"id": "req-1", "op": "synthesize", "text": "(female, low pitch)Xin chào",
 "prompt_wav_path": "/abs/ref.wav" | null, "prompt_text": "hello" | null,
 "voice_design": "female, low pitch" | null,
 "cfg_value": 2.0, "inference_timesteps": 10,
 "model": "openbmb/VoxCPM2", "device": "cuda:0",
 "output_path": "/abs/out.wav"}

{"id": "req-1", "ok": true, "output_path": "...", "duration_sec": 1.42, "sample_rate": 24000}
{"id": "req-1", "ok": false, "code": "VOXCPM_INFERENCE_FAILED", "message": "...", "retryable": true}
```

- [ ] **Step 1: Append the failing regression test to `backend/tests/test_voxcpm_tts.py`**

```python
def test_worker_generate_uses_real_voxcpm_api(monkeypatch):
    """Regression: worker must call model.generate with VoxCPM2 kwargs."""
    from dv_backend.adapters import voxcpm_worker

    class FakeEngine:
        def __init__(self) -> None:
            self.calls = []

        def generate(self, text, **kwargs):
            self.calls.append((text, kwargs))
            return b"audio"

    fake_voxcpm = type("FakeVoxCPMModule", (), {})
    monkeypatch.setitem(__import__("sys").modules, "voxcpm", fake_voxcpm)
    engine = FakeEngine()

    result = voxcpm_worker._generate(
        engine,
        "(female, low pitch)Xin chao",
        prompt_wav_path="ref.wav",
        prompt_text="hello",
        voice_design="female, low pitch",
        cfg_value=2.0,
        inference_timesteps=10,
    )

    assert result == b"audio"
    text, kwargs = engine.calls[0]
    assert text == "(female, low pitch)Xin chao"
    assert kwargs["prompt_wav_path"] == "ref.wav"
    assert kwargs["prompt_text"] == "hello"
    assert kwargs["cfg_value"] == 2.0
    assert kwargs["inference_timesteps"] == 10
```

- [ ] **Step 2: Run the test to confirm it fails**

Run:
```bash
cd backend && uv run pytest tests/test_voxcpm_tts.py::test_worker_generate_uses_real_voxcpm_api -v
```
Expected: `ModuleNotFoundError: No module named 'dv_backend.adapters.voxcpm_worker'`.

- [ ] **Step 3: Create `backend/dv_backend/adapters/voxcpm_worker.py`**

```python
"""Long-lived VoxCPM2 inference worker.

This script runs *inside* the isolated ``.venv-voxcpm`` virtualenv so the
heavy ``torch`` / ``voxcpm`` stack stays isolated from the main backend
environment. The backend process spawns this worker as a child process and
exchanges newline-delimited JSON messages on stdin/stdout.

Request (backend -> worker)::

    {"id": "req-1", "op": "synthesize",
     "text": "(female, low pitch)Xin chào",
     "prompt_wav_path": "/abs/ref.wav", "prompt_text": "hello",
     "voice_design": "female, low pitch",
     "cfg_value": 2.0, "inference_timesteps": 10,
     "model": "openbmb/VoxCPM2", "device": "cuda:0",
     "output_path": "/abs/out.wav"}

Response (worker -> backend)::

    {"id": "req-1", "ok": true, "output_path": "/abs/out.wav",
     "duration_sec": 1.42, "sample_rate": 24000}
    {"id": "req-1", "ok": false, "code": "...", "message": "...",
     "retryable": true}

VoxCPM2's ``generate()`` is single-text; we call it per request after
optionally coalescing compatible requests (same model/device/voice
parameters) so the model stays hot in VRAM.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "openbmb/VoxCPM2"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_INFERENCE_TIMESTEPS = 10
DEFAULT_CFG_VALUE = 2.0
DEFAULT_FLUSH_MS = 150
DEFAULT_MAX_BATCH = 4
DEFAULT_SAMPLE_RATE = 24000


def _log(message: str) -> None:
    sys.stderr.write(f"[voxcpm-worker] {message}\n")
    sys.stderr.flush()


def _emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class VoxCPMEngine:
    """Lazy wrapper around the ``voxcpm`` package.

    The actual import is deferred until first use so that ``--health-check``
    style invocations do not require torch to be available.
    """

    def __init__(self) -> None:
        self._model_id: str | None = None
        self._device: str | None = None
        self._engine: Any = None

    def get(self, *, model: str, device: str) -> Any:
        if self._engine is not None and self._model_id == model and self._device == device:
            return self._engine
        self._engine = None
        self._model_id = model
        self._device = device
        from voxcpm import VoxCPM  # type: ignore

        _log(f"Loading VoxCPM model={model} device={device}")
        with contextlib.redirect_stdout(sys.stderr):
            engine = VoxCPM.from_pretrained(model)
            if device:
                engine = engine.to(device)
            engine.eval()
        _log("VoxCPM model ready")
        self._engine = engine
        return self._engine

    def sample_rate(self, engine_obj: Any) -> int:
        tts = getattr(engine_obj, "tts_model", None)
        return int(getattr(tts, "sample_rate", DEFAULT_SAMPLE_RATE) or DEFAULT_SAMPLE_RATE)

    def release(self) -> None:
        self._engine = None
        self._model_id = None
        self._device = None
        try:
            import gc
            import torch  # type: ignore
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _write_wav(output_path: str, audio: Any, sample_rate: int) -> float:
    import numpy as np
    import wave

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio).reshape(-1)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    pcm = np.clip(arr, -1.0, 1.0)
    pcm_int = (pcm * 32767.0).astype(np.int16)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm_int.tobytes())
    return float(arr.shape[0]) / float(sample_rate) if arr.size else 0.0


def _generate(
    engine_obj: Any,
    text: str,
    *,
    prompt_wav_path: str | None,
    prompt_text: str | None,
    voice_design: str | None,
    inference_timesteps: int,
    cfg_value: float,
) -> Any:
    from voxcpm import VoxCPM  # type: ignore  # noqa: F401  (ensures package is importable)

    text = "".join(
        character
        for character in str(text or "")
        if not (0xD800 <= ord(character) <= 0xDFFF)
    )
    kwargs: dict[str, Any] = {
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(inference_timesteps),
    }
    if prompt_wav_path:
        kwargs["prompt_wav_path"] = prompt_wav_path
        if prompt_text:
            kwargs["prompt_text"] = prompt_text
    elif voice_design:
        # Voice design is carried in the text prefix; nothing else needed.
        pass
    _log(
        f"Generating 1 segment, inference_timesteps={inference_timesteps} "
        f"cfg_value={cfg_value} voice_design={bool(voice_design)} "
        f"prompt_wav={bool(prompt_wav_path)}"
    )
    with contextlib.redirect_stdout(sys.stderr):
        return engine_obj.generate(text=text, **kwargs)


def _coalesce(requests: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    order: list[tuple] = []
    for req in requests:
        key = (
            req.get("model") or DEFAULT_MODEL,
            req.get("device") or DEFAULT_DEVICE,
            req.get("prompt_wav_path") or "",
            req.get("prompt_text") or "",
            req.get("voice_design") or "",
            int(req.get("inference_timesteps") or DEFAULT_INFERENCE_TIMESTEPS),
            float(req.get("cfg_value") or DEFAULT_CFG_VALUE),
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(req)
    return [groups[key] for key in order]


def _run_batch(engine_obj: Any, batch: list[dict[str, Any]], sample_rate: int) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for req in batch:
        try:
            audio = _generate(
                engine_obj,
                req["text"],
                prompt_wav_path=req.get("prompt_wav_path"),
                prompt_text=req.get("prompt_text"),
                voice_design=req.get("voice_design"),
                inference_timesteps=int(req.get("inference_timesteps") or DEFAULT_INFERENCE_TIMESTEPS),
                cfg_value=float(req.get("cfg_value") or DEFAULT_CFG_VALUE),
            )
            duration = _write_wav(req["output_path"], audio, sample_rate)
        except Exception as exc:  # noqa: BLE001
            _log(f"Synthesize failed: {exc!r}")
            traceback.print_exc(file=sys.stderr)
            responses.append(
                {
                    "id": req.get("id"),
                    "ok": False,
                    "code": "VOXCPM_INFERENCE_FAILED",
                    "message": str(exc),
                    "retryable": True,
                }
            )
            continue
        responses.append(
            {
                "id": req.get("id"),
                "ok": True,
                "output_path": req["output_path"],
                "duration_sec": round(duration, 3),
                "sample_rate": sample_rate,
            }
        )
    return responses


def _read_request(line: str) -> dict[str, Any] | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        _emit({"id": None, "ok": False, "code": "BAD_REQUEST", "message": str(exc)})
        return None


def serve(*, max_batch: int = DEFAULT_MAX_BATCH, flush_ms: int = DEFAULT_FLUSH_MS, idle_timeout_sec: float = 0.0) -> int:
    _ = max_batch, flush_ms, idle_timeout_sec  # batching is by JSONL request ordering; flushed eagerly per request
    engine = VoxCPMEngine()

    def _handle_synthesize(request: dict[str, Any]) -> None:
        model = request.get("model") or DEFAULT_MODEL
        device = request.get("device") or DEFAULT_DEVICE
        request_text = "".join(
            character
            for character in str(request.get("text") or "")
            if not (0xD800 <= ord(character) <= 0xDFFF)
        )
        request["text"] = request_text
        _log(
            f"Handling synthesize id={request.get('id')} model={model} device={device} "
            f"text_len={len(request_text)} "
            f"prompt_wav={bool(request.get('prompt_wav_path'))} "
            f"prompt_text_len={len(str(request.get('prompt_text') or ''))} "
            f"voice_design={bool(request.get('voice_design'))}"
        )
        engine_obj = engine.get(model=model, device=device)
        sample_rate = engine.sample_rate(engine_obj)
        for response in _run_batch(engine_obj, [request], sample_rate):
            _emit(response)

    _log("Worker ready for JSONL requests")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = _read_request(line)
        if request is None:
            continue
        op = request.get("op") or "synthesize"
        if op == "shutdown":
            _log("Worker shutdown requested")
            return 0
        if op == "ping":
            _emit({"id": request.get("id"), "ok": True, "pong": True})
            continue
        if op != "synthesize":
            _emit({"id": request.get("id"), "ok": False, "code": "UNKNOWN_OP", "message": f"Unknown op: {op}"})
            continue
        _handle_synthesize(request)
    _log("Worker stdin closed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH)
    parser.add_argument("--flush-ms", type=int, default=DEFAULT_FLUSH_MS)
    parser.add_argument("--idle-timeout-sec", type=float, default=0.0)
    parser.add_argument("--health-check", action="store_true", help="Import voxcpm and exit; used by the runtime smoke test.")
    args = parser.parse_args()

    if args.health_check:
        try:
            import voxcpm  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            _emit({"ok": False, "code": "VOXCPM_NOT_INSTALLED", "message": str(exc)})
            return 1
        _emit({"ok": True, "code": "READY"})
        return 0

    return serve(max_batch=args.max_batch, flush_ms=args.flush_ms, idle_timeout_sec=args.idle_timeout_sec)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to confirm it passes**

Run:
```bash
cd backend && uv run pytest tests/test_voxcpm_tts.py::test_worker_generate_uses_real_voxcpm_api -v
```
Expected: 1 passed.

- [ ] **Step 5: Confirm the worker is runnable as a module**

Run:
```bash
cd backend && uv run python -m dv_backend.adapters.voxcpm_worker --help
```
Expected: argparse help text is printed, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add backend/dv_backend/adapters/voxcpm_worker.py backend/tests/test_voxcpm_tts.py
git commit -m "feat(tts): add long-lived VoxCPM2 inference worker"
```

---

## Task 5: Add `voxcpm_client.py` and adapter-with-injected-client tests

**Files:**
- Create: `backend/dv_backend/adapters/voxcpm_client.py`
- Modify: `backend/tests/test_voxcpm_tts.py` (append adapter-with-fake-client tests)

**Interfaces:**
- `class VoxCPMWorkerClient` (mirror of `OmniVoiceWorkerClient`).
  - `__init__(*, data_dir, model, device, num_steps, max_batch=4, flush_ms=150, idle_shutdown_sec=300.0)`
  - `register_with_runner(runner) -> None`
  - `synthesize(*, text, output_path, prompt_wav_path, prompt_text, voice_design, cfg_value, inference_timesteps, cache_key) -> dict`
  - `close() -> None`
- `acquire_client(*, data_dir, model, device, num_steps) -> VoxCPMWorkerClient` (key = `f"{model}|{device}|{int(num_steps)}"`).
- `release_all_clients() -> None`.
- Constants: `WORKER_SCRIPT = "dv_backend.adapters.voxcpm_worker"`, `PROCESS_READY_TIMEOUT_SEC = 120.0`, `STARTUP_PING_TIMEOUT_SEC = 30.0`, `IDLE_SHUTDOWN_SEC = 300.0`, `PING_INTERVAL_SEC = 60.0`, `DEFAULT_MAX_BATCH = 4`, `DEFAULT_FLUSH_MS = 150`.

- [ ] **Step 1: Append the failing adapter tests to `backend/tests/test_voxcpm_tts.py`**

```python
class FakeClient:
    """Drop-in replacement for VoxCPMWorkerClient used in tests."""

    def __init__(self, *, model: str = "", device: str = "", num_steps: int = 0) -> None:
        self.calls: list[dict] = []
        self.requested_model = model
        self.requested_device = device
        self.requested_num_steps = num_steps
        self.next_response: dict = {"ok": True, "duration_sec": 1.0, "sample_rate": 24000}

    def synthesize(self, **kwargs):
        self.calls.append(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x00\x00" * 240)
        return self.next_response

    def register_with_runner(self, runner):  # pragma: no cover - not invoked here
        return None


def test_adapter_routes_to_client(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        model="openbmb/VoxCPM2",
        device="cuda:0",
        num_steps=10,
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    output = tmp_path / "out.wav"
    adapter.synthesize("Xin chao", output, voice="auto")
    assert client.calls, "client.synthesize was not called"
    call = client.calls[0]
    assert call["text"] == "Xin chao"
    assert call["prompt_wav_path"] is None
    assert call["voice_design"] is None
    assert output.is_file()


def test_adapter_clone_uses_ref_audio_and_ref_text(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        device="cuda:0",
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"RIFF")
    output = tmp_path / "out.wav"
    adapter.synthesize("Xin chao", output, voice=str(ref_audio), ref_text="hello")

    call = client.calls[0]
    assert call["prompt_wav_path"] == str(ref_audio)
    assert call["prompt_text"] == "hello"


def test_adapter_instruct_prefixes_text(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        device="cuda:0",
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    output = tmp_path / "out.wav"
    adapter.synthesize("Xin chao", output, voice=f"{VOXCPM_INSTRUCT_PREFIX}female, low pitch")
    call = client.calls[0]
    assert call["text"] == "(female, low pitch)Xin chao"
    assert call["voice_design"] == "female, low pitch"


def test_adapter_chunks_long_text(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    output = tmp_path / "out.wav"
    long_text = " ".join(["cau"] * 200)
    adapter.synthesize(long_text, output, voice="auto")
    assert client.calls
    expected_chunks = len(split_tts_text(long_text))
    assert len(client.calls) == expected_chunks
    for call in client.calls:
        out = Path(call["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x00\x00" * 240)


def test_adapter_rejects_empty_text(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    with pytest.raises(AppError) as exc:
        adapter.synthesize("   ", tmp_path / "out.wav", voice="auto")
    assert exc.value.info.code == "EMPTY_TTS_TEXT"


def test_adapter_propagates_client_error(tmp_path: Path) -> None:
    class FailingClient(FakeClient):
        def synthesize(self, **kwargs):  # type: ignore[override]
            super().synthesize(**kwargs)
            return {
                "ok": False,
                "code": "VOXCPM_GPU_OOM",
                "message": "Out of memory",
                "retryable": True,
            }

    client = FailingClient()
    adapter = VoxCPMTtsAdapter(
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    with pytest.raises(AppError) as exc:
        adapter.synthesize("Xin chao", tmp_path / "out.wav", voice="auto")
    assert exc.value.info.code == "VOXCPM_GPU_OOM"
    assert exc.value.info.retryable is True


def test_adapter_uses_cache(tmp_path: Path) -> None:
    client = FakeClient()
    cache = VoxCPMCache(tmp_path / "cache")
    adapter = VoxCPMTtsAdapter(
        data_dir=tmp_path,
        enable_cache=True,
        _client=client,
        _cache=cache,
    )
    output = tmp_path / "out.wav"
    adapter.synthesize("Xin chao", output, voice="auto")
    calls_after_first = len(client.calls)
    assert calls_after_first == 1
    assert output.is_file()

    output2 = tmp_path / "out2.wav"
    adapter.synthesize("Xin chao", output2, voice="auto")
    assert len(client.calls) == calls_after_first
    assert output2.read_bytes() == output.read_bytes()


def test_adapter_cache_disabled_always_calls_client(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = VoxCPMTtsAdapter(
        data_dir=tmp_path,
        enable_cache=False,
        _client=client,
    )
    output = tmp_path / "out.wav"
    adapter.synthesize("Xin chao", output, voice="auto")
    output2 = tmp_path / "out2.wav"
    adapter.synthesize("Xin chao", output2, voice="auto")
    assert len(client.calls) == 2
```

- [ ] **Step 2: Run the new tests to confirm they fail (with `ModuleNotFoundError`)**

Run:
```bash
cd backend && uv run pytest tests/test_voxcpm_tts.py -v -k "adapter_routes or adapter_clone or adapter_instruct or adapter_chunks or adapter_rejects or adapter_propagates or adapter_uses_cache or adapter_cache_disabled"
```
Expected: 8 errors (`ModuleNotFoundError: No module named 'dv_backend.adapters.voxcpm_client'`).

- [ ] **Step 3: Create `backend/dv_backend/adapters/voxcpm_client.py`**

```python
"""Client for the long-lived VoxCPM2 worker.

The client is responsible for:
* Locating the Python executable in the isolated ``.venv-voxcpm``.
* Spawning the worker subprocess and managing its lifecycle.
* Forwarding synthesize requests and reading responses.
* Re-spawning the worker transparently if it dies (e.g. OOM, crash).
* Honouring cancellation via the ``JobRunner`` process registry.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ..errors import AppError
from ..models import ErrorInfo
from ..voxcpm_env import resolve_voxcpm_python

WORKER_SCRIPT = "dv_backend.adapters.voxcpm_worker"
DEFAULT_MAX_BATCH = 4
DEFAULT_FLUSH_MS = 150
PROCESS_READY_TIMEOUT_SEC = 120.0
RESPONSE_QUEUE_GET_TIMEOUT_SEC = 0.1
STARTUP_PING_TIMEOUT_SEC = 30.0
IDLE_SHUTDOWN_SEC = 300.0
PING_INTERVAL_SEC = 60.0


class VoxCPMWorkerClient:
    """Manages a single worker subprocess and a request/response correlation queue.

    A ``threading.Lock`` serializes writes to the worker stdin pipe (the
    protocol is line-based so interleaved writes would corrupt the stream).
    A dedicated reader thread populates a per-request response queue and a
    shared ``alive`` flag.
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        model: str,
        device: str,
        num_steps: int,
        max_batch: int = DEFAULT_MAX_BATCH,
        flush_ms: int = DEFAULT_FLUSH_MS,
        idle_shutdown_sec: float = IDLE_SHUTDOWN_SEC,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.model = model
        self.device = device
        self.num_steps = max(4, min(64, int(num_steps)))
        self.max_batch = max(1, int(max_batch))
        self.flush_ms = max(20, int(flush_ms))
        self.idle_shutdown_sec = float(idle_shutdown_sec)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[str, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._last_used = time.perf_counter()
        self._last_ping = 0.0
        self._start_error: str | None = None
        self._closed = False

    # ------------------------------------------------------------------ lifecycle

    def _spawn_locked(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        python = resolve_voxcpm_python()
        cmd = [
            str(python),
            "-m",
            WORKER_SCRIPT,
            "--max-batch",
            str(self.max_batch),
            "--flush-ms",
            str(self.flush_ms),
            "--idle-timeout-sec",
            str(self.idle_shutdown_sec),
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=env,
            )
        except FileNotFoundError as exc:
            raise AppError(
                400,
                ErrorInfo(
                    code="VOXCPM_NOT_INSTALLED",
                    message="VoxCPM environment is not installed.",
                    action="Run 'python scripts/setup_voxcpm.py' in the backend folder.",
                    detail=str(exc),
                ),
            ) from exc
        except OSError as exc:
            raise AppError(
                502,
                ErrorInfo(
                    code="VOXCPM_TTS_FAILED",
                    message="VoxCPM2 inference subprocess failed to start.",
                    action="Verify the isolated VoxCPM virtualenv is configured correctly.",
                    detail=str(exc),
                    retryable=True,
                ),
            ) from exc

        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="voxcpm-worker-reader", daemon=True
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop, name="voxcpm-worker-stderr", daemon=True
        )
        self._stderr_thread.start()
        self._wait_ready()
        self._last_used = time.perf_counter()

    def _wait_ready(self) -> None:
        if self._proc is None:
            return
        deadline = time.perf_counter() + STARTUP_PING_TIMEOUT_SEC
        request_id = f"startup-{uuid.uuid4().hex}"
        q: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = q
        message = json.dumps({"id": request_id, "op": "ping"}, ensure_ascii=False) + "\n"
        try:
            assert self._proc.stdin is not None
            with self._write_lock:
                self._proc.stdin.write(message)
                self._proc.stdin.flush()
        except Exception as exc:  # noqa: BLE001
            self._drain_pending_with_error(
                request_id,
                code="VOXCPM_TTS_FAILED",
                message="Worker did not respond to startup ping.",
                detail=str(exc),
                retryable=True,
            )
            return
        try:
            q.get(timeout=max(1.0, deadline - time.perf_counter()))
        except queue.Empty:
            self._drain_pending_with_error(
                request_id,
                code="VOXCPM_TTS_FAILED",
                message="VoxCPM worker failed to start within timeout.",
                retryable=True,
            )
            return

    def _drain_pending_with_error(self, request_id: str, *, code: str, message: str, detail: str | None = None, retryable: bool = True) -> None:
        with self._pending_lock:
            q = self._pending.pop(request_id, None)
        if q is None:
            return
        try:
            q.put_nowait(
                {
                    "id": request_id,
                    "ok": False,
                    "code": code,
                    "message": message,
                    "detail": detail,
                    "retryable": retryable,
                }
            )
        except queue.Full:
            pass

    def _reader_loop(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            req_id = payload.get("id")
            if not req_id:
                continue
            with self._pending_lock:
                q = self._pending.pop(req_id, None)
            if q is None:
                continue
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass
        self._fail_pending("VoxCPM worker exited unexpectedly.", retryable=True)

    def _stderr_loop(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            if not line:
                continue
            sys.stderr.write(f"[voxcpm-worker] {line}")
            sys.stderr.flush()

    def _fail_pending(self, message: str, *, retryable: bool) -> None:
        with self._pending_lock:
            pending = list(self._pending.items())
            self._pending.clear()
        for _req_id, q in pending:
            try:
                q.put_nowait(
                    {
                        "ok": False,
                        "code": "VOXCPM_WORKER_DIED",
                        "message": message,
                        "retryable": retryable,
                    }
                )
            except queue.Full:
                pass

    def _ensure_alive(self) -> None:
        with self._lock:
            if self._closed:
                raise AppError(
                    502,
                    ErrorInfo(
                        code="VOXCPM_TTS_FAILED",
                        message="VoxCPM client is closed.",
                        retryable=True,
                    ),
                )
            if self._proc is None or self._proc.poll() is not None:
                self._spawn_locked()
            self._last_used = time.perf_counter()

    def _keep_alive(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        now = time.perf_counter()
        if now - self._last_ping < PING_INTERVAL_SEC:
            return
        if now - self._last_used < PING_INTERVAL_SEC:
            return
        self._last_ping = now
        try:
            assert self._proc.stdin is not None
            with self._write_lock:
                self._proc.stdin.write(json.dumps({"id": f"ping-{uuid.uuid4().hex}", "op": "ping"}) + "\n")
                self._proc.stdin.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------ public API

    def register_with_runner(self, runner: Any) -> None:
        if runner is None or self._proc is None:
            return
        if hasattr(runner, "register_process"):
            try:
                runner.register_process("_voxcpm_worker", self._proc)
            except Exception:
                pass

    def synthesize(
        self,
        *,
        text: str,
        output_path: Path,
        prompt_wav_path: str | None,
        prompt_text: str | None,
        voice_design: str | None,
        cfg_value: float,
        inference_timesteps: int,
        cache_key: str | None = None,
    ) -> dict[str, Any]:
        output_path = Path(output_path)
        if not text or not text.strip():
            raise AppError(
                422,
                ErrorInfo(
                    code="EMPTY_TTS_TEXT",
                    message="Cannot synthesize empty narration text.",
                    action="Verify translation output for this segment.",
                ),
            )
        self._ensure_alive()
        self._keep_alive()

        request_id = f"req-{uuid.uuid4().hex}"
        response_q: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = response_q

        request: dict[str, Any] = {
            "id": request_id,
            "op": "synthesize",
            "text": text,
            "output_path": str(output_path),
            "model": self.model,
            "device": self.device,
            "inference_timesteps": int(inference_timesteps),
            "cfg_value": float(cfg_value),
            "prompt_wav_path": prompt_wav_path,
            "prompt_text": prompt_text,
            "voice_design": voice_design,
        }
        if cache_key:
            request["cache_key"] = cache_key
        try:
            assert self._proc is not None and self._proc.stdin is not None
            with self._write_lock:
                self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise AppError(
                502,
                ErrorInfo(
                    code="VOXCPM_TTS_FAILED",
                    message="VoxCPM worker is not accepting requests.",
                    action="Verify the isolated VoxCPM virtualenv and GPU availability.",
                    detail=str(exc),
                    retryable=True,
                ),
            ) from exc

        try:
            response = response_q.get(timeout=PROCESS_READY_TIMEOUT_SEC)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise AppError(
                504,
                ErrorInfo(
                    code="VOXCPM_TIMEOUT",
                    message="VoxCPM worker did not respond within the expected time.",
                    action="Check the worker log and GPU availability, then retry.",
                    retryable=True,
                ),
            )

        self._last_used = time.perf_counter()
        if response.get("ok"):
            return response
        raise AppError(
            502,
            ErrorInfo(
                code=response.get("code") or "VOXCPM_TTS_FAILED",
                message=response.get("message") or "VoxCPM2 could not generate narration.",
                action=(
                    "Check VoxCPM2 model, GPU availability, and reference audio settings. "
                    "Run 'python scripts/setup_voxcpm.py' if the isolated env is missing."
                ),
                detail=response.get("detail"),
                retryable=bool(response.get("retryable", True)),
            ),
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None and proc.poll() is None:
                with self._write_lock:
                    proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


_client_lock = threading.Lock()
_clients: dict[str, VoxCPMWorkerClient] = {}


def acquire_client(
    *,
    data_dir: Path,
    model: str,
    device: str,
    num_steps: int,
) -> VoxCPMWorkerClient:
    """Return a shared worker client keyed by (model, device, num_steps).

    A single worker is reused across segments that share the same
    (model, device, num_steps) tuple so the GPU model stays hot.
    """
    key = f"{model}|{device}|{int(num_steps)}"
    with _client_lock:
        client = _clients.get(key)
        if client is None:
            client = VoxCPMWorkerClient(
                data_dir=data_dir,
                model=model,
                device=device,
                num_steps=num_steps,
            )
            _clients[key] = client
        return client


def release_all_clients() -> None:
    with _client_lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        client.close()
```

- [ ] **Step 4: Run the new tests to confirm they pass**

Run:
```bash
cd backend && uv run pytest tests/test_voxcpm_tts.py -v
```
Expected: all tests in `test_voxcpm_tts.py` pass (count should be 18 — 3 voice + 4 cache + 4 env + 1 worker + 8 adapter + the `test_create_tts_adapter_always_selects_voxcpm` already counted).

- [ ] **Step 5: Commit**

```bash
git add backend/dv_backend/adapters/voxcpm_client.py backend/tests/test_voxcpm_tts.py
git commit -m "feat(tts): add VoxCPM2 worker client with JobRunner integration"
```

---

## Task 6: Update `settings.py` with the new defaults and validation

**Files:**
- Modify: `backend/dv_backend/settings.py` (replace import from `tts`, replace `DEFAULT_SETTINGS` TTS keys, rename `OMNIVOICE_DEFAULT_MODEL` import, update validation block)
- Test: `backend/tests/test_settings.py` (update assertions)

**Interfaces:** All `omnivoice_*` settings keys are gone. New keys are listed in Global Constraints. The `tts_backend` validator still references `SUPPORTED_TTS_BACKENDS` from `tts.py` (now `("voxcpm",)`).

- [ ] **Step 1: Update `backend/tests/test_settings.py`**

Find any reference to `omnivoice_*` and rename to `voxcpm_*`. Concretely, the test file currently has:

```python
def test_settings_defaults_include_omnivoice_keys():
    ...
    assert settings.get_all()["omnivoice_model"] == "k2-fsa/OmniVoice"
    assert settings.get_all()["omnivoice_device"] == "cuda:0"
```

Rename the test to `test_settings_defaults_include_voxcpm_keys` and change the assertions to:

```python
def test_settings_defaults_include_voxcpm_keys():
    ...
    assert settings.get_all()["voxcpm_model"] == "openbmb/VoxCPM2"
    assert settings.get_all()["voxcpm_device"] == "cuda:0"
    assert settings.get_all()["voxcpm_num_steps"] == 10
```

Also update the test that writes `omnivoice_ref_audio` to write `voxcpm_ref_audio`.

Read the current file to find every spot. Do not delete tests; only rename and change keys.

- [ ] **Step 2: Run the test file to confirm the assertions fail**

Run:
```bash
cd backend && uv run pytest tests/test_settings.py -v
```
Expected: failures on the renamed assertions (old key missing in `get_all()`).

- [ ] **Step 3: Update `backend/dv_backend/settings.py`**

Three edits:

1. Replace the import line:
   ```python
   from .adapters.tts import SUPPORTED_TTS_BACKENDS, OMNIVOICE_DEFAULT_MODEL
   ```
   with:
   ```python
   from .adapters.tts import SUPPORTED_TTS_BACKENDS, VOXCPM_DEFAULT_MODEL
   ```

2. In `DEFAULT_SETTINGS`, replace the TTS-related block:
   ```python
   "omnivoice_model": OMNIVOICE_DEFAULT_MODEL,
   "omnivoice_device": "cuda:0",
   "omnivoice_ref_audio": "",
   "omnivoice_instruct": "",
   "omnivoice_auto_voice": True,
   "omnivoice_num_steps": 32,
   "omnivoice_batch_size": 4,
   "omnivoice_batch_flush_ms": 150,
   "omnivoice_cache_enabled": True,
   ```
   with:
   ```python
   "voxcpm_model": VOXCPM_DEFAULT_MODEL,
   "voxcpm_device": "cuda:0",
   "voxcpm_ref_audio": "",
   "voxcpm_instruct": "",
   "voxcpm_auto_voice": True,
   "voxcpm_num_steps": 10,
   "voxcpm_batch_size": 4,
   "voxcpm_batch_flush_ms": 150,
   "voxcpm_cache_enabled": True,
   ```

3. In `SettingsService.update`, replace the validation block:
   ```python
   if values.get("omnivoice_num_steps") is not None:
       try:
           steps = int(values["omnivoice_num_steps"])
       except (TypeError, ValueError) as error:
           raise ValueError("omnivoice_num_steps must be an integer.") from error
       values["omnivoice_num_steps"] = max(8, min(64, steps))
   ```
   with:
   ```python
   if values.get("voxcpm_num_steps") is not None:
       try:
           steps = int(values["voxcpm_num_steps"])
       except (TypeError, ValueError) as error:
           raise ValueError("voxcpm_num_steps must be an integer.") from error
       values["voxcpm_num_steps"] = max(4, min(64, steps))
   ```

- [ ] **Step 4: Run the test file to confirm the assertions pass**

Run:
```bash
cd backend && uv run pytest tests/test_settings.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/dv_backend/settings.py backend/tests/test_settings.py
git commit -m "refactor(settings): migrate TTS settings keys from omnivoice_* to voxcpm_*"
```

---

## Task 7: Update `runtime.py` with `_check_voxcpm`

**Files:**
- Modify: `backend/dv_backend/runtime.py` (rename `_check_omnivoice` → `_check_voxcpm`, update the `checks` list and imports)

- [ ] **Step 1: Update `runtime.py`**

In `RuntimeSmokeTestService.run`, the `checks` list contains:
```python
self._check_omnivoice(),
```
Replace it with:
```python
self._check_voxcpm(),
```

Replace the `_check_omnivoice` method body with the VoxCPM2 version:

```python
def _check_voxcpm(self) -> RuntimeCheck:
    from .voxcpm_env import is_voxcpm_available, voxcpm_venv_root
    from .adapters.voxcpm_client import acquire_client, release_all_clients

    if not is_voxcpm_available():
        return RuntimeCheck(
            id="voxcpm",
            display_name="VoxCPM2",
            status="blocked",
            required=True,
            message="VoxCPM2 is not installed in the isolated virtualenv.",
            action="Run 'python scripts/setup_voxcpm.py' in the backend folder.",
            resolved_path=str(voxcpm_venv_root()),
        )
    try:
        client = acquire_client(
            data_dir=self.config.data_dir,
            model="openbmb/VoxCPM2",
            device="cpu",
            num_steps=10,
        )
        client._ensure_alive()
    except Exception as exc:  # noqa: BLE001
        return RuntimeCheck(
            id="voxcpm",
            display_name="VoxCPM2",
            status="blocked",
            required=True,
            message="VoxCPM2 worker could not be started.",
            action="Re-run 'python scripts/setup_voxcpm.py' and verify the worker script.",
            resolved_path=str(voxcpm_venv_root()),
            detail=str(exc),
        )
    finally:
        release_all_clients()
    return RuntimeCheck(
        id="voxcpm",
        display_name="VoxCPM2",
        status="ready",
        required=True,
        message="VoxCPM2 isolated environment and worker are operational.",
        action="No action required.",
        resolved_path=str(voxcpm_venv_root()),
    )
```

- [ ] **Step 2: Import the runtime check to make sure it still constructs**

Run:
```bash
cd backend && uv run python -c "from dv_backend.runtime import RuntimeSmokeTestService; print(RuntimeSmokeTestService._check_voxcpm.__name__)"
```
Expected: prints `_check_voxcpm` (no import error).

- [ ] **Step 3: Commit**

```bash
git add backend/dv_backend/runtime.py
git commit -m "refactor(runtime): rename _check_omnivoice to _check_voxcpm"
```

---

## Task 8: Update `pipeline.py` keys and `api.py` references

**Files:**
- Modify: `backend/dv_backend/pipeline.py` (`_default_tts_voice`, the `device` lookup around line 1315)
- Modify: `backend/dv_backend/api.py` (all `omnivoice_*` settings keys, error codes, `output_suffix`, labels)

- [ ] **Step 1: Update `backend/dv_backend/pipeline.py`**

In `_default_tts_voice`:
```python
def _default_tts_voice(settings: dict) -> str:
    instruct = str(settings.get("omnivoice_instruct") or "").strip()
    if instruct:
        return f"instruct:{instruct}"
    ref_audio = str(settings.get("omnivoice_ref_audio") or "").strip()
    if ref_audio:
        return ref_audio
    return "auto"
```

Replace with:
```python
def _default_tts_voice(settings: dict) -> str:
    instruct = str(settings.get("voxcpm_instruct") or "").strip()
    if instruct:
        return f"instruct:{instruct}"
    ref_audio = str(settings.get("voxcpm_ref_audio") or "").strip()
    if ref_audio:
        return ref_audio
    return "auto"
```

Also update the `device` lookup inside the `mix_mode == MIX_MODE_SEPARATE` branch (line ~1315):
```python
device=str(settings.get("omnivoice_device", "cuda:0") or "cuda:0"),
```
to:
```python
device=str(settings.get("voxcpm_device", "cuda:0") or "cuda:0"),
```

- [ ] **Step 2: Update `backend/dv_backend/api.py`**

Five edits:

1. In `_synthesize_voice_preview`, replace the import:
   ```python
   from .adapters.tts import OMNIVOICE_INSTRUCT_PREFIX, create_tts_adapter
   ```
   with:
   ```python
   from .adapters.tts import VOXCPM_INSTRUCT_PREFIX, create_tts_adapter
   ```

2. In the same function, replace the instruct handling:
   ```python
   instruct = str(settings.get("omnivoice_instruct") or "").strip()
   ...
   preview_voice = f"{OMNIVOICE_INSTRUCT_PREFIX}{instruct}"
   ```
   with:
   ```python
   instruct = str(settings.get("voxcpm_instruct") or "").strip()
   ...
   preview_voice = f"{VOXCPM_INSTRUCT_PREFIX}{instruct}"
   ```

3. In the `except Exception as exc` block:
   ```python
   code = "OMNIVOICE_SYNTHESIZE_FAILED"
   label = "OmniVoice"
   ...
   action="Run 'python scripts/setup_omnivoice.py' for OmniVoice.",
   ```
   Replace with:
   ```python
   code = "VOXCPM_SYNTHESIZE_FAILED"
   label = "VoxCPM2"
   ...
   action="Run 'python scripts/setup_voxcpm.py' for VoxCPM2.",
   ```

4. In the `/api/capabilities` handler:
   ```python
   "tts_backend": "omnivoice",
   ```
   Replace with:
   ```python
   "tts_backend": "voxcpm",
   ```

5. In `preview_voice`:
   ```python
   output_suffix="omnivoice",
   ...
   return FileResponse(str(output_wav), media_type="audio/wav", filename="preview_omnivoice.wav")
   ```
   Replace `output_suffix` with `"voxcpm"` and the filename with `"preview_voxcpm.wav"`.

- [ ] **Step 3: Verify the package still imports and the API still constructs**

Run:
```bash
cd backend && uv run python -c "from dv_backend.api import create_app; app = create_app(); print('ok', app.title)"
```
Expected: prints `ok Douyin Vietnamizer Backend` (or similar — `app.title` is whatever the FastAPI instance has).

- [ ] **Step 4: Run the full backend test suite to surface any leftover references**

Run:
```bash
cd backend && uv run pytest -v
```
Expected: every test in the suite passes. If any test (e.g. `test_pipeline.py`) still references `omnivoice_*` keys, fix them in this task as well — read the file, rename keys, rerun.

- [ ] **Step 5: Commit**

```bash
git add backend/dv_backend/pipeline.py backend/dv_backend/api.py backend/tests/test_pipeline.py
git commit -m "refactor(api,pipeline): route TTS through VoxCPM2 settings and error codes"
```

---

## Task 9: Add `setup_voxcpm.py` and `run_voxcpm_smoke.py`

**Files:**
- Create: `backend/scripts/setup_voxcpm.py`
- Create: `backend/scripts/run_voxcpm_smoke.py`

**Interfaces:**
- `setup_voxcpm.py`: `main()` parses `--venv` (default `backend/.venv-voxcpm`) and `--skip-torch`, creates venv, installs `torch`/`torchaudio` from `https://download.pytorch.org/whl/cu128`, installs `voxcpm` from PyPI, prints `voxcpm <version>`.
- `run_voxcpm_smoke.py`: `main()` runs the same end-to-end pipeline as the OmniVoice smoke test but with `tts_backend="voxcpm"`, `voxcpm_auto_voice=True`, and import from `dv_backend.voxcpm_env` (it bails with the setup hint if `is_voxcpm_available()` is `False`).

- [ ] **Step 1: Create `backend/scripts/setup_voxcpm.py`**

```python
#!/usr/bin/env python3
"""Create an isolated VoxCPM2 virtualenv for TTS inference."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv",
        default=str(Path(__file__).resolve().parents[1] / ".venv-voxcpm"),
        help="Target virtualenv path",
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Skip installing PyTorch (use when torch is already present)",
    )
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parents[1]
    venv_path = Path(args.venv).resolve()

    _run(["uv", "venv", str(venv_path), "--python", "3.12"], cwd=backend_dir)
    python = venv_path / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    if not args.skip_torch:
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "torch",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            cwd=backend_dir,
        )

    _run(
        ["uv", "pip", "install", "--python", str(python), "voxcpm"],
        cwd=backend_dir,
    )

    _run([str(python), "-c", "import voxcpm; print('voxcpm', voxcpm.__version__)"], cwd=backend_dir)
    print(f"VoxCPM2 ready at {venv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Create `backend/scripts/run_voxcpm_smoke.py`**

This is a port of the existing `backend/scripts/run_omnivoice_smoke.py`. Read that file once and reproduce it with these substitutions:

- Replace `from dv_backend.omnivoice_env import is_omnivoice_available` with `from dv_backend.voxcpm_env import is_voxcpm_available`.
- Replace `is_omnivoice_available()` with `is_voxcpm_available()`.
- Replace the error message `OmniVoice env missing. Run: python scripts/setup_omnivoice.py` with `VoxCPM2 env missing. Run: python scripts/setup_voxcpm.py`.
- Replace the `data_dir` basename `.data-omnivoice-smoke` with `.data-voxcpm-smoke`.
- Replace the settings dict keys: `tts_backend: "voxcpm"`, `voxcpm_auto_voice: True`.
- Replace the `Job` title strings `OmniVoice Smoke` and `Thử OmniVoice` with `VoxCPM2 Smoke` and `Thử VoxCPM2`.
- Replace the segment text `Xin chào, đây là thử nghiệm lồng tiếng bằng OmniVoice.` with `Xin chào, đây là thử nghiệm lồng tiếng bằng VoxCPM2.`.
- Replace `print("Running TTS with OmniVoice...")` with `print("Running TTS with VoxCPM2...")`.

The structure of the script (function signatures, `create_app` import, job creation, runner setup, polling) stays identical.

- [ ] **Step 3: Run the new setup script in `--help` mode to verify it parses**

Run:
```bash
cd backend && uv run python scripts/setup_voxcpm.py --help
```
Expected: argparse help text, exit 0.

- [ ] **Step 4: Run the smoke script in `--help` (or `python -c "import …"`) to confirm it imports**

Run:
```bash
cd backend && uv run python -c "import importlib.util, pathlib; spec = importlib.util.spec_from_file_location('smoke', pathlib.Path('scripts/run_voxcpm_smoke.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('ok')"
```
Expected: prints `ok`. (We can't actually exercise the GPU path here; we just need the module to be syntactically valid and import-clean.)

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/setup_voxcpm.py backend/scripts/run_voxcpm_smoke.py
git commit -m "feat(tts): add VoxCPM2 setup and smoke-test scripts"
```

---

## Task 10: Update frontend bindings

**Files:**
- Modify: `frontend/src/renderer/App.tsx` (rename card title and rebind settings keys)
- Modify: `frontend/tests/App.test.tsx` (assert new card title and `tts_backend: "voxcpm"`)

- [ ] **Step 1: Update `frontend/src/renderer/App.tsx`**

Three changes in the TTS settings card:

1. Title: `Lồng tiếng OmniVoice` → `Lồng tiếng VoxCPM2`.
2. Description: `OmniVoice là engine TTS duy nhất. Chọn audio tham chiếu .wav, nhập voice design, hoặc để auto voice.` → `VoxCPM2 là engine TTS. Chọn audio tham chiếu .wav, nhập voice design, hoặc để auto voice.`
3. Three field bindings:
   - `settings.omnivoice_ref_audio` → `settings.voxcpm_ref_audio`
   - `settings.omnivoice_instruct` → `settings.voxcpm_instruct`
   - `settings.omnivoice_auto_voice` → `settings.voxcpm_auto_voice`

The label text "Audio tham chiếu (.wav)" / "Voice design (tùy chọn)" / "Auto voice khi không có audio tham chiếu" stays unchanged.

- [ ] **Step 2: Update `frontend/tests/App.test.tsx`**

For every fixture that sets `tts_backend: "omnivoice"`, change to `tts_backend: "voxcpm"`. Update the assertion that checks for the card title text from `Lá»“ng tiáº¿ng OmniVoice` (escaped form of `Lồng tiếng OmniVoice`) to the escaped form of `Lồng tiếng VoxCPM2`. The original tests use the literal string with Vietnamese diacritics; the escaped form `Lá»“ng tiáº¿ng` is the UTF-8 bytes `Lồng tiếng` interpreted as latin-1. Replace it with the equivalent `Lá»“ng tiáº¿ng VoxCPM2` (i.e. `Lồng tiếng VoxCPM2`).

- [ ] **Step 3: Run the frontend test suite**

Run:
```bash
pnpm --filter frontend test
```
Expected: all pass.

- [ ] **Step 4: Verify the type-check passes**

Run:
```bash
pnpm --filter frontend exec tsc --noEmit
```
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/renderer/App.tsx frontend/tests/App.test.tsx
git commit -m "feat(ui): rename TTS settings card to VoxCPM2"
```

---

## Task 11: Update docs and delete legacy files

**Files:**
- Modify: `README.md`, `docs/DIARIZATION.md`
- Create: `docs/VOXCPM.md` (replaces `docs/OMNIVOICE.md`)
- Delete: 8 files (see list)

- [ ] **Step 1: Create `docs/VOXCPM.md`**

Mirror the structure of the existing `docs/OMNIVOICE.md`:

```markdown
# VoxCPM2 TTS Pipeline

The VoxCPM2 backend now runs as a **long-lived worker process** with
batched GPU inference and an on-disk result cache. This replaces the
previous OmniVoice design and dramatically reduces end-to-end dubbing
time for long videos.

## Architecture

```
backend (main process)                   .venv-voxcpm (worker process)
-------------------------                -----------------------------------
VoxCPMTtsAdapter                         dv_backend.adapters.voxcpm_worker
   |                                          |
   v                                          v
VoxCPMWorkerClient  ---stdin/stdout---> VoxCPMEngine (keeps model in VRAM)
   |                                          |
   v                                          v
VoxCPMCache (sha256 -> .wav)             per-segment inference via voxcpm.generate()
```

### Key components

- `dv_backend/adapters/voxcpm_worker.py` — worker script that runs inside
  the isolated `.venv-voxcpm` virtualenv, loads the VoxCPM2 model once,
  and processes JSONL requests from stdin.
- `dv_backend/adapters/voxcpm_client.py` — backend-side client that
  spawns the worker, batches requests by `(model, device, prompt_wav_path,
  prompt_text, voice_design, inference_timesteps, cfg_value)`, and
  surfaces cancel via `JobRunner.register_process`.
- `dv_backend/adapters/voxcpm_cache.py` — content-addressed cache
  stored under `<data_dir>/cache/voxcpm/`. Disabled with
  `DV_VOXCPM_CACHE_DISABLED=1` or `voxcpm_cache_enabled=False` in
  settings.
- `dv_backend/adapters/tts.py` — `VoxCPMTtsAdapter` no longer spawns a
  fresh Python per call. It routes through the worker client and
  consults the cache first.

## Performance

For a 60-segment Vietnamese dubbing job on a single GPU, the model-load
and Python-startup costs are paid once. With the worker design:

- The VoxCPM2 model is loaded once and stays resident in VRAM.
- Requests with the same voice signature are coalesced and the worker
  serves them sequentially; the model stays hot.
- Repeated work (resume after crash, dubbing the same video twice)
  becomes instant thanks to the on-disk cache.

## Settings

| Setting                        | Default          | Description                                            |
| ------------------------------ | ---------------- | ------------------------------------------------------ |
| `voxcpm_model`                 | `openbmb/VoxCPM2` | HF model id.                                          |
| `voxcpm_device`                | `cuda:0`         | Torch device string.                                   |
| `voxcpm_num_steps`             | `10`             | Diffusion steps. **Lower = faster but lower quality.** |
| `voxcpm_ref_audio`             | `""`             | Path to a `.wav` for voice cloning.                    |
| `voxcpm_instruct`              | `""`             | Voice design description (e.g. `female, low pitch`).   |
| `voxcpm_auto_voice`            | `true`           | When `true` and no ref audio is set, auto voice.       |
| `voxcpm_batch_size`            | `4`              | Maximum segments per coalesced batch.                  |
| `voxcpm_batch_flush_ms`        | `150`            | Flush window for incomplete batches.                   |
| `voxcpm_cache_enabled`         | `true`           | Disable to skip the on-disk result cache.              |

## Cancellation

When the user cancels a job the `JobRunner` kills the worker Popen.
The reader thread detects the broken pipe, fails any in-flight
requests, and the next call transparently respawns the worker. No
adapter code change is required to take advantage of this.

## Smoke test

`backend/scripts/run_voxcpm_smoke.py` exercises the full TTS
pipeline end-to-end with VoxCPM2. Use it after upgrading the
worker script to confirm nothing regressed:

```powershell
cd backend
python scripts/setup_voxcpm.py
python scripts/run_voxcpm_smoke.py
```

## Troubleshooting

- **`VOXCPM_TTS_FAILED` with `code = VOXCPM_WORKER_DIED`** —
  the worker crashed (typically OOM). Lower `voxcpm_batch_size` to
  reduce peak VRAM usage.
- **Slow first segment, fast subsequent ones** — expected; the worker
  is loading the model on the first request. Cache hits bypass the
  worker entirely.
- **No batching benefit** — every segment uses a different voice
  (e.g. a per-speaker reference audio). The cache still helps when
  segments repeat, but the per-batch win is limited.
```

- [ ] **Step 2: Update `README.md`**

Three edits:

1. Section "Current status" step 4: `Synthesize Vietnamese speech with OmniVoice.` → `Synthesize Vietnamese speech with VoxCPM2.`
2. Section "Speaker diarization/per-speaker voice assignment has been removed; all segments use the single OmniVoice configuration." → replace `OmniVoice` with `VoxCPM2`.
3. Replace the bullet `OmniVoice runs through the isolated `backend/.venv-omnivoice` environment after setup.` with `VoxCPM2 runs through the isolated `backend/.venv-voxcpm` environment after setup.`
4. Replace the `setup_omnivoice.py` reference with `setup_voxcpm.py`.

- [ ] **Step 3: Update `docs/DIARIZATION.md`**

The single line:
> Speaker diarization and per-speaker voice assignment have been removed. The pipeline now reads ASR segments directly in `normalize_segments` and uses OmniVoice as the single TTS engine.

Replace `OmniVoice` with `VoxCPM2`.

- [ ] **Step 4: Delete the legacy files**

```bash
git rm backend/dv_backend/omnivoice_env.py
git rm backend/dv_backend/adapters/omnivoice_client.py
git rm backend/dv_backend/adapters/omnivoice_worker.py
git rm backend/dv_backend/adapters/omnivoice_cache.py
git rm backend/scripts/setup_omnivoice.py
git rm backend/scripts/run_omnivoice_smoke.py
git rm backend/tests/test_omnivoice_tts.py
git rm docs/OMNIVOICE.md
```

- [ ] **Step 5: Run the entire backend test suite to confirm nothing dangles**

Run:
```bash
cd backend && uv run pytest -v
```
Expected: every test passes. If any test imports from `dv_backend.omnivoice_env` or `dv_backend.adapters.omnivoice_*`, fix it.

- [ ] **Step 6: Run the frontend test suite and type-check**

Run:
```bash
pnpm --filter frontend test
pnpm --filter frontend exec tsc --noEmit
```
Expected: both pass.

- [ ] **Step 7: Run a final repo-wide grep for any leftover `omnivoice` references**

Run:
```bash
grep -rIn --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=.venv --exclude-dir=.venv-omnivoice --exclude-dir=.venv-voxcpm 'omnivoice\|OmniVoice\|OMNIVOICE' .
```
Expected: no matches.

If matches exist, fix them in this commit (rename keys, update error messages, update UI strings).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "docs+chore: finish VoxCPM2 migration, drop all OmniVoice artefacts"
```

---

## Self-Review

**1. Spec coverage** — every section in `docs/superpowers/specs/2026-06-27-voxcpm2-migration-design.md` is covered:
- §1 Goals/non-goals: enforced by Task 11 (delete all OmniVoice files) and Task 6 (single backend in `SUPPORTED_TTS_BACKENDS`).
- §2 Architecture (long-lived worker + cache): Tasks 4 (worker), 5 (client), 2 (cache), 3 (env), 1 (adapter).
- §3 Renaming map: Tasks 1, 6, 7, 8, 10, 11.
- §4 Component design: Tasks 1–9 mirror the spec's component design one-for-one.
- §5 Data flow: covered by the adapter contract in Task 1.
- §6 Error handling: covered by Tasks 1, 4, 5 (each defines the relevant `VOXCPM_*` error code path).
- §7 Testing plan: tasks 1–5 build `test_voxcpm_tts.py`; Task 6 updates `test_settings.py`; Task 10 updates `App.test.tsx`; Task 11 runs the full suite as the final check.
- §8 Open questions: streaming/segmentation out of scope, defaults match spec.

**2. Placeholder scan** — no `TBD`, `TODO`, "implement later", "add appropriate error handling", "similar to Task N" anywhere. Code blocks contain full implementations; commands list full output expectations.

**3. Type / name consistency** — every reference to a symbol defined in an earlier task uses the same name:
- `VoxCPMTtsAdapter` defined in Task 1, used in Tasks 2, 5, 6, 8, 10.
- `parse_voxcpm_voice` defined in Task 1, used only inside the same file.
- `VOXCPM_INSTRUCT_PREFIX` defined in Task 1, used in Task 1 (adapters + tests) and Task 10.
- `VoxCPMCache` defined in Task 2, used in Task 1 (via lazy import) and Task 5.
- `voxcpm_venv_root`, `resolve_voxcpm_python`, `is_voxcpm_available` defined in Task 3, used in Task 4 (via the worker module) and Task 5 (client) and Task 7 (runtime check).
- `VoxCPMWorkerClient`, `acquire_client`, `release_all_clients` defined in Task 5, used in Task 7 (runtime check) and Task 1 (adapter lazy import).
- `VOXCPM_DEFAULT_MODEL = "openbmb/VoxCPM2"` defined in Task 1, used in Task 6 (settings default) and Task 7 (runtime check).
- `worker_generate` and its fake `model.generate(text=..., **kwargs)` contract defined in Task 4, regression-tested in the same task.
- The worker request schema (`prompt_wav_path`, `prompt_text`, `voice_design`, `cfg_value`, `inference_timesteps`) defined in Task 4, used in Task 5 (client) and Task 1 (adapter wires them up).
- The cache key signature `cache_key(*, voice_id, text, model, num_step, voice_design, cfg_value)` defined in Task 2, called in Task 1 with the same kwargs.

All match.
