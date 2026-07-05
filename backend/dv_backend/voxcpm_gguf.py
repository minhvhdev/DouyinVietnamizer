"""GGUF model and voxcpm2-cli resolution for VoxCPM2 inference."""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

VOXCPM_GGUF_REPO = "DennisHuang648/VoxCPM2-GGUF"
VOXCPM_DEFAULT_MODEL = "gguf-q8"
VOXCPM_BASELM_Q8 = "VoxCPM2-BaseLM-Q8_0.gguf"
VOXCPM_ACOUSTIC_F16 = "VoxCPM2-Acoustic-F16.gguf"
VOXCPM_DEFAULT_SAMPLE_RATE = 48_000

_LEGACY_MODEL_ALIASES = frozenset(
    {
        "openbmb/VoxCPM2",
        "OpenBMB/VoxCPM2",
        "OpenBMB/voxcpm2",
    }
)


def normalize_voxcpm_model_id(model: str | None) -> str:
    configured = (model or VOXCPM_DEFAULT_MODEL).strip() or VOXCPM_DEFAULT_MODEL
    if configured in _LEGACY_MODEL_ALIASES:
        return VOXCPM_DEFAULT_MODEL
    return configured


def _models_root() -> Path | None:
    override = os.environ.get("DV_MODELS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return None


def _vendor_dir() -> Path | None:
    override = os.environ.get("DV_VENDOR_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    vendor = project_root / "vendor"
    return vendor if vendor.is_dir() else None


def resolve_voxcpm_cli() -> Path:
    override = os.environ.get("DV_VOXCPM_CLI", "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"DV_VOXCPM_CLI does not exist: {path}")

    vendor = _vendor_dir()
    if vendor is not None:
        for relative in (
            "voxcpm2/voxcpm2-cli.exe",
            "voxcpm2/voxcpm2-cli",
            "tools/voxcpm2/voxcpm2-cli.exe",
            "tools/voxcpm2/voxcpm2-cli",
        ):
            candidate = vendor / relative
            if candidate.is_file():
                return candidate

    found = shutil.which("voxcpm2-cli")
    if found:
        return Path(found).resolve()

    raise FileNotFoundError(
        "voxcpm2-cli was not found. Build llama.cpp-omni (target voxcpm2-cli), "
        "place it under vendor/voxcpm2/, or set DV_VOXCPM_CLI."
    )


def _resolve_gguf_filenames(model_id: str) -> tuple[str, str]:
    baselm = os.environ.get("DV_VOXCPM_BASELM", "").strip()
    acoustic = os.environ.get("DV_VOXCPM_ACOUSTIC", "").strip()
    if baselm and acoustic:
        return baselm, acoustic
    if model_id == "gguf-f16":
        return "VoxCPM2-BaseLM-F16.gguf", VOXCPM_ACOUSTIC_F16
    return VOXCPM_BASELM_Q8, VOXCPM_ACOUSTIC_F16


def _search_roots(model: str | None) -> list[Path]:
    roots: list[Path] = []
    configured = (model or "").strip()
    if configured and configured not in {VOXCPM_DEFAULT_MODEL, "gguf-f16", *_LEGACY_MODEL_ALIASES}:
        path = Path(configured).expanduser()
        if path.is_file() and path.suffix.lower() == ".gguf":
            roots.append(path.parent)
        elif path.is_dir():
            roots.append(path.resolve())

    models_root = _models_root()
    if models_root is not None:
        roots.append(models_root / "voxcpm2")
        roots.append(models_root / "voxcpm2" / "VoxCPM2")

    backend_models = Path(__file__).resolve().parents[1] / "models" / "voxcpm2"
    roots.append(backend_models)
    return roots


def resolve_voxcpm_gguf_paths(model: str | None = None) -> tuple[Path, Path]:
    """Return ``(baselm_gguf, acoustic_gguf)`` for the configured model id."""
    model_id = normalize_voxcpm_model_id(model)
    baselm_name, acoustic_name = _resolve_gguf_filenames(model_id)

    configured = (model or "").strip()
    if configured and Path(configured).is_file() and Path(configured).suffix.lower() == ".gguf":
        baselm = Path(configured).expanduser().resolve()
        acoustic = baselm.parent / acoustic_name
        if acoustic.is_file():
            return baselm, acoustic
        raise FileNotFoundError(
            f"VoxCPM acoustic GGUF not found next to {baselm}: expected {acoustic}"
        )

    missing: list[str] = []
    for root in _search_roots(model):
        baselm = root / baselm_name
        acoustic = root / acoustic_name
        if baselm.is_file() and acoustic.is_file():
            return baselm.resolve(), acoustic.resolve()
        if not baselm.is_file():
            missing.append(str(baselm))
        if not acoustic.is_file():
            missing.append(str(acoustic))

    detail = ", ".join(dict.fromkeys(missing)) or "no search roots"
    raise FileNotFoundError(
        "VoxCPM2 GGUF weights were not found. "
        f"Expected {baselm_name} and {acoustic_name}. "
        f"Searched: {detail}. "
        "Run: python scripts/setup_voxcpm.py"
    )


def resolve_llama_tts_server() -> Path:
    override = os.environ.get("DV_VOXCPM_TTS_SERVER", "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"DV_VOXCPM_TTS_SERVER does not exist: {path}")

    vendor = _vendor_dir()
    if vendor is not None:
        for relative in (
            "voxcpm2/llama-tts-server.exe",
            "voxcpm2/llama-tts-server",
            "tools/voxcpm2/llama-tts-server.exe",
            "tools/voxcpm2/llama-tts-server",
        ):
            candidate = vendor / relative
            if candidate.is_file():
                return candidate

    cli = resolve_voxcpm_cli()
    sibling = cli.parent / ("llama-tts-server.exe" if sys.platform == "win32" else "llama-tts-server")
    if sibling.is_file():
        return sibling

    raise FileNotFoundError(
        "llama-tts-server was not found. Build llama.cpp-omni (target llama-tts-server) "
        "and place it next to voxcpm2-cli."
    )


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _cli_env(cli_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    cli_dir_str = str(cli_dir)
    env["PATH"] = cli_dir_str if not env.get("PATH") else cli_dir_str + os.pathsep + env["PATH"]
    env.setdefault("PYTHONUTF8", "1")
    return env


def _wav_duration(output_path: str | Path) -> tuple[float, int]:
    with wave.open(str(output_path), "rb") as wav_file:
        sample_rate = int(wav_file.getframerate() or VOXCPM_DEFAULT_SAMPLE_RATE)
        frames = int(wav_file.getnframes())
        if sample_rate <= 0 or frames <= 0:
            raise ValueError(f"Generated WAV is empty: {output_path}")
        return frames / float(sample_rate), sample_rate


class GgufTtsServerSession:
    """Long-lived llama-tts-server process with model loaded once."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._port: int | None = None
        self._baselm: Path | None = None
        self._acoustic: Path | None = None
        self._device: str | None = None

    def ensure_running(self, *, baselm: Path, acoustic: Path, device: str) -> None:
        device_key = (device or "cuda:0").strip() or "cuda:0"
        if (
            self._proc is not None
            and self._proc.poll() is None
            and self._port
            and self._baselm == baselm
            and self._acoustic == acoustic
            and self._device == device_key
        ):
            return

        self.shutdown()
        server = resolve_llama_tts_server()
        port = _pick_free_port()
        cmd = [
            str(server),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--voxcpm2-base-lm",
            str(baselm),
            "--voxcpm2-acoustic",
            str(acoustic),
        ]
        if device_key.lower() == "cpu":
            cmd.extend(["--voxcpm2-n-gpu-layers", "0"])
        env = _cli_env(server.parent)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.perf_counter() + 180.0
        while time.perf_counter() < deadline:
            if self._proc.poll() is not None:
                stderr = (self._proc.stderr.read() if self._proc.stderr else "") or ""
                raise RuntimeError(f"llama-tts-server exited during startup: {stderr[-2000:]}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2.0) as resp:
                    if resp.status == 200:
                        self._port = port
                        self._baselm = baselm
                        self._acoustic = acoustic
                        self._device = device_key
                        return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.5)
        raise RuntimeError("llama-tts-server failed to become ready within 180s")

    def synthesize(
        self,
        *,
        text: str,
        output_path: str,
        cfg_value: float,
        inference_timesteps: int,
        reference_wav_path: str | None = None,
        mode: str = "design",
        timeout_sec: float = 300.0,
    ) -> tuple[float, int]:
        if self._port is None:
            raise RuntimeError("TTS server is not running")
        payload: dict[str, object] = {
            "model": "voxcpm",
            "input": text,
            "cfg_value": float(cfg_value),
            "inference_timesteps": int(inference_timesteps),
            "response_format": "wav",
        }
        resolved_mode = (mode or "design").strip().lower() or "design"
        if resolved_mode in {"reference", "ultimate"} and reference_wav_path:
            payload["reference_audio"] = base64.b64encode(Path(reference_wav_path).read_bytes()).decode("ascii")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{self._port}/v1/audio/speech",
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=max(30.0, float(timeout_sec))) as response:
                audio = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-tts-server HTTP {exc.code}: {detail}") from exc
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(audio)
        return _wav_duration(out)

    def shutdown(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        self._port = None
        self._baselm = None
        self._acoustic = None
        self._device = None


def is_gguf_runtime_ready(*, model: str | None = None) -> bool:
    try:
        resolve_voxcpm_gguf_paths(model)
        try:
            resolve_llama_tts_server()
            return True
        except FileNotFoundError:
            resolve_voxcpm_cli()
            return True
    except FileNotFoundError:
        return False


def build_voxcpm_cli_command(
    cli: Path,
    *,
    text: str,
    output_path: str,
    baselm: Path,
    acoustic: Path,
    device: str,
    cfg_value: float,
    inference_timesteps: int,
    reference_wav_path: str | None = None,
    prompt_wav_path: str | None = None,
    prompt_text: str | None = None,
    mode: str = "design",
) -> list[str]:
    """Build a ``voxcpm2-cli`` argv list for one synthesis request."""
    cmd = [str(cli), "-t", text, "-o", str(output_path)]
    if (device or "").strip().lower() == "cpu":
        cmd.append("--cpu")
    cmd.extend(
        [
            "--cfg",
            str(float(cfg_value)),
            "--timesteps",
            str(int(inference_timesteps)),
        ]
    )

    resolved_mode = (mode or "design").strip().lower() or "design"
    if resolved_mode == "reference":
        anchor = reference_wav_path or prompt_wav_path
        if anchor:
            cmd.extend(["-r", anchor])
    elif resolved_mode == "ultimate":
        anchor = reference_wav_path or prompt_wav_path
        if anchor:
            cmd.extend(["--prompt-wav", anchor])
        anchor_text = (prompt_text or "").strip()
        if anchor_text:
            cmd.extend(["--prompt-text", anchor_text])
    elif prompt_wav_path:
        cmd.extend(["-r", prompt_wav_path])
        if prompt_text:
            cmd.extend(["--prompt-text", prompt_text])

    cmd.extend([str(baselm), str(acoustic)])
    return cmd


def resolve_worker_python() -> Path:
    """Python executable used to run the JSONL worker process."""
    env_override = os.environ.get("DV_VOXCPM_PYTHON", "").strip()
    if env_override:
        path = Path(env_override).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"DV_VOXCPM_PYTHON does not exist: {path}")

    venv_override = os.environ.get("DV_VOXCPM_VENV", "").strip()
    if venv_override:
        venv_root = Path(venv_override).expanduser().resolve()
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

    return Path(sys.executable).resolve()
