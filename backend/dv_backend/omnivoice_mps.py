"""Device planning for OmniVoice, including Apple Silicon MPS placement."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module, metadata
import os
import platform
import sys
from typing import Any, cast


@dataclass(frozen=True)
class OmniVoiceDevicePlan:
    requested_device: str
    resolved_device: str
    model_dtype: str
    audio_tokenizer_device: str
    reason: str


class OmniVoiceDeviceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _mps_capabilities(torch_module: Any) -> tuple[bool, bool]:
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is None:
        return False, False
    is_built = getattr(mps, "is_built", None)
    is_available = getattr(mps, "is_available", None)
    built = bool(is_built()) if callable(is_built) else False
    available = bool(is_available()) if callable(is_available) else False
    return built, available


def plan_omnivoice_device(
    device: str | None,
    *,
    platform_name: str | None = None,
    machine: str | None = None,
    torch_module: Any | None = None,
    allow_cpu_fallback: bool = False,
) -> OmniVoiceDevicePlan:
    """Resolve OmniVoice's main-model and tokenizer placement.

    OmniVoice 0.2.x supports a hybrid Apple Silicon path: the main model runs
    on MPS in float16 while its Higgs audio tokenizer remains on CPU.
    """
    requested = (device or "auto").strip().lower() or "auto"
    current_platform = (platform_name or sys.platform).lower()
    current_machine = (machine or platform.machine()).lower()
    if torch_module is None:
        torch_module = import_module("torch")

    if current_platform == "darwin":
        if current_machine not in {"arm64", "aarch64"}:
            raise OmniVoiceDeviceError(
                "OMNIVOICE_MPS_UNSUPPORTED_ARCH",
                "OmniVoice MPS is supported only on Apple Silicon (arm64).",
            )
        if requested == "cpu":
            return OmniVoiceDevicePlan(requested, "cpu", "float32", "cpu", "explicit_cpu")
        if requested not in {"auto", "default", "mps"} and not requested.startswith("cuda"):
            raise OmniVoiceDeviceError(
                "OMNIVOICE_DEVICE_INVALID",
                f"Unsupported OmniVoice device on Apple Silicon: {requested}",
            )

        built, available = _mps_capabilities(torch_module)
        if available:
            return OmniVoiceDevicePlan(
                requested,
                "mps",
                "float16",
                "cpu",
                "apple_silicon_mps",
            )
        if allow_cpu_fallback:
            return OmniVoiceDevicePlan(
                requested,
                "cpu",
                "float32",
                "cpu",
                "explicit_cpu_fallback",
            )
        code = "OMNIVOICE_MPS_UNAVAILABLE" if built else "OMNIVOICE_MPS_NOT_BUILT"
        raise OmniVoiceDeviceError(
            code,
            "OmniVoice requires an MPS-enabled PyTorch runtime on Apple Silicon. "
            "Set an explicit CPU fallback only if slow CPU inference is acceptable.",
        )

    if requested == "mps":
        _built, available = _mps_capabilities(torch_module)
        if available:
            return OmniVoiceDevicePlan(requested, "mps", "float16", "cpu", "mps")
        raise OmniVoiceDeviceError(
            "OMNIVOICE_MPS_UNAVAILABLE",
            "MPS was requested but is not available in this runtime.",
        )

    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda.is_available()) if cuda and hasattr(cuda, "is_available") else False
    if requested in {"auto", "default"}:
        requested = "cuda:0"
    if requested.startswith("cuda"):
        if cuda_available:
            return OmniVoiceDevicePlan(requested, requested, "float16", requested, "cuda")
        return OmniVoiceDevicePlan(requested, "cpu", "float32", "cpu", "cuda_unavailable")
    if requested == "cpu":
        return OmniVoiceDevicePlan(requested, "cpu", "float32", "cpu", "explicit_cpu")
    raise OmniVoiceDeviceError(
        "OMNIVOICE_DEVICE_INVALID",
        f"Unsupported OmniVoice device: {requested}",
    )


def omnivoice_torch_dtype(torch_module: Any, device: str):
    """Return the upstream-compatible dtype for an OmniVoice main model."""
    if str(device).startswith("cuda") or str(device) == "mps":
        return torch_module.float16
    return torch_module.float32


def omnivoice_runtime_capabilities(torch_module: Any) -> dict[str, Any]:
    """Describe accelerator capabilities without loading model weights."""
    built, available = _mps_capabilities(torch_module)
    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda.is_available()) if cuda and hasattr(cuda, "is_available") else False
    return {
        "platform": sys.platform,
        "machine": platform.machine().lower(),
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
        "cuda_available": cuda_available,
        "mps_built": built,
        "mps_available": available,
    }


def validate_mps_operator_fallback_environment(
    environ: Mapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
) -> bool:
    """Reject implicit MPS operator fallback unless separately opted in."""
    env = environ if environ is not None else os.environ
    fallback_enabled = str(env.get("PYTORCH_ENABLE_MPS_FALLBACK", "")).strip() == "1"
    operator_opt_in = (
        str(env.get("DV_OMNIVOICE_ALLOW_MPS_OPERATOR_FALLBACK", "")).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if fallback_enabled and (platform_name or sys.platform).lower() == "darwin" and not operator_opt_in:
        raise OmniVoiceDeviceError(
            "OMNIVOICE_MPS_FALLBACK_FORBIDDEN",
            "PYTORCH_ENABLE_MPS_FALLBACK=1 enables implicit CPU operator fallback. "
            "Unset it, or explicitly set DV_OMNIVOICE_ALLOW_MPS_OPERATOR_FALLBACK=1 "
            "for a diagnostic run.",
        )
    return fallback_enabled


def inspect_module_placement(module: Any) -> dict[str, Any]:
    """Inspect CPU/float32 placement across module parameters and buffers."""
    tensors: list[tuple[str, Any]] = []
    for method_name in ("named_parameters", "named_buffers"):
        method = getattr(module, method_name, None)
        if callable(method):
            entries = cast(Iterable[tuple[str, Any]], method(recurse=True))
            tensors.extend((str(name), tensor) for name, tensor in entries)

    devices: set[str] = set()
    floating_dtypes: set[str] = set()
    violations: list[str] = []
    for name, tensor in tensors:
        device = str(getattr(tensor, "device", "unknown"))
        dtype = str(getattr(tensor, "dtype", "unknown"))
        devices.add(device)
        is_floating = getattr(tensor, "is_floating_point", None)
        floating = bool(is_floating()) if callable(is_floating) else False
        if not device.startswith("cpu"):
            violations.append(f"{name}:device={device}")
        if floating:
            floating_dtypes.add(dtype)
            if dtype not in {"torch.float32", "float32"}:
                violations.append(f"{name}:dtype={dtype}")

    if not tensors:
        module_device = str(getattr(module, "device", "unknown"))
        devices.add(module_device)
        if not module_device.startswith("cpu"):
            violations.append(f"module:device={module_device}")
    return {
        "devices": sorted(devices),
        "floating_dtypes": sorted(floating_dtypes),
        "tensor_count": len(tensors),
        "violations": violations,
    }


def omnivoice_runtime_versions() -> dict[str, str]:
    """Return exact versions needed to reproduce an OmniVoice runtime."""
    versions = {"python": platform.python_version()}
    for package in ("omnivoice", "torch", "torchaudio", "transformers", "accelerate"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions
