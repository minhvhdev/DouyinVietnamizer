import ctypes
import os
import sys
from pathlib import Path

def detect_vulkan() -> bool:
    """Probes the system for Vulkan support. Windows-only probe; returns False elsewhere."""
    if sys.platform != "win32":
        return False
    try:
        # Drivers supporting Vulkan place vulkan-1.dll in System32
        vulkan_lib = ctypes.windll.LoadLibrary("vulkan-1.dll")
        return vulkan_lib is not None
    except Exception:
        return False

def detect_cpu_avx2() -> bool:
    """Checks if the CPU supports AVX2. Windows: IsProcessorFeaturePresent. Other: positive default."""
    if sys.platform == "win32":
        try:
            # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40 in Windows SDK
            kernel32 = ctypes.windll.kernel32
            return kernel32.IsProcessorFeaturePresent(40) != 0
        except Exception:
            # Fallback to True or check AVX (36) if AVX2 call fails or is unsupported
            try:
                return kernel32.IsProcessorFeaturePresent(36) != 0
            except Exception:
                return False
    # macOS / linux: assume modern CPU (Apple Silicon = ARMv8.4 with i8mm/dotprod).
    return True

def detect_espeak() -> bool:
    """Checks for eSpeak NG. Windows: Program Files scan. Other: returns False (optional dep)."""
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")

        paths_to_check = [
            Path(program_files) / "eSpeak NG" / "libespeak-ng.dll",
            Path(program_files_x86) / "eSpeak NG" / "libespeak-ng.dll",
        ]
        for p in paths_to_check:
            if p.is_file():
                return True
        return False
    return False

def mps_available() -> bool:
    try:
        import torch

        return sys.platform == "darwin" and bool(torch.backends.mps.is_available())
    except Exception:
        return False


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def accelerator_available() -> bool:
    """Returns True if a GPU backend is available: NVIDIA CUDA or Apple MPS."""
    return cuda_available() or mps_available()


def detect_cuda() -> bool:
    """Backward-compatible alias for accelerator availability checks."""
    return accelerator_available()


def default_inference_device() -> str:
    if cuda_available():
        return "cuda:0"
    if mps_available():
        return "mps"
    return "cpu"


def resolve_inference_device(device: str | None = "") -> str:
    """Map configured device strings to an available torch device."""
    requested = (device or "").strip()
    if requested.startswith("cuda"):
        if cuda_available():
            return requested if requested != "cuda" else "cuda:0"
    elif requested == "mps":
        if mps_available():
            return "mps"
    elif requested == "cpu":
        return "cpu"
    return default_inference_device()


def inference_dtype_for_device(device: str):
    import torch

    if device == "mps":
        return torch.float16
    return torch.bfloat16


def get_hardware_report() -> dict:
    """Aggregates hardware status and returns a recommended configuration profile."""
    cuda = detect_cuda()
    vulkan = detect_vulkan()
    avx2 = detect_cpu_avx2()
    espeak = detect_espeak()

    if cuda:
        recommendation = "gpu_cuda"
    elif vulkan:
        recommendation = "gpu_vulkan"
    elif avx2:
        recommendation = "cpu_avx2"
    else:
        recommendation = "cpu_legacy"

    return {
        "cuda_supported": cuda,
        "vulkan_supported": vulkan,
        "avx2_supported": avx2,
        "espeak_installed": espeak,
        "recommendation": recommendation,
    }
