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

def detect_cuda() -> bool:
    """Returns True if a GPU backend is available: NVIDIA CUDA (any OS) or Apple MPS (macOS)."""
    try:
        import torch

        if torch.cuda.is_available():
            return True
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            return True
        return False
    except Exception:
        return False


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
