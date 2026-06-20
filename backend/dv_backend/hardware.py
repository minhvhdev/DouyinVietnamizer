import ctypes
import os
from pathlib import Path

def detect_vulkan() -> bool:
    """Probes the system for Vulkan support by attempting to load vulkan-1.dll."""
    try:
        # Drivers supporting Vulkan place vulkan-1.dll in System32
        vulkan_lib = ctypes.windll.LoadLibrary("vulkan-1.dll")
        return vulkan_lib is not None
    except Exception:
        return False

def detect_cpu_avx2() -> bool:
    """Checks if the CPU supports the AVX2 instruction set on Windows."""
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

def detect_espeak() -> bool:
    """Checks if eSpeak NG is installed in standard Windows Program Files directories."""
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

def detect_cuda() -> bool:
    """Checks whether CUDA is available for Qwen3-ASR GPU inference."""
    try:
        import torch

        return bool(torch.cuda.is_available())
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
