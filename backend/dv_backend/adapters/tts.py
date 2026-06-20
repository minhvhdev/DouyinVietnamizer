from pathlib import Path

from ..errors import AppError
from ..models import ErrorInfo


VIENEU_PRESET_VOICES = (
    "Ngọc Lan",
    "Gia Bảo",
    "Thái Sơn",
    "Đức Trí",
    "Mỹ Duyên",
    "Trúc Ly",
    "Xuân Vĩnh",
    "Trọng Hữu",
    "Bình An",
    "Ngọc Linh",
)
LEGACY_INVALID_PRESET_VOICES = {"Phương Trang", "Mai Vy"}


class VieNeuTtsAdapter:
    DEFAULT_MODEL_REPO = "pnnbao-ump/VieNeu-TTS-v3-Turbo"

    def __init__(
        self,
        vieneu_class=None,
        *,
        device: str = "cuda",
    ) -> None:
        self.vieneu_class = vieneu_class
        self.device = device
        self._engine = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        requested = device.strip().lower() or "cuda"
        if requested in {"cpu", "onnx"}:
            raise AppError(
                400,
                ErrorInfo(
                    code="VIENEU_GPU_REQUIRED",
                    message="VieNeu-TTS is configured for GPU-only mode.",
                    action="Set vieneu_device to cuda and install PyTorch with CUDA 12.8+ (cu128) for RTX 50-series GPUs.",
                ),
            )
        if requested == "auto":
            requested = "cuda"
        if not requested.startswith("cuda"):
            return requested

        try:
            import torch
        except ImportError as error:
            raise AppError(
                400,
                ErrorInfo(
                    code="TORCH_NOT_INSTALLED",
                    message="PyTorch is required for GPU VieNeu-TTS.",
                    action="Run 'uv sync' in the backend folder.",
                    detail=str(error),
                ),
            ) from error

        if not torch.cuda.is_available():
            raise AppError(
                502,
                ErrorInfo(
                    code="CUDA_NOT_AVAILABLE",
                    message="CUDA GPU is not available for VieNeu-TTS.",
                    action="Install an NVIDIA driver and PyTorch wheels built with CUDA 12.8+ (cu128).",
                ),
            )

        try:
            torch.zeros(1, device="cuda")
        except Exception as error:
            major, minor = torch.cuda.get_device_capability(0)
            raise AppError(
                502,
                ErrorInfo(
                    code="CUDA_INCOMPATIBLE",
                    message="GPU CUDA kernels failed to run on this device.",
                    action=(
                        f"RTX 50-series (sm_{major}{minor}) needs PyTorch cu128 or newer. "
                        "Re-run backend setup: cd backend && uv sync --group dev"
                    ),
                    detail=str(error),
                ),
            ) from error

        return "cuda"

    def _get_engine(self):
        if self._engine is None:
            if self.vieneu_class:
                self._engine = self.vieneu_class()
            else:
                from vieneu import Vieneu

                device = self._resolve_device(self.device)
                self._engine = Vieneu(mode="v3turbo", device=device)
        return self._engine

    def synthesize(self, text: str, output_path: Path, *, voice: str) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tts = self._get_engine()
            ref_path = Path(voice)
            if ref_path.is_file():
                audio = tts.infer(text, ref_audio=str(ref_path))
            else:
                audio = tts.infer(text, voice=voice)
            tts.save(audio, str(output_path))
        except AppError:
            raise
        except ValueError as cause:
            message = str(cause)
            if "not found" in message.lower() and "voice" in message.lower():
                raise AppError(
                    400,
                    ErrorInfo(
                        code="VIENEU_VOICE_NOT_FOUND",
                        message=f"VieNeu-TTS preset voice is not available: {voice!r}.",
                        action=(
                            "Open Settings and choose a built-in voice such as "
                            f"{VIENEU_PRESET_VOICES[6]} or upload a cloned .wav reference."
                        ),
                        detail=message,
                    ),
                ) from cause
            raise AppError(
                502,
                ErrorInfo(
                    code="VIENEU_TTS_FAILED",
                    message="VieNeu-TTS could not generate narration on GPU.",
                    action="Ensure PyTorch cu128 is installed and the GPU driver supports CUDA 12.8+.",
                    detail=message,
                    retryable=True,
                ),
            ) from cause
        except ImportError as e:
            raise AppError(
                400,
                ErrorInfo(
                    code="VIENEU_NOT_INSTALLED",
                    message="VieNeu-TTS library is not installed in the current environment.",
                    action="Run 'uv sync' in the backend folder to install VieNeu-TTS.",
                    detail=str(e),
                ),
            ) from e
        except Exception as cause:
            raise AppError(
                502,
                ErrorInfo(
                    code="VIENEU_TTS_FAILED",
                    message="VieNeu-TTS could not generate narration on GPU.",
                    action="Ensure PyTorch cu128 is installed and the GPU driver supports CUDA 12.8+.",
                    detail=str(cause),
                    retryable=True,
                ),
            ) from cause
