import pytest

from scripts.setup_omnivoice import omnivoice_runtime_packages


def test_apple_silicon_runtime_packages_are_exactly_pinned() -> None:
    assert omnivoice_runtime_packages("darwin", "arm64") == [
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "soundfile",
        "omnivoice==0.2.0",
    ]


def test_intel_macos_is_out_of_scope() -> None:
    with pytest.raises(ValueError, match="Apple Silicon"):
        omnivoice_runtime_packages("darwin", "x86_64")


def test_non_macos_keeps_platform_torch_while_pinning_omnivoice() -> None:
    packages = omnivoice_runtime_packages("win32", "amd64")

    assert packages[:2] == ["torch", "torchaudio"]
    assert packages[-1] == "omnivoice==0.2.0"
