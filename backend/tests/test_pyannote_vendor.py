from pathlib import Path

from dv_backend.pyannote_vendor import (
    PYANNOTE_MODEL_DIRNAME,
    PYANNOTE_REQUIRED_WEIGHTS,
    pyannote_model_dir,
    validate_pyannote_model_dir,
)


def _stub_pyannote_weights(model_dir: Path) -> None:
    for rel in PYANNOTE_REQUIRED_WEIGHTS:
        path = model_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")


def test_validate_pyannote_model_dir_requires_config(tmp_path: Path) -> None:
    model_dir = pyannote_model_dir(tmp_path / "vendor")
    assert validate_pyannote_model_dir(model_dir) is not None

    model_dir.mkdir(parents=True)
    assert validate_pyannote_model_dir(model_dir) is not None

    (model_dir / "config.yaml").write_text("pipeline:\n", encoding="utf-8")
    assert validate_pyannote_model_dir(model_dir) is not None

    _stub_pyannote_weights(model_dir)
    assert validate_pyannote_model_dir(model_dir) is None


def test_pyannote_model_dir_layout(tmp_path: Path) -> None:
    expected = tmp_path / "vendor" / "pyannote" / PYANNOTE_MODEL_DIRNAME
    assert pyannote_model_dir(tmp_path / "vendor") == expected
