from pathlib import Path

from dv_backend.config import AppConfig


def test_data_dir_override_is_used(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DV_DATA_DIR", str(tmp_path))

    config = AppConfig.from_env()

    assert config.data_dir == tmp_path
    assert config.database_path == tmp_path / "app.db"


def test_error_log_path_uses_logs_subdir(tmp_path: Path) -> None:
    config = AppConfig(tmp_path)

    assert config.error_log_path == tmp_path / "logs" / "backend-error.log"

