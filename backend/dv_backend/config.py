from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path

    @property
    def database_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "logs" / "backend.log"

    @property
    def error_log_path(self) -> Path:
        return self.data_dir / "logs" / "backend-error.log"

    @classmethod
    def from_env(cls) -> "AppConfig":
        override = os.environ.get("DV_DATA_DIR")
        if override:
            return cls(Path(override))
        local_app_data = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return cls(Path(local_app_data) / "DouyinVietnamizer")

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "jobs").mkdir(exist_ok=True)

