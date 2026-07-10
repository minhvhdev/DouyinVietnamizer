import os
import shutil
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

# Pinned stable URLs for vendor binaries
DEFAULT_URLS = {
    "ffmpeg": "https://github.com/GyanD/codexffmpeg/releases/download/6.0/ffmpeg-6.0-essentials_build.zip",
}

QWEN3_MODELS = [
    ("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR-1.7B"),
    ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner-0.6B"),
]

# Thread-safe global status manager
class BootstrapManager:
    _lock = threading.Lock()
    _status = {
        "status": "idle",       # idle, downloading, extracting, completed, failed
        "current_task": "",
        "download_percent": 0.0,
        "download_speed_kb": 0.0,
        "downloaded_bytes": 0,
        "total_bytes": 0,
        "error_message": "",
        "logs": []
    }
    _thread = None

    @classmethod
    def get_status(cls) -> dict:
        with cls._lock:
            return dict(cls._status)

    @classmethod
    def add_log(cls, message: str):
        with cls._lock:
            print(f"[BOOTSTRAP] {message}")
            cls._status["logs"].append(f"[{time.strftime('%H:%M:%S')}] {message}")
            if len(cls._status["logs"]) > 200:
                cls._status["logs"].pop(0)

    @classmethod
    def update(cls, **kwargs):
        with cls._lock:
            for k, v in kwargs.items():
                if k in cls._status:
                    cls._status[k] = v

    @classmethod
    def start_bootstrap(cls, profile: str, vendor_dir: Path, default_manifest_path: Path | None = None):
        with cls._lock:
            if cls._status["status"] in ("downloading", "extracting"):
                cls.add_log("Bootstrap is already running.")
                return False
            
            cls._status = {
                "status": "downloading",
                "current_task": "Starting initialization...",
                "download_percent": 0.0,
                "download_speed_kb": 0.0,
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "error_message": "",
                "logs": []
            }
            
            cls._thread = threading.Thread(
                target=cls._run_bootstrap,
                args=(profile, vendor_dir, default_manifest_path),
                daemon=True
            )
            cls._thread.start()
            return True

    @classmethod
    def _run_bootstrap(cls, profile: str, vendor_dir: Path, default_manifest_path: Path | None):
        cls.add_log(f"Starting bootstrap environment setup with profile: {profile}")
        cls.add_log(f"Vendor directory: {vendor_dir}")

        try:
            vendor_dir.mkdir(parents=True, exist_ok=True)

            # 1. Ensure manifest.json exists in vendor folder
            manifest_dest = vendor_dir / "manifest.json"
            if not manifest_dest.is_file():
                cls.add_log("manifest.json not found in vendor folder, restoring...")
                if default_manifest_path and default_manifest_path.is_file():
                    cls.add_log(f"Copying default manifest from {default_manifest_path}")
                    shutil.copy2(default_manifest_path, manifest_dest)
                else:
                    cls.add_log("No default manifest found. Writing fallback manifest.json.")
                    cls._write_fallback_manifest(manifest_dest)

            # 2. Define download schedule based on profile
            # All profiles require media/download tools and Qwen3-ASR model weights.
            tasks = ["ffmpeg"]

            # Temporary folder for zip downloads
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                for item in tasks:
                    url = DEFAULT_URLS[item]
                    cls.add_log(f"Downloading {item} from {url}...")
                    cls.update(current_task=f"Tải xuống {item}...")

                    # Destination file
                    dest_file = temp_path / f"{item}_download"
                    if url.endswith(".zip"):
                        dest_file = dest_file.with_suffix(".zip")
                    elif url.endswith(".exe"):
                        dest_file = dest_file.with_suffix(".exe")
                    else:
                        dest_file = dest_file.with_suffix(".bin")

                    cls._download_file(url, dest_file)

                    cls.update(status="extracting", current_task=f"Giải nén {item}...")
                    cls.add_log(f"Processing and extracting {item} to vendor directory...")
                    cls._process_and_extract(item, dest_file, vendor_dir)

            cls.update(status="extracting", current_task="Tải mô hình Qwen3-ASR 1.7B...")
            cls._download_qwen_models(vendor_dir)

            cls.update(status="completed", current_task="Thiết lập môi trường hoàn tất!", download_percent=100.0)
            cls.add_log("Bootstrap completed successfully. System is ready.")

        except Exception as e:
            cls.update(status="failed", current_task="Lỗi thiết lập", error_message=str(e))
            cls.add_log(f"FATAL ERROR during bootstrap: {str(e)}")

    @classmethod
    def _download_file(cls, url: str, dest_path: Path):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        req = urllib.request.Request(url, headers=headers)
        
        start_time = time.time()
        last_update = start_time
        downloaded = 0

        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', 0))
            cls.update(total_bytes=total_size, downloaded_bytes=0, download_percent=0.0)

            with open(dest_path, 'wb') as f:
                block_size = 1024 * 64
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)
                    
                    now = time.time()
                    if now - last_update > 0.5 or downloaded == total_size:
                        speed = (downloaded / 1024.0) / (now - start_time + 0.001)
                        percent = (downloaded / total_size * 100.0) if total_size > 0 else 0.0
                        cls.update(
                            downloaded_bytes=downloaded,
                            download_percent=round(percent, 1),
                            download_speed_kb=round(speed, 1)
                        )
                        cls.add_log(f"Downloading: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB) at {speed:.1f} KB/s")
                        last_update = now

    @classmethod
    def _process_and_extract(cls, item: str, file_path: Path, vendor_dir: Path):
        if item == "ffmpeg":
            dest_dir = vendor_dir / "ffmpeg"
            dest_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(file_path, "r") as zip_ref:
                for member in zip_ref.namelist():
                    if member.endswith("ffmpeg.exe"):
                        with zip_ref.open(member) as source, open(dest_dir / "ffmpeg.exe", "wb") as target:
                            shutil.copyfileobj(source, target)
                        cls.add_log("ffmpeg.exe extracted.")
                    elif member.endswith("ffprobe.exe"):
                        with zip_ref.open(member) as source, open(dest_dir / "ffprobe.exe", "wb") as target:
                            shutil.copyfileobj(source, target)
                        cls.add_log("ffprobe.exe extracted.")

    @classmethod
    def _download_qwen_models(cls, vendor_dir: Path) -> None:
        from huggingface_hub import snapshot_download

        models_root = vendor_dir / "qwen3-asr"
        models_root.mkdir(parents=True, exist_ok=True)
        for repo_id, folder_name in QWEN3_MODELS:
            dest_dir = models_root / folder_name
            if dest_dir.is_dir() and any(dest_dir.iterdir()):
                cls.add_log(f"{folder_name} already present, skipping download.")
                continue
            cls.add_log(f"Downloading {repo_id}...")
            cls.update(current_task=f"Tải mô hình {folder_name}...")
            snapshot_download(repo_id, local_dir=dest_dir)
            cls.add_log(f"{folder_name} installed under qwen3-asr/.")

    @classmethod
    def _write_fallback_manifest(cls, dest_path: Path):
        manifest_data = {
            "schema_version": 1,
            "tools": [
                {
                    "id": "ffmpeg",
                    "display_name": "FFmpeg",
                    "executable": "ffmpeg/ffmpeg.exe",
                    "dev_command": "ffmpeg",
                    "version_args": ["-version"],
                    "version_contains": "ffmpeg",
                    "required": True,
                    "capability": "media"
                }
            ]
        }
        import json
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        cls.add_log("Fallback manifest.json written successfully.")
