import io
import json
import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dv_backend.bootstrap import BootstrapManager
from dv_backend.hardware import get_hardware_report


def test_hardware_detection_prefers_cuda():
    with patch("dv_backend.hardware.detect_cuda", return_value=True), \
         patch("dv_backend.hardware.detect_vulkan", return_value=True), \
         patch("dv_backend.hardware.detect_cpu_avx2", return_value=True), \
         patch("dv_backend.hardware.detect_espeak", return_value=True):

        report = get_hardware_report()
        assert report["cuda_supported"] is True
        assert report["recommendation"] == "gpu_cuda"


def test_bootstrap_downloads_and_extracts(tmp_path):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("ffmpeg.exe", b"mock_ffmpeg")
        zip_file.writestr("ffprobe.exe", b"mock_ffprobe")

    zip_data = zip_buffer.getvalue()

    def mock_urlopen(req, *args, **kwargs):
        resp = MagicMock()
        stream = io.BytesIO(zip_data)
        resp.read.side_effect = lambda block_size=None: stream.read(block_size) if block_size is not None else stream.read()
        resp.info.return_value.get.return_value = str(len(zip_data))
        resp.__enter__.return_value = resp
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen), \
         patch("urllib.request.Request"), \
         patch("dv_backend.bootstrap.BootstrapManager._download_qwen_models") as mock_qwen:

         vendor_dir = tmp_path / "vendor"
         default_manifest = tmp_path / "default_manifest.json"

         manifest_content = {
             "schema_version": 1,
             "tools": []
         }
         with open(default_manifest, "w", encoding="utf-8") as f:
             json.dump(manifest_content, f)

         success = BootstrapManager.start_bootstrap(
             profile="gpu_cuda",
             vendor_dir=vendor_dir,
             default_manifest_path=default_manifest
         )
         assert success is True

         start_time = time.time()
         completed = False
         while time.time() - start_time < 5.0:
             status = BootstrapManager.get_status()
             if status["status"] == "completed":
                 completed = True
                 break
             elif status["status"] == "failed":
                 pytest.fail(f"Bootstrap failed: {status['error_message']}")
             time.sleep(0.1)

         assert completed is True
         assert (vendor_dir / "manifest.json").is_file()
         assert (vendor_dir / "ffmpeg" / "ffmpeg.exe").is_file()
         assert (vendor_dir / "ffmpeg" / "ffprobe.exe").is_file()
         assert (vendor_dir / "yt-dlp" / "yt-dlp.exe").is_file()
         mock_qwen.assert_called_once_with(vendor_dir)
