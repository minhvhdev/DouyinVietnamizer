from pathlib import Path

from dv_backend.checkpoints import save_checkpoint
from dv_backend.config import AppConfig
from dv_backend.database import Database
from dv_backend.pipeline import qc_step


def test_qc_writes_json_and_html_reports(tmp_path: Path) -> None:
    config = AppConfig(tmp_path)
    config.ensure_directories()
    database = Database(config.database_path)
    database.migrate()
    job_id = "job-qc"
    output = tmp_path / "jobs" / job_id / "output" / "dubbed.mp4"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"video")
    save_checkpoint(tmp_path, job_id, "normalize_segments", {"segments": [{"index": 0}]})
    save_checkpoint(tmp_path, job_id, "duration_repair", {
        "segments": [{
            "index": 0,
            "translation": "Xin chao",
            "repaired_method": "time_stretch_1.2x",
            "duration_budget": 1.0,
            "repaired_duration": 1.1,
        }]
    })
    save_checkpoint(tmp_path, job_id, "render", {"output_path": str(output)})

    report = qc_step(job_id, config, database, runner=None)

    assert report["warnings"]
    assert (tmp_path / "jobs" / job_id / "artifacts" / "qc_report.json").is_file()
    html = (tmp_path / "jobs" / job_id / "artifacts" / "qc_report.html").read_text(encoding="utf-8")
    assert "Douyin Vietnamizer QC Report" in html
    assert "time_stretch_1.2x" in html
