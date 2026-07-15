from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from dv_backend.api import create_app
from dv_backend.checkpoints import load_checkpoint, save_checkpoint
from dv_backend.config import AppConfig
from dv_backend import pipeline
from dv_backend.segment_export_ops import (
    ACTIVE_EXPORT_STATUSES,
    SegmentArtifactUnavailableError,
)
import pytest

from test_segment_export_api import (
    edit_first_segment,
    make_completed_job,
    write_wav,
)


def _queue_export_with_output(client: TestClient, tmp_path: Path) -> tuple[str, int, Path]:
    job_id = make_completed_job(client, tmp_path)
    output = tmp_path / "jobs" / job_id / "output" / "dubbed.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"previous-output")
    plan_version = edit_first_segment(client, job_id)
    client.app.state.runner.start_job = Mock()
    response = client.post(
        f"/api/jobs/{job_id}/segments/export",
        json={"expected_plan_version": plan_version},
    )
    assert response.status_code == 202
    return job_id, plan_version, output


def test_recover_interrupted_active_export_restores_and_allows_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(AppConfig(tmp_path)))
    job_id, plan_version, output = _queue_export_with_output(client, tmp_path)

    # Simulate crash after prepare: checkpoints rewritten, export still active.
    state = load_checkpoint(tmp_path, job_id, "segment_export_state")
    assert state["status"] in ACTIVE_EXPORT_STATUSES
    staging_id = f"{job_id}__segment_export_{plan_version}"
    staging_dir = tmp_path / "jobs" / staging_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "marker.txt").write_text("staged", encoding="utf-8")

    def fake_tts(staging_job_id, config, database, runner):
        cp = load_checkpoint(tmp_path, staging_job_id, "translate")
        result = []
        for segment in cp["segments"]:
            raw = staging_dir / f"{segment['segment_id']}-raw.wav"
            write_wav(raw)
            result.append(
                {**segment, "tts_raw_path": str(raw), "tts_path": str(raw)}
            )
        return {"segments": result}

    def fake_duration(staging_job_id, config, database, runner):
        cp = load_checkpoint(tmp_path, staging_job_id, "tts")
        result = []
        for segment in cp["segments"]:
            repaired = staging_dir / f"{segment['segment_id']}-repaired.wav"
            write_wav(repaired)
            result.append(
                {
                    **segment,
                    "tts_path": str(repaired),
                    "repaired_duration": 0.1,
                    "placement_start": segment["start"],
                }
            )
        return {"segments": result, "release_eligible": True}

    monkeypatch.setattr(pipeline, "tts_step", fake_tts)
    monkeypatch.setattr(pipeline, "duration_repair_step", fake_duration)
    client.app.state.segment_exports.prepare_pending(job_id)

    state = load_checkpoint(tmp_path, job_id, "segment_export_state")
    assert state["status"] == "prepared"
    state["status"] = "running"
    save_checkpoint(tmp_path, job_id, "segment_export_state", state)

    # Corrupt live output + rewrite checkpoints as if mid-render crash.
    output.write_bytes(b"partial-new-output")
    save_checkpoint(
        tmp_path,
        job_id,
        "tts",
        {"segments": [{"index": 99, "translation": "corrupt-mid-export"}]},
    )
    with client.app.state.database.connection:
        client.app.state.database.connection.execute(
            "UPDATE jobs SET status = 'interrupted', current_step = 'mix' WHERE id = ?",
            (job_id,),
        )

    recovered = client.app.state.segment_exports.recover_interrupted(job_id)

    assert recovered["status"] == "failed"
    assert output.read_bytes() == b"previous-output"
    assert (
        load_checkpoint(tmp_path, job_id, "tts")["segments"][0]["translation"]
        == "Đoạn 0"
    )
    export_state = load_checkpoint(tmp_path, job_id, "segment_export_state")
    assert export_state["status"] == "failed"
    plan = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()
    assert plan["diff"]["has_changes"] is True
    assert plan["plan_version"] == plan_version
    assert plan["applied_plan_version"] != plan_version

    # Retry export must be allowed after recovery.
    client.app.state.runner.start_job = Mock()
    retry = client.post(
        f"/api/jobs/{job_id}/segments/export",
        json={"expected_plan_version": plan_version},
    )
    assert retry.status_code == 202
    assert retry.json()["status"] == "queued"


def test_recover_interrupted_scans_orphaned_active_exports_on_startup(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(AppConfig(tmp_path)))
    job_id, _plan_version, output = _queue_export_with_output(client, tmp_path)
    state = load_checkpoint(tmp_path, job_id, "segment_export_state")
    state["status"] = "prepared"
    save_checkpoint(tmp_path, job_id, "segment_export_state", state)
    save_checkpoint(
        tmp_path,
        job_id,
        "duration_repair",
        {"segments": [{"index": 0, "translation": "half-prepared"}]},
    )
    with client.app.state.database.connection:
        client.app.state.database.connection.execute(
            "UPDATE jobs SET status = 'completed', current_step = NULL WHERE id = ?",
            (job_id,),
        )

    results = client.app.state.segment_exports.recover_interrupted()

    assert any(item.get("job_id") == job_id for item in results)
    assert output.read_bytes() == b"previous-output"
    assert (
        load_checkpoint(tmp_path, job_id, "duration_repair")["segments"][0][
            "translation"
        ]
        == "Đoạn 0"
    )
    assert (
        load_checkpoint(tmp_path, job_id, "segment_export_state")["status"]
        == "failed"
    )


def test_prepare_fails_when_reusable_wav_is_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(AppConfig(tmp_path)))
    job_id = make_completed_job(client, tmp_path)
    initial = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()
    reversed_segments = list(reversed(initial["draft_segments"]))
    saved = client.put(
        f"/api/jobs/{job_id}/segments/edit-plan",
        json={
            "expected_plan_version": initial["plan_version"],
            "segments": [
                {
                    "segment_id": segment["segment_id"],
                    "start_ms": segment["start_ms"],
                    "end_ms": segment["end_ms"],
                    "spoken_text": segment["spoken_text"],
                }
                for segment in reversed_segments
            ],
        },
    ).json()
    client.app.state.runner.start_job = Mock()
    client.post(
        f"/api/jobs/{job_id}/segments/export",
        json={"expected_plan_version": saved["plan_version"]},
    )
    raw = Path(
        load_checkpoint(tmp_path, job_id, "duration_repair")["segments"][0][
            "tts_raw_path"
        ]
    )
    raw.write_bytes(b"")
    monkeypatch.setattr(
        pipeline,
        "tts_step",
        Mock(side_effect=AssertionError("must not full-TTS on empty reusable wav")),
    )
    monkeypatch.setattr(
        pipeline,
        "duration_repair_step",
        Mock(side_effect=AssertionError("must not repair on empty reusable wav")),
    )

    with pytest.raises(SegmentArtifactUnavailableError):
        client.app.state.segment_exports.prepare_pending(job_id)


def test_prepare_fails_when_manifest_path_mismatches_available_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(AppConfig(tmp_path)))
    job_id = make_completed_job(client, tmp_path)
    initial = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()
    kept = initial["draft_segments"][1]
    duration_cp = load_checkpoint(tmp_path, job_id, "duration_repair")
    reused = duration_cp["segments"][1]
    save_checkpoint(
        tmp_path,
        job_id,
        "segment_audio_manifest",
        {
            "schema_version": 1,
            "job_id": job_id,
            "plan_version": 1,
            "entries": {
                kept["segment_id"]: {
                    "segment_id": kept["segment_id"],
                    "index": 1,
                    "raw_wav_path": str(tmp_path / "missing-manifest-raw.wav"),
                    "repaired_wav_path": reused["tts_path"],
                    "spoken_text": reused["translation"],
                    "start_ms": 1000,
                    "end_ms": 2000,
                }
            },
        },
    )
    saved = client.put(
        f"/api/jobs/{job_id}/segments/edit-plan",
        json={
            "expected_plan_version": initial["plan_version"],
            "segments": [
                {
                    "segment_id": kept["segment_id"],
                    "start_ms": kept["start_ms"],
                    "end_ms": kept["end_ms"],
                    "spoken_text": kept["spoken_text"],
                }
            ],
        },
    ).json()
    client.app.state.runner.start_job = Mock()
    client.post(
        f"/api/jobs/{job_id}/segments/export",
        json={"expected_plan_version": saved["plan_version"]},
    )
    monkeypatch.setattr(
        pipeline,
        "tts_step",
        Mock(side_effect=AssertionError("must not full-TTS on manifest mismatch")),
    )
    monkeypatch.setattr(
        pipeline,
        "duration_repair_step",
        Mock(side_effect=AssertionError("must not repair on manifest mismatch")),
    )

    with pytest.raises(SegmentArtifactUnavailableError):
        client.app.state.segment_exports.prepare_pending(job_id)


def test_finalize_success_cleans_staging_dir_keeps_applied_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(AppConfig(tmp_path)))
    job_id = make_completed_job(client, tmp_path)
    output = tmp_path / "jobs" / job_id / "output" / "dubbed.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"applied-output")
    plan_version = edit_first_segment(client, job_id)
    client.app.state.runner.start_job = Mock()
    client.post(
        f"/api/jobs/{job_id}/segments/export",
        json={"expected_plan_version": plan_version},
    )

    staging_id = f"{job_id}__segment_export_{plan_version}"
    staging_dir = tmp_path / "jobs" / staging_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "leftover.txt").write_text("cleanup-me", encoding="utf-8")
    applied_raw = Path(
        load_checkpoint(tmp_path, job_id, "tts")["segments"][0]["tts_raw_path"]
    )
    assert applied_raw.is_file()

    def fake_tts(staging_job_id, config, database, runner):
        cp = load_checkpoint(tmp_path, staging_job_id, "translate")
        result = []
        for segment in cp["segments"]:
            raw = staging_dir / f"{segment['segment_id']}-raw.wav"
            write_wav(raw)
            result.append(
                {**segment, "tts_raw_path": str(raw), "tts_path": str(raw)}
            )
        return {"segments": result}

    def fake_duration(staging_job_id, config, database, runner):
        cp = load_checkpoint(tmp_path, staging_job_id, "tts")
        result = []
        for segment in cp["segments"]:
            repaired = staging_dir / f"{segment['segment_id']}-repaired.wav"
            write_wav(repaired)
            result.append(
                {
                    **segment,
                    "tts_path": str(repaired),
                    "repaired_duration": 0.1,
                    "placement_start": segment["start"],
                }
            )
        return {"segments": result, "release_eligible": True}

    monkeypatch.setattr(pipeline, "tts_step", fake_tts)
    monkeypatch.setattr(pipeline, "duration_repair_step", fake_duration)
    client.app.state.segment_exports.prepare_pending(job_id)

    # Successful export writes a new output; applied WAV from prior pipeline must remain.
    output.write_bytes(b"new-export-output")
    result = client.app.state.segment_exports.finalize_success(job_id)

    assert result["status"] == "succeeded"
    assert not staging_dir.exists()
    assert applied_raw.is_file()
    assert output.is_file()
    assert output.read_bytes() == b"new-export-output"
    backup = (
        tmp_path
        / "jobs"
        / job_id
        / "output"
        / f"dubbed.pre_export_v{plan_version}.mp4"
    )
    assert backup.is_file()
    assert backup.read_bytes() == b"applied-output"
