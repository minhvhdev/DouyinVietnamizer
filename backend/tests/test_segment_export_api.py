from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock
import wave

from fastapi.testclient import TestClient
import pytest

from dv_backend.api import create_app
from dv_backend.checkpoints import load_checkpoint, save_checkpoint
from dv_backend.config import AppConfig
from dv_backend import pipeline
from dv_backend.segment_export_ops import SegmentArtifactUnavailableError


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 1600)


def make_completed_job(client: TestClient, tmp_path: Path) -> str:
    sample = tmp_path / "sample.mp4"
    sample.write_bytes(b"fake-video")
    job = client.app.state.jobs.create_imported(
        sample,
        original_filename="sample.mp4",
    )
    with client.app.state.database.connection:
        client.app.state.database.connection.execute(
            "UPDATE jobs SET status = 'completed' WHERE id = ?",
            (job.id,),
        )
    segments = []
    for index in range(2):
        raw = tmp_path / "jobs" / job.id / "artifacts" / "tts" / f"raw-{index}.wav"
        repaired = (
            tmp_path / "jobs" / job.id / "artifacts" / "tts" / f"repaired-{index}.wav"
        )
        write_wav(raw)
        write_wav(repaired)
        segments.append({
            "index": index,
            "start": float(index),
            "end": float(index + 1),
            "translation": f"Đoạn {index}",
            "text": f"原文 {index}",
            "tts_spoken_text": f"Đoạn {index}",
            "tts_raw_path": str(raw),
            "tts_path": str(repaired),
            "tts_duration": 0.1,
            "repaired_duration": 0.1,
        })
    save_checkpoint(tmp_path, job.id, "translate", {"segments": segments})
    save_checkpoint(tmp_path, job.id, "tts", {"segments": segments})
    save_checkpoint(tmp_path, job.id, "duration_repair", {"segments": segments})
    save_checkpoint(tmp_path, job.id, "align_final_dub", {"segments": segments})
    save_checkpoint(tmp_path, job.id, "mix", {"mixed_wav_path": "old.wav"})
    save_checkpoint(tmp_path, job.id, "render", {"output_path": "old.mp4"})
    save_checkpoint(tmp_path, job.id, "qc", {"output_video_path": "old.mp4"})
    return job.id


def edit_first_segment(client: TestClient, job_id: str) -> int:
    initial = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()
    draft = initial["draft_segments"]
    draft[0]["spoken_text"] = "Bản dịch mới"
    payload = {
        "expected_plan_version": initial["plan_version"],
        "segments": [
            {
                "segment_id": segment["segment_id"],
                "start_ms": segment["start_ms"],
                "end_ms": segment["end_ms"],
                "spoken_text": segment["spoken_text"],
            }
            for segment in draft
        ],
    }
    response = client.put(
        f"/api/jobs/{job_id}/segments/edit-plan",
        json=payload,
    )
    assert response.status_code == 200
    return response.json()["plan_version"]


def test_export_noop_returns_unchanged_without_runner(tmp_path: Path) -> None:
    client = TestClient(create_app(AppConfig(tmp_path)))
    job_id = make_completed_job(client, tmp_path)
    plan = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()
    start_job = Mock()
    client.app.state.runner.start_job = start_job

    response = client.post(
        f"/api/jobs/{job_id}/segments/export",
        json={"expected_plan_version": plan["plan_version"]},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "unchanged"
    start_job.assert_not_called()
    assert load_checkpoint(tmp_path, job_id, "segment_export_state") is None


def test_export_captures_version_and_schedules_only_downstream(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(AppConfig(tmp_path)))
    job_id = make_completed_job(client, tmp_path)
    plan_version = edit_first_segment(client, job_id)
    start_job = Mock()
    client.app.state.runner.start_job = start_job

    response = client.post(
        f"/api/jobs/{job_id}/segments/export",
        json={"expected_plan_version": plan_version},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    state = load_checkpoint(tmp_path, job_id, "segment_export_state")
    assert state["captured_plan_version"] == plan_version
    assert state["status"] == "queued"
    rows = client.app.state.database.connection.execute(
        "SELECT name, status FROM job_steps WHERE job_id = ?",
        (job_id,),
    ).fetchall()
    statuses = {row["name"]: row["status"] for row in rows}
    assert statuses["tts"] == "completed"
    assert statuses["duration_repair"] == "completed"
    assert statuses["align_final_dub"] == "pending"
    assert statuses["mix"] == "pending"
    assert statuses["render"] == "pending"
    assert statuses["qc"] == "pending"
    start_job.assert_called_once_with(job_id)


def test_export_rejects_stale_version_and_active_export(tmp_path: Path) -> None:
    client = TestClient(create_app(AppConfig(tmp_path)))
    job_id = make_completed_job(client, tmp_path)
    plan_version = edit_first_segment(client, job_id)
    client.app.state.runner.start_job = Mock()

    stale = client.post(
        f"/api/jobs/{job_id}/segments/export",
        json={"expected_plan_version": plan_version - 1},
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "plan_version_conflict"

    accepted = client.post(
        f"/api/jobs/{job_id}/segments/export",
        json={"expected_plan_version": plan_version},
    )
    assert accepted.status_code == 202
    active = client.post(
        f"/api/jobs/{job_id}/segments/export",
        json={"expected_plan_version": plan_version},
    )
    assert active.status_code == 409
    assert active.json()["error"]["code"] == "segment_export_in_progress"


def test_prepare_text_change_synthesizes_and_repairs_only_changed_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(AppConfig(tmp_path)))
    job_id = make_completed_job(client, tmp_path)
    initial = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()
    changed_id = initial["draft_segments"][0]["segment_id"]
    unchanged_id = initial["draft_segments"][1]["segment_id"]
    plan_version = edit_first_segment(client, job_id)
    client.app.state.runner.start_job = Mock()
    accepted = client.post(
        f"/api/jobs/{job_id}/segments/export",
        json={"expected_plan_version": plan_version},
    )
    assert accepted.status_code == 202

    tts_calls: list[list[str]] = []
    duration_calls: list[list[str]] = []

    def fake_tts(staging_id, config, database, runner):
        cp = load_checkpoint(tmp_path, staging_id, "translate")
        ids = [segment["segment_id"] for segment in cp["segments"]]
        tts_calls.append(ids)
        result = []
        for segment in cp["segments"]:
            raw = tmp_path / "staged" / f"{segment['segment_id']}-raw.wav"
            write_wav(raw)
            result.append({**segment, "tts_raw_path": str(raw), "tts_path": str(raw)})
        return {"segments": result}

    def fake_duration(staging_id, config, database, runner):
        cp = load_checkpoint(tmp_path, staging_id, "tts")
        ids = [segment["segment_id"] for segment in cp["segments"]]
        duration_calls.append(ids)
        result = []
        for segment in cp["segments"]:
            repaired = tmp_path / "staged" / f"{segment['segment_id']}-repaired.wav"
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

    result = client.app.state.segment_exports.prepare_pending(job_id)

    assert result["status"] == "prepared"
    assert tts_calls == [[changed_id]]
    assert duration_calls == [[changed_id]]
    tts_cp = load_checkpoint(tmp_path, job_id, "tts")
    duration_cp = load_checkpoint(tmp_path, job_id, "duration_repair")
    assert [segment["segment_id"] for segment in tts_cp["segments"]] == [
        changed_id,
        unchanged_id,
    ]
    assert [segment["segment_id"] for segment in duration_cp["segments"]] == [
        changed_id,
        unchanged_id,
    ]
    assert duration_cp["segments"][1]["tts_raw_path"].endswith("raw-1.wav")


def test_prepare_timing_only_reuses_raw_and_repairs_one_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(AppConfig(tmp_path)))
    job_id = make_completed_job(client, tmp_path)
    initial = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()
    changed_id = initial["draft_segments"][0]["segment_id"]
    payload_segments = []
    for position, segment in enumerate(initial["draft_segments"]):
        payload_segments.append(
            {
                "segment_id": segment["segment_id"],
                "start_ms": segment["start_ms"] + (100 if position == 0 else 0),
                "end_ms": segment["end_ms"] + (100 if position == 0 else 0),
                "spoken_text": segment["spoken_text"],
            }
        )
    saved = client.put(
        f"/api/jobs/{job_id}/segments/edit-plan",
        json={
            "expected_plan_version": initial["plan_version"],
            "segments": payload_segments,
        },
    ).json()
    client.app.state.runner.start_job = Mock()
    client.post(
        f"/api/jobs/{job_id}/segments/export",
        json={"expected_plan_version": saved["plan_version"]},
    )
    tts = Mock(side_effect=AssertionError("timing-only must not synthesize"))
    duration_ids: list[list[str]] = []

    def fake_duration(staging_id, config, database, runner):
        cp = load_checkpoint(tmp_path, staging_id, "tts")
        duration_ids.append([segment["segment_id"] for segment in cp["segments"]])
        result = []
        for segment in cp["segments"]:
            repaired = tmp_path / "staged" / f"{segment['segment_id']}-timing.wav"
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

    monkeypatch.setattr(pipeline, "tts_step", tts)
    monkeypatch.setattr(pipeline, "duration_repair_step", fake_duration)

    client.app.state.segment_exports.prepare_pending(job_id)

    tts.assert_not_called()
    assert duration_ids == [[changed_id]]
    target = load_checkpoint(tmp_path, job_id, "duration_repair")["segments"][0]
    assert target["tts_raw_path"].endswith("raw-0.wav")
    assert target["start"] == pytest.approx(0.1)


def test_prepare_delete_and_reorder_calls_no_tts_or_duration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(AppConfig(tmp_path)))
    job_id = make_completed_job(client, tmp_path)
    initial = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()
    kept = initial["draft_segments"][1]
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
    tts = Mock(side_effect=AssertionError("delete must not synthesize"))
    duration = Mock(side_effect=AssertionError("delete must not repair"))
    monkeypatch.setattr(pipeline, "tts_step", tts)
    monkeypatch.setattr(pipeline, "duration_repair_step", duration)

    client.app.state.segment_exports.prepare_pending(job_id)

    tts.assert_not_called()
    duration.assert_not_called()
    target = load_checkpoint(tmp_path, job_id, "duration_repair")["segments"]
    assert [segment["segment_id"] for segment in target] == [kept["segment_id"]]
    assert target[0]["index"] == 0


def test_prepare_fails_when_reusable_raw_wav_is_missing(
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
    raw.unlink()
    monkeypatch.setattr(
        pipeline,
        "tts_step",
        Mock(side_effect=AssertionError("reorder must not synthesize")),
    )
    monkeypatch.setattr(
        pipeline,
        "duration_repair_step",
        Mock(side_effect=AssertionError("reorder must not repair")),
    )

    with pytest.raises(SegmentArtifactUnavailableError):
        client.app.state.segment_exports.prepare_pending(job_id)


def test_finalize_success_promotes_applied_without_incrementing_plan_version(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(AppConfig(tmp_path)))
    job_id = make_completed_job(client, tmp_path)
    plan_version = edit_first_segment(client, job_id)
    client.app.state.runner.start_job = Mock()
    client.post(
        f"/api/jobs/{job_id}/segments/export",
        json={"expected_plan_version": plan_version},
    )
    state = load_checkpoint(tmp_path, job_id, "segment_export_state")
    state["status"] = "prepared"
    state["candidate_segments"] = [{"segment_id": "x"}]
    save_checkpoint(tmp_path, job_id, "segment_export_state", state)

    result = client.app.state.segment_exports.finalize_success(job_id)

    assert result["status"] == "succeeded"
    plan = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()
    assert plan["plan_version"] == plan_version
    assert plan["applied_plan_version"] == plan_version
    assert plan["diff"]["has_changes"] is False
    assert (
        load_checkpoint(tmp_path, job_id, "segment_export_state")["status"]
        == "succeeded"
    )


def test_finalize_failure_restores_output_and_keeps_applied(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(AppConfig(tmp_path)))
    job_id = make_completed_job(client, tmp_path)
    output = tmp_path / "jobs" / job_id / "output" / "dubbed.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"old-output")
    plan_version = edit_first_segment(client, job_id)
    client.app.state.runner.start_job = Mock()
    client.post(
        f"/api/jobs/{job_id}/segments/export",
        json={"expected_plan_version": plan_version},
    )
    state = load_checkpoint(tmp_path, job_id, "segment_export_state")
    backup_path = Path(state["previous_output_path"])
    assert backup_path.is_file()
    assert backup_path.read_bytes() == b"old-output"
    output.write_bytes(b"corrupt-output")
    save_checkpoint(
        tmp_path,
        job_id,
        "tts",
        {"segments": [{"index": 9, "translation": "corrupt"}]},
    )

    result = client.app.state.segment_exports.finalize_failure(
        job_id, error="ALIGN_FAILED"
    )

    assert result["status"] == "failed"
    assert output.read_bytes() == b"old-output"
    assert (
        load_checkpoint(tmp_path, job_id, "tts")["segments"][0]["translation"]
        == "Đoạn 0"
    )
    plan = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()
    assert plan["applied_plan_version"] != plan["plan_version"]
    assert plan["diff"]["has_changes"] is True
