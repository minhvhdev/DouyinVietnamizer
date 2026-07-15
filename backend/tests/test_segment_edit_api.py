from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from dv_backend.api import create_app
from dv_backend.checkpoints import load_checkpoint, save_checkpoint
from dv_backend.config import AppConfig


def make_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(AppConfig(tmp_path)))


def completed_job(client: TestClient, tmp_path: Path, *, count: int = 2) -> str:
    sample = tmp_path / "sample.mp4"
    sample.write_bytes(b"fake-video")
    job = client.app.state.jobs.create_imported(
        sample,
        original_filename="sample.mp4",
    )
    job_id = job.id
    with client.app.state.database.connection:
        client.app.state.database.connection.execute(
            "UPDATE jobs SET status = 'completed' WHERE id = ?",
            (job_id,),
        )
    save_checkpoint(
        tmp_path,
        job_id,
        "translate",
        {
            "segments": [
                {
                    "index": index,
                    "start": float(index),
                    "end": float(index + 1),
                    "translation": f"Đoạn {index}",
                    "text": f"原文 {index}",
                }
                for index in range(count)
            ]
        },
    )
    return job_id


def test_get_edit_plan_initializes_once_with_stable_ids(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    job_id = completed_job(client, tmp_path)

    first = client.get(f"/api/jobs/{job_id}/segments/edit-plan")
    second = client.get(f"/api/jobs/{job_id}/segments/edit-plan")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["plan_version"] == 1
    assert load_checkpoint(tmp_path, job_id, "segment_edit_plan") is not None


def test_get_edit_plan_rejects_unknown_and_non_completed_jobs(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    missing = client.get("/api/jobs/missing/segments/edit-plan")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "JOB_NOT_FOUND"

    sample = tmp_path / "queued.mp4"
    sample.write_bytes(b"fake")
    job = client.app.state.jobs.create_imported(sample, original_filename="queued.mp4")
    response = client.get(f"/api/jobs/{job.id}/segments/edit-plan")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "SEGMENT_EDIT_JOB_NOT_COMPLETED"


def test_put_edit_plan_supports_text_timing_add_delete_and_reorder(
    tmp_path: Path,
) -> None:
    client = make_client(tmp_path)
    job_id = completed_job(client, tmp_path, count=3)
    initial = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()
    first, _, third = initial["draft_segments"]

    response = client.put(
        f"/api/jobs/{job_id}/segments/edit-plan",
        json={
            "expected_plan_version": 1,
            "segments": [
                {
                    "segment_id": third["segment_id"],
                    "start_ms": 2100,
                    "end_ms": 2900,
                    "spoken_text": "Đoạn ba đã sửa",
                },
                {
                    "segment_id": None,
                    "start_ms": 3000,
                    "end_ms": 3500,
                    "spoken_text": "Đoạn mới",
                },
                {
                    "segment_id": first["segment_id"],
                    "start_ms": first["start_ms"],
                    "end_ms": first["end_ms"],
                    "spoken_text": first["spoken_text"],
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan_version"] == 2
    assert payload["applied_plan_version"] == 1
    assert payload["draft_segments"][0]["segment_id"] == third["segment_id"]
    assert payload["draft_segments"][1]["origin"] == "user"
    assert payload["draft_segments"][1]["segment_id"]
    assert payload["diff"]["has_changes"] is True
    assert payload["diff"]["structural_changed"] is True


def test_put_noop_does_not_increment_version(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    job_id = completed_job(client, tmp_path, count=1)
    initial = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()
    row = initial["draft_segments"][0]

    response = client.put(
        f"/api/jobs/{job_id}/segments/edit-plan",
        json={
            "expected_plan_version": 1,
            "segments": [
                {
                    "segment_id": row["segment_id"],
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "spoken_text": row["spoken_text"],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["plan_version"] == 1


def test_put_stale_version_returns_structured_conflict(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    job_id = completed_job(client, tmp_path, count=1)
    row = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()[
        "draft_segments"
    ][0]

    response = client.put(
        f"/api/jobs/{job_id}/segments/edit-plan",
        json={
            "expected_plan_version": 0,
            "segments": [
                {
                    "segment_id": row["segment_id"],
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "spoken_text": row["spoken_text"],
                }
            ],
        },
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "SEGMENT_EDIT_VERSION_CONFLICT"


def test_put_unknown_id_and_invalid_timing_do_not_write(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    job_id = completed_job(client, tmp_path, count=1)
    initial = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()
    stored_before = load_checkpoint(tmp_path, job_id, "segment_edit_plan")

    unknown = client.put(
        f"/api/jobs/{job_id}/segments/edit-plan",
        json={
            "expected_plan_version": 1,
            "segments": [
                {
                    "segment_id": "invented",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "spoken_text": "X",
                }
            ],
        },
    )
    invalid = client.put(
        f"/api/jobs/{job_id}/segments/edit-plan",
        json={
            "expected_plan_version": 1,
            "segments": [
                {
                    "segment_id": initial["draft_segments"][0]["segment_id"],
                    "start_ms": 1000,
                    "end_ms": 1000,
                    "spoken_text": "X",
                }
            ],
        },
    )

    assert unknown.status_code == 422, unknown.text
    assert unknown.json()["error"]["code"] == "SEGMENT_EDIT_UNKNOWN_ID"
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "SEGMENT_EDIT_INVALID"
    assert load_checkpoint(tmp_path, job_id, "segment_edit_plan") == stored_before


def test_put_rejects_client_provenance_fields(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    job_id = completed_job(client, tmp_path, count=1)

    response = client.put(
        f"/api/jobs/{job_id}/segments/edit-plan",
        json={
            "expected_plan_version": 1,
            "segments": [
                {
                    "segment_id": None,
                    "start_ms": 0,
                    "end_ms": 1000,
                    "spoken_text": "X",
                    "origin": "pipeline",
                    "source_segment_index": 42,
                }
            ],
        },
    )

    assert response.status_code == 422


@patch("dv_backend.api.JobRunner.start_job")
def test_put_has_no_pipeline_or_output_side_effects(
    start_job,
    tmp_path: Path,
) -> None:
    client = make_client(tmp_path)
    job_id = completed_job(client, tmp_path, count=1)
    output = tmp_path / "jobs" / job_id / "output" / "dubbed.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"old-output")
    initial = client.get(f"/api/jobs/{job_id}/segments/edit-plan").json()
    row = initial["draft_segments"][0]
    start_job.reset_mock()

    response = client.put(
        f"/api/jobs/{job_id}/segments/edit-plan",
        json={
            "expected_plan_version": 1,
            "segments": [
                {
                    "segment_id": row["segment_id"],
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "spoken_text": "Changed",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert client.app.state.jobs.get(job_id).status == "completed"
    assert output.read_bytes() == b"old-output"
    start_job.assert_not_called()
