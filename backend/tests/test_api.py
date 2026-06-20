from pathlib import Path

from fastapi.testclient import TestClient

from dv_backend.api import create_app
from dv_backend.config import AppConfig


def make_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(AppConfig(tmp_path)))


def test_health_and_capabilities(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    health = client.get("/api/health")
    capabilities = client.get("/api/capabilities")

    assert health.json()["status"] == "ok"
    assert capabilities.json()["cpu_mode"] is False
    assert capabilities.json()["asr_backend"] == "qwen3_asr"
    assert len(capabilities.json()["implemented_steps"]) == 12


def test_create_and_list_job(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    created = client.post("/api/jobs", json={"source_url": "https://v.douyin.com/demo/"})
    listed = client.get("/api/jobs")

    assert created.status_code == 201
    assert listed.json()[0]["id"] == created.json()["id"]
    assert len(created.json()["steps"]) == 12


def test_delete_finished_job(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    app = client.app
    job_id = app.state.jobs.create("https://v.douyin.com/demo/").id
    with app.state.database.connection:
        app.state.database.connection.execute(
            "UPDATE jobs SET status = 'completed' WHERE id = ?",
            (job_id,),
        )

    response = client.delete(f"/api/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json() == {"status": "deleted"}
    assert all(job["id"] != job_id for job in client.get("/api/jobs").json())


def test_delete_running_job_is_rejected(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    app = client.app
    job_id = app.state.jobs.create("https://v.douyin.com/demo/").id
    with app.state.database.connection:
        app.state.database.connection.execute(
            "UPDATE jobs SET status = 'running' WHERE id = ?",
            (job_id,),
        )

    response = client.delete(f"/api/jobs/{job_id}")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "JOB_NOT_DELETABLE"


def test_invalid_url_returns_actionable_error(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.post("/api/jobs", json={"source_url": "https://example.com/video"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_DOUYIN_URL"
    assert response.json()["error"]["action"]
    assert response.json()["error"]["retryable"] is False


def test_settings_and_events(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    updated = client.put("/api/settings", json={"translation_api_base": "https://api.example.com/v1"})
    events = client.get("/api/events")

    assert updated.json()["translation_api_base"] == "https://api.example.com/v1"
    assert client.get("/api/settings").json()["translation_api_base"] == "https://api.example.com/v1"
    assert events.json()[0]["code"] == "SETTINGS_UPDATED"


def test_loopback_renderer_origin_is_allowed(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.options(
        "/api/jobs",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"




def test_invalid_url_returns_actionable_error(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.post("/api/jobs", json={"source_url": "https://example.com/video"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_DOUYIN_URL"
    assert response.json()["error"]["action"]
    assert response.json()["error"]["retryable"] is False


def test_settings_and_events(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    updated = client.put("/api/settings", json={"translation_api_base": "https://api.example.com/v1"})
    events = client.get("/api/events")

    assert updated.json()["translation_api_base"] == "https://api.example.com/v1"
    assert client.get("/api/settings").json()["translation_api_base"] == "https://api.example.com/v1"
    assert events.json()[0]["code"] == "SETTINGS_UPDATED"


def test_loopback_renderer_origin_is_allowed(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.options(
        "/api/jobs",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_runtime_status_and_smoke_test(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    status = client.get("/api/runtime/status")
    rerun = client.post("/api/runtime/smoke-test")

    assert status.status_code == 200
    assert status.json()["status"] in {"ready", "warning", "blocked"}
    assert rerun.json()["checks"]


def test_cloned_voices_crud(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    
    resp = client.get("/api/cloned-voices")
    assert resp.status_code == 200
    assert resp.json() == []
    
    wav_content = b"fake wav audio data"
    files = {"file": ("test_voice.wav", wav_content, "audio/wav")}
    data = {"name": "Giong Test"}
    
    resp = client.post("/api/cloned-voices", data=data, files=files)
    assert resp.status_code == 201
    created = resp.json()
    assert created["name"] == "Giong Test"
    assert created["id"] is not None
    assert created["wav_filename"] == f"{created['id']}.wav"
    
    voice_dir = tmp_path / "cloned_voices"
    saved_file = voice_dir / created["wav_filename"]
    assert saved_file.is_file()
    assert saved_file.read_bytes() == wav_content
    
    resp = client.get("/api/cloned-voices")
    assert len(resp.json()) == 1
    assert resp.json()[0]["id"] == created["id"]
    
    resp = client.get(f"/api/cloned-voices/{created['id']}/wav")
    assert resp.status_code == 200
    assert resp.content == wav_content
    
    resp = client.post("/api/cloned-voices", data={"name": "Giong Test"}, files={"file": ("t.wav", b"data")})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "VOICE_NAME_EXISTS"
    
    resp = client.delete(f"/api/cloned-voices/{created['id']}")
    assert resp.status_code == 200
    assert not saved_file.is_file()
    
    resp = client.get("/api/cloned-voices")
    assert resp.json() == []


def test_job_files(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    app = client.app
    job_id = app.state.jobs.create("https://v.douyin.com/demo/").id
    
    # Check empty list when no files exist yet
    resp = client.get(f"/api/jobs/{job_id}/files")
    assert resp.status_code == 200
    assert resp.json() == []
    
    # Create some mock files in the job directory
    job_dir = tmp_path / "jobs" / job_id
    output_dir = job_dir / "output"
    artifacts_dir = job_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    (output_dir / "dubbed.mp4").write_bytes(b"mock mp4 data")
    (artifacts_dir / "bgm.wav").write_bytes(b"mock bgm data")
    
    resp = client.get(f"/api/jobs/{job_id}/files")
    assert resp.status_code == 200
    files_list = resp.json()
    assert len(files_list) == 2
    
    keys = {f["key"] for f in files_list}
    assert "dubbed_video" in keys
    assert "bgm" in keys
    
    # Test streaming file content
    resp = client.get(f"/api/jobs/{job_id}/files/bgm")
    assert resp.status_code == 200
    assert resp.content == b"mock bgm data"
    
    # Test 404 for missing file key
    resp = client.get(f"/api/jobs/{job_id}/files/vocals")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "FILE_NOT_FOUND"
    
    # Test 400 for invalid file key
    resp = client.get(f"/api/jobs/{job_id}/files/invalid_key")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_FILE_KEY"

