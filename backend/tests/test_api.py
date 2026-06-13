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
    assert capabilities.json()["cpu_mode"] is True
    assert len(capabilities.json()["implemented_steps"]) == 12


def test_create_and_list_job(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    created = client.post("/api/jobs", json={"source_url": "https://v.douyin.com/demo/"})
    listed = client.get("/api/jobs")

    assert created.status_code == 201
    assert listed.json()[0]["id"] == created.json()["id"]
    assert len(created.json()["steps"]) == 12


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
