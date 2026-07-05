from dv_backend import runtime


class _DummyRunner:
    def kill_managed_processes(self) -> list[str]:
        return ["job-1:1234"]


class _DummyGpuManager:
    def reset(self) -> dict[str, object]:
        return {"resident_models": [{"family": "asr", "device": "cuda:0", "model": "Qwen/Qwen3-ASR-1.7B"}]}

    def snapshot(self) -> dict[str, object]:
        return {"resident_models": [], "lease_history_size": 0, "eviction_count": 0}


def test_release_vram_resources_runs_cleanup(monkeypatch) -> None:
    from dv_backend.adapters import asr, voxcpm_client
    from dv_backend import gpu_manager

    calls: list[str] = []

    monkeypatch.setattr(voxcpm_client, "release_all_clients", lambda: calls.append("clients"))
    monkeypatch.setattr(asr, "reset_model_cache", lambda: calls.append("asr"))
    monkeypatch.setattr(gpu_manager, "global_gpu_manager", lambda: _DummyGpuManager())
    monkeypatch.setattr(runtime, "_terminate_gpu_helper_processes", lambda: ["llama-tts-server.exe (42)"])
    monkeypatch.setattr(runtime, "_clear_torch_cuda_state", lambda: calls.append("torch"))
    monkeypatch.setattr(
        runtime,
        "collect_runtime_gpu_status",
        lambda: runtime.RuntimeGpuStatus(
            cuda_supported=False,
            active_voxcpm_clients=0,
            resident_models=[],
            helper_processes=[],
        ),
    )

    result = runtime.release_vram_resources(runner=_DummyRunner())

    assert result.status == "ok"
    assert set(result.released) >= {
        "managed_job_processes",
        "voxcpm_clients",
        "qwen3_asr_cache",
        "gpu_manager_state",
        "gpu_helper_processes",
        "torch_cuda_cache",
    }
    assert result.terminated_processes == ["job-1:1234", "llama-tts-server.exe (42)"]
    assert calls == ["clients", "asr", "torch"]
