import json
import multiprocessing
import subprocess
import wave
from pathlib import Path

import pytest


def _hold_gpu_lease(lock_dir: str, ready, release) -> None:
    from dv_backend.gpu_lease import gpu_lease

    with gpu_lease("holder", device="cuda:0", lock_dir=Path(lock_dir)):
        ready.set()
        release.wait(5)


def _signal_after_gpu_lease(lock_dir: str, ready) -> None:
    from dv_backend.gpu_lease import gpu_lease

    with gpu_lease("waiter", device="cuda:0", lock_dir=Path(lock_dir)):
        ready.set()


def write_wav(path: Path, *, duration: float = 1.0, sample_rate: int = 16000, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * int(duration * sample_rate * channels))


def test_telemetry_sink_writes_jsonl_and_ignores_metric_errors(tmp_path: Path) -> None:
    from dv_backend.telemetry import TelemetrySink

    class BadValue:
        pass

    sink = TelemetrySink(tmp_path, "job-1")
    sink.record("vad", {"audio_duration_sec": 2.0, "wall_time_ms": 1000})
    sink.record("asr", {"bad": BadValue()})

    lines = (tmp_path / "jobs" / "job-1" / "artifacts" / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["job_id"] == "job-1"
    assert payload["step"] == "vad"
    assert payload["real_time_factor"] == 0.5


def test_measure_step_records_wall_time_even_when_body_raises(tmp_path: Path) -> None:
    from dv_backend.telemetry import TelemetrySink

    sink = TelemetrySink(tmp_path, "job-1")
    with pytest.raises(RuntimeError):
        with sink.measure("tts", audio_duration_sec=1.0):
            raise RuntimeError("boom")

    payload = json.loads((tmp_path / "jobs" / "job-1" / "artifacts" / "telemetry.jsonl").read_text(encoding="utf-8"))
    assert payload["step"] == "tts"
    assert payload["status"] == "failed"
    assert payload["wall_time_ms"] >= 0


def test_telemetry_sink_rotates_when_max_exceeded(tmp_path: Path) -> None:
    from dv_backend.telemetry import TelemetrySink

    sink = TelemetrySink(tmp_path, "job-1", max_file_bytes=64)
    for index in range(20):
        sink.record("vad", {"audio_duration_sec": float(index), "wall_time_ms": 1000.0})
    rotated = sink.path.with_suffix(sink.path.suffix + ".1")
    assert rotated.exists()
    assert 0 < sink.path.stat().st_size < rotated.stat().st_size + 64


def test_telemetry_sink_redacts_sensitive_fields(tmp_path: Path) -> None:
    from dv_backend.telemetry import TelemetrySink

    sink = TelemetrySink(tmp_path, "job-1")
    sink.record(
        "tts",
        {
            "audio_duration_sec": 1.0,
            "wall_time_ms": 100.0,
            "text": "private transcript",
            "api_key": "sk-secret",
        },
    )
    payload = json.loads(sink.path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["text"] == "[redacted]"
    assert payload["api_key"] == "[redacted]"
    assert payload["audio_duration_sec"] == 1.0


def test_probe_duration_prefers_ffprobe_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dv_backend.audio_probe import get_audio_duration

    audio = tmp_path / "audio.wav"
    write_wav(audio, duration=1.0)

    def fake_run(cmd, capture_output, text, encoding, errors, timeout, check):
        assert "ffprobe" in str(cmd[0])
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"format": {"duration": "2.345"}}), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert get_audio_duration(audio, ffprobe_path=Path("ffprobe")) == pytest.approx(2.345)


def test_probe_duration_falls_back_to_wav_header(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dv_backend.audio_probe import get_audio_duration

    audio = tmp_path / "audio.wav"
    write_wav(audio, duration=1.25)

    def fake_run(*_args, **_kwargs):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert get_audio_duration(audio, ffprobe_path=Path("ffprobe")) == pytest.approx(1.25)


def test_probe_path_resolution_prefers_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dv_backend.audio_probe import resolved_probe_path

    explicit = tmp_path / "ffprobe.exe"
    explicit.write_text("probe")
    monkeypatch.setenv("DV_FFPROBE_PATH", str(explicit))
    resolved = resolved_probe_path(None)
    assert resolved is not None
    assert resolved.suffix == ".exe"


def test_gpu_lease_serializes_across_processes(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    holder_ready = ctx.Event()
    release_holder = ctx.Event()
    waiter_ready = ctx.Event()
    holder = ctx.Process(target=_hold_gpu_lease, args=(str(tmp_path), holder_ready, release_holder))
    waiter = ctx.Process(target=_signal_after_gpu_lease, args=(str(tmp_path), waiter_ready))
    holder.start()
    try:
        assert holder_ready.wait(5)
        waiter.start()
        assert not waiter_ready.wait(0.3)
        release_holder.set()
        assert waiter_ready.wait(5)
    finally:
        release_holder.set()
        holder.join(5)
        waiter.join(5)
        if holder.is_alive():
            holder.terminate()
        if waiter.is_alive():
            waiter.terminate()
    assert holder.exitcode == 0
    assert waiter.exitcode == 0


def test_gpu_manager_serializes_same_device_and_tracks_warm_start(tmp_path: Path) -> None:
    from dv_backend.gpu_manager import GpuModelManager

    manager = GpuModelManager(lock_dir=tmp_path)
    first = manager.acquire("asr", "cuda:0", "model-a")
    with first as lease:
        assert lease.cold_start is True
        assert lease.queue_wait_ms >= 0
    with manager.acquire("asr", "cuda:0", "model-a") as lease:
        assert lease.cold_start is False
    manager.evict("asr", "cuda:0", reason="oom")
    with manager.acquire("asr", "cuda:0", "model-a") as lease:
        assert lease.cold_start is True
        assert manager.evictions[-1]["reason"] == "oom"


def test_gpu_manager_loader_failure_releases_lock() -> None:
    from dv_backend.gpu_manager import GpuModelManager

    manager = GpuModelManager()

    def boom() -> None:
        raise RuntimeError("no model")

    with pytest.raises(RuntimeError):
        with manager.acquire("asr", "cuda:0", "model-a", loader=boom):
            pass
    # lock should be released; next acquire should not hang
    with manager.acquire("asr", "cuda:0", "model-a") as lease:
        assert lease.cold_start is True


def test_gpu_manager_max_resident_evicts_oldest() -> None:
    from dv_backend.gpu_manager import GpuModelManager

    manager = GpuModelManager()
    manager.max_resident_families = 1
    with manager.acquire("asr", "cuda:0", "model-a"):
        pass
    with manager.acquire("tts", "cuda:0", "model-b"):
        pass
    assert any(ev["reason"] == "max_resident_exceeded" for ev in manager.evictions)


def test_gpu_manager_keeps_warm_when_enabled() -> None:
    from dv_backend.gpu_manager import GpuModelManager

    manager = GpuModelManager()
    manager.keep_warm = False
    with manager.acquire("asr", "cuda:0", "model-a"):
        pass
    assert ("asr", "cuda:0") not in manager._loaded  # type: ignore[attr-defined]


def test_gpu_manager_records_vram_when_torch_available(monkeypatch: pytest.MonkeyPatch) -> None:
    from dv_backend import gpu_manager

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def memory_allocated() -> int:
            return 1024 * 1024 * 256

    class FakeTorch:
        cuda = FakeCuda

    monkeypatch.setattr(gpu_manager, "_torch", FakeTorch)
    lease = gpu_manager.GpuModelManager().acquire("asr", "cuda:0", "m")
    assert lease.vram_after_mb == 256.0


def test_sparse_asr_builds_chunks_and_rebases_timestamps() -> None:
    from dv_backend.sparse_asr import build_sparse_chunks, rebase_sparse_segments

    chunks = build_sparse_chunks(
        [{"start": 1.0, "end": 2.0}, {"start": 2.1, "end": 3.0}, {"start": 10.0, "end": 11.0}],
        total_duration=20.0,
        merge_gap_sec=0.25,
        padding_sec=0.2,
        max_chunk_sec=30.0,
    )
    assert chunks == [
        {"source_start": 0.8, "source_end": 3.2},
        {"source_start": 9.8, "source_end": 11.2},
    ]
    rebased = rebase_sparse_segments(chunks[0], [{"start": 0.1, "end": 0.9, "text": "你好"}])
    assert rebased == [{"start": 0.9, "end": 1.7, "text": "你好"}]


def test_build_stitched_timeline_uses_contiguous_stitched_spans() -> None:
    from dv_backend.sparse_asr import build_stitched_timeline, stitched_timeline_duration

    timeline = build_stitched_timeline([
        {"source_start": 0.8, "source_end": 3.2},
        {"source_start": 0.0, "source_end": 0.0},
        {"source_start": 9.8, "source_end": 11.2},
    ])
    assert timeline == [
        {
            "source_start": 0.8,
            "source_end": 3.2,
            "stitched_start": 0.0,
            "stitched_end": 2.4,
        },
        {
            "source_start": 9.8,
            "source_end": 11.2,
            "stitched_start": 2.4,
            "stitched_end": 3.8,
        },
    ]
    assert stitched_timeline_duration(timeline) == 3.8


def test_map_stitched_segments_to_source_handles_cross_chunk_timestamps() -> None:
    from dv_backend.sparse_asr import (
        build_stitched_timeline,
        map_stitched_segments_to_source,
    )

    timeline = build_stitched_timeline([
        {"source_start": 0.8, "source_end": 3.2},
        {"source_start": 9.8, "source_end": 11.2},
    ])
    mapped = map_stitched_segments_to_source(
        timeline,
        [
            {"start": 0.2, "end": 2.0, "text": "first"},
            {"start": 2.0, "end": 3.4, "text": "cross"},
            {"start": 2.6, "end": 3.0, "text": "second_tail"},
        ],
    )
    assert mapped == [
        {"start": 1.0, "end": 2.8, "text": "first"},
        {"start": 2.8, "end": 10.8, "text": "cross"},
        {"start": 10.0, "end": 10.4, "text": "second_tail"},
    ]


def test_sparse_asr_falls_back_when_vad_quality_is_suspicious() -> None:
    from dv_backend.sparse_asr import should_use_sparse_asr

    decision = should_use_sparse_asr(
        [{"start": 0.0, "end": 0.1} for _ in range(100)],
        total_duration=10.0,
        min_silence_ratio=0.2,
    )
    assert decision.use_sparse is False
    assert "fragment" in decision.reason


def test_semantic_split_prefers_aligned_punctuation_near_vad_boundary() -> None:
    from dv_backend.segmentation import split_segment_semantically

    segment = {"start": 0.0, "end": 6.0, "text": "你好世界。今天去北京123号。"}
    aligned_units = [
        {"text": "你好世界。", "start": 0.0, "end": 2.9},
        {"text": "今天去北京123号。", "start": 3.0, "end": 6.0},
    ]
    parts = split_segment_semantically(segment, [{"start": 0.0, "end": 3.1}, {"start": 3.2, "end": 6.0}], aligned_units)
    assert [part["text"] for part in parts] == ["你好世界。", "今天去北京123号。"]
    assert parts[0]["split_method"] == "alignment_semantic"
    assert parts[0]["split_confidence"] > 0.8
    assert parts[1]["start"] >= parts[0]["end"]


def test_duration_safety_classifies_stretch_and_tail_energy() -> None:
    from dv_backend.duration_safety import classify_stretch, tail_has_speech

    assert classify_stretch(1.08, max_safe=1.25).risk == "normal"
    assert classify_stretch(1.30, max_safe=1.25).risk == "warning"
    assert classify_stretch(1.45, max_safe=1.25).risk == "danger"

    samples = [0.0] * 1000 + [0.4] * 200
    assert tail_has_speech(samples, sample_rate=1000, tail_ms=200) is True
    assert tail_has_speech([0.0] * 1200, sample_rate=1000, tail_ms=200) is False


def test_translation_duration_estimate_and_metadata() -> None:
    from dv_backend.translation_duration import annotate_translation_duration

    segment = {"duration_budget": 2.0, "translation": "Xin chào mọi người."}
    annotated = annotate_translation_duration(segment, speaking_rate_wps=3.0)
    assert annotated["estimated_translation_duration"] > 0
    assert annotated["duration_fit_prediction"] in {"fits", "risky", "over_budget"}
    assert annotated["translation_was_duration_constrained"] is True


def test_translation_timing_guidance_uses_alignment_and_budget() -> None:
    from dv_backend.translation_duration import build_translation_timing_guidance

    guidance = build_translation_timing_guidance(
        {"text": "你好世界", "duration_budget": 1.5},
        aligned_units=[
            {"start": 0.0, "end": 0.2, "text": "你"},
            {"start": 0.2, "end": 0.4, "text": "好"},
            {"start": 0.4, "end": 0.6, "text": "世"},
            {"start": 0.6, "end": 0.8, "text": "界"},
        ],
        speaking_rate_wps=3.2,
    )

    assert guidance == {
        "source_speech_units": 4,
        "target_vi_syllables": 5,
        "target_vi_syllable_range": [4, 6],
    }


def test_tts_conversion_strategy_defaults_to_lazy_mix() -> None:
    from dv_backend.tts_conversion import conversion_strategy_from_settings

    assert conversion_strategy_from_settings({}) == "lazy_mix"
    assert conversion_strategy_from_settings({"tts_conversion_strategy": "lazy_mix"}) == "lazy_mix"
    assert conversion_strategy_from_settings({"tts_conversion_strategy": "bogus"}) == "lazy_mix"


def test_tts_conversion_lazy_mix_records_zero_processes() -> None:
    from dv_backend.tts_conversion import convert_segments, describe

    class _Cfg:
        def __init__(self, data_dir: Path) -> None:
            self.data_dir = data_dir

    result = convert_segments(
        _Cfg(Path("data")),
        pipeline_module=__import__("dv_backend.pipeline", fromlist=["pipeline"]),
        job_id="n/a",
        runner=None,
        segments=[{"index": 0}, {"index": 1}],
        settings={"tts_conversion_strategy": "lazy_mix"},
    )
    payload = describe(result)
    assert payload["conversion_strategy"] == "lazy_mix"
    assert payload["conversion_process_count"] == 0
    assert payload["conversion_input_count"] == 2
