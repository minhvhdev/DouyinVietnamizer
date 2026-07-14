from pathlib import Path

from dv_backend.pipeline import (
    _apply_uniform_reading_speed,
    _collect_proposed_speed_factors,
    _propose_then_apply_uniform_speed,
)


def test_collect_proposed_speed_factors_takes_max_without_rewriting(tmp_path) -> None:
    segments = [
        {
            "index": 0,
            "tts_spoken_text": "needs max",
            "timing_overflow_sec": 1.0,
            "timing_available_duration": 1.0,
            "repaired_duration": 1.5,
        },
        {
            "index": 1,
            "tts_spoken_text": "mild",
            "timing_overflow_sec": 0.2,
            "timing_available_duration": 1.0,
            "repaired_duration": 1.1,
        },
        {
            "index": 2,
            "tts_spoken_text": "ok",
            "timing_overflow_sec": 0.0,
            "timing_available_duration": 1.0,
            "repaired_duration": 0.8,
        },
    ]
    target = _collect_proposed_speed_factors(segments, absolute_max_rate=1.2)
    assert target == 1.2
    assert segments[0]["proposed_speed_factor"] == 1.2
    assert segments[1]["proposed_speed_factor"] == 1.1
    assert segments[2]["proposed_speed_factor"] == 1.0


def test_apply_uniform_reading_speed_from_base_once(tmp_path, monkeypatch) -> None:
    calls: list[tuple[Path, str]] = []

    def fake_filter(ffmpeg_path, source, dest, filter_expr, job_id, runner):  # noqa: ARG001
        calls.append((Path(source), filter_expr))
        dest.write_bytes(b"RIFF")

    monkeypatch.setattr("dv_backend.pipeline._run_ffmpeg_audio_filter", fake_filter)
    monkeypatch.setattr("dv_backend.pipeline.get_wav_duration", lambda _p: 0.9)
    monkeypatch.setattr("dv_backend.pipeline.compute_placement_starts", lambda segs: segs)
    monkeypatch.setattr("dv_backend.pipeline.schedule_soft_placements", lambda segs: segs)

    base_a = tmp_path / "tts_speed_base_0.wav"
    base_b = tmp_path / "tts_speed_base_1.wav"
    base_a.write_bytes(b"a")
    base_b.write_bytes(b"b")
    segments = [
        {
            "index": 0,
            "tts_spoken_text": "alpha",
            "tts_path": str(tmp_path / "tts_repaired_0.wav"),
            "tts_speed_base_path": str(base_a),
            "proposed_speed_factor": 1.2,
            "repaired_duration": 1.0,
        },
        {
            "index": 1,
            "tts_spoken_text": "beta",
            "tts_path": str(tmp_path / "tts_repaired_1.wav"),
            "tts_speed_base_path": str(base_b),
            "proposed_speed_factor": 1.0,
            "repaired_duration": 1.0,
        },
    ]
    target = _apply_uniform_reading_speed(
        segments=segments,
        target_rate=1.2,
        ffmpeg_path=tmp_path / "ffmpeg",
        tts_dir=tmp_path,
        job_id="job",
        runner=None,
    )
    assert target == 1.2
    assert segments[0]["soft_speed_factor"] == 1.2
    assert segments[1]["soft_speed_factor"] == 1.2
    assert len(calls) == 2
    assert all(src.name.startswith("tts_speed_base_") for src, _ in calls)
    assert all("atempo=" in expr for _, expr in calls)


def test_propose_then_apply_uniform_speed_uses_max(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_filter(ffmpeg_path, source, dest, filter_expr, job_id, runner):  # noqa: ARG001
        calls.append(filter_expr)
        dest.write_bytes(b"RIFF")

    monkeypatch.setattr("dv_backend.pipeline._run_ffmpeg_audio_filter", fake_filter)

    def fake_duration(path):
        name = Path(path).name
        if "0" in name:
            return 1.5
        return 0.8

    monkeypatch.setattr("dv_backend.pipeline.get_wav_duration", fake_duration)

    def fake_place(segs):
        for s in segs:
            if int(s["index"]) == 0:
                s["timing_overflow_sec"] = 1.0
                s["timing_available_duration"] = 1.0
            else:
                s["timing_overflow_sec"] = 0.0
                s["timing_available_duration"] = 2.0
        return segs

    monkeypatch.setattr("dv_backend.pipeline.compute_placement_starts", fake_place)
    monkeypatch.setattr("dv_backend.pipeline.schedule_soft_placements", fake_place)

    wav_a = tmp_path / "tts_repaired_0.wav"
    wav_b = tmp_path / "tts_repaired_1.wav"
    wav_a.write_bytes(b"a")
    wav_b.write_bytes(b"b")
    segments = [
        {
            "index": 0,
            "tts_spoken_text": "long",
            "tts_path": str(wav_a),
            "repaired_duration": 1.5,
        },
        {
            "index": 1,
            "tts_spoken_text": "short",
            "tts_path": str(wav_b),
            "repaired_duration": 0.8,
        },
    ]
    target = _propose_then_apply_uniform_speed(
        segments=segments,
        absolute_max_rate=1.2,
        ffmpeg_path=tmp_path / "ffmpeg",
        tts_dir=tmp_path,
        job_id="job",
        runner=None,
    )
    assert target == 1.2
    assert segments[0]["soft_speed_factor"] == 1.2
    assert segments[1]["soft_speed_factor"] == 1.2
    assert (tmp_path / "tts_speed_base_0.wav").is_file()
    assert (tmp_path / "tts_speed_base_1.wav").is_file()
    assert len(calls) == 2
