"""End-to-end duration lock: tempo-fit, pad-fit, impossible-fit through production paths."""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from dv_backend.duration_repair_executor import (
    RewriteOutcome,
    attach_repair_execution_to_segment,
    execute_segment_duration_repair,
)
from dv_backend.subtitle_timing import (
    annotate_subtitle_playback_windows,
    hash_subtitle_track_body,
    load_canonical_subtitle_track,
    resolve_subtitle_track,
    write_canonical_subtitle_track,
)
from dv_backend.timing_placement import (
    compute_placement_starts,
    segment_effective_end,
    segment_effective_start,
    segment_playback_interval,
)
from dv_backend.timing_profile import attach_timing_profiles, build_timing_profile
from dv_backend.timing_qc_metrics import compute_timing_qc_metrics
from dv_backend.tts_attempt_budget import TtsAttemptBudget


TOLERANCE_SEC = 0.08


def _profile(*, start: float, original: float, next_start: float | None, budget: float) -> dict[str, float]:
    built = build_timing_profile(
        {
            "start": start,
            "end": start + original,
            "original_duration": original,
            "duration_budget": budget,
        },
        next_segment_start=next_start,
    )
    return {k: float(v) for k, v in built.items() if isinstance(v, (int, float))}


def _write_wav(path: Path, duration: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        frames = int(16000 * duration)
        handle.writeframes(b"\x00\x01" * frames)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass
class E2ERepairOps:
    durations: dict[str, float]
    calls: list[str] = field(default_factory=list)
    rewrite_improves: bool = True

    def probe_wav_duration(self, path: Path) -> float | None:
        return self.durations.get(str(path))

    def probe_speech_duration(self, path: Path) -> float:
        return float(self.probe_wav_duration(path) or 0.0)

    def apply_rewrite_shorten(self, *, input_path: Path) -> RewriteOutcome:
        self.calls.append("rewrite_shorten")
        if not self.rewrite_improves:
            return RewriteOutcome(success=False, no_improvement=True, reason="rewrite_no_improvement")
        out = input_path.parent / "rewritten.wav"
        self.durations[str(out)] = self.probe_wav_duration(input_path) or 0.0
        return RewriteOutcome(success=False, no_improvement=True, reason="rewrite_no_improvement")

    def apply_rewrite_lengthen(self, *, input_path: Path) -> RewriteOutcome:
        self.calls.append("rewrite_lengthen")
        return RewriteOutcome(success=False, no_improvement=True, reason="rewrite_no_improvement")

    def apply_tempo(self, *, input_path: Path, factor: float, output_path: Path) -> bool:
        self.calls.append(f"tempo:{factor}")
        base = self.probe_wav_duration(input_path) or 0.0
        self.durations[str(output_path)] = round(base / factor, 3)
        return True

    def apply_pad(
        self, *, input_path: Path, target_duration: float, output_path: Path, current_duration: float
    ) -> bool:
        self.calls.append(f"pad:{target_duration}")
        self.durations[str(output_path)] = round(target_duration, 3)
        return True

    def apply_outer_silence_trim(
        self,
        *,
        input_path: Path,
        target_duration: float,
        output_path: Path,
        current_duration: float,
        speech_duration: float,
    ) -> bool:
        self.calls.append("trim_silence")
        self.durations[str(output_path)] = round(target_duration, 3)
        return True

    def apply_global_speed(self, *, input_path: Path) -> tuple[Path, float]:
        return input_path, float(self.probe_wav_duration(input_path) or 0.0)


def _run_repair(
    *,
    segment: dict,
    profile: dict[str, float],
    raw_path: Path,
    raw_duration: float,
    settings: dict,
    rewrite_improves: bool = True,
) -> None:
    ops = E2ERepairOps(durations={str(raw_path): raw_duration}, rewrite_improves=rewrite_improves)
    fit_max = max(
        float(profile.get("speech_target_duration") or 0.0),
        float(profile.get("hard_max_duration") or 0.0),
    )
    result = execute_segment_duration_repair(
        segment=segment,
        profile=profile,
        settings=settings,
        ops=ops,
        segment_budget=TtsAttemptBudget(max_rewrite_attempts=1),
        exact_timing_enabled=True,
        tolerance_sec=0.05,
        fit_max=fit_max,
        repair_target=float(profile.get("speech_target_duration") or 0.0),
        orig_file=raw_path,
        allow_spoken_text_mutation=bool(settings.get("allow_spoken_text_mutation", False)),
    )
    attach_repair_execution_to_segment(segment, result)
    segment["tts_path"] = str(raw_path)
    segment["raw_tts_sha256"] = _sha256_file(raw_path)
    segment["_repair_calls"] = ops.calls


def test_duration_lock_abc_end_to_end(tmp_path: Path) -> None:
    settings = {
        "exact_timing_enabled": True,
        "allow_spoken_text_mutation": True,
        "timing_max_llm_rewrite_attempts": 1,
    }
    tts_dir = tmp_path / "artifacts" / "tts"
    raw_a = tts_dir / "tts_0.wav"
    raw_b = tts_dir / "tts_1.wav"
    raw_c = tts_dir / "tts_2.wav"
    _write_wav(raw_a, 2.1)
    _write_wav(raw_b, 1.0)
    _write_wav(raw_c, 2.8)
    raw_hashes = {
        "a": _sha256_file(raw_a),
        "b": _sha256_file(raw_b),
        "c": _sha256_file(raw_c),
    }

    profile_a = _profile(start=0.0, original=1.8, next_start=1.8, budget=1.8)
    profile_b = _profile(start=1.8, original=2.0, next_start=3.8, budget=2.0)
    profile_c = _profile(start=3.8, original=1.2, next_start=5.0, budget=1.2)

    seg_a: dict = {
        "index": 0,
        "start": 0.0,
        "end": 1.8,
        "original_duration": 1.8,
        "duration_budget": 1.8,
        "translation": "Segment A tempo fit.",
        "timing_profile": profile_a,
    }
    seg_b: dict = {
        "index": 1,
        "start": 1.8,
        "end": 3.8,
        "original_duration": 2.0,
        "duration_budget": 2.0,
        "translation": "Segment B pad fit.",
        "timing_profile": profile_b,
    }
    seg_c: dict = {
        "index": 2,
        "start": 3.8,
        "end": 5.0,
        "original_duration": 1.2,
        "duration_budget": 1.2,
        "translation": "Segment C impossible.",
        "timing_profile": profile_c,
    }

    _run_repair(segment=seg_a, profile=profile_a, raw_path=raw_a, raw_duration=2.1, settings=settings)
    _run_repair(segment=seg_b, profile=profile_b, raw_path=raw_b, raw_duration=1.0, settings=settings)
    _run_repair(
        segment=seg_c,
        profile=profile_c,
        raw_path=raw_c,
        raw_duration=2.8,
        settings=settings,
        rewrite_improves=False,
    )

    segments = [seg_a, seg_b, seg_c]
    source_starts = [s["start"] for s in segments]
    source_ends = [s["end"] for s in segments]

    # Segment A — tempo-fit
    assert seg_a["duration_fit_status"] == "fit"
    assert any(call.startswith("tempo:") for call in seg_a["_repair_calls"])
    assert sum(1 for call in seg_a["_repair_calls"] if call.startswith("tempo:")) == 1
    assert seg_a["repaired_duration"] == pytest.approx(2.1 / 1.2, abs=0.15)
    assert seg_a["initial_planned_action"] == seg_a["decision_history"][0]["action"]
    assert "tempo" in seg_a["applied_actions"]

    # Segment B — pad-fit
    assert seg_b["duration_fit_status"] == "fit"
    assert any(call.startswith("pad:") for call in seg_b["_repair_calls"])
    assert seg_b["repaired_duration"] == pytest.approx(2.0, abs=TOLERANCE_SEC)

    # Segment C — impossible-fit
    assert seg_c["duration_fit_status"] == "unresolved"
    assert seg_c.get("unresolved_reason")
    assert not any(call.startswith("tempo:") and float(call.split(":")[1]) > 1.2 for call in seg_c["_repair_calls"])

    attach_timing_profiles(segments, settings=settings)
    compute_placement_starts(segments)

    for segment in segments:
        assert segment["start"] == source_starts[segment["index"]]
        assert segment["end"] == source_ends[segment["index"]]
        start = segment_effective_start(segment)
        end = segment_effective_end(segment)
        assert end == pytest.approx(start + float(segment["repaired_duration"]), abs=TOLERANCE_SEC)

    # No overlap after placement
    intervals = [segment_playback_interval(s) for s in segments]
    for idx in range(1, len(intervals)):
        assert intervals[idx][0] >= intervals[idx - 1][1] - TOLERANCE_SEC

    # Subtitle + QC artifact path
    for segment in segments:
        segment["tts_spoken_text"] = segment.get("translation")
    annotate_subtitle_playback_windows(segments)
    track = resolve_subtitle_track(segments, tts_asr_align=False)
    track_path = write_canonical_subtitle_track(
        tmp_path,
        cues=track["cues"],
        segment_indices=[int(s["index"]) for s in segments],
    )
    loaded = load_canonical_subtitle_track(tmp_path)
    assert loaded is not None
    assert loaded["content_hash"] == hash_subtitle_track_body(
        cues=track["cues"],
        segment_indices=[0, 1, 2],
    )
    qc_track = load_canonical_subtitle_track(tmp_path)
    assert qc_track is not None
    assert qc_track["cues"] == track["cues"]

    render_tuples = {
        (cue.get("text"), round(float(cue["start"]), 3), round(float(cue["end"]), 3))
        for cue in track["cues"]
    }
    qc_tuples = {
        (cue.get("text"), round(float(cue["start"]), 3), round(float(cue["end"]), 3))
        for cue in qc_track["cues"]
    }
    assert render_tuples == qc_tuples

    sorted_segments = sorted(segments, key=segment_effective_start)
    sorted_cues = sorted(track["cues"], key=lambda cue: float(cue["start"]))
    assert len(sorted_cues) >= len(sorted_segments)
    for segment, cue in zip(sorted_segments, sorted_cues[: len(sorted_segments)]):
        start, end = segment_playback_interval(segment)
        assert float(cue["start"]) >= start - TOLERANCE_SEC
        assert float(cue["end"]) <= end + TOLERANCE_SEC

    metrics = compute_timing_qc_metrics(segments, settings=settings)
    unresolved_count = sum(1 for s in segments if s.get("duration_fit_status") == "unresolved")
    fit_count = sum(1 for s in segments if s.get("duration_fit_status") == "fit")
    assert unresolved_count >= 1
    assert fit_count >= 2
    assert metrics.get("segment_count", len(segments)) >= 3

    # Raw TTS cache safety
    assert _sha256_file(raw_a) == raw_hashes["a"]
    assert _sha256_file(raw_b) == raw_hashes["b"]
    assert _sha256_file(raw_c) == raw_hashes["c"]
    assert track_path.is_file()


def test_probe_failure_marks_unresolved(tmp_path: Path) -> None:
    profile = _profile(start=0.0, original=1.8, next_start=3.0, budget=1.8)
    wav = tmp_path / "seg.wav"
    _write_wav(wav, 2.0)

    @dataclass
    class BrokenOps(E2ERepairOps):
        def probe_wav_duration(self, path: Path) -> float | None:
            return None

    segment = {
        "index": 0,
        "start": 0.0,
        "end": 1.8,
        "original_duration": 1.8,
        "duration_budget": 1.8,
        "timing_profile": profile,
    }
    ops = BrokenOps(durations={str(wav): 2.0})
    result = execute_segment_duration_repair(
        segment=segment,
        profile=profile,
        settings={"exact_timing_enabled": True},
        ops=ops,
        segment_budget=TtsAttemptBudget(),
        exact_timing_enabled=True,
        tolerance_sec=0.05,
        fit_max=float(profile["hard_max_duration"]),
        repair_target=float(profile["speech_target_duration"]),
        orig_file=wav,
    )
    assert result.duration_fit_status == "unresolved"
    assert result.unresolved_reason == "repair_measurement_failed"


def test_subtitle_artifact_mismatch_qc_failed(tmp_path: Path) -> None:
    cues = [{"start": 0.0, "end": 1.0, "text": "A", "segment_index": 0}]
    write_canonical_subtitle_track(tmp_path, cues=cues, segment_indices=[0])
    path = tmp_path / "artifacts" / "subtitle_track.json"
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cues"][0]["text"] = "Tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_canonical_subtitle_track(tmp_path) is None
