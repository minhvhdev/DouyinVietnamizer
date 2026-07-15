"""Policy-to-execution parity tests for duration repair executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dv_backend.duration_repair_executor import (
    DurationRepairExecutionResult,
    RewriteOutcome,
    execute_segment_duration_repair,
)
from dv_backend.timing_profile import build_timing_profile
from dv_backend.tts_attempt_budget import TtsAttemptBudget


def _profile(*, start: float, original: float, next_start: float) -> dict[str, float]:
    built = build_timing_profile(
        {"start": start, "end": start + original, "original_duration": original},
        next_segment_start=next_start,
    )
    return {k: float(v) for k, v in built.items() if isinstance(v, (int, float))}


@dataclass
class MockDurationRepairOps:
  calls: list[str] = field(default_factory=list)
  durations: dict[str, float] = field(default_factory=dict)
  rewrite_text: str | None = None
  tempo_factor: float = 1.2
  working: str = "input"

  def probe_wav_duration(self, path: Path) -> float | None:
    key = str(path)
    return self.durations.get(key, self.durations.get(self.working, 2.0))

  def probe_speech_duration(self, path: Path) -> float:
    return self.probe_wav_duration(path) or 0.0

  def apply_rewrite_shorten(self, *, input_path: Path) -> RewriteOutcome:
    self.calls.append("rewrite_shorten")
    out = input_path.parent / "rewritten.wav"
    self.durations[str(out)] = 1.8
    return RewriteOutcome(
      success=True,
      output_path=out,
      new_speech_duration=1.8,
      new_wav_duration=1.8,
      new_translation=self.rewrite_text or "shorter",
      method_label="rewrite_shorten",
    )

  def apply_rewrite_lengthen(self, *, input_path: Path) -> RewriteOutcome:
    self.calls.append("rewrite_lengthen")
    out = input_path.parent / "rewritten.wav"
    self.durations[str(out)] = 2.2
    return RewriteOutcome(
      success=True,
      output_path=out,
      new_speech_duration=2.2,
      new_wav_duration=2.2,
      new_translation=self.rewrite_text or "longer",
      method_label="rewrite_lengthen",
    )

  def apply_tempo(self, *, input_path: Path, factor: float, output_path: Path) -> bool:
    self.calls.append(f"tempo:{factor}")
    self.durations[str(output_path)] = round((self.probe_wav_duration(input_path) or 0.0) / factor, 3)
    return True

  def apply_pad(
    self, *, input_path: Path, target_duration: float, output_path: Path, current_duration: float
  ) -> bool:
    self.calls.append(f"pad:{target_duration}")
    self.durations[str(output_path)] = target_duration
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
    self.durations[str(output_path)] = target_duration
    return True

  def apply_global_speed(self, *, input_path: Path) -> tuple[Path, float]:
    return input_path, self.probe_wav_duration(input_path) or 0.0


@pytest.mark.parametrize(
    ("speech", "next_start", "exact", "mutation", "expected_action"),
    [
      (3.0, 9.0, False, False, "accept"),
      (9.0, 8.0, True, True, "rewrite_shorten"),
      (9.0, 8.0, True, False, "tempo"),
      (0.4, 8.0, True, False, "pad"),
    ],
)
def test_executor_runs_policy_action(
    speech: float,
    next_start: float,
    exact: bool,
    mutation: bool,
    expected_action: str,
) -> None:
  profile = _profile(start=0.0, original=2.0, next_start=next_start)
  ops = MockDurationRepairOps(durations={"input": speech})
  segment = {"index": 0, "start": 0.0, "end": 2.0, "original_duration": 2.0, "timing_profile": profile}
  settings = {
    "exact_timing_enabled": exact,
    "allow_spoken_text_mutation": mutation,
    "timing_max_llm_rewrite_attempts": 1,
  }
  with patch("dv_backend.duration_repair_executor.decide_duration_repair") as decide:
    decide.return_value = {
      "action": expected_action,
      "reason": "test",
      "tempo_factor": 1.2,
      "pad_target_duration": profile["speech_target_duration"],
      "classification": "too_long",
      "duration_miss": True,
      "placement_shift_only": False,
    }
    result = execute_segment_duration_repair(
      segment=segment,
      profile=profile,
      settings=settings,
      ops=ops,
      segment_budget=TtsAttemptBudget(),
      exact_timing_enabled=exact,
      tolerance_sec=0.05,
      fit_max=float(profile["hard_max_duration"]),
      repair_target=float(profile["speech_target_duration"]),
      orig_file=Path("input"),
      allow_spoken_text_mutation=mutation,
    )
  if expected_action == "accept":
    assert "accept" in str(result.decision_history[-1].get("action"))
  else:
    assert any(expected_action in call for call in ops.calls)


def test_tempo_applied_only_once() -> None:
  profile = _profile(start=0.0, original=2.0, next_start=8.0)
  ops = MockDurationRepairOps(durations={"input": 9.0})
  segment = {"index": 0, "start": 0.0, "end": 2.0, "original_duration": 2.0, "timing_profile": profile}
  settings = {"exact_timing_enabled": True, "allow_spoken_text_mutation": False}
  result = execute_segment_duration_repair(
    segment=segment,
    profile=profile,
    settings=settings,
    ops=ops,
    segment_budget=TtsAttemptBudget(),
    exact_timing_enabled=True,
    tolerance_sec=0.05,
    fit_max=float(profile["hard_max_duration"]),
    repair_target=float(profile["speech_target_duration"]),
    orig_file=Path("input"),
  )
  tempo_calls = [call for call in ops.calls if call.startswith("tempo:")]
  assert len(tempo_calls) == 1
  assert result.tempo_applied is True


def test_placement_shift_only_accepts_without_ops(tmp_path: Path) -> None:
  profile = _profile(start=0.0, original=3.0, next_start=9.0)
  wav = tmp_path / "seg.wav"
  wav.write_bytes(b"x")
  ops = MockDurationRepairOps(durations={str(wav): 3.0})
  segment = {
    "index": 0,
    "start": 0.0,
    "end": 3.0,
    "original_duration": 3.0,
    "timing_profile": profile,
    "placement_drift_sec": 0.35,
  }
  result = execute_segment_duration_repair(
    segment=segment,
    profile=profile,
    settings={},
    ops=ops,
    segment_budget=TtsAttemptBudget(),
    exact_timing_enabled=True,
    tolerance_sec=0.05,
    fit_max=float(profile["hard_max_duration"]),
    repair_target=float(profile["speech_target_duration"]),
    orig_file=wav,
  )
  assert ops.calls == []
  assert result.duration_fit_status == "fit"
  assert result.initial_planned_action == "accept"


def test_measurement_failure_marks_unresolved(tmp_path: Path) -> None:
  profile = _profile(start=0.0, original=2.0, next_start=8.0)
  wav = tmp_path / "seg.wav"
  wav.write_bytes(b"x")

  @dataclass
  class BrokenProbeOps(MockDurationRepairOps):
    def probe_wav_duration(self, path: Path) -> float | None:
      return None

  ops = BrokenProbeOps(durations={str(wav): 2.0})
  segment = {"index": 0, "start": 0.0, "end": 2.0, "original_duration": 2.0, "timing_profile": profile}
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
