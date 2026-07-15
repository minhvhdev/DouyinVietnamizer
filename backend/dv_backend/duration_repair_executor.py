"""Policy-driven duration repair execution — single source of truth from decide_duration_repair."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .duration_fit_policy import (
    acceptable_duration_fit,
    classify_duration_fit,
    decide_duration_repair,
    policy_from_settings,
    timing_profile_from_segment,
)
from .tts_attempt_budget import TtsAttemptBudget

IMPROVEMENT_EPSILON_SEC = 0.03
MAX_REPAIR_ITERATIONS = 12


@dataclass
class RewriteOutcome:
    success: bool
    output_path: Path | None = None
    new_speech_duration: float | None = None
    new_wav_duration: float | None = None
    new_translation: str | None = None
    method_label: str | None = None
    no_improvement: bool = False
    reason: str | None = None


@dataclass
class DurationRepairOps(Protocol):
    def probe_wav_duration(self, path: Path) -> float | None: ...

    def probe_speech_duration(self, path: Path) -> float: ...

    def apply_rewrite_shorten(self, *, input_path: Path) -> RewriteOutcome: ...

    def apply_rewrite_lengthen(self, *, input_path: Path) -> RewriteOutcome: ...

    def apply_tempo(self, *, input_path: Path, factor: float, output_path: Path) -> bool: ...

    def apply_pad(
        self, *, input_path: Path, target_duration: float, output_path: Path, current_duration: float
    ) -> bool: ...

    def apply_outer_silence_trim(
        self,
        *,
        input_path: Path,
        target_duration: float,
        output_path: Path,
        current_duration: float,
        speech_duration: float,
    ) -> bool: ...


@dataclass
class DurationRepairExecutionResult:
    working_path: Path
    repaired_duration: float
    speech_duration: float
    decision_history: list[dict[str, Any]] = field(default_factory=list)
    applied_actions: list[str] = field(default_factory=list)
    initial_planned_action: str = "accept"
    final_repair_action: str = "accepted"
    duration_fit_status: str = "fit"
    unresolved_reason: str | None = None
    fit_methods: list[str] = field(default_factory=list)
    quality_warning: str | None = None
    duration_repair_risk: str = "none"
    needs_review: bool = False
    time_stretch_factor: float = 1.0
    repair_attempts: int = 0
    re_synthesis_count: int = 0
    accepted_without_repair: bool = False
    tempo_applied: bool = False


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _summarize_final_action(applied: list[str]) -> str:
    if not applied:
        return "accepted"
    if len(applied) == 1:
        return applied[0]
    return "_then_".join(applied)


def _rewrite_improved(
    *,
    before_speech: float,
    after_speech: float | None,
    before_text: str,
    after_text: str | None,
    target: float,
    action: str,
) -> bool:
    if after_speech is None:
        return False
    if _normalize_text(before_text) and _normalize_text(before_text) == _normalize_text(after_text or ""):
        return False
    delta_before = abs(before_speech - target)
    delta_after = abs(after_speech - target)
    if action == "rewrite_shorten":
        return after_speech < before_speech - IMPROVEMENT_EPSILON_SEC or delta_after + IMPROVEMENT_EPSILON_SEC < delta_before
    if action == "rewrite_lengthen":
        return after_speech > before_speech + IMPROVEMENT_EPSILON_SEC or delta_after + IMPROVEMENT_EPSILON_SEC < delta_before
    return after_speech != before_speech


def execute_segment_duration_repair(
    *,
    segment: dict[str, Any],
    profile: dict[str, float],
    settings: dict[str, Any],
    ops: DurationRepairOps,
    segment_budget: TtsAttemptBudget,
    exact_timing_enabled: bool,
    tolerance_sec: float,
    fit_max: float,
    repair_target: float,
    orig_file: Path,
    allow_spoken_text_mutation: bool | None = None,
) -> DurationRepairExecutionResult:
    """Run policy → execute → probe loop until accept or unresolved."""
    policy = policy_from_settings(settings)
    if not profile.get("speech_target_duration"):
        profile = timing_profile_from_segment(segment)

    working_path = orig_file
    decision_history: list[dict[str, Any]] = []
    applied_actions: list[str] = []
    fit_methods: list[str] = []
    quality_warning: str | None = None
    duration_repair_risk = "none"
    needs_review = False
    time_stretch_factor = 1.0
    repair_attempts = 0
    re_synthesis_count = 0
    tempo_applied = False
    rewrite_count = segment_budget.rewrite_attempts
    unresolved_reason: str | None = None
    duration_fit_status = "fit"
    allow_mutation = (
        bool(settings.get("allow_spoken_text_mutation", False))
        if allow_spoken_text_mutation is None
        else allow_spoken_text_mutation
    )

    initial_decision = decide_duration_repair(
        speech_duration=ops.probe_speech_duration(working_path),
        timing_profile=profile,
        segment=segment,
        policy=policy,
        settings=settings,
        rewrite_attempts=rewrite_count,
        max_rewrite_attempts=segment_budget.max_rewrite_attempts,
        exact_timing_enabled=exact_timing_enabled,
        allow_spoken_text_mutation=allow_mutation,
        tolerance_sec=tolerance_sec,
    )
    initial_planned_action = str(initial_decision.get("action") or "accept")

    for _iteration in range(MAX_REPAIR_ITERATIONS):
        speech_duration = ops.probe_speech_duration(working_path)
        wav_duration = ops.probe_wav_duration(working_path)
        if wav_duration is None:
            unresolved_reason = "repair_measurement_failed"
            duration_fit_status = "unresolved"
            quality_warning = quality_warning or unresolved_reason
            decision_history.append(
                {
                    "action": "unresolved",
                    "reason": unresolved_reason,
                    "measured_duration": None,
                }
            )
            break

        decision = decide_duration_repair(
            speech_duration=speech_duration,
            timing_profile=profile,
            segment=segment,
            policy=policy,
            settings=settings,
            rewrite_attempts=rewrite_count,
            max_rewrite_attempts=segment_budget.max_rewrite_attempts,
            exact_timing_enabled=exact_timing_enabled,
            allow_spoken_text_mutation=allow_mutation,
            tolerance_sec=tolerance_sec,
        )
        action = str(decision.get("action") or "accept")
        history_entry = {
            **decision,
            "measured_speech_duration": round(speech_duration, 3),
            "measured_wav_duration": round(wav_duration, 3),
        }
        decision_history.append(history_entry)

        if action == "accept":
            if (
                exact_timing_enabled
                and fit_max > 0
                and wav_duration > fit_max + tolerance_sec
                and speech_duration <= fit_max + tolerance_sec
                and (wav_duration - speech_duration) > tolerance_sec
            ):
                trim_out = working_path.parent / f"tts_exact_trim_{segment.get('index', 0)}.wav"
                if ops.apply_outer_silence_trim(
                    input_path=working_path,
                    target_duration=fit_max,
                    output_path=trim_out,
                    current_duration=wav_duration,
                    speech_duration=speech_duration,
                ):
                    working_path = trim_out
                    applied_actions.append("trim_silence")
                    fit_methods.append("outer_silence_trim")
                    repair_attempts += 1
                    continue
            break

        if action == "unresolved":
            unresolved_reason = str(decision.get("reason") or "unresolved")
            duration_fit_status = "unresolved"
            if fit_max > 0 and speech_duration > fit_max + tolerance_sec:
                duration_repair_risk = "danger"
                needs_review = True
                quality_warning = quality_warning or "residual_speech_over_window"
            break

        if action in {"rewrite_shorten", "rewrite_lengthen"}:
            before_text = str(segment.get("translation") or "")
            before_speech = speech_duration
            outcome = (
                ops.apply_rewrite_shorten(input_path=working_path)
                if action == "rewrite_shorten"
                else ops.apply_rewrite_lengthen(input_path=working_path)
            )
            rewrite_count += 1
            repair_attempts += 1
            if outcome.success and outcome.output_path is not None:
                target = float(profile.get("speech_target_duration") or repair_target or fit_max)
                improved = _rewrite_improved(
                    before_speech=before_speech,
                    after_speech=outcome.new_speech_duration,
                    before_text=before_text,
                    after_text=outcome.new_translation or before_text,
                    target=target,
                    action=action,
                )
                if not improved or outcome.no_improvement:
                    applied_actions.append(f"{action}_no_improvement")
                    quality_warning = quality_warning or (
                        outcome.reason or "rewrite_no_improvement"
                    )
                    continue
                working_path = outcome.output_path
                if outcome.new_translation:
                    segment["translation"] = outcome.new_translation
                if outcome.method_label:
                    fit_methods.append(outcome.method_label)
                applied_actions.append(action)
                re_synthesis_count += 1
                segment_budget.rewrite_attempts = rewrite_count
                continue
            applied_actions.append(f"{action}_failed")
            quality_warning = quality_warning or (outcome.reason or f"{action}_failed")
            continue

        if action == "tempo":
            if tempo_applied:
                unresolved_reason = "tempo_already_applied"
                duration_fit_status = "unresolved"
                break
            factor = float(decision.get("tempo_factor") or 1.0)
            if factor <= 1.0:
                unresolved_reason = str(decision.get("reason") or "tempo_factor_invalid")
                duration_fit_status = "unresolved"
                break
            tempo_out = working_path.parent / f"tts_stretch_{segment.get('index', 0)}.wav"
            if not ops.apply_tempo(input_path=working_path, factor=factor, output_path=tempo_out):
                unresolved_reason = "tempo_apply_failed"
                duration_fit_status = "unresolved"
                break
            working_path = tempo_out
            tempo_applied = True
            time_stretch_factor = round(factor, 3)
            segment["automatic_tempo_factor"] = time_stretch_factor
            applied_actions.append("tempo")
            fit_methods.append(f"time_stretch_{round(factor, 2)}x")
            repair_attempts += 1
            continue

        if action == "pad":
            target = float(decision.get("pad_target_duration") or repair_target or 0.0)
            if target <= 0 or speech_duration >= target - tolerance_sec:
                break
            pad_out = working_path.parent / f"tts_exact_{segment.get('index', 0)}.wav"
            if not ops.apply_pad(
                input_path=working_path,
                target_duration=target,
                output_path=pad_out,
                current_duration=wav_duration,
            ):
                unresolved_reason = "pad_apply_failed"
                duration_fit_status = "unresolved"
                break
            working_path = pad_out
            applied_actions.append("pad")
            fit_methods.append("tail_silence_pad")
            repair_attempts += 1
            continue

        unresolved_reason = f"unknown_action:{action}"
        duration_fit_status = "unresolved"
        break

    final_wav = ops.probe_wav_duration(working_path)
    final_speech = ops.probe_speech_duration(working_path)
    if final_wav is None:
        duration_fit_status = "unresolved"
        unresolved_reason = unresolved_reason or "repair_measurement_failed"
        final_wav = float(segment.get("tts_duration") or 0.0)

    final_fit = classify_duration_fit(final_speech, profile, policy=policy, segment=segment)
    if duration_fit_status != "unresolved" and not acceptable_duration_fit(final_fit):
        if unresolved_reason is None:
            duration_fit_status = "unresolved"
            unresolved_reason = f"final_fit:{final_fit}"

    if fit_max > 0 and final_speech > fit_max + tolerance_sec:
        duration_repair_risk = "danger"
        needs_review = True
        quality_warning = quality_warning or "residual_speech_over_window"
        if duration_fit_status == "fit":
            duration_fit_status = "unresolved"
            unresolved_reason = unresolved_reason or "residual_speech_over_window"

    accepted_without_repair = not applied_actions and duration_fit_status == "fit"
    final_repair_action = _summarize_final_action(applied_actions) if applied_actions else "accepted"

    return DurationRepairExecutionResult(
        working_path=working_path,
        repaired_duration=round(final_wav, 2),
        speech_duration=round(final_speech, 3),
        decision_history=decision_history,
        applied_actions=applied_actions,
        initial_planned_action=initial_planned_action,
        final_repair_action=final_repair_action,
        duration_fit_status=duration_fit_status,
        unresolved_reason=unresolved_reason,
        fit_methods=fit_methods,
        quality_warning=quality_warning,
        duration_repair_risk=duration_repair_risk,
        needs_review=needs_review,
        time_stretch_factor=time_stretch_factor,
        repair_attempts=repair_attempts,
        re_synthesis_count=re_synthesis_count,
        accepted_without_repair=accepted_without_repair,
        tempo_applied=tempo_applied,
    )


def attach_repair_execution_to_segment(segment: dict[str, Any], result: DurationRepairExecutionResult) -> None:
    """Persist execution metadata on segment checkpoint fields."""
    segment["duration_repair_decision"] = result.decision_history[-1] if result.decision_history else {}
    segment["initial_planned_action"] = result.initial_planned_action
    segment["decision_history"] = result.decision_history
    segment["applied_actions"] = result.applied_actions
    segment["final_repair_action"] = result.final_repair_action
    segment["duration_fit_status"] = result.duration_fit_status
    segment["unresolved_reason"] = result.unresolved_reason
    segment["planned_repair_action"] = result.initial_planned_action
    segment["repaired_duration"] = result.repaired_duration
    segment["repaired_method"] = "+".join(result.fit_methods) if result.fit_methods else "none"
    segment["duration_repair_risk"] = result.duration_repair_risk
    segment["needs_review"] = result.needs_review
    segment["quality_warning"] = result.quality_warning
    segment["time_stretch_factor"] = result.time_stretch_factor
    segment["repair_attempts"] = result.repair_attempts
    segment["re_synthesis_count"] = result.re_synthesis_count
    segment["accepted_without_repair"] = result.accepted_without_repair
