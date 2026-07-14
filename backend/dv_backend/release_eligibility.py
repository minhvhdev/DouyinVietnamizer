"""Hard release_eligible gate for formal mix/render/export paths."""

from __future__ import annotations

from typing import Any

from .checkpoints import load_checkpoint
from .config import AppConfig
from .errors import AppError
from .models import ErrorInfo
from .timing_placement import segments_with_voiced_overlap
from .timing_review import flag_infeasible_segments, list_timing_review_segments


def compute_release_eligible(
    segments: list[dict[str, Any]],
    *,
    absolute_max_rate: float = 1.2,
) -> bool:
    flagged = flag_infeasible_segments(list(segments), absolute_max_rate=absolute_max_rate)
    overlaps = segments_with_voiced_overlap(segments)
    remaining = list_timing_review_segments(segments, absolute_max_rate=absolute_max_rate)
    return not flagged and not overlaps and len(remaining) == 0


def resolve_release_eligible(
    config: AppConfig,
    job_id: str,
    *,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return release eligibility from duration_repair checkpoint, recomputing if needed.

    Live remaining/overlap counts are authoritative for blocking. A stale checkpoint
    ``release_eligible=false`` must not block when remaining=0 and overlaps=0
    (common after timing-review logic fixes or speed-fitting that cleared the queue).
    Empty-segment jobs may still trust an explicit checkpoint True (normalize drop-all).
    """
    cfg = settings or {}
    absolute_max = float(cfg.get("edge_tts_overflow_speed_hard_max", 1.2) or 1.2)
    absolute_max = max(1.0, min(1.2, absolute_max))
    repair_cp = load_checkpoint(config.data_dir, job_id, "duration_repair") or {}
    segments = list(repair_cp.get("segments") or [])
    if not segments:
        align_cp = load_checkpoint(config.data_dir, job_id, "align_final_dub") or {}
        segments = list(align_cp.get("segments") or [])

    remaining = list_timing_review_segments(segments, absolute_max_rate=absolute_max)
    overlaps = segments_with_voiced_overlap(segments)
    remaining_count = len(remaining)
    overlap_count = len(overlaps)
    live_clean = remaining_count == 0 and overlap_count == 0

    if not live_clean:
        eligible = False
    elif not segments:
        # Empty after normalize can still be formally releasable if checkpoint says so.
        eligible = bool(repair_cp.get("release_eligible", True))
    else:
        # Segments present and review queue empty — allow even if checkpoint is stale false.
        eligible = True

    return {
        "release_eligible": eligible,
        "remaining_count": remaining_count,
        "overlap_count": overlap_count,
        "source": "duration_repair" if repair_cp else "recomputed",
    }


def assert_formal_release_allowed(
    config: AppConfig,
    job_id: str,
    *,
    settings: dict[str, Any] | None = None,
    stage: str = "formal_output",
) -> dict[str, Any]:
    """Block formal mix/render/export when release_eligible is false."""
    info = resolve_release_eligible(config, job_id, settings=settings)
    if info["release_eligible"]:
        return info
    raise AppError(
        409,
        ErrorInfo(
            code="RELEASE_ELIGIBLE_BLOCKED",
            message=(
                f"Formal {stage} blocked because release_eligible=false "
                f"(remaining={info['remaining_count']}, overlaps={info['overlap_count']})."
            ),
            action="Finish timing review until release_eligible=true, or use a QA preview artifact only.",
            detail=(
                f"stage={stage},remaining={info['remaining_count']},"
                f"overlaps={info['overlap_count']}"
            ),
        ),
    )
