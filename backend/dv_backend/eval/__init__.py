"""Offline evaluation, benchmarking, and A/B experiment helpers (not used in production pipeline)."""

from .dubbing_quality_score import score_segment_quality
from .evaluation_audio import export_evaluation_audio
from .experiment_comparability import validate_experiment_comparability
from .timing_eval_dashboard import build_dashboard_payload, export_dashboard_html
from .timing_experiment import clone_job_prefix, experiment_dir, load_manifest

__all__ = [
    "build_dashboard_payload",
    "clone_job_prefix",
    "experiment_dir",
    "export_dashboard_html",
    "export_evaluation_audio",
    "load_manifest",
    "score_segment_quality",
    "validate_experiment_comparability",
]
