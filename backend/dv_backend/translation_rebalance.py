"""Translation-time fragment spill detection, clustering, and repair validation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from .duration_predictor import count_vietnamese_syllables
from .semantic_safeguards import evaluate_semantic_safeguards

logger = logging.getLogger(__name__)

MAX_REPAIR_SYLLABLE_PENALTY_REGRESSION = 4

HANGING_ENDINGS = (
    "có hay không",
    "bởi vì",
    "nếu",
    "nhưng",
    "và",
    "để",
    "của",
    "với",
    "một",
)
_FRAGMENT_START = re.compile(r"^(\.\.\.|…|,|;|:|\s)+")
_CONTINUATION_STARTS = (
    "cách giải quyết",
    "việc đó",
    "mà hắn",
    "mà cô",
    "mà anh",
    "để rồi",
    "thì sao",
    "hay không",
    "được không",
    "câu trả lời",
)
_OPEN_PREV_ENDINGS = (
    "...",
    "…",
    ":",
    ",",
    ";",
    "—",
    "-",
)
_ZH_OPEN_ENDINGS = ("有没有", "是否", "因为", "但是", "可是", "如果", "所以", "而且", "：", ":")
_ZH_CONTINUATION_STARTS = ("别的", "的话", "的是", "了吗", "吗", "呢", "着")


@dataclass(frozen=True)
class FragmentCluster:
    cluster_id: int
    mutable_indices: tuple[int, ...]
    context_before_index: int | None
    context_after_index: int | None


@dataclass(frozen=True)
class RepairMetrics:
    internal_spills: int
    boundary_spills: int
    syllable_penalty: int
    semantic_critical: bool = False


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or "", flags=re.UNICODE))


def _unbalanced_pair_chars(text: str, open_ch: str, close_ch: str) -> bool:
    return (text or "").count(open_ch) != (text or "").count(close_ch)


def _prev_looks_open(prev: str) -> bool:
    if prev.endswith(_OPEN_PREV_ENDINGS):
        return True
    for ending in HANGING_ENDINGS:
        if prev.endswith(ending):
            return True
    return False


def _hanging_start(prev: str, nxt: str) -> bool:
    if prev.endswith((".", "!", "?", "。", "！", "？")) and not _prev_looks_open(prev):
        return False
    if _FRAGMENT_START.match(nxt):
        return True
    for start in _CONTINUATION_STARTS:
        if not nxt.startswith(start):
            continue
        if _prev_looks_open(prev):
            return True
        # Incomplete previous clause (no sentence terminator) + continuation head
        if not prev.endswith((".", "!", "?", "。", "！", "？")) and _word_count(prev) <= 10:
            return True
    return False


def _unbalanced_punctuation(prev: str, nxt: str) -> bool:
    joined_prev = prev
    for open_ch, close_ch in (('"', '"'), ("(", ")"), ("「", "」"), ("『", "』")):
        if _unbalanced_pair_chars(joined_prev, open_ch, close_ch):
            return True
    if prev.endswith((":", "：")) and not nxt.endswith((".", "!", "?", "。", "！", "？")):
        # colon explanation pushed to next; only if next looks like continuation fragment
        if _word_count(nxt) <= 12 or _FRAGMENT_START.match(nxt) or any(
            nxt.startswith(s) for s in _CONTINUATION_STARTS
        ):
            return True
    if ("?" not in prev and "？" not in prev) and prev.endswith(("có hay không", "hay không", "whether")):
        if "?" in nxt or "？" in nxt:
            return True
    return False


def _short_tail_or_head(prev: str, nxt: str) -> bool:
    next_words = _word_count(nxt)
    if next_words == 0:
        return False
    # Short head on next + open prev
    if next_words <= 3 and _prev_looks_open(prev):
        return True
    # Short dangling clause after a mid-line colon (not completed list clauses)
    colon = max(prev.rfind(":"), prev.rfind("："))
    if colon > 0:
        after = prev[colon + 1 :].strip()
        if (
            after
            and _word_count(after) <= 3
            and next_words >= 2
            and not after.endswith((".", "!", "?", "。", "！", "？"))
        ):
            return True
    return False


def _source_boundary_hint(previous_source: str | None, next_source: str | None) -> bool:
    prev_s = (previous_source or "").strip()
    next_s = (next_source or "").strip()
    if not prev_s or not next_s:
        return False
    if any(prev_s.endswith(end) for end in _ZH_OPEN_ENDINGS):
        return True
    if any(next_s.startswith(start) for start in _ZH_CONTINUATION_STARTS):
        # Only with some open signal on previous
        if any(prev_s.endswith(end) for end in ("有没有", "是否", "因为", "但是", "：", ":", "，", ",")):
            return True
    return False


def looks_like_fragment_spill(
    prev_text: str,
    next_text: str,
    previous_source: str | None = None,
    next_source: str | None = None,
) -> bool:
    """Heuristic: previous/next timing slots split one thought across the boundary."""
    prev = _norm(prev_text)
    nxt = _norm(next_text)
    if not prev or not nxt:
        return False

    # Legacy high-precision signals
    if prev.endswith(("...", "…", ":")):
        return True
    if _FRAGMENT_START.match(nxt):
        return True
    for ending in HANGING_ENDINGS:
        if prev.endswith(ending):
            return True

    hanging = _hanging_start(prev, nxt)
    unbalanced = _unbalanced_punctuation(prev, nxt)
    short = _short_tail_or_head(prev, nxt)
    if hanging or unbalanced or short:
        return True

    # Source boundary is supporting evidence only — never alone.
    if _source_boundary_hint(previous_source, next_source) and (
        not prev.endswith((".", "!", "?", "。", "！", "？"))
        or any(nxt.startswith(start) for start in _CONTINUATION_STARTS)
    ):
        return True
    return False


def find_fragment_spill_pairs(
    translations: list[str],
    sources: list[str] | None = None,
) -> list[tuple[int, int]]:
    """Return adjacent index pairs (i, i+1) that look like fragment spill."""
    pairs: list[tuple[int, int]] = []
    for index in range(len(translations) - 1):
        prev_src = sources[index] if sources and index < len(sources) else None
        next_src = sources[index + 1] if sources and index + 1 < len(sources) else None
        if looks_like_fragment_spill(
            translations[index],
            translations[index + 1],
            previous_source=prev_src,
            next_source=next_src,
        ):
            pairs.append((index, index + 1))
    return pairs


def build_fragment_clusters(
    translations: list[str],
    sources: list[str] | None = None,
) -> list[FragmentCluster]:
    """Merge adjacent spill pairs into non-overlapping clusters with read-only borders."""
    pairs = find_fragment_spill_pairs(translations, sources=sources)
    if not pairs:
        return []

    groups: list[list[int]] = []
    current = [pairs[0][0], pairs[0][1]]
    for left, right in pairs[1:]:
        if left <= current[-1]:
            if right > current[-1]:
                current.append(right)
        else:
            groups.append(current)
            current = [left, right]
    groups.append(current)

    clusters: list[FragmentCluster] = []
    for cluster_id, indices in enumerate(groups):
        unique = tuple(sorted(set(indices)))
        before = unique[0] - 1 if unique[0] > 0 else None
        after = unique[-1] + 1 if unique[-1] < len(translations) - 1 else None
        clusters.append(
            FragmentCluster(
                cluster_id=cluster_id,
                mutable_indices=unique,
                context_before_index=before,
                context_after_index=after,
            )
        )
    return clusters


def count_internal_spills(
    translations: list[str],
    indices: tuple[int, ...],
    sources: list[str] | None = None,
) -> int:
    count = 0
    for left, right in zip(indices, indices[1:], strict=False):
        prev_src = sources[left] if sources and left < len(sources) else None
        next_src = sources[right] if sources and right < len(sources) else None
        if looks_like_fragment_spill(
            translations[left],
            translations[right],
            previous_source=prev_src,
            next_source=next_src,
        ):
            count += 1
    return count


def count_boundary_spills(
    translations: list[str],
    cluster: FragmentCluster,
    sources: list[str] | None = None,
) -> int:
    count = 0
    first = cluster.mutable_indices[0]
    last = cluster.mutable_indices[-1]
    if cluster.context_before_index is not None:
        before = cluster.context_before_index
        if looks_like_fragment_spill(
            translations[before],
            translations[first],
            previous_source=sources[before] if sources and before < len(sources) else None,
            next_source=sources[first] if sources and first < len(sources) else None,
        ):
            count += 1
    if cluster.context_after_index is not None:
        after = cluster.context_after_index
        if looks_like_fragment_spill(
            translations[last],
            translations[after],
            previous_source=sources[last] if sources and last < len(sources) else None,
            next_source=sources[after] if sources and after < len(sources) else None,
        ):
            count += 1
    return count


def _syllable_range_penalty(text: str, syllable_range: list[int] | tuple[int, int] | None) -> int:
    if not syllable_range or len(syllable_range) < 2:
        return 0
    low = int(syllable_range[0])
    high = int(syllable_range[1])
    if high < low:
        low, high = high, low
    count = count_vietnamese_syllables(text)
    if count < low:
        return low - count
    if count > high:
        return count - high
    return 0


def cluster_fit_score(
    translations: list[str],
    segments: list[dict[str, Any]],
    cluster: FragmentCluster,
) -> tuple[int, int, int]:
    """Lower is better: (internal_spills, boundary_spills, syllable_penalty_sum)."""
    sources = [str(segment.get("text") or "") for segment in segments]
    internal = count_internal_spills(translations, cluster.mutable_indices, sources=sources)
    boundary = count_boundary_spills(translations, cluster, sources=sources)
    syllable_penalty = 0
    for index in cluster.mutable_indices:
        syllable_penalty += _syllable_range_penalty(
            translations[index],
            segments[index].get("target_vi_syllable_range"),
        )
    return (internal, boundary, syllable_penalty)


def should_accept_repair(old_metrics: RepairMetrics, new_metrics: RepairMetrics) -> bool:
    """Deterministic acceptance comparator (lower spill is primary)."""
    if new_metrics.semantic_critical:
        return False
    if new_metrics.boundary_spills > old_metrics.boundary_spills:
        return False
    if new_metrics.boundary_spills > 0 and old_metrics.boundary_spills == 0:
        return False
    if new_metrics.internal_spills >= old_metrics.internal_spills:
        return False
    # Internal spills must decrease; syllable may regress slightly when clearing spills.
    if new_metrics.syllable_penalty > old_metrics.syllable_penalty + MAX_REPAIR_SYLLABLE_PENALTY_REGRESSION:
        return False
    return True


def build_cluster_payloads(
    segments: list[dict[str, Any]],
    clusters: list[FragmentCluster],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for cluster in clusters:
        mutable = []
        for index in cluster.mutable_indices:
            segment = segments[index]
            mutable.append(
                {
                    "segment_id": segment.get("index", index),
                    "slot_index": index,
                    "source_text": str(segment.get("text") or ""),
                    "current_translation": str(segment.get("translation") or ""),
                    "speech_target_duration_sec": (segment.get("timing_profile") or {}).get(
                        "speech_target_duration"
                    ),
                    "target_vi_syllable_range": segment.get("target_vi_syllable_range"),
                }
            )

        def _context(index: int | None) -> dict[str, Any] | None:
            if index is None:
                return None
            segment = segments[index]
            return {
                "segment_id": segment.get("index", index),
                "slot_index": index,
                "source_text": str(segment.get("text") or ""),
                "translation": str(segment.get("translation") or ""),
                "read_only": True,
            }

        payloads.append(
            {
                "cluster_id": cluster.cluster_id,
                "mutable_segments": mutable,
                "context_before": _context(cluster.context_before_index),
                "context_after": _context(cluster.context_after_index),
            }
        )
    return payloads


def parse_repair_clusters(
    data: Any,
    *,
    expected_clusters: list[FragmentCluster],
    segments: list[dict[str, Any]],
) -> dict[int, dict[Any, str]]:
    """Parse repair JSON into {cluster_id: {segment_id: translation}}."""
    if isinstance(data, dict):
        for key in ("clusters", "repairs", "results"):
            nested = data.get(key)
            if isinstance(nested, list):
                data = nested
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError("Repair response must be a JSON array of clusters.")

    expected_by_id = {cluster.cluster_id: cluster for cluster in expected_clusters}
    parsed: dict[int, dict[Any, str]] = {}

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Repair cluster item must be an object.")
        cluster_id = item.get("cluster_id")
        if cluster_id is None:
            raise ValueError("Repair cluster missing cluster_id.")
        cluster_id = int(cluster_id)
        if cluster_id not in expected_by_id:
            raise ValueError(f"Unexpected cluster_id={cluster_id}")
        cluster = expected_by_id[cluster_id]
        raw_segments = item.get("segments") or item.get("mutable_segments") or item.get("translations")
        if not isinstance(raw_segments, list):
            raise ValueError(f"Cluster {cluster_id} missing segments array.")

        expected_ids = [segments[i].get("index", i) for i in cluster.mutable_indices]
        if len(raw_segments) != len(expected_ids):
            raise ValueError(
                f"Cluster {cluster_id} segment count mismatch: "
                f"expected={len(expected_ids)} got={len(raw_segments)}"
            )

        mapping: dict[Any, str] = {}
        for offset, raw in enumerate(raw_segments):
            if not isinstance(raw, dict):
                raise ValueError(f"Cluster {cluster_id} segment entry must be object.")
            segment_id = raw.get("segment_id", expected_ids[offset])
            expected_id = expected_ids[offset]
            if segment_id != expected_id:
                raise ValueError(
                    f"Cluster {cluster_id} segment_id mismatch at offset {offset}: "
                    f"{segment_id} != {expected_id}"
                )
            text = str(raw.get("translation") or raw.get("text") or "").strip()
            if not text:
                raise ValueError(f"Cluster {cluster_id} empty translation for segment_id={segment_id}")
            mapping[expected_id] = text

        if set(mapping) != set(expected_ids):
            raise ValueError(f"Cluster {cluster_id} segment_id set mismatch.")
        parsed[cluster_id] = mapping

    if set(parsed) != set(expected_by_id):
        missing = set(expected_by_id) - set(parsed)
        raise ValueError(f"Repair response missing clusters: {sorted(missing)}")
    return parsed


def validate_cluster_repair(
    segments: list[dict[str, Any]],
    cluster: FragmentCluster,
    repaired_by_id: dict[Any, str],
) -> tuple[bool, str | None, list[str]]:
    """Return (accepted, reject_reason, mutable_new_texts)."""
    old_translations = [str(segment.get("translation") or "") for segment in segments]
    candidate_translations = list(old_translations)
    mutable_sources: list[str] = []
    mutable_old: list[str] = []
    mutable_new: list[str] = []

    for index in cluster.mutable_indices:
        segment = segments[index]
        segment_id = segment.get("index", index)
        if segment_id not in repaired_by_id:
            return False, "id_mismatch", []
        new_text = str(repaired_by_id[segment_id]).strip()
        if not new_text:
            return False, "empty_translation", []
        candidate_translations[index] = new_text
        mutable_sources.append(str(segment.get("text") or ""))
        mutable_old.append(old_translations[index])
        mutable_new.append(new_text)

    old_score = cluster_fit_score(old_translations, segments, cluster)
    new_score = cluster_fit_score(candidate_translations, segments, cluster)

    joined_source = " ".join(mutable_sources)
    joined_old = " ".join(mutable_old)
    joined_new = " ".join(mutable_new)
    semantic = evaluate_semantic_safeguards(
        joined_new,
        source_text=joined_source,
        reference_text=joined_old,
    )
    semantic_critical = bool(semantic.get("critical_violation"))

    old_metrics = RepairMetrics(
        internal_spills=old_score[0],
        boundary_spills=old_score[1],
        syllable_penalty=old_score[2],
        semantic_critical=False,
    )
    new_metrics = RepairMetrics(
        internal_spills=new_score[0],
        boundary_spills=new_score[1],
        syllable_penalty=new_score[2],
        semantic_critical=semantic_critical,
    )

    if semantic_critical:
        reasons = ",".join(semantic.get("rejection_reasons") or [])
        return False, f"semantic_regression:{reasons}", []
    if new_metrics.internal_spills > 0 and old_metrics.internal_spills > 0:
        if new_metrics.internal_spills >= old_metrics.internal_spills:
            return False, "internal_spill_remaining", []
    if new_metrics.boundary_spills > old_metrics.boundary_spills:
        return False, "boundary_regression", []
    if new_metrics.boundary_spills > 0 and old_metrics.boundary_spills == 0:
        return False, "boundary_regression", []
    if new_metrics.syllable_penalty > old_metrics.syllable_penalty + MAX_REPAIR_SYLLABLE_PENALTY_REGRESSION:
        return False, "syllable_regression", []
    if not should_accept_repair(old_metrics, new_metrics):
        return False, "score_not_improved", []
    return True, None, mutable_new


def apply_accepted_cluster_repair(
    segments: list[dict[str, Any]],
    cluster: FragmentCluster,
    repaired_by_id: dict[Any, str],
) -> None:
    for index in cluster.mutable_indices:
        segment = segments[index]
        segment_id = segment.get("index", index)
        text = repaired_by_id[segment_id]
        segment["translation"] = text
        candidates = segment.get("translation_candidates")
        if isinstance(candidates, list) and candidates:
            selected = dict(candidates[0]) if isinstance(candidates[0], dict) else {}
            selected.update(
                {
                    "text": text,
                    "style": selected.get("style") or "natural",
                    "meaning_notes": list(selected.get("meaning_notes") or []),
                    "candidate_source": "fragment_rebalance",
                }
            )
            segment["translation_candidates"] = [selected]
            segment["selected_candidate_index"] = 0
            segment["selected_candidate_style"] = selected.get("style")
            segment["translation_candidate_source"] = "fragment_rebalance"
        segment["fragment_rebalance_applied"] = True


def rebalance_fragment_spills(
    segments: list[dict[str, Any]],
    *,
    repair_fn: Callable[..., Any],
    source_lang: str,
    target_lang: str,
) -> dict[str, Any]:
    """Detect clusters, call repair_fn at most once, validate and overlay per cluster.

    repair_fn(cluster_payloads, source=..., target=...) -> raw response text or already-parsed list/dict.
    """
    translations = [str(segment.get("translation") or "") for segment in segments]
    sources = [str(segment.get("text") or "") for segment in segments]
    pairs = find_fragment_spill_pairs(translations, sources=sources)
    clusters = build_fragment_clusters(translations, sources=sources)
    diagnostics: dict[str, Any] = {
        "fragment_pairs_detected": len(pairs),
        "repair_cluster_count": len(clusters),
        "cluster_count": len(clusters),
        "repair_calls": 0,
        "repair_requested": False,
        "repair_cluster_accepted": [],
        "repair_cluster_rejected": [],
        "accepted_clusters": [],
        "rejected_clusters": [],
        "skipped": False,
    }
    if not clusters:
        diagnostics["skipped"] = True
        return diagnostics

    payloads = build_cluster_payloads(segments, clusters)
    diagnostics["repair_calls"] = 1
    diagnostics["repair_requested"] = True
    try:
        raw = repair_fn(payloads, source=source_lang, target=target_lang)
    except Exception as error:
        logger.warning("Fragment rebalance repair call failed: %s", error)
        diagnostics["error"] = "api_error"
        diagnostics["repair_api_failed"] = True
        diagnostics["repair_rejection_reason"] = f"api_error:{type(error).__name__}"
        return diagnostics

    try:
        if isinstance(raw, (dict, list)):
            data = raw
        else:
            from .translation_candidate_llm import parse_fragment_repair_response

            data = parse_fragment_repair_response(str(raw))
        repaired = parse_repair_clusters(data, expected_clusters=clusters, segments=segments)
    except Exception as error:
        logger.warning("Fragment rebalance parse failed: %s", error)
        reason = "schema_mismatch" if isinstance(error, ValueError) and "mismatch" in str(error) else "parse_error"
        if isinstance(error, ValueError) and "segment_id" in str(error):
            reason = "id_mismatch"
        diagnostics["error"] = reason
        diagnostics["repair_rejection_reason"] = f"{reason}:{type(error).__name__}"
        return diagnostics

    for cluster in clusters:
        mapping = repaired.get(cluster.cluster_id) or {}
        accepted, reason, _ = validate_cluster_repair(segments, cluster, mapping)
        if not accepted:
            entry = {"cluster_id": cluster.cluster_id, "reason": reason or "unknown"}
            diagnostics["rejected_clusters"].append(entry)
            diagnostics["repair_cluster_rejected"].append(entry)
            continue
        apply_accepted_cluster_repair(segments, cluster, mapping)
        diagnostics["accepted_clusters"].append(cluster.cluster_id)
        diagnostics["repair_cluster_accepted"].append(cluster.cluster_id)

    return diagnostics
