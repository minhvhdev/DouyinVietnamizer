"""Inspect problematic segments for ChatGPT report."""
from pathlib import Path

from dv_backend.checkpoints import load_checkpoint
from dv_backend.config import AppConfig

JOB = "f7620299-9f3c-42bd-a1e6-6f3ec0b542a2"


def show(segs, idxs):
    by = {int(s["index"]): s for s in segs}
    for i in idxs:
        for j in range(i - 1, i + 2):
            s = by.get(j)
            if not s:
                continue
            print(f"\n--- #{j+1} index={j} start={s.get('start')} end={s.get('end')} place={s.get('placement_start')} rd={s.get('repaired_duration')} ---")
            print("cn:", (s.get("text") or "")[:120])
            print("vi:", (s.get("translation") or "")[:120])
            print("spoken:", (s.get("tts_spoken_text") or "")[:120])
            print("method:", s.get("repaired_method"), "status:", s.get("timing_status"))
            print("cluster:", s.get("cluster_source_indices"))
            print("tts_path:", s.get("tts_path"))
            p = Path(str(s.get("tts_path") or ""))
            print("tts_exists:", p.is_file(), "size:", p.stat().st_size if p.is_file() else None)


def main() -> None:
    cfg = AppConfig.from_env()
    for step in ("duration_repair", "align_final_dub"):
        cp = load_checkpoint(cfg.data_dir, JOB, step) or {}
        segs = cp.get("segments") or []
        print(f"\n===== {step} n={len(segs)} =====")
        # UI shows 1-based #27, #37, #72, #73
        show(segs, [26, 36, 71, 72])

        # Find ellipsis-leading / mid-word style joins
        bad = []
        ordered = sorted(segs, key=lambda s: int(s["index"]))
        for a, b in zip(ordered, ordered[1:]):
            ta = str(a.get("tts_spoken_text") or a.get("translation") or "").strip()
            tb = str(b.get("tts_spoken_text") or b.get("translation") or "").strip()
            if not ta or not tb:
                continue
            score = 0
            if ta.endswith("...") or ta.endswith("…") or ta.endswith("gãy...") or ta.endswith("gãy…"):
                score += 1
            if tb.startswith("...") or tb.startswith("…") or (tb[:1].islower() and not ta.endswith((".", "!", "?"))):
                score += 1
            if ta.rstrip(".").endswith(("gãy", "gãy ")) and tb.lower().startswith("chân"):
                score += 2
            if score >= 1 and (tb.startswith("...") or tb[:1].islower() or ta.endswith("...")):
                bad.append((a.get("index"), b.get("index"), ta[-40:], tb[:40]))
        print("\nsuspicious_breaks", len(bad))
        for row in bad[:25]:
            print(" ", row)


if __name__ == "__main__":
    main()
