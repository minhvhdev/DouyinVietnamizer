def test_merge_incomplete_sentence_segments_respects_max_duration() -> None:
    from dv_backend.segmentation import merge_incomplete_sentence_segments

    merged = merge_incomplete_sentence_segments(
        [
            {"start": 0.0, "end": 4.0, "text": "第一段没有句号"},
            {"start": 4.1, "end": 8.0, "text": "第二段也没有"},
            {"start": 8.1, "end": 12.0, "text": "第三段继续"},
        ],
        max_merged_duration_sec=8.0,
    )
    assert len(merged) == 2
    assert merged[0]["end"] == 8.0
    assert merged[1]["text"].startswith("第三段")


def test_merge_orphan_word_cues_joins_neighbors() -> None:
    from dv_backend.subtitle_timing import _merge_orphan_word_cues

    cues = [
        {"start": 0.0, "end": 1.0, "text": "Bình tĩnh, phải kiên trì đến khi Trấn Ma"},
        {"start": 1.0, "end": 1.4, "text": "Ty"},
        {"start": 1.4, "end": 2.0, "text": "tới."},
    ]
    merged = _merge_orphan_word_cues(cues, max_chars=80, language="vi")
    assert len(merged) == 1
    assert "Trấn Ma Ty tới." in merged[0]["text"]


def test_split_segments_by_alignment_pauses_cuts_near_gap() -> None:
    from dv_backend.segmentation import split_segments_by_alignment_pauses

    # Continuous speech 0-6s with a clear pause around mid clause boundary.
    units = []
    t = 0.0
    chars = list("手的时候你脸上还挂着那该死的笑容到底是如何看待死去的百姓如何")
    for index, ch in enumerate(chars):
        start = t
        end = t + 0.12
        units.append({"start": round(start, 2), "end": round(end, 2), "text": ch})
        t = end
        if index == 15:  # after 容 of 笑容
            t += 0.45

    segments = [
        {
            "start": 0.0,
            "end": 6.0,
            "text": "".join(chars),
            "index": 0,
        }
    ]
    split = split_segments_by_alignment_pauses(segments, units, target_seconds=4.0)
    assert len(split) >= 2
    assert "笑容" in split[0]["text"]
    assert not split[0]["text"].endswith("笑")
    assert split[1]["text"].startswith("到底") or "到底" in split[1]["text"]
    assert split[0]["split_method"] == "alignment_pause"
