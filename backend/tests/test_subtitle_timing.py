import json
from pathlib import Path

from dv_backend.subtitle_timing import (
    allocate_proportional_cues,
    build_subtitle_cues,
    filter_subtitle_segments,
    hash_subtitle_cues,
    hash_subtitle_track_body,
    load_canonical_subtitle_track,
    map_chunks_to_asr_timeline,
    resolve_overlapping_cues,
    resolve_subtitle_track,
    segment_subtitle_end,
    segment_subtitle_start,
    split_for_subtitle_display,
    split_translation_sentences,
    write_canonical_subtitle_track,
)


def test_split_translation_sentences_without_trailing_space() -> None:
    assert split_translation_sentences("Câu một. Câu hai! Câu ba?") == [
        "Câu một.",
        "Câu hai!",
        "Câu ba?",
    ]


def test_split_for_subtitle_display_splits_commas_when_no_periods() -> None:
    chunks = split_for_subtitle_display(
        "Anh ấy đi ra thật nhanh, nhìn lại phía sau một lần nữa, "
        "rồi bước tiếp về phía con phố đông đúc phía trước"
    )
    assert len(chunks) >= 2
    assert all(len(chunk) <= 58 for chunk in chunks)


def test_build_subtitle_cues_shows_one_sentence_at_a_time() -> None:
    cues = build_subtitle_cues(
        [
            {
                "start": 1.0,
                "placement_start": 1.0,
                "repaired_duration": 9.0,
                "translation": "Câu thứ nhất. Câu thứ hai! Câu thứ ba?",
            }
        ]
    )
    assert len(cues) == 3
    assert cues[0]["text"] == "Câu thứ nhất."
    assert cues[1]["text"] == "Câu thứ hai!"
    assert cues[2]["text"] == "Câu thứ ba?"
    assert cues[0]["start"] == 1.0
    assert abs(cues[0]["end"] - cues[1]["start"]) < 0.001
    assert abs(cues[1]["end"] - cues[2]["start"]) < 0.001
    assert abs(cues[2]["end"] - 10.0) < 0.001


def test_map_chunks_to_asr_timeline_uses_unit_boundaries() -> None:
    cues = map_chunks_to_asr_timeline(
        ["Ngắn.", "Dài hơn nhiều."],
        [
            {"text": "Ngắn.", "start": 0.0, "end": 1.0},
            {"text": "Dài", "start": 1.0, "end": 2.0},
            {"text": "hơn", "start": 2.0, "end": 3.0},
            {"text": "nhiều.", "start": 3.0, "end": 4.0},
        ],
        window_start=5.0,
        window_duration=4.0,
    )
    assert cues is not None
    assert len(cues) == 2
    assert cues[0]["start"] == 5.0
    assert cues[0]["end"] == 6.0
    assert cues[1]["start"] == 6.0
    assert cues[1]["end"] == 9.0


def test_resolve_overlapping_cues_trims_previous_end() -> None:
    resolved = resolve_overlapping_cues(
        [
            {"start": 1.0, "end": 4.0, "text": "A"},
            {"start": 3.5, "end": 6.0, "text": "B"},
        ]
    )
    assert resolved[0]["end"] == 3.5
    assert resolved[1]["start"] == 3.5


def test_subtitle_playback_window_matches_mix_cap() -> None:
    segments = [
        {
            "index": 0,
            "start": 85.6,
            "placement_start": 85.6,
            "repaired_duration": 84.12,
            "translation": "Câu một. Câu hai.",
        },
        {
            "index": 1,
            "start": 144.72,
            "placement_start": 144.72,
            "repaired_duration": 7.74,
            "translation": "Câu ba.",
        },
    ]
    cues = build_subtitle_cues(segments)
    assert cues[0]["start"] == 85.6
    assert cues[-2]["end"] <= 144.72 + 0.05
    assert all(cue["end"] - cue["start"] >= 0.24 for cue in cues)


def test_enforce_monotonic_cues_rescales_overlapping_asr_chunks() -> None:
    from dv_backend.subtitle_timing import enforce_monotonic_cues

    normalized = enforce_monotonic_cues(
        [
            {"start": 10.0, "end": 10.2, "text": "A"},
            {"start": 10.1, "end": 10.3, "text": "B"},
            {"start": 10.15, "end": 10.35, "text": "C"},
        ],
        window_start=10.0,
        window_end=13.0,
    )
    assert len(normalized) == 3
    assert normalized[0]["start"] == 10.0
    assert normalized[-1]["end"] == 13.0
    assert normalized[1]["start"] >= normalized[0]["end"] - 0.001
    assert all(cue["end"] - cue["start"] >= 0.24 for cue in normalized)


def test_split_for_subtitle_display_skips_dot_chunks() -> None:
    chunks = split_for_subtitle_display("Nghe chưa? Ngươi dám.")
    assert "." not in chunks


def test_asr_quality_gate_falls_back_when_many_min_duration_cues() -> None:
    from dv_backend.subtitle_timing import _asr_cues_are_usable

    cues = [{"start": 0.0, "end": 0.25, "text": f"C{i}."} for i in range(8)]
    assert not _asr_cues_are_usable(cues, window_duration=10.0, chunk_count=8)
    assert _asr_cues_are_usable(
        [{"start": 0.0, "end": 2.0, "text": "A"}, {"start": 2.0, "end": 4.0, "text": "B"}],
        window_duration=4.0,
        chunk_count=2,
    )


def test_clip_aligned_units_caps_timestamps_to_wav_duration() -> None:
    from dv_backend.subtitle_timing import _clip_aligned_units

    clipped = _clip_aligned_units(
        [
            {"text": "A", "start": 0.0, "end": 5.0},
            {"text": "B", "start": 8.0, "end": 104.0},
        ],
        max_duration=10.0,
    )
    assert clipped[0]["end"] == 5.0
    assert clipped[1]["start"] == 8.0
    assert clipped[1]["end"] == 10.0


def test_resolve_subtitle_speech_window_uses_aligned_units(tmp_path: Path) -> None:
    import wave

    from dv_backend.subtitle_timing import resolve_subtitle_speech_window

    wav_path = tmp_path / "clip.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000 * 60)

    start, end = resolve_subtitle_speech_window(
        window_start=10.0,
        window_end=70.0,
        wav_path=wav_path,
        ffmpeg_path=None,
        aligned_units=[
            {"text": "A", "start": 0.0, "end": 20.0},
            {"text": "B", "start": 20.0, "end": 25.0},
        ],
    )
    assert start == 10.0
    assert abs(end - 35.0) < 0.01


def test_allocate_proportional_cues_weights_longer_sentence() -> None:
    cues = allocate_proportional_cues(["Ngắn.", "Dài hơn nhiều so với câu trước."], 1.0, 10.0)
    assert len(cues) == 2
    assert (cues[1]["end"] - cues[1]["start"]) > (cues[0]["end"] - cues[0]["start"])


def _dub_word(text: str, start: float, end: float) -> dict:
    return {
        "text": text,
        "start": start,
        "end": end,
        "absolute_start": start,
        "absolute_end": end,
    }


def test_dub_words_short_sentence_single_cue() -> None:
    from dv_backend.subtitle_timing import build_cues_from_dub_words

    segment = {
        "index": 0,
        "placement_start": 0.0,
        "repaired_duration": 1.4,
        "dub_words": [
            _dub_word("Xin", 0.0, 0.4),
            _dub_word("chào", 0.4, 0.9),
            _dub_word("bạn.", 0.9, 1.4),
        ],
    }
    cues = build_cues_from_dub_words(segment, settings={}, language="vi")
    assert len(cues) == 1
    assert cues[0]["text"] == "Xin chào bạn."


def test_dub_words_long_line_splits_into_multiple_cues() -> None:
    from dv_backend.subtitle_timing import build_cues_from_dub_words

    words = [_dub_word(f"word{i:02d}", i * 0.4, i * 0.4 + 0.35) for i in range(12)]
    segment = {"index": 0, "placement_start": 0.0, "repaired_duration": 5.2, "dub_words": words}
    # Tight per-line limit forces multiple cues even inside one uninterrupted phrase.
    settings = {"subtitle_max_chars_per_line": 20, "subtitle_max_lines_per_cue": 1}
    cues = build_cues_from_dub_words(segment, settings=settings, language="vi")
    assert len(cues) >= 2
    assert all(len(cue["text"]) <= 20 for cue in cues)


def test_dub_words_min_cue_duration_respected() -> None:
    from dv_backend.subtitle_timing import build_cues_from_dub_words

    segment = {
        "index": 0,
        "placement_start": 0.0,
        "dub_words": [_dub_word("Ừ.", 0.0, 0.1)],
    }
    settings = {"subtitle_min_cue_duration_ms": 800}
    cues = build_cues_from_dub_words(segment, settings=settings, language="vi")
    assert len(cues) == 1
    assert (cues[0]["end"] - cues[0]["start"]) >= 0.8 - 1e-6


def test_fallback_without_dub_words_respects_max_chars_setting() -> None:
    long_text = (
        "Đây là một câu rất dài không có dấu chấm ở giữa nên phải được chia nhỏ "
        "thành nhiều dòng phụ đề khác nhau để dễ đọc hơn trên màn hình"
    )
    cues = build_subtitle_cues(
        [
            {
                "index": 0,
                "start": 0.0,
                "placement_start": 0.0,
                "repaired_duration": 12.0,
                "translation": long_text,
            }
        ],
        settings={"subtitle_max_chars_per_line": 24, "subtitle_max_lines_per_cue": 1},
    )
    assert len(cues) >= 2
    assert all(len(cue["text"]) <= 24 for cue in cues)


def test_thai_dub_words_join_without_spaces() -> None:
    from dv_backend.subtitle_timing import build_cues_from_dub_words

    segment = {
        "index": 0,
        "placement_start": 0.0,
        "repaired_duration": 1.0,
        "dub_words": [
            _dub_word("สวัสดี", 0.0, 0.5),
            _dub_word("ครับ", 0.5, 1.0),
        ],
    }
    cues = build_cues_from_dub_words(segment, settings={}, language="th")
    assert len(cues) == 1
    # Thai must not be joined with ASCII spaces.
    assert " " not in cues[0]["text"]
    assert cues[0]["text"] == "สวัสดีครับ"


def test_thai_display_split_uses_character_limit() -> None:
    thai_text = "สวัสดีครับทุกคนวันนี้เรามาพูดคุยเรื่องการพากย์เสียงภาษาไทยกันนะครับ"
    chunks = split_for_subtitle_display(thai_text, max_chars=15, language="th")
    assert len(chunks) >= 2
    assert all(len(chunk) <= 15 for chunk in chunks)
    assert all(" " not in chunk for chunk in chunks)


def test_mixed_alignment_each_segment_gets_cues() -> None:
    segments = [
        {
            "index": 0,
            "placement_start": 0.0,
            "repaired_duration": 2.0,
            "tts_spoken_text": "Segment A có dub words",
            "dub_words": [_dub_word("Một", 0.0, 0.5), _dub_word("hai.", 0.5, 1.0)],
        },
        {
            "index": 1,
            "start": 2.0,
            "placement_start": 2.5,
            "repaired_duration": 1.5,
            "tts_spoken_text": "Segment B thiếu words",
            "dub_words": [],
        },
        {
            "index": 2,
            "start": 4.0,
            "placement_start": 4.0,
            "repaired_duration": 1.8,
            "tts_spoken_text": "Segment C không có field dub",
        },
    ]
    cues = build_subtitle_cues(segments, tts_asr_align=False)
    assert len(cues) >= 3
    assert any(float(cue["start"]) < 1.0 for cue in cues)
    assert any(2.4 <= float(cue["start"]) <= 2.6 for cue in cues)


def test_invalid_dub_words_fallback_to_proportional() -> None:
    from dv_backend.final_dub_alignment import segment_has_usable_dub_words

    segment = {
        "index": 0,
        "placement_start": 1.0,
        "repaired_duration": 2.0,
        "tts_spoken_text": "fallback khi words invalid",
        "dub_words": [{"text": "bad", "start": float("nan"), "end": 1.0}],
    }
    assert segment_has_usable_dub_words(segment) is False
    cues = build_subtitle_cues([segment], tts_asr_align=False)
    assert len(cues) >= 1
    assert float(cues[0]["start"]) >= 1.0


def test_zero_placement_start_uses_playback_not_source() -> None:
    segment = {
        "index": 0,
        "start": 3.0,
        "placement_start": 0.0,
        "repaired_duration": 2.0,
        "tts_spoken_text": "zero placement",
    }
    cues = build_subtitle_cues([segment], tts_asr_align=False)
    assert float(cues[0]["start"]) == 0.0


def test_tts_spoken_text_only_segment_still_builds_cues() -> None:
    from dv_backend.subtitle_timing import resolve_spoken_subtitle_text

    segment = {
        "index": 0,
        "placement_start": 0.0,
        "repaired_duration": 2.0,
        "tts_spoken_text": "Chỉ có spoken text",
    }
    assert resolve_spoken_subtitle_text(segment) == "Chỉ có spoken text"
    cues = build_subtitle_cues([segment], tts_asr_align=False)
    assert len(cues) >= 1


def test_filter_subtitle_segments_includes_tts_spoken_text_only() -> None:
    segments = [
        {"index": 0, "tts_spoken_text": "Chỉ spoken", "placement_start": 0.0, "repaired_duration": 2.0},
        {"index": 1, "translation": "Có translation", "placement_start": 2.0, "repaired_duration": 2.0},
    ]
    assert len(filter_subtitle_segments(segments)) == 2


def test_resolve_subtitle_track_is_deterministic_for_render_and_qc() -> None:
    segments = [
        {
            "index": 0,
            "translation": "A valid dub.",
            "placement_start": 0.0,
            "repaired_duration": 2.0,
            "dub_words": [
                {
                    "text": "A",
                    "start": 0.0,
                    "end": 0.4,
                    "absolute_start": 0.0,
                    "absolute_end": 0.4,
                    "alignment": "exact",
                },
                {
                    "text": "valid",
                    "start": 0.4,
                    "end": 1.0,
                    "absolute_start": 0.4,
                    "absolute_end": 1.0,
                    "alignment": "exact",
                },
            ],
            "dub_alignment_status": "aligned",
        },
        {
            "index": 1,
            "translation": "Fallback segment.",
            "placement_start": 2.0,
            "repaired_duration": 2.0,
            "dub_words": [],
        },
        {
            "index": 2,
            "tts_spoken_text": "Chỉ có spoken text.",
            "placement_start": 4.0,
            "repaired_duration": 2.0,
        },
    ]
    render_track = resolve_subtitle_track(segments, tts_asr_align=False)
    qc_track = resolve_subtitle_track(segments, tts_asr_align=False)
    assert render_track["segments"] == qc_track["segments"]
    assert render_track["cues"] == qc_track["cues"]
    assert len(render_track["segments"]) == 3
    assert len(render_track["cues"]) >= 3


def test_canonical_subtitle_track_roundtrip(tmp_path: Path) -> None:
    cues = [
        {"start": 0.0, "end": 1.0, "text": "A"},
        {"start": 1.0, "end": 2.0, "text": "B"},
    ]
    path = write_canonical_subtitle_track(tmp_path, cues=cues, segment_indices=[0, 1])
    assert path.is_file()
    loaded = load_canonical_subtitle_track(tmp_path)
    assert loaded is not None
    assert loaded["cues"] == cues
    assert loaded["content_hash"] == hash_subtitle_track_body(
        cues=cues,
        segment_indices=[0, 1],
    )
    assert loaded["cue_count"] == 2


def test_canonical_subtitle_track_rejects_tampered_hash(tmp_path: Path) -> None:
    write_canonical_subtitle_track(tmp_path, cues=[{"start": 0.0, "end": 1.0, "text": "A"}])
    path = tmp_path / "artifacts" / "subtitle_track.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cues"][0]["text"] = "Tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_canonical_subtitle_track(tmp_path) is None
