from dv_backend.segment_mix import (
    annotate_segment_mix_caps,
    build_background_narration_mix_filter,
    build_narration_amix_filter,
    build_narration_segment_filter,
    duration_lock_audio_chain,
    format_mix_target_duration,
    scaled_segment_fades,
)


def test_scaled_segment_fades_shrinks_for_short_clips() -> None:
    fade_in, fade_out, fade_out_start = scaled_segment_fades(0.06)
    assert fade_in <= 0.012
    assert fade_out <= 0.028
    assert fade_out_start >= 0.0


def test_annotate_segment_mix_caps_prevents_overlap() -> None:
    entries = [
        {"placement_start": 0.0, "clip_duration": 3.5},
        {"placement_start": 3.0, "clip_duration": 2.0},
    ]
    annotate_segment_mix_caps(entries)
    assert entries[0]["max_duration"] == 2.975
    assert entries[1]["max_duration"] is None
    assert entries[0]["mix_would_clip_sec"] == 0.525


def test_build_narration_segment_filter_does_not_hard_clip_by_default() -> None:
    expr = build_narration_segment_filter(
        1,
        placement_start=2.5,
        clip_duration=3.6,
        max_duration=3.0,
    )
    assert "atrim=" not in expr
    assert "afade=t=in" in expr
    assert "afade=t=out" in expr
    assert "adelay=2500" in expr
    assert expr.endswith("[seg1]")


def test_build_narration_segment_filter_legacy_hard_clip_opt_in() -> None:
    expr = build_narration_segment_filter(
        1,
        placement_start=2.5,
        clip_duration=3.6,
        max_duration=3.0,
        allow_hard_clip=True,
    )
    assert "atrim=0:3.000" in expr


def test_build_narration_amix_filter_uses_dropout_transition() -> None:
    expr = build_narration_amix_filter(2)
    assert "amix=inputs=2" in expr
    assert "dropout_transition=0.040" in expr


def test_duration_lock_uses_high_precision_atrim() -> None:
    assert duration_lock_audio_chain(5.123456789) == "apad,atrim=0:5.123457,asetpts=PTS-STARTPTS"
    assert format_mix_target_duration(5.0) == "5.000000"


def test_build_background_narration_mix_filter_locks_both_inputs_before_amix() -> None:
    duck = build_background_narration_mix_filter(duck=True, target_duration_sec=5.0)
    bg_only = build_background_narration_mix_filter(duck=False, target_duration_sec=5.0)
    assert duck.count("apad,atrim=0:5.000000") == 2
    assert bg_only.count("apad,atrim=0:5.000000") == 2
    assert "sidechaincompress" in duck
    assert "sidechaincompress" not in bg_only
    assert duck.endswith("[mixed]")
    assert bg_only.endswith("[mixed]")
