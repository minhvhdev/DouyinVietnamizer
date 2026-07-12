import shutil
import subprocess
from pathlib import Path

import pytest

from dv_backend.adapters.subtitles import (
    build_ass_content,
    build_subtitle_cues,
    ffmpeg_subtitles_filter,
    format_ass_time,
    hex_to_ass_color,
    normalize_font_size,
    normalize_hex_color,
    normalize_position,
    segment_subtitle_end,
    split_translation_sentences,
    subtitles_filter_available,
    write_ass_file,
)


def test_format_ass_time() -> None:
    assert format_ass_time(0) == "0:00:00.00"
    assert format_ass_time(61.5) == "0:01:01.50"


def test_hex_to_ass_color_white_opaque() -> None:
    assert hex_to_ass_color("#FFFFFF", opacity_percent=100) == "&H00FFFFFF"


def test_hex_to_ass_color_with_opacity() -> None:
    assert hex_to_ass_color("#000000", opacity_percent=0) == "&HFF000000"


def test_segment_subtitle_end_prefers_repaired_duration() -> None:
    segment = {"start": 1.0, "end": 3.0, "repaired_duration": 1.5}
    assert segment_subtitle_end(segment) == 2.5


def test_segment_subtitle_uses_placement_start() -> None:
    segment = {"start": 2.0, "placement_start": 1.5, "repaired_duration": 2.0}
    assert segment_subtitle_end(segment) == 3.5
    cues = build_subtitle_cues([{**segment, "translation": "Xin chào bạn."}])
    assert cues[0]["start"] == 1.5
    assert cues[0]["end"] == 3.5


def test_build_ass_content_contains_dialogue() -> None:
    content = build_ass_content(
        [
            {
                "start": 0.0,
                "end": 8.0,
                "repaired_duration": 8.0,
                "translation": "Câu thứ nhất. Câu thứ hai! Câu thứ ba?",
            },
        ],
        font_size=48,
        font_color="#FFFFFF",
        background_color="#000000",
        background_opacity=70,
        background_padding=12,
        position="bottom",
        edge_margin=80,
        play_res_x=1080,
        play_res_y=1920,
    )
    assert content.count("Dialogue:") == 3
    assert "Câu thứ nhất." in content
    assert "Câu thứ hai!" in content
    assert "Câu thứ ba?" in content


def test_build_subtitle_cues_splits_by_sentence_and_allocates_time() -> None:
    cues = build_subtitle_cues(
        [
            {
                "start": 1.0,
                "repaired_duration": 9.0,
                "translation": "Ngắn. Dài hơn nhiều so với câu trước.",
            }
        ]
    )
    assert len(cues) == 2
    assert cues[0]["text"] == "Ngắn."
    assert cues[1]["text"] == "Dài hơn nhiều so với câu trước."
    assert cues[0]["start"] == 1.0
    assert abs(cues[0]["end"] - cues[1]["start"]) < 0.001
    assert abs(cues[1]["end"] - 10.0) < 0.001
    assert (cues[1]["end"] - cues[1]["start"]) > (cues[0]["end"] - cues[0]["start"])


def test_split_translation_sentences() -> None:
    assert split_translation_sentences("A. B! C?") == ["A.", "B!", "C?"]
    assert split_translation_sentences("  ") == []


def test_build_ass_content_position_alignment() -> None:
    bottom = build_ass_content(
        [{"start": 0.0, "end": 1.0, "translation": "A"}],
        font_size=40,
        font_color="#FFFFFF",
        background_color="#000000",
        background_opacity=100,
        background_padding=None,
        position="bottom",
        edge_margin=80,
        play_res_x=1080,
        play_res_y=1920,
    )
    center = build_ass_content(
        [{"start": 0.0, "end": 1.0, "translation": "A"}],
        font_size=40,
        font_color="#FFFFFF",
        background_color="#000000",
        background_opacity=100,
        background_padding=None,
        position="center",
        edge_margin=120,
        play_res_x=1080,
        play_res_y=1920,
    )
    assert ",2,40,40,80,1" in bottom
    assert ",5,40,40,0,1" in center


def test_build_ass_content_edge_margin_for_top_and_bottom() -> None:
    bottom = build_ass_content(
        [{"start": 0.0, "end": 1.0, "translation": "A"}],
        font_size=40,
        font_color="#FFFFFF",
        background_color="#000000",
        background_opacity=100,
        background_padding=None,
        position="bottom",
        edge_margin=0,
        play_res_x=1080,
        play_res_y=1920,
    )
    top = build_ass_content(
        [{"start": 0.0, "end": 1.0, "translation": "A"}],
        font_size=40,
        font_color="#FFFFFF",
        background_color="#000000",
        background_opacity=100,
        background_padding=None,
        position="top",
        edge_margin=0,
        play_res_x=1080,
        play_res_y=1920,
    )
    center = build_ass_content(
        [{"start": 0.0, "end": 1.0, "translation": "A"}],
        font_size=40,
        font_color="#FFFFFF",
        background_color="#000000",
        background_opacity=100,
        background_padding=None,
        position="center",
        edge_margin=120,
        play_res_x=1080,
        play_res_y=1920,
    )
    assert ",2,40,40,0,1" in bottom
    assert ",8,40,40,0,1" in top
    assert ",5,40,40,0,1" in center


def test_build_ass_content_uses_box_background_with_padding() -> None:
    brown = hex_to_ass_color("#2A1A00", opacity_percent=100)
    content = build_ass_content(
        [{"start": 0.0, "end": 1.0, "translation": "Hello"}],
        font_size=48,
        font_color="#FFFFFF",
        background_color="#2A1A00",
        background_opacity=100,
        background_padding=14,
        position="bottom",
        edge_margin=80,
        play_res_x=1080,
        play_res_y=1920,
    )
    assert brown in content
    assert ",4,14,0," in content
    assert "{\\bord14\\xbord14\\ybord14\\shad0" in content


def test_write_ass_file(tmp_path: Path) -> None:
    path = tmp_path / "subtitles.ass"
    write_ass_file(
        path,
        [{"start": 0.0, "end": 1.0, "translation": "Hello"}],
        {
            "subtitle_font_size": 48,
            "subtitle_font_color": "#FFFFFF",
            "subtitle_background_color": "#000000",
            "subtitle_background_opacity": 70,
            "subtitle_position": "bottom",
        },
        play_res_x=1080,
        play_res_y=1920,
    )
    assert path.is_file()
    assert "Hello" in path.read_text(encoding="utf-8-sig")


def test_normalizers() -> None:
    assert normalize_font_size("999") == 120
    assert normalize_font_size("bad") == 48
    assert normalize_hex_color("bad", fallback="#ABCDEF") == "#ABCDEF"
    assert normalize_position("BOTTOM") == "bottom"
    assert normalize_position("left") == "bottom"


def test_ffmpeg_subtitles_filter_escapes_path(tmp_path: Path) -> None:
    ass_path = tmp_path / "sub titles.ass"
    ass_path.write_text("test", encoding="utf-8")
    filter_value = ffmpeg_subtitles_filter(ass_path)
    assert filter_value.startswith("subtitles=filename='")
    if len(ass_path.drive) == 2 and ass_path.drive[1] == ":":
        assert r"\:" in filter_value


def test_subtitles_filter_available_detects_missing_filter(tmp_path: Path, monkeypatch) -> None:
    ffmpeg_path = tmp_path / "ffmpeg"
    ffmpeg_path.write_text("fake", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=" T. a drawtext\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert subtitles_filter_available(ffmpeg_path) is False

    def fake_run_with_subtitles(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=" T. subtitles V->V\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run_with_subtitles)
    assert subtitles_filter_available(ffmpeg_path) is True


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_ass_background_box_renders_with_ffmpeg(tmp_path: Path) -> None:
    ass_path = tmp_path / "test.ass"
    frame_path = tmp_path / "frame.png"
    content = build_ass_content(
        [{"start": 0.5, "end": 2.5, "translation": "Subtitle box test."}],
        font_size=48,
        font_color="#FFFFFF",
        background_color="#000000",
        background_opacity=100,
        background_padding=12,
        position="bottom",
        edge_margin=80,
        play_res_x=1280,
        play_res_y=720,
    )
    ass_path.write_text(content, encoding="utf-8-sig")
    vf = ffmpeg_subtitles_filter(ass_path)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=orange:s=1280x720:d=3",
            "-vf",
            vf,
            "-frames:v",
            "1",
            "-ss",
            "1",
            str(frame_path),
        ],
        check=True,
        capture_output=True,
    )

    from PIL import Image

    image = Image.open(frame_path).convert("RGB")
    width, height = image.size
    dark = 0
    total = 0
    for y in range(int(height * 0.72), int(height * 0.92)):
        for x in range(int(width * 0.15), int(width * 0.85)):
            r, g, b = image.getpixel((x, y))
            total += 1
            if r < 80 and g < 80 and b < 80:
                dark += 1
    assert dark / max(total, 1) > 0.01
