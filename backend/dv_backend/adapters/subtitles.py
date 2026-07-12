import json
import re
import subprocess
from pathlib import Path

from ..subtitle_timing import (
    build_subtitle_cues as build_timed_subtitle_cues,
    segment_subtitle_end,
    segment_subtitle_start,
    split_translation_sentences,
)

SUPPORTED_SUBTITLE_POSITIONS = {"bottom", "center", "top"}
DEFAULT_SUBTITLE_FONT_SIZE = 48
DEFAULT_SUBTITLE_FONT_COLOR = "#FFFFFF"
DEFAULT_SUBTITLE_BACKGROUND_COLOR = "#000000"
DEFAULT_SUBTITLE_BACKGROUND_OPACITY = 95
DEFAULT_SUBTITLE_BACKGROUND_PADDING = 12
DEFAULT_SUBTITLE_EDGE_MARGIN = 80
DEFAULT_SUBTITLE_POSITION = "bottom"

_ASS_ALIGNMENT = {
    "bottom": 2,
    "center": 5,
    "top": 8,
}


def normalize_edge_margin(value: object) -> int:
    try:
        margin = int(value)
    except (TypeError, ValueError):
        margin = DEFAULT_SUBTITLE_EDGE_MARGIN
    return max(0, min(margin, 300))


def resolve_margin_v(position: str, edge_margin: int) -> int:
    if position == "center":
        return 0
    return max(0, edge_margin)


def normalize_hex_color(value: str, *, fallback: str) -> str:
    candidate = (value or fallback).strip()
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", candidate):
        return fallback
    return candidate.upper()


def normalize_font_size(value: object) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        size = DEFAULT_SUBTITLE_FONT_SIZE
    return max(16, min(size, 120))


def normalize_background_opacity(value: object) -> int:
    try:
        opacity = int(value)
    except (TypeError, ValueError):
        opacity = DEFAULT_SUBTITLE_BACKGROUND_OPACITY
    return max(0, min(opacity, 100))


def normalize_background_padding(value: object, *, font_size: int) -> int:
    if value in (None, "", "auto"):
        return max(8, int(round(font_size * 0.22)))
    try:
        padding = int(value)
    except (TypeError, ValueError):
        padding = DEFAULT_SUBTITLE_BACKGROUND_PADDING
    return max(4, min(padding, 40))


def normalize_position(value: object) -> str:
    position = str(value or DEFAULT_SUBTITLE_POSITION).strip().lower()
    if position not in SUPPORTED_SUBTITLE_POSITIONS:
        return DEFAULT_SUBTITLE_POSITION
    return position


def hex_to_ass_color(hex_color: str, *, opacity_percent: int = 100) -> str:
    normalized = normalize_hex_color(hex_color, fallback="#FFFFFF")
    red = int(normalized[1:3], 16)
    green = int(normalized[3:5], 16)
    blue = int(normalized[5:7], 16)
    alpha = int(round((100 - opacity_percent) / 100 * 255))
    return f"&H{alpha:02X}{blue:02X}{green:02X}{red:02X}"


def escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def format_ass_time(seconds: float) -> str:
    total = max(0.0, float(seconds))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60
    whole_secs = int(secs)
    centis = int(round((secs - whole_secs) * 100))
    if centis == 100:
        whole_secs += 1
        centis = 0
    return f"{hours}:{minutes:02d}:{whole_secs:02d}.{centis:02d}"


def build_subtitle_cues(
    segments: list[dict],
    *,
    job_dir: Path | None = None,
    settings: dict | None = None,
    vendor_dir: Path | None = None,
    ffmpeg_path: Path | None = None,
    transcribe_fn=None,
    tts_asr_align: bool = False,
) -> list[dict]:
    return build_timed_subtitle_cues(
        segments,
        job_dir=job_dir,
        settings=settings,
        vendor_dir=vendor_dir,
        ffmpeg_path=ffmpeg_path,
        transcribe_fn=transcribe_fn,
        tts_asr_align=tts_asr_align,
    )


def ass_box_tags(outline_padding: int, back_colour: str) -> str:
    return (
        f"{{\\bord{outline_padding}\\xbord{outline_padding}\\ybord{outline_padding}"
        f"\\shad0\\3c{back_colour}&\\4c{back_colour}&}}"
    )


def build_ass_content(
    segments: list[dict],
    *,
    font_size: int,
    font_color: str,
    background_color: str,
    background_opacity: int,
    background_padding: int | None,
    position: str,
    edge_margin: int,
    play_res_x: int,
    play_res_y: int,
    job_dir: Path | None = None,
    settings: dict | None = None,
    vendor_dir: Path | None = None,
    ffmpeg_path: Path | None = None,
    transcribe_fn=None,
    tts_asr_align: bool = False,
) -> str:
    position = normalize_position(position)
    font_size = normalize_font_size(font_size)
    font_color = normalize_hex_color(font_color, fallback=DEFAULT_SUBTITLE_FONT_COLOR)
    background_color = normalize_hex_color(
        background_color,
        fallback=DEFAULT_SUBTITLE_BACKGROUND_COLOR,
    )
    background_opacity = normalize_background_opacity(background_opacity)
    scaled_font_size = max(16, int(round(font_size * (play_res_y / 1080))))
    outline_padding = normalize_background_padding(
        background_padding,
        font_size=scaled_font_size,
    )

    primary_colour = hex_to_ass_color(font_color)
    back_colour = hex_to_ass_color(background_color, opacity_percent=background_opacity)
    alignment = _ASS_ALIGNMENT[position]
    margin_v = resolve_margin_v(position, normalize_edge_margin(edge_margin))
    box_tags = ass_box_tags(outline_padding, back_colour)

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        (
            "Style: Default,Arial,"
            f"{scaled_font_size},{primary_colour},&H000000FF,{back_colour},{back_colour},"
            "0,0,0,0,100,100,0,0,4,"
            f"{outline_padding},0,"
            f"{alignment},40,40,{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for cue in build_subtitle_cues(
        segments,
        job_dir=job_dir,
        settings=settings,
        vendor_dir=vendor_dir,
        ffmpeg_path=ffmpeg_path,
        transcribe_fn=transcribe_fn,
        tts_asr_align=tts_asr_align,
    ):
        start = format_ass_time(float(cue["start"]))
        end = format_ass_time(float(cue["end"]))
        lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,"
            f"{box_tags}{escape_ass_text(str(cue['text']))}"
        )

    return "\n".join(lines) + "\n"


def write_ass_file(
    output_path: Path,
    segments: list[dict],
    settings: dict,
    *,
    play_res_x: int,
    play_res_y: int,
    job_dir: Path | None = None,
    vendor_dir: Path | None = None,
    ffmpeg_path: Path | None = None,
    transcribe_fn=None,
    tts_asr_align: bool = False,
) -> Path:
    content = build_ass_content(
        segments,
        font_size=settings.get("subtitle_font_size", DEFAULT_SUBTITLE_FONT_SIZE),
        font_color=settings.get("subtitle_font_color", DEFAULT_SUBTITLE_FONT_COLOR),
        background_color=settings.get(
            "subtitle_background_color",
            DEFAULT_SUBTITLE_BACKGROUND_COLOR,
        ),
        background_opacity=settings.get(
            "subtitle_background_opacity",
            DEFAULT_SUBTITLE_BACKGROUND_OPACITY,
        ),
        background_padding=settings.get("subtitle_background_padding"),
        position=settings.get("subtitle_position", DEFAULT_SUBTITLE_POSITION),
        edge_margin=settings.get("subtitle_edge_margin", DEFAULT_SUBTITLE_EDGE_MARGIN),
        play_res_x=play_res_x,
        play_res_y=play_res_y,
        job_dir=job_dir,
        settings=settings,
        vendor_dir=vendor_dir,
        ffmpeg_path=ffmpeg_path,
        transcribe_fn=transcribe_fn,
        tts_asr_align=tts_asr_align,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8-sig")
    return output_path


def ffmpeg_escape_path(path: Path) -> str:
    escaped = path.resolve().as_posix()
    if len(escaped) >= 2 and escaped[1] == ":":
        escaped = escaped[0] + r"\:" + escaped[2:]
    return escaped.replace("'", r"'\''")


def ffmpeg_subtitles_filter(ass_path: Path) -> str:
    return f"subtitles=filename='{ffmpeg_escape_path(ass_path)}'"


def subtitles_filter_available(ffmpeg_path: Path) -> bool:
    """Return True when ffmpeg was built with libass (subtitles/ass filter)."""
    try:
        completed = subprocess.run(
            [str(ffmpeg_path), "-filters"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        output = f"{completed.stdout}\n{completed.stderr}".lower()
        return " subtitles " in output or " ass " in output
    except (OSError, subprocess.TimeoutExpired):
        return False


def probe_video_dimensions(ffmpeg_path: Path, video_path: Path) -> tuple[int, int]:
    ffprobe = ffmpeg_path.with_name(
        "ffprobe.exe" if ffmpeg_path.name.lower().startswith("ffmpeg") else "ffprobe"
    )
    if not ffprobe.is_file():
        ffprobe = Path("ffprobe")

    cmd = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        payload = json.loads(completed.stdout or "{}")
        stream = (payload.get("streams") or [{}])[0]
        width = int(stream.get("width") or 1080)
        height = int(stream.get("height") or 1920)
        return max(width, 1), max(height, 1)
    except (subprocess.CalledProcessError, ValueError, TypeError, json.JSONDecodeError):
        return 1080, 1920
