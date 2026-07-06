import json
import sys
from pathlib import Path, PurePosixPath
import shutil

from pydantic import BaseModel, Field, field_validator


class VendorTool(BaseModel):
    id: str
    display_name: str
    executable: str
    dev_command: str
    version_args: list[str]
    version_contains: str
    success_exit_codes: list[int] = Field(default_factory=lambda: [0])
    required: bool
    capability: str

    @field_validator("executable")
    @classmethod
    def executable_is_safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("executable must be a safe relative path")
        return value


class VendorManifest(BaseModel):
    schema_version: int
    tools: list[VendorTool]

    @classmethod
    def load(cls, path: Path) -> "VendorManifest":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8-sig")))


class ResolvedTool(BaseModel):
    path: Path | None
    source: str


def macos_ffmpeg_full_path() -> Path | None:
    if sys.platform != "darwin":
        return None
    for candidate in (
        Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"),
        Path("/usr/local/opt/ffmpeg-full/bin/ffmpeg"),
    ):
        if candidate.is_file():
            return candidate
    return None


def prefer_macos_ffmpeg_full(tool: VendorTool, resolved: ResolvedTool) -> ResolvedTool:
    if tool.id != "ffmpeg" or resolved.path is None:
        return resolved
    full_path = macos_ffmpeg_full_path()
    if full_path is None:
        return resolved
    from .adapters.subtitles import subtitles_filter_available

    if subtitles_filter_available(full_path) and not subtitles_filter_available(resolved.path):
        return ResolvedTool(path=full_path, source="path")
    return resolved


class VendorResolver:
    def __init__(self, vendor_dir: Path, allow_path_tools: bool = True) -> None:
        self.vendor_dir = vendor_dir
        self.allow_path_tools = allow_path_tools

    def resolve(self, tool: VendorTool) -> ResolvedTool:
        executable = tool.executable.replace("\\", "/")
        candidates = [executable]
        if executable.startswith("tools/"):
            candidates.append(executable.removeprefix("tools/"))

        for relative in candidates:
            bundled = self.vendor_dir / relative
            if bundled.is_file():
                return ResolvedTool(path=bundled, source="bundled")
            if sys.platform == "darwin" and relative.endswith(".exe"):
                alt_bundled = self.vendor_dir / relative[:-4]
                if alt_bundled.is_file():
                    return ResolvedTool(path=alt_bundled, source="bundled")

        if self.allow_path_tools:
            found = shutil.which(tool.dev_command)
            if found:
                return prefer_macos_ffmpeg_full(tool, ResolvedTool(path=Path(found), source="path"))
        return ResolvedTool(path=None, source="missing")
