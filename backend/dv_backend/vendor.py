import json
from pathlib import Path, PurePosixPath
import shutil

from pydantic import BaseModel, field_validator


class VendorTool(BaseModel):
    id: str
    display_name: str
    executable: str
    dev_command: str
    version_args: list[str]
    version_contains: str
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


class VendorResolver:
    def __init__(self, vendor_dir: Path, allow_path_tools: bool = False) -> None:
        self.vendor_dir = vendor_dir
        self.allow_path_tools = allow_path_tools

    def resolve(self, tool: VendorTool) -> ResolvedTool:
        bundled = self.vendor_dir / tool.executable
        if bundled.is_file():
            return ResolvedTool(path=bundled, source="bundled")
        if self.allow_path_tools:
            found = shutil.which(tool.dev_command)
            if found:
                return ResolvedTool(path=Path(found), source="path")
        return ResolvedTool(path=None, source="missing")
