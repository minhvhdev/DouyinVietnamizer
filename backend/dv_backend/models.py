from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    source_url: str = Field(min_length=1)


class JobRerun(BaseModel):
    keep_steps: list[str] = Field(default_factory=list)


class JobStep(BaseModel):
    name: str
    position: int
    status: str
    checkpoint_path: str | None = None


class Job(BaseModel):
    id: str
    source_url: str
    title: str | None = None
    title_vi: str | None = None
    status: str
    current_step: str | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None
    created_at: str
    updated_at: str
    steps: list[JobStep] = []


class ErrorInfo(BaseModel):
    code: str
    message: str
    action: str
    detail: str | None = None
    retryable: bool = False
    log_path: str | None = None

