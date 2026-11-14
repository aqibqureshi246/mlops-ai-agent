from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EmptyInput(StrictInput):
    pass


class Evidence(BaseModel):
    id: str
    tool: str
    status: Literal["ok", "error", "denied", "timeout"]
    observed_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    request_id: str


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class ActionRequest(StrictInput):
    tool: str
    arguments: dict[str, Any]


class ChatRequest(StrictInput):
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


class ConfirmationRequest(StrictInput):
    token: str = Field(min_length=16, max_length=200)
    confirm: bool


class AgentResponse(BaseModel):
    request_id: str
    answer: str
    evidence: list[Evidence]
    uncertainties: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    pending_actions: list[dict[str, Any]] = Field(default_factory=list)
