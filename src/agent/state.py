from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.mcp.state import ConfigState

type Phase = Literal[
    "init",
    "geometry",
    "envelope",
    "schedule",
    "hvac",
    "loads",
    "validate",
    "simulate",
    "done",
]


class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    user_input: str = Field(default="")
    image_path: str | None = Field(default=None)

    config_state: ConfigState = Field(default_factory=ConfigState)

    current_phase: Phase = Field(default="init")
    validation_errors: list[str] = Field(default_factory=list)
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
