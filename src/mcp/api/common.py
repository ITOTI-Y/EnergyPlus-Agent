from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolInput(BaseModel):
    model_config = ConfigDict(
        validate_by_name=True,
        validate_by_alias=True,
        extra="forbid",
    )


def to_payload(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(by_alias=True, exclude_none=True)
