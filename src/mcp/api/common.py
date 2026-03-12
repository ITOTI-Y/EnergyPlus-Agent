from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolInput(BaseModel):
    """Base input model for all MCP tool parameters.

    Provides strict validation with alias support. All tool-specific
    input schemas should inherit from this class.
    """

    model_config = ConfigDict(
        validate_by_name=True,
        validate_by_alias=True,
        extra="forbid",
    )


def to_payload(model: BaseModel) -> dict[str, Any]:
    """Convert a validated input model to a dictionary payload for tool operations.

    Args:
        model: Validated Pydantic model instance.

    Returns:
        Dictionary with alias keys and None values excluded.
    """
    return model.model_dump(by_alias=True, exclude_none=True)
