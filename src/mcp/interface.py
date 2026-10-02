from pydantic import BaseModel, Field


class ToolResponse(BaseModel):
    """Standardized response object returned by all MCP tool operations."""

    success: bool = Field(..., description="Whether the tool call was successful.")
    message: str = Field(..., description="The message from the tool call.")
    data: dict | list | None = Field(
        default=None, description="The data from the tool call."
    )

    def to_mcp_response(self) -> dict:
        return {"result": self.model_dump()}
