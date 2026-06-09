from typing import Any

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


class SchemaValidationError(BaseModel):
    """Represents a single field-level validation error."""

    field: str = Field(..., description="The field that caused the validation error.")
    message: str = Field(..., description="The message from the validation error.")


class ConfigSummary(BaseModel):
    """Summary snapshot of the current EnergyPlus IDF state."""

    building: dict[str, Any] | None = None
    site_location: dict[str, Any] | None = None
    zones_count: int = 0
    materials_count: int = 0
    constructions_count: int = 0
    surfaces_count: int = 0
    fenestrations_count: int = 0
    schedules_count: int = 0
    hvac_thermostats_count: int = 0
    hvac_ideal_loads_count: int = 0
    simulation_control: dict[str, Any] | None = None
    run_period: dict[str, Any] | None = None
    global_geometry_rules: dict[str, Any] | None = None
