from pydantic import BaseModel, Field


class ToolResponse(BaseModel):
    success: bool = Field(..., description="Whether the tool call was successful.")
    message: str = Field(..., description="The message from the tool call.")
    data: dict | list | None = Field(
        default=None, description="The data from the tool call."
    )


class ValidationError(BaseModel):
    field: str = Field(..., description="The field that caused the validation error.")
    message: str = Field(..., description="The message from the validation error.")

class ConfigSummary(BaseModel):
    zones_count: int = Field(default=0, description="The number of zones in the configuration.")
    materials_count: int = Field(default=0, description="The number of materials in the configuration.")
    constructions_count: int = Field(default=0, description="The number of constructions in the configuration.")
    surfaces_count: int = Field(default=0, description="The number of surfaces in the configuration.")
    fenestrations_count: int = Field(default=0, description="The number of fenestrations in the configuration.")
    schedules_count: int = Field(default=0, description="The number of schedules in the configuration.")
    hvac_thermostats_count: int = Field(default=0, description="The number of HVAC thermostats in the configuration.")
    hvac_ideal_loads_count: int = Field(default=0, description="The number of HVAC ideal loads in the configuration.")
    has_simulation_control: bool = Field(default=False, description="Whether the configuration has a simulation control.")
    has_site_location: bool = Field(default=False, description="Whether the configuration has a site location.")
    has_run_period: bool = Field(default=False, description="Whether the configuration has a run period.")
