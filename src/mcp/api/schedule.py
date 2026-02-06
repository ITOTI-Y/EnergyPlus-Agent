from fastmcp import FastMCP
from pydantic import Field

from src.mcp.api.common import ToolInput, to_payload
from src.mcp.tools import ScheduleCompactTool, ScheduleTypeLimitsTool


class ScheduleTypeLimitsInput(ToolInput):
    name: str = Field(alias="Name")
    lower_limit_value: float = Field(alias="Lower Limit Value")
    upper_limit_value: float = Field(alias="Upper Limit Value")
    numeric_type: str = Field(alias="Numeric Type")
    unit_type: str = Field(alias="Unit Type")


class ScheduleCompactInput(ToolInput):
    name: str = Field(alias="Name")
    schedule_type_limits_name: str = Field(alias="Schedule Type Limits Name")
    times: list[dict] = Field(alias="Data")


def register_schedule_tools(
    mcp: FastMCP,
    schedule_type_limits_tool: ScheduleTypeLimitsTool,
    schedule_compact_tool: ScheduleCompactTool,
) -> None:
    @mcp.tool
    def create_schedule_type_limits(
        name: str,
        lower_limit_value: float,
        upper_limit_value: float,
        numeric_type: str,
        unit_type: str,
    ) -> dict:
        payload = to_payload(
            ScheduleTypeLimitsInput.model_validate(
                {
                    "name": name,
                    "lower_limit_value": lower_limit_value,
                    "upper_limit_value": upper_limit_value,
                    "numeric_type": numeric_type,
                    "unit_type": unit_type,
                }
            )
        )
        return schedule_type_limits_tool.create(payload).to_mcp_response()

    @mcp.tool
    def get_schedule_type_limits(name: str) -> dict:
        return schedule_type_limits_tool.read(name).to_mcp_response()

    @mcp.tool
    def update_schedule_type_limits(
        name: str,
        lower_limit_value: float,
        upper_limit_value: float,
        numeric_type: str,
        unit_type: str,
    ) -> dict:
        payload = to_payload(
            ScheduleTypeLimitsInput.model_validate(
                {
                    "name": name,
                    "lower_limit_value": lower_limit_value,
                    "upper_limit_value": upper_limit_value,
                    "numeric_type": numeric_type,
                    "unit_type": unit_type,
                }
            )
        )
        return schedule_type_limits_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def delete_schedule_type_limits(name: str) -> dict:
        return schedule_type_limits_tool.delete(name).to_mcp_response()

    @mcp.tool
    def list_schedule_type_limits() -> dict:
        return schedule_type_limits_tool.list_all().to_mcp_response()

    @mcp.tool
    def create_schedule_compact(
        name: str,
        schedule_type_limits_name: str,
        times: list[dict],
    ) -> dict:
        payload = to_payload(
            ScheduleCompactInput.model_validate(
                {
                    "name": name,
                    "schedule_type_limits_name": schedule_type_limits_name,
                    "times": times,
                }
            )
        )
        return schedule_compact_tool.create(payload).to_mcp_response()

    @mcp.tool
    def get_schedule_compact(name: str) -> dict:
        return schedule_compact_tool.read(name).to_mcp_response()

    @mcp.tool
    def update_schedule_compact(
        name: str,
        schedule_type_limits_name: str,
        times: list[dict],
    ) -> dict:
        payload = to_payload(
            ScheduleCompactInput.model_validate(
                {
                    "name": name,
                    "schedule_type_limits_name": schedule_type_limits_name,
                    "times": times,
                }
            )
        )
        return schedule_compact_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def delete_schedule_compact(name: str) -> dict:
        return schedule_compact_tool.delete(name).to_mcp_response()

    @mcp.tool
    def list_schedule_compacts() -> dict:
        return schedule_compact_tool.list_all().to_mcp_response()
