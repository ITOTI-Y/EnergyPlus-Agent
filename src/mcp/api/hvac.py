from fastmcp import FastMCP
from pydantic import Field

from src.mcp.api.common import ToolInput, to_payload
from src.mcp.tools import IdealLoadsSystemTool, ThermostatTool


class ThermostatCreateInput(ToolInput):
    name: str = Field(alias="Name")
    heating_setpoint_schedule_name: str = Field(alias="Heating Setpoint Schedule Name")
    cooling_setpoint_schedule_name: str = Field(alias="Cooling Setpoint Schedule Name")


class ThermostatUpdateInput(ToolInput):
    name: str = Field(alias="Name")
    heating_setpoint_schedule_name: str | None = Field(
        default=None,
        alias="Heating Setpoint Schedule Name",
    )
    cooling_setpoint_schedule_name: str | None = Field(
        default=None,
        alias="Cooling Setpoint Schedule Name",
    )


class IdealLoadsSystemCreateInput(ToolInput):
    zone_name: str = Field(alias="Zone Name")
    template_thermostat_name: str = Field(alias="Template Thermostat Name")
    system_availability_schedule_name: str | None = Field(
        default=None,
        alias="System Availability Schedule Name",
    )


class IdealLoadsSystemUpdateInput(ToolInput):
    zone_name: str = Field(alias="Zone Name")
    template_thermostat_name: str | None = Field(
        default=None,
        alias="Template Thermostat Name",
    )
    system_availability_schedule_name: str | None = Field(
        default=None,
        alias="System Availability Schedule Name",
    )


def register_hvac_tools(
    mcp: FastMCP,
    thermostat_tool: ThermostatTool,
    ideal_loads_system_tool: IdealLoadsSystemTool,
) -> None:
    @mcp.tool
    def create_hvac_thermostat(
        name: str,
        heating_setpoint_schedule_name: str,
        cooling_setpoint_schedule_name: str,
    ) -> dict:
        payload = to_payload(
            ThermostatCreateInput.model_validate(
                {
                    "name": name,
                    "heating_setpoint_schedule_name": heating_setpoint_schedule_name,
                    "cooling_setpoint_schedule_name": cooling_setpoint_schedule_name,
                }
            )
        )
        return thermostat_tool.create(payload).to_mcp_response()

    @mcp.tool
    def get_hvac_thermostat(name: str) -> dict:
        return thermostat_tool.read(name).to_mcp_response()

    @mcp.tool
    def update_hvac_thermostat(
        name: str,
        heating_setpoint_schedule_name: str | None = None,
        cooling_setpoint_schedule_name: str | None = None,
    ) -> dict:
        payload = to_payload(
            ThermostatUpdateInput.model_validate(
                {
                    "name": name,
                    "heating_setpoint_schedule_name": heating_setpoint_schedule_name,
                    "cooling_setpoint_schedule_name": cooling_setpoint_schedule_name,
                }
            )
        )
        return thermostat_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def delete_hvac_thermostat(name: str) -> dict:
        return thermostat_tool.delete(name).to_mcp_response()

    @mcp.tool
    def list_hvac_thermostats() -> dict:
        return thermostat_tool.list_all().to_mcp_response()

    @mcp.tool
    def create_hvac_ideal_loads_system(
        zone_name: str,
        template_thermostat_name: str,
        system_availability_schedule_name: str | None = None,
    ) -> dict:
        payload = to_payload(
            IdealLoadsSystemCreateInput.model_validate(
                {
                    "zone_name": zone_name,
                    "template_thermostat_name": template_thermostat_name,
                    "system_availability_schedule_name": system_availability_schedule_name,
                }
            )
        )
        return ideal_loads_system_tool.create(payload).to_mcp_response()

    @mcp.tool
    def get_hvac_ideal_loads_system(zone_name: str) -> dict:
        return ideal_loads_system_tool.read(zone_name).to_mcp_response()

    @mcp.tool
    def update_hvac_ideal_loads_system(
        zone_name: str,
        template_thermostat_name: str | None = None,
        system_availability_schedule_name: str | None = None,
    ) -> dict:
        payload = to_payload(
            IdealLoadsSystemUpdateInput.model_validate(
                {
                    "zone_name": zone_name,
                    "template_thermostat_name": template_thermostat_name,
                    "system_availability_schedule_name": system_availability_schedule_name,
                }
            )
        )
        return ideal_loads_system_tool.update(zone_name, payload).to_mcp_response()

    @mcp.tool
    def delete_hvac_ideal_loads_system(zone_name: str) -> dict:
        return ideal_loads_system_tool.delete(zone_name).to_mcp_response()

    @mcp.tool
    def list_hvac_ideal_loads_systems() -> dict:
        return ideal_loads_system_tool.list_all().to_mcp_response()
