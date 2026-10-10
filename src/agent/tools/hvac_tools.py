from idfpy.models.hvac_templates import (
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
)
from idfpy.models.schedules import ScheduleCompact
from idfpy.models.thermal_zones import Zone
from langchain_core.tools import BaseTool

from src.agent.tools._share import list_tool, model_tool, ok
from src.modeling import objects
from src.modeling.hvac import check_zone_free, find_ideal_loads
from src.state.config_state import ConfigState


def make_hvac_tools(config: ConfigState) -> list[BaseTool]:
    idf = config.idf

    @model_tool
    def create_thermostat(
        name: str,
        heating_setpoint_schedule_name: str,
        cooling_setpoint_schedule_name: str,
    ) -> str:
        """Create an HVACTemplate:Thermostat.

        Args:
            name: Unique thermostat name.
            heating_setpoint_schedule_name: Existing Schedule:Compact for heating setpoints (C).
            cooling_setpoint_schedule_name: Existing Schedule:Compact for cooling setpoints (C).
        """
        thermostat = objects.create(
            idf,
            HVACTemplateThermostat(
                name=name,
                heating_setpoint_schedule_name=heating_setpoint_schedule_name,
                cooling_setpoint_schedule_name=cooling_setpoint_schedule_name,
            ),
        )
        return ok(
            f"Thermostat '{name}' created.", thermostat.model_dump(exclude_none=True)
        )

    @model_tool
    def create_ideal_loads_system(
        zone_name: str,
        template_thermostat_name: str,
        system_availability_schedule_name: str | None = None,
    ) -> str:
        """Create an HVACTemplate:Zone:IdealLoadsAirSystem (one per zone).

        Args:
            zone_name: Existing Zone name; identifies the system.
            template_thermostat_name: Existing HVACTemplate:Thermostat name.
            system_availability_schedule_name: Optional availability Schedule:Compact.
        """
        check_zone_free(idf, zone_name)
        system = objects.create(
            idf,
            HVACTemplateZoneIdealLoadsAirSystem(
                zone_name=zone_name,
                template_thermostat_name=template_thermostat_name,
                system_availability_schedule_name=system_availability_schedule_name,
            ),
        )
        return ok(
            f"IdealLoadsAirSystem for zone '{zone_name}' created.",
            system.model_dump(exclude_none=True),
        )

    @model_tool
    def delete_thermostat(name: str) -> str:
        """Delete a thermostat; refused while an IdealLoadsAirSystem uses it."""
        objects.delete(idf, objects.get(idf, HVACTemplateThermostat, name), name)
        return ok(f"Thermostat '{name}' deleted.")

    @model_tool
    def delete_ideal_loads_system(zone_name: str) -> str:
        """Delete the IdealLoadsAirSystem of a zone."""
        key, system = find_ideal_loads(idf, zone_name)
        objects.delete(idf, system, key)
        return ok(f"IdealLoadsAirSystem for zone '{zone_name}' deleted.")

    return [
        create_thermostat,
        create_ideal_loads_system,
        list_tool(
            idf, "list_thermostats", HVACTemplateThermostat, "List all thermostats."
        ),
        list_tool(
            idf,
            "list_ideal_loads_systems",
            HVACTemplateZoneIdealLoadsAirSystem,
            "List all IdealLoadsAirSystem entries (keyed by zone_name).",
        ),
        delete_thermostat,
        delete_ideal_loads_system,
        list_tool(idf, "list_zones", Zone, "List zones that can get an HVAC system."),
        list_tool(
            idf,
            "list_schedules",
            ScheduleCompact,
            "List Schedule:Compact for setpoint and availability references.",
        ),
    ]
