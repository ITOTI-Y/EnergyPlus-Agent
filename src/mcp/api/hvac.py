from fastmcp import FastMCP
from idfpy.models.hvac_templates import (
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
)

from src.mcp.api.common import Outcome, dump, given, model_tool
from src.modeling import objects
from src.modeling.hvac import check_zone_free, find_ideal_loads
from src.state.config_state import ConfigState


def register_hvac_tools(mcp: FastMCP, state: ConfigState) -> None:
    """Register HVACTemplate thermostat and ideal loads tools."""
    idf = state.idf
    tool = model_tool(mcp)

    @tool
    def create_hvac_thermostat(
        name: str,
        heating_setpoint_schedule_name: str,
        cooling_setpoint_schedule_name: str,
    ) -> Outcome:
        """Create an HVACTemplate:Thermostat from two setpoint schedules (C)."""
        thermostat = HVACTemplateThermostat(
            name=name,
            heating_setpoint_schedule_name=heating_setpoint_schedule_name,
            cooling_setpoint_schedule_name=cooling_setpoint_schedule_name,
        )
        return f"Thermostat '{name}' created.", dump(objects.create(idf, thermostat))

    @tool
    def get_hvac_thermostat(name: str) -> Outcome:
        """Read a thermostat by name."""
        return (
            f"Thermostat '{name}' read.",
            dump(objects.get(idf, HVACTemplateThermostat, name)),
        )

    @tool
    def update_hvac_thermostat(
        name: str,
        new_name: str | None = None,
        heating_setpoint_schedule_name: str | None = None,
        cooling_setpoint_schedule_name: str | None = None,
    ) -> Outcome:
        """Update a thermostat; a new name is applied to its ideal loads systems."""
        thermostat = objects.update(
            idf,
            objects.get(idf, HVACTemplateThermostat, name),
            given(
                name=new_name,
                heating_setpoint_schedule_name=heating_setpoint_schedule_name,
                cooling_setpoint_schedule_name=cooling_setpoint_schedule_name,
            ),
        )
        return f"Thermostat '{name}' updated.", dump(thermostat)

    @tool
    def delete_hvac_thermostat(name: str) -> Outcome:
        """Delete a thermostat; refused while an ideal loads system uses it."""
        objects.delete(idf, objects.get(idf, HVACTemplateThermostat, name), name)
        return f"Thermostat '{name}' deleted.", None

    @tool
    def list_hvac_thermostats() -> Outcome:
        """List all thermostats."""
        return "Listed thermostats.", objects.dumps(
            idf.all_of_type(HVACTemplateThermostat)
        )

    @tool
    def create_hvac_ideal_loads_system(
        zone_name: str,
        template_thermostat_name: str,
        system_availability_schedule_name: str | None = None,
    ) -> Outcome:
        """Create the HVACTemplate:Zone:IdealLoadsAirSystem of a zone."""
        check_zone_free(idf, zone_name)
        system = HVACTemplateZoneIdealLoadsAirSystem(
            zone_name=zone_name,
            template_thermostat_name=template_thermostat_name,
            system_availability_schedule_name=system_availability_schedule_name,
        )
        return (
            f"Ideal loads system for zone '{zone_name}' created.",
            dump(objects.create(idf, system)),
        )

    @tool
    def get_hvac_ideal_loads_system(zone_name: str) -> Outcome:
        """Read the ideal loads system of a zone."""
        _, system = find_ideal_loads(idf, zone_name)
        return f"Ideal loads system for zone '{zone_name}' read.", dump(system)

    @tool
    def update_hvac_ideal_loads_system(
        zone_name: str,
        template_thermostat_name: str | None = None,
        system_availability_schedule_name: str | None = None,
    ) -> Outcome:
        """Update the ideal loads system of a zone."""
        _, system = find_ideal_loads(idf, zone_name)
        objects.update(
            idf,
            system,
            given(
                template_thermostat_name=template_thermostat_name,
                system_availability_schedule_name=system_availability_schedule_name,
            ),
        )
        return f"Ideal loads system for zone '{zone_name}' updated.", dump(system)

    @tool
    def delete_hvac_ideal_loads_system(zone_name: str) -> Outcome:
        """Delete the ideal loads system of a zone."""
        key, system = find_ideal_loads(idf, zone_name)
        objects.delete(idf, system, key)
        return f"Ideal loads system for zone '{zone_name}' deleted.", None

    @tool
    def list_hvac_ideal_loads_systems() -> Outcome:
        """List all ideal loads systems."""
        return "Listed ideal loads systems.", objects.dumps(
            idf.all_of_type(HVACTemplateZoneIdealLoadsAirSystem)
        )
