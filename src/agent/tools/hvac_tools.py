from idfpy.models.hvac_templates import (
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
)
from idfpy.models.schedules import ScheduleCompact
from langchain_core.tools import BaseTool

from src.agent.tools._share import (
    create_in_zones,
    list_tool,
    list_zone_names_tool,
    model_tool,
    ok,
)
from src.modeling import objects
from src.modeling.hvac import (
    check_setpoints_free,
    check_zone_free,
    find_ideal_loads,
)
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

        A pair of setpoint schedules has one thermostat: a second one for the
        same pair is refused, naming the existing one to use.
        """
        check_setpoints_free(
            idf, heating_setpoint_schedule_name, cooling_setpoint_schedule_name
        )
        thermostat = objects.create(
            idf,
            HVACTemplateThermostat(
                name=name,
                heating_setpoint_schedule_name=heating_setpoint_schedule_name,
                cooling_setpoint_schedule_name=cooling_setpoint_schedule_name,
            ),
        )
        return ok(f"Thermostat '{thermostat.name}' created.")

    @model_tool
    def create_ideal_loads_systems(
        zone_names: list[str],
        template_thermostat_name: str,
        system_availability_schedule_name: str | None = None,
    ) -> str:
        """Create an HVACTemplate:Zone:IdealLoadsAirSystem for each zone.

        Call it once per group of zones with the same thermostat.

        Args:
            zone_names: Existing Zone names, each without a system yet.
            template_thermostat_name: Existing HVACTemplate:Thermostat name.
            system_availability_schedule_name: Optional availability Schedule:Compact.
        """

        def system(zone: str) -> HVACTemplateZoneIdealLoadsAirSystem:
            check_zone_free(idf, zone)
            return HVACTemplateZoneIdealLoadsAirSystem(
                zone_name=zone,
                template_thermostat_name=template_thermostat_name,
                system_availability_schedule_name=system_availability_schedule_name,
            )

        return create_in_zones(idf, "IdealLoadsAirSystem", zone_names, system)

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
        create_ideal_loads_systems,
        list_tool(
            idf, "list_thermostats", HVACTemplateThermostat, "List all thermostats."
        ),
        list_tool(
            idf,
            "list_ideal_loads_systems",
            HVACTemplateZoneIdealLoadsAirSystem,
            "List IdealLoadsAirSystem entries: zone, thermostat, availability.",
            (
                "zone_name",
                "template_thermostat_name",
                "system_availability_schedule_name",
            ),
        ),
        delete_thermostat,
        delete_ideal_loads_system,
        list_zone_names_tool(idf, "List zone names that can get an HVAC system."),
        list_tool(
            idf,
            "list_schedules",
            ScheduleCompact,
            "List Schedule:Compact names and type limits, for setpoint and "
            "availability references.",
            ("name", "schedule_type_limits_name"),
        ),
    ]
