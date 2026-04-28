import json

from langchain_core.tools import BaseTool, tool

from idfpy.models.hvac_templates import HVACTemplateThermostat, HVACTemplateZoneIdealLoadsAirSystem
from src.mcp.state import ConfigState


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def make_hvac_tools(config: ConfigState) -> list[BaseTool]:
    idf = config._idf

    @tool
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
        if idf.has("HVACTemplate:Thermostat", name):
            return _err(f"Thermostat '{name}' already exists.")
        try:
            idf.add(HVACTemplateThermostat(
                name=name,
                heating_setpoint_schedule_name=heating_setpoint_schedule_name,
                cooling_setpoint_schedule_name=cooling_setpoint_schedule_name,
            ))
            return _ok(
                f"Thermostat '{name}' created successfully.",
                idf.get("HVACTemplate:Thermostat", name).model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating thermostat '{name}': {e}")

    @tool
    def create_ideal_loads_system(
        zone_name: str,
        template_thermostat_name: str,
        system_availability_schedule_name: str | None = None,
    ) -> str:
        """Create an HVACTemplate:Zone:IdealLoadsAirSystem (one per zone).

        Args:
            zone_name: Existing Zone name. Acts as the identity key (no separate Name).
            template_thermostat_name: Existing HVACTemplate:Thermostat name.
            system_availability_schedule_name: Optional availability Schedule:Compact.
        """
        existing = idf.all_of_type("HVACTemplate:Zone:IdealLoadsAirSystem")
        if any(obj.zone_name == zone_name for obj in existing.values()):
            return _err(f"IdealLoadsAirSystem for zone '{zone_name}' already exists.")
        try:
            idf.add(HVACTemplateZoneIdealLoadsAirSystem(
                zone_name=zone_name,
                template_thermostat_name=template_thermostat_name,
                system_availability_schedule_name=system_availability_schedule_name or None,
            ))
            return _ok(
                f"IdealLoadsAirSystem for zone '{zone_name}' created successfully.",
                {"zone_name": zone_name, "template_thermostat_name": template_thermostat_name},
            )
        except Exception as e:
            return _err(f"Error creating IdealLoadsAirSystem for zone '{zone_name}': {e}")

    @tool
    def list_thermostats() -> str:
        """List all thermostats."""
        items = [t.model_dump() for t in idf.all_of_type("HVACTemplate:Thermostat").values()]
        return _ok(f"Listed {len(items)} thermostats.", items)

    @tool
    def list_ideal_loads_systems() -> str:
        """List all IdealLoadsAirSystem entries (keyed by zone_name)."""
        items = [
            obj.model_dump()
            for obj in idf.all_of_type("HVACTemplate:Zone:IdealLoadsAirSystem").values()
        ]
        return _ok(f"Listed {len(items)} IdealLoadsAirSystem entries.", items)

    @tool
    def delete_thermostat(name: str) -> str:
        """Delete a thermostat. Fails if referenced by an IdealLoadsSystem."""
        if not idf.has("HVACTemplate:Thermostat", name):
            return _err(f"Thermostat '{name}' not found.")
        refs = []
        for obj in idf.all_of_type("HVACTemplate:Zone:IdealLoadsAirSystem").values():
            if obj.template_thermostat_name == name:
                refs.append(f"IdealLoadsSystem:{obj.zone_name}")
        if refs:
            return _err(
                f"Thermostat '{name}' is referenced by IdealLoadsAirSystem.",
                {"references": refs},
            )
        idf.remove("HVACTemplate:Thermostat", name)
        return _ok(f"Thermostat '{name}' deleted successfully.")

    @tool
    def delete_ideal_loads_system(zone_name: str) -> str:
        """Delete an IdealLoadsSystem by its zone_name."""
        items = idf.all_of_type("HVACTemplate:Zone:IdealLoadsAirSystem")
        key = next((k for k, v in items.items() if v.zone_name == zone_name), None)
        if key is None:
            return _err(f"IdealLoadsAirSystem for zone '{zone_name}' not found.")
        idf.remove("HVACTemplate:Zone:IdealLoadsAirSystem", key)
        return _ok(f"IdealLoadsAirSystem for zone '{zone_name}' deleted successfully.")

    @tool
    def list_zones() -> str:
        """Read-only: list zones an IdealLoadsAirSystem can be attached to."""
        items = [z.model_dump() for z in idf.all_of_type("Zone").values()]
        return _ok(f"Listed {len(items)} zones.", items)

    @tool
    def list_schedules() -> str:
        """Read-only: list Schedule:Compact objects (setpoint / availability references)."""
        items = [s.model_dump() for s in idf.all_of_type("Schedule:Compact").values()]
        return _ok(f"Listed {len(items)} schedules.", items)

    return [
        create_thermostat,
        create_ideal_loads_system,
        list_thermostats,
        list_ideal_loads_systems,
        delete_thermostat,
        delete_ideal_loads_system,
        list_zones,
        list_schedules,
    ]
