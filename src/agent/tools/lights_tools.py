import json
from typing import Literal

from idfpy.models import Lights, ScheduleCompact, Zone
from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def make_lights_tools(config: ConfigState) -> list[BaseTool]:
    idf = config._idf

    @tool
    def create_light(
        name: str,
        zone_name: str,
        schedule_name: str,
        design_level_calculation_method: Literal[
            "LightingLevel", "Watts/Area", "Watts/Person"
        ] = "Watts/Area",
        lighting_level: float = 0.0,
        watts_per_floor_area: float = 0.0,
        watts_per_person: float = 0.0,
        fraction_radiant: float = 0.0,
        fraction_visible: float = 0.0,
    ) -> str:
        """Create a Lights (lighting load) object.

        Args:
            name: Unique lights object name.
            zone_name: Existing Zone name.
            schedule_name: Existing Schedule:Compact (Fraction).
            design_level_calculation_method: LightingLevel / Watts/Area / Watts/Person.
            lighting_level: Absolute watts (when method=LightingLevel).
            watts_per_floor_area: W/m^2 (when method=Watts/Area).
            watts_per_person: W/person (when method=Watts/Person).
            fraction_radiant: Radiant fraction (0-1).
            fraction_visible: Visible light fraction (0-1).
        """
        if idf is None:
            raise ValueError("IDF is None")
        if idf.has(Lights, name):
            return _err(f"Lights '{name}' already exists.")
        # Reference checks: emit missing_ref so the agent's detect_upstream_gap
        # can back-hop to the owning phase (zone / schedule) to create it.
        if zone_name and not idf.has(Zone, zone_name):
            return _err(
                f"Zone '{zone_name}' not found.",
                {"missing_ref": "Zone", "missing_name": zone_name},
            )
        if schedule_name and not idf.has("Schedule:Compact", schedule_name):
            return _err(
                f"Schedule:Compact '{schedule_name}' not found.",
                {"missing_ref": "Schedule:Compact", "missing_name": schedule_name},
            )
        try:
            idf.add(
                Lights(
                    name=name,
                    zone_or_zonelist_or_space_or_spacelist_name=zone_name,
                    schedule_name=schedule_name,
                    design_level_calculation_method=design_level_calculation_method,
                    lighting_level=lighting_level if lighting_level != 0.0 else None,
                    watts_per_floor_area=watts_per_floor_area
                    if watts_per_floor_area != 0.0
                    else None,
                    watts_per_person=watts_per_person
                    if watts_per_person != 0.0
                    else None,
                    fraction_radiant=fraction_radiant,
                    fraction_visible=fraction_visible,
                )
            )
            data = idf.get(Lights, name)
            if data is None:
                raise ValueError("Lights not found")
            return _ok(
                f"Lights '{name}' created successfully.",
                data.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating lights '{name}': {e}")

    @tool
    def list_lights() -> str:
        """List all Lights objects."""
        if idf is None:
            raise ValueError("IDF is None")
        items = [lt.model_dump() for lt in idf.all_of_type(Lights).values()]
        return _ok(f"Listed {len(items)} Lights objects.", items)

    @tool
    def update_light(
        name: str,
        zone_name: str | None = None,
        schedule_name: str | None = None,
        design_level_calculation_method: Literal[
            "LightingLevel", "Watts/Area", "Watts/Person"
        ]
        | None = None,
        lighting_level: float | None = None,
        watts_per_floor_area: float | None = None,
        watts_per_person: float | None = None,
        fraction_radiant: float | None = None,
        fraction_visible: float | None = None,
    ) -> str:
        """Update fields of an existing Lights object by name.

        Only non-None fields are written. Pass only the fields you want to
        change (e.g. to lower LPD, set watts_per_floor_area).

        Args:
            name: Existing Lights object name.
            zone_name: New Zone name.
            schedule_name: New Schedule:Compact name.
            design_level_calculation_method: LightingLevel / Watts/Area / Watts/Person.
            lighting_level / watts_per_floor_area / watts_per_person:
                Load values (use the one matching the calculation method).
            fraction_radiant / fraction_visible: Light distribution fractions (0-1).
        """
        if idf is None:
            raise ValueError("IDF is None")
        obj = idf.get(Lights, name)
        if obj is None:
            return _err(f"Lights '{name}' not found.")
        try:
            if zone_name is not None:
                if not idf.has(Zone, zone_name):
                    return _err(f"Zone '{zone_name}' not found.")
                obj.zone_or_zonelist_or_space_or_spacelist_name = zone_name
            if schedule_name is not None:
                if not idf.has(ScheduleCompact, schedule_name):
                    return _err(f"Schedule '{schedule_name}' not found.")
                obj.schedule_name = schedule_name
            if design_level_calculation_method is not None:
                obj.design_level_calculation_method = design_level_calculation_method
            if lighting_level is not None:
                obj.lighting_level = lighting_level
            if watts_per_floor_area is not None:
                obj.watts_per_floor_area = watts_per_floor_area
            if watts_per_person is not None:
                obj.watts_per_person = watts_per_person
            if fraction_radiant is not None:
                obj.fraction_radiant = fraction_radiant
            if fraction_visible is not None:
                obj.fraction_visible = fraction_visible
            return _ok(f"Lights '{name}' updated successfully.", obj.model_dump())
        except Exception as e:
            return _err(f"Error updating lights '{name}': {e}")

    @tool
    def delete_light(name: str) -> str:
        """Delete a Lights object."""
        if idf is None:
            raise ValueError("IDF is None")
        if not idf.has(Lights, name):
            return _err(f"Lights '{name}' not found.")
        idf.remove("Lights", name)
        return _ok(f"Lights '{name}' deleted successfully.")

    @tool
    def list_zones() -> str:
        """Read-only: list zones a Lights load can be assigned to."""
        if idf is None:
            raise ValueError("IDF is None")
        items = [z.model_dump() for z in idf.all_of_type(Zone).values()]
        return _ok(f"Listed {len(items)} zones.", items)

    @tool
    def list_schedules() -> str:
        """Read-only: list Schedule:Compact (for schedule_name reference)."""
        if idf is None:
            raise ValueError("IDF is None")
        items = [s.model_dump() for s in idf.all_of_type(ScheduleCompact).values()]
        return _ok(f"Listed {len(items)} schedules.", items)

    return [
        create_light,
        list_lights,
        update_light,
        delete_light,
        list_zones,
        list_schedules,
    ]
