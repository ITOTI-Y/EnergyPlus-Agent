import json
from typing import Literal

from idfpy.models.internal_gains import Lights
from idfpy.models.schedules import ScheduleCompact
from idfpy.models.thermal_zones import Zone
from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def make_lights_tools(config: ConfigState) -> list[BaseTool]:

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
        idf = config.idf
        if idf.has("Lights", name):
            return _err(f"Lights '{name}' already exists.")
        try:
            light = Lights(
                name=name,
                zone_or_zonelist_or_space_or_spacelist_name=zone_name,
                schedule_name=schedule_name,
                design_level_calculation_method=design_level_calculation_method,
                lighting_level=lighting_level,
                watts_per_floor_area=watts_per_floor_area,
                watts_per_person=watts_per_person,
                fraction_radiant=fraction_radiant,
                fraction_visible=fraction_visible,
            )
            idf.add(light)
            return _ok(
                f"Lights '{name}' created successfully.",
                light.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating lights '{name}': {e}")

    @tool
    def list_lights() -> str:
        """List all Lights objects."""
        idf = config.idf
        items = [lt.model_dump() for lt in idf.all_of_type(Lights).values()]
        return _ok(f"Listed {len(items)} Lights objects.", items)

    @tool
    def delete_light(name: str) -> str:
        """Delete a Lights object."""
        idf = config.idf
        if not idf.has("Lights", name):
            return _err(f"Lights '{name}' not found.")
        idf.remove("Lights", name)
        return _ok(f"Lights '{name}' deleted successfully.")

    @tool
    def list_zones() -> str:
        """Read-only: list zones a Lights load can be assigned to."""
        idf = config.idf
        items = [z.model_dump() for z in idf.all_of_type(Zone).values()]
        return _ok(f"Listed {len(items)} zones.", items)

    @tool
    def list_schedules() -> str:
        """Read-only: list Schedule:Compact (for schedule_name reference)."""
        idf = config.idf
        items = [s.model_dump() for s in idf.all_of_type(ScheduleCompact).values()]
        return _ok(f"Listed {len(items)} schedules.", items)

    return [create_light, list_lights, delete_light, list_zones, list_schedules]
