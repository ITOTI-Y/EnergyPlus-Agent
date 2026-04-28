import json

from langchain_core.tools import BaseTool, tool

from idfpy.models.internal_gains import Lights
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
        design_level_calculation_method: str = "Watts/Area",
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
        if idf.has("Lights", name):
            return _err(f"Lights '{name}' already exists.")
        try:
            idf.add(Lights(
                name=name,
                zone_or_zonelist_or_space_or_spacelist_name=zone_name,
                schedule_name=schedule_name,
                design_level_calculation_method=design_level_calculation_method,
                lighting_level=lighting_level if lighting_level != 0.0 else None,
                watts_per_floor_area=watts_per_floor_area if watts_per_floor_area != 0.0 else None,
                watts_per_person=watts_per_person if watts_per_person != 0.0 else None,
                fraction_radiant=fraction_radiant,
                fraction_visible=fraction_visible,
            ))
            return _ok(
                f"Lights '{name}' created successfully.",
                idf.get("Lights", name).model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating lights '{name}': {e}")

    @tool
    def list_lights() -> str:
        """List all Lights objects."""
        items = [lt.model_dump() for lt in idf.all_of_type("Lights").values()]
        return _ok(f"Listed {len(items)} Lights objects.", items)

    @tool
    def delete_light(name: str) -> str:
        """Delete a Lights object."""
        if not idf.has("Lights", name):
            return _err(f"Lights '{name}' not found.")
        idf.remove("Lights", name)
        return _ok(f"Lights '{name}' deleted successfully.")

    @tool
    def list_zones() -> str:
        """Read-only: list zones a Lights load can be assigned to."""
        items = [z.model_dump() for z in idf.all_of_type("Zone").values()]
        return _ok(f"Listed {len(items)} zones.", items)

    @tool
    def list_schedules() -> str:
        """Read-only: list Schedule:Compact (for schedule_name reference)."""
        items = [s.model_dump() for s in idf.all_of_type("Schedule:Compact").values()]
        return _ok(f"Listed {len(items)} schedules.", items)

    return [create_light, list_lights, delete_light, list_zones, list_schedules]
