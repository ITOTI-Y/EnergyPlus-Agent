import json
from typing import Literal

from idfpy.models.internal_gains import People
from idfpy.models.schedules import ScheduleCompact
from idfpy.models.thermal_zones import Zone
from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def make_people_tools(config: ConfigState) -> list[BaseTool]:

    @tool
    def create_people(
        name: str,
        zone_name: str,
        number_of_people_schedule_name: str,
        activity_level_schedule_name: str,
        number_of_people_calculation_method: Literal[
            "People", "People/Area", "Area/Person"
        ] = "People",
        number_of_people: float = 0.0,
        people_per_floor_area: float = 0.0,
        floor_area_per_person: float = 0.0,
        fraction_radiant: float = 0.3,
    ) -> str:
        """Create a People (occupancy load) object.

        Args:
            name: Unique people object name.
            zone_name: Existing Zone name this load applies to.
            number_of_people_schedule_name: Existing Schedule:Compact (Fraction).
            activity_level_schedule_name: Existing Schedule:Compact (Activity Level, W/person).
            number_of_people_calculation_method: People / People/Area / Area/Person.
            number_of_people: Absolute count (use when method=People).
            people_per_floor_area: people/m^2 (use when method=People/Area).
            floor_area_per_person: m^2/person (use when method=Area/Person).
            fraction_radiant: Radiant fraction of sensible heat (0-1).
        """
        idf = config.idf
        if idf.has("People", name):
            return _err(f"People '{name}' already exists.")
        try:
            people = People(
                name=name,
                zone_or_zonelist_or_space_or_spacelist_name=zone_name,
                number_of_people_schedule_name=number_of_people_schedule_name,
                activity_level_schedule_name=activity_level_schedule_name,
                number_of_people_calculation_method=number_of_people_calculation_method,
                number_of_people=number_of_people,
                people_per_floor_area=people_per_floor_area,
                floor_area_per_person=floor_area_per_person,
                fraction_radiant=fraction_radiant,
            )
            idf.add(people)
            return _ok(
                f"People '{name}' created successfully.",
                people.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating people '{name}': {e}")

    @tool
    def list_people() -> str:
        """List all People objects."""
        idf = config.idf
        items = [p.model_dump() for p in idf.all_of_type(People).values()]
        return _ok(f"Listed {len(items)} People objects.", items)

    @tool
    def delete_people(name: str) -> str:
        """Delete a People object."""
        idf = config.idf
        if not idf.has("People", name):
            return _err(f"People '{name}' not found.")
        idf.remove("People", name)
        return _ok(f"People '{name}' deleted successfully.")

    @tool
    def list_zones() -> str:
        """Read-only: list zones an occupancy load can be assigned to."""
        idf = config.idf
        items = [z.model_dump() for z in idf.all_of_type(Zone).values()]
        return _ok(f"Listed {len(items)} zones.", items)

    @tool
    def list_schedules() -> str:
        """Read-only: list Schedule:Compact (for number_of_people and activity_level refs)."""
        idf = config.idf
        items = [s.model_dump() for s in idf.all_of_type(ScheduleCompact).values()]
        return _ok(f"Listed {len(items)} schedules.", items)

    return [create_people, list_people, delete_people, list_zones, list_schedules]
