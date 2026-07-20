import json
from typing import Literal

from idfpy.models import People, ScheduleCompact, Zone
from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def make_people_tools(config: ConfigState) -> list[BaseTool]:
    idf = config._idf

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
        if idf is None:
            raise ValueError("IDF is None")
        if idf.has(People, name):
            return _err(f"People '{name}' already exists.")
        # Reference checks: emit missing_ref so the agent's detect_upstream_gap
        # can back-hop to the owning phase (zone / schedule) to create it.
        if zone_name and not idf.has(Zone, zone_name):
            return _err(
                f"Zone '{zone_name}' not found.",
                {"missing_ref": "Zone", "missing_name": zone_name},
            )
        if number_of_people_schedule_name and not idf.has(
            ScheduleCompact, number_of_people_schedule_name
        ):
            return _err(
                f"Schedule:Compact '{number_of_people_schedule_name}' not found.",
                {
                    "missing_ref": "Schedule:Compact",
                    "missing_name": number_of_people_schedule_name,
                },
            )
        if activity_level_schedule_name and not idf.has(
            ScheduleCompact, activity_level_schedule_name
        ):
            return _err(
                f"Schedule:Compact '{activity_level_schedule_name}' not found.",
                {
                    "missing_ref": "Schedule:Compact",
                    "missing_name": activity_level_schedule_name,
                },
            )
        try:
            idf.add(
                People(
                    name=name,
                    zone_or_zonelist_or_space_or_spacelist_name=zone_name,
                    number_of_people_schedule_name=number_of_people_schedule_name,
                    activity_level_schedule_name=activity_level_schedule_name,
                    number_of_people_calculation_method=number_of_people_calculation_method,
                    number_of_people=number_of_people
                    if number_of_people != 0.0
                    else None,
                    people_per_floor_area=people_per_floor_area
                    if people_per_floor_area != 0.0
                    else None,
                    floor_area_per_person=floor_area_per_person
                    if floor_area_per_person != 0.0
                    else None,
                    fraction_radiant=fraction_radiant,
                )
            )
            data = idf.get(People, name)
            if data is None:
                raise ValueError("People not found")
            return _ok(
                f"People '{name}' created successfully.",
                data.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating people '{name}': {e}")

    @tool
    def list_people() -> str:
        """List all People objects."""
        if idf is None:
            raise ValueError("IDF is None")
        items = [p.model_dump() for p in idf.all_of_type(People).values()]
        return _ok(f"Listed {len(items)} People objects.", items)

    @tool
    def update_people(
        name: str,
        zone_name: str | None = None,
        number_of_people_schedule_name: str | None = None,
        activity_level_schedule_name: str | None = None,
        number_of_people_calculation_method: Literal[
            "People", "People/Area", "Area/Person"
        ]
        | None = None,
        number_of_people: float | None = None,
        people_per_floor_area: float | None = None,
        floor_area_per_person: float | None = None,
        fraction_radiant: float | None = None,
    ) -> str:
        """Update fields of an existing People object by name.

        Only non-None fields are written. Pass only the fields you want to
        change (e.g. to alter occupancy density, set number_of_people or
        people_per_floor_area; to change the schedule, set the *_schedule_name).

        Args:
            name: Existing People object name.
            zone_name: New Zone name.
            number_of_people_schedule_name / activity_level_schedule_name: New schedules.
            number_of_people_calculation_method: People / People/Area / Area/Person.
            number_of_people / people_per_floor_area / floor_area_per_person:
                Load values (use the one matching the calculation method).
            fraction_radiant: Radiant fraction of sensible heat (0-1).
        """
        if idf is None:
            raise ValueError("IDF is None")
        obj = idf.get(People, name)
        if obj is None:
            return _err(f"People '{name}' not found.")
        try:
            if zone_name is not None:
                obj.zone_or_zonelist_or_space_or_spacelist_name = zone_name
            if number_of_people_schedule_name is not None:
                obj.number_of_people_schedule_name = number_of_people_schedule_name
            if activity_level_schedule_name is not None:
                obj.activity_level_schedule_name = activity_level_schedule_name
            if number_of_people_calculation_method is not None:
                obj.number_of_people_calculation_method = (
                    number_of_people_calculation_method
                )
            if number_of_people is not None:
                obj.number_of_people = number_of_people
            if people_per_floor_area is not None:
                obj.people_per_floor_area = people_per_floor_area
            if floor_area_per_person is not None:
                obj.floor_area_per_person = floor_area_per_person
            if fraction_radiant is not None:
                obj.fraction_radiant = fraction_radiant
            return _ok(f"People '{name}' updated successfully.", obj.model_dump())
        except Exception as e:
            return _err(f"Error updating people '{name}': {e}")

    @tool
    def delete_people(name: str) -> str:
        """Delete a People object."""
        if idf is None:
            raise ValueError("IDF is None")
        if not idf.has(People, name):
            return _err(f"People '{name}' not found.")
        idf.remove(People, name)
        return _ok(f"People '{name}' deleted successfully.")

    @tool
    def list_zones() -> str:
        """Read-only: list zones an occupancy load can be assigned to."""
        if idf is None:
            raise ValueError("IDF is None")
        items = [z.model_dump() for z in idf.all_of_type(Zone).values()]
        return _ok(f"Listed {len(items)} zones.", items)

    @tool
    def list_schedules() -> str:
        """Read-only: list Schedule:Compact (for number_of_people and activity_level refs)."""
        if idf is None:
            raise ValueError("IDF is None")
        items = [s.model_dump() for s in idf.all_of_type(ScheduleCompact).values()]
        return _ok(f"Listed {len(items)} schedules.", items)

    return [
        create_people,
        list_people,
        update_people,
        delete_people,
        list_zones,
        list_schedules,
    ]
