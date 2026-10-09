from typing import Literal

from idfpy.models.internal_gains import People
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
from src.state.config_state import ConfigState


def make_people_tools(config: ConfigState) -> list[BaseTool]:
    idf = config.idf

    @model_tool
    def create_people(
        zone_names: list[str],
        number_of_people_schedule_name: str,
        activity_level_schedule_name: str,
        number_of_people_calculation_method: Literal[
            "People", "People/Area", "Area/Person"
        ] = "People",
        number_of_people: float = 0.0,
        people_per_floor_area: float = 0.0,
        floor_area_per_person: float = 0.0,
        fraction_radiant: float = 0.3,
        name_suffix: str = "People",
    ) -> str:
        """Create one People object, named '<zone>_<name_suffix>', per zone.

        Call it once per group of zones with the same values.

        Args:
            zone_names: Existing Zone names.
            number_of_people_schedule_name: Existing Schedule:Compact (Fraction).
            activity_level_schedule_name: Existing Schedule:Compact (ActivityLevel limits, W/person).
            number_of_people_calculation_method: People / People/Area / Area/Person.
            number_of_people: Absolute count per zone (use when method=People).
            people_per_floor_area: people/m^2 (use when method=People/Area).
            floor_area_per_person: m^2/person (use when method=Area/Person).
            fraction_radiant: Radiant fraction of sensible heat (0-1).
            name_suffix: Name suffix; give another one for a second People
                object in the same zones.
        """
        return create_in_zones(
            idf,
            "People",
            zone_names,
            lambda zone: People(
                name=f"{zone}_{name_suffix}",
                zone_or_zonelist_or_space_or_spacelist_name=zone,
                number_of_people_schedule_name=number_of_people_schedule_name,
                activity_level_schedule_name=activity_level_schedule_name,
                number_of_people_calculation_method=number_of_people_calculation_method,
                number_of_people=number_of_people,
                people_per_floor_area=people_per_floor_area,
                floor_area_per_person=floor_area_per_person,
                fraction_radiant=fraction_radiant,
            ),
        )

    @model_tool
    def delete_people(name: str) -> str:
        """Delete a People object."""
        objects.delete(idf, objects.get(idf, People, name), name)
        return ok(f"People '{name}' deleted.")

    return [
        create_people,
        list_tool(
            idf,
            "list_people",
            People,
            "List People objects: name, zone, schedules and density.",
            (
                "name",
                "zone_or_zonelist_or_space_or_spacelist_name",
                "number_of_people_schedule_name",
                "activity_level_schedule_name",
                "number_of_people_calculation_method",
                "number_of_people",
                "people_per_floor_area",
                "floor_area_per_person",
            ),
        ),
        delete_people,
        list_zone_names_tool(idf, "List zone names an occupancy load can use."),
        list_tool(
            idf,
            "list_schedules",
            ScheduleCompact,
            "List Schedule:Compact names and type limits, for occupancy and "
            "activity references.",
            ("name", "schedule_type_limits_name"),
        ),
    ]
