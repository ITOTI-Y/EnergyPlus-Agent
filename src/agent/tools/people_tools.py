from typing import Literal

from idfpy.models import People, ScheduleCompact
from langchain_core.tools import BaseTool

from src.agent.tools._share import (
    create_in_zones,
    list_tool,
    list_zone_names_tool,
    model_tool,
    ok,
    zones_of,
)
from src.modeling import objects
from src.modeling.loads import require_level
from src.state.config_state import ConfigState


def make_people_tools(
    config: ConfigState, plan_zones: dict[str, list[str]]
) -> list[BaseTool]:
    idf = config.idf

    @model_tool
    def create_people(
        number_of_people_schedule_name: str,
        activity_level_schedule_name: str,
        plans: list[str] | None = None,
        zone_names: list[str] | None = None,
        number_of_people_calculation_method: Literal[
            "People", "People/Area", "Area/Person"
        ] = "People",
        number_of_people: float | None = None,
        people_per_floor_area: float | None = None,
        floor_area_per_person: float | None = None,
        fraction_radiant: float = 0.3,
        name_suffix: str = "People",
    ) -> str:
        """Create one People object, named '<zone>_<name_suffix>', per zone.

        Call it once per group of zones with the same values.

        Args:
            plans: Zone plan keys: every zone made from these plans, on
                every storey. Give plans, zone_names or both.
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
        require_level(
            number_of_people_calculation_method,
            number_of_people=number_of_people,
            people_per_floor_area=people_per_floor_area,
            floor_area_per_person=floor_area_per_person,
        )
        return create_in_zones(
            idf,
            "People",
            zones_of(plans, zone_names, plan_zones),
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
