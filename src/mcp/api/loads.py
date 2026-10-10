from typing import Literal

from fastmcp import FastMCP
from idfpy.models.internal_gains import Lights, People

from src.mcp.api.common import Outcome, dump, given, model_tool
from src.modeling import objects
from src.state.config_state import ConfigState

type PeopleMethod = Literal["People", "People/Area", "Area/Person"]
type LightingMethod = Literal["LightingLevel", "Watts/Area", "Watts/Person"]


def register_load_tools(mcp: FastMCP, state: ConfigState) -> None:
    """Register People and Lights tools."""
    idf = state.idf
    tool = model_tool(mcp)

    @tool
    def create_people(
        name: str,
        zone_name: str,
        number_of_people_schedule_name: str,
        activity_level_schedule_name: str,
        number_of_people_calculation_method: PeopleMethod = "People",
        number_of_people: float = 0.0,
        people_per_floor_area: float = 0.0,
        floor_area_per_person: float = 0.0,
        fraction_radiant: float = 0.3,
    ) -> Outcome:
        """Create a People load.

        Args:
            name: Unique name.
            zone_name: Existing zone.
            number_of_people_schedule_name: Fraction schedule.
            activity_level_schedule_name: Activity level schedule (W/person).
            number_of_people_calculation_method: Which of the next three applies.
            number_of_people: Absolute count.
            people_per_floor_area: people/m^2.
            floor_area_per_person: m^2/person.
            fraction_radiant: Radiant fraction of sensible heat (0-1).
        """
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
        return f"People '{name}' created.", dump(objects.create(idf, people))

    @tool
    def get_people(name: str) -> Outcome:
        """Read a People load by name."""
        return f"People '{name}' read.", dump(objects.get(idf, People, name))

    @tool
    def update_people(
        name: str,
        new_name: str | None = None,
        zone_name: str | None = None,
        number_of_people_schedule_name: str | None = None,
        activity_level_schedule_name: str | None = None,
        number_of_people_calculation_method: PeopleMethod | None = None,
        number_of_people: float | None = None,
        people_per_floor_area: float | None = None,
        floor_area_per_person: float | None = None,
        fraction_radiant: float | None = None,
    ) -> Outcome:
        """Update a People load; omitted fields stay unchanged."""
        people = objects.update(
            idf,
            objects.get(idf, People, name),
            given(
                name=new_name,
                zone_or_zonelist_or_space_or_spacelist_name=zone_name,
                number_of_people_schedule_name=number_of_people_schedule_name,
                activity_level_schedule_name=activity_level_schedule_name,
                number_of_people_calculation_method=number_of_people_calculation_method,
                number_of_people=number_of_people,
                people_per_floor_area=people_per_floor_area,
                floor_area_per_person=floor_area_per_person,
                fraction_radiant=fraction_radiant,
            ),
        )
        return f"People '{name}' updated.", dump(people)

    @tool
    def delete_people(name: str) -> Outcome:
        """Delete a People load."""
        objects.delete(idf, objects.get(idf, People, name), name)
        return f"People '{name}' deleted.", None

    @tool
    def list_people() -> Outcome:
        """List all People loads."""
        return "Listed People.", objects.dumps(idf.all_of_type(People))

    @tool
    def create_light(
        name: str,
        zone_name: str,
        schedule_name: str,
        design_level_calculation_method: LightingMethod = "Watts/Area",
        lighting_level: float = 0.0,
        watts_per_floor_area: float = 0.0,
        watts_per_person: float = 0.0,
        fraction_radiant: float = 0.0,
        fraction_visible: float = 0.0,
    ) -> Outcome:
        """Create a Lights load.

        Args:
            name: Unique name.
            zone_name: Existing zone.
            schedule_name: Fraction schedule.
            design_level_calculation_method: Which of the next three applies.
            lighting_level: Watts.
            watts_per_floor_area: W/m^2.
            watts_per_person: W/person.
            fraction_radiant: Radiant fraction (0-1).
            fraction_visible: Visible fraction (0-1).
        """
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
        return f"Lights '{name}' created.", dump(objects.create(idf, light))

    @tool
    def get_light(name: str) -> Outcome:
        """Read a Lights load by name."""
        return f"Lights '{name}' read.", dump(objects.get(idf, Lights, name))

    @tool
    def update_light(
        name: str,
        new_name: str | None = None,
        zone_name: str | None = None,
        schedule_name: str | None = None,
        design_level_calculation_method: LightingMethod | None = None,
        lighting_level: float | None = None,
        watts_per_floor_area: float | None = None,
        watts_per_person: float | None = None,
        fraction_radiant: float | None = None,
        fraction_visible: float | None = None,
    ) -> Outcome:
        """Update a Lights load; omitted fields stay unchanged."""
        light = objects.update(
            idf,
            objects.get(idf, Lights, name),
            given(
                name=new_name,
                zone_or_zonelist_or_space_or_spacelist_name=zone_name,
                schedule_name=schedule_name,
                design_level_calculation_method=design_level_calculation_method,
                lighting_level=lighting_level,
                watts_per_floor_area=watts_per_floor_area,
                watts_per_person=watts_per_person,
                fraction_radiant=fraction_radiant,
                fraction_visible=fraction_visible,
            ),
        )
        return f"Lights '{name}' updated.", dump(light)

    @tool
    def delete_light(name: str) -> Outcome:
        """Delete a Lights load."""
        objects.delete(idf, objects.get(idf, Lights, name), name)
        return f"Lights '{name}' deleted.", None

    @tool
    def list_lights() -> Outcome:
        """List all Lights loads."""
        return "Listed Lights.", objects.dumps(idf.all_of_type(Lights))
