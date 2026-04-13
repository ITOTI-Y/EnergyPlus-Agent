from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState
from src.mcp.tools.people import PeopleTool


def make_people_tools(config: ConfigState) -> list[BaseTool]:
    pt = PeopleTool(config)

    @tool
    def create_people(
        name: str,
        zone_name: str,
        number_of_people_schedule_name: str,
        activity_level_schedule_name: str,
        number_of_people_calculation_method: str = "People",
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
        return pt.create(
            {
                "Name": name,
                "Zone or ZoneList or Space or SpaceList Name": zone_name,
                "Number of People Schedule Name": number_of_people_schedule_name,
                "Activity Level Schedule Name": activity_level_schedule_name,
                "Number of People Calculation Method": number_of_people_calculation_method,
                "Number of People": number_of_people,
                "People per Floor Area": people_per_floor_area,
                "Floor Area per Person": floor_area_per_person,
                "Fraction Radiant": fraction_radiant,
            }
        ).model_dump_json()

    @tool
    def list_people() -> str:
        """List all People objects."""
        return pt.list_all().model_dump_json()

    @tool
    def delete_people(name: str) -> str:
        """Delete a People object."""
        return pt.delete(name).model_dump_json()

    return [create_people, list_people, delete_people]
