from typing import Literal

from idfpy.models.internal_gains import Lights
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


def make_lights_tools(config: ConfigState) -> list[BaseTool]:
    idf = config.idf

    @model_tool
    def create_lights(
        zone_names: list[str],
        schedule_name: str,
        design_level_calculation_method: Literal[
            "LightingLevel", "Watts/Area", "Watts/Person"
        ] = "Watts/Area",
        lighting_level: float = 0.0,
        watts_per_floor_area: float = 0.0,
        watts_per_person: float = 0.0,
        fraction_radiant: float = 0.0,
        fraction_visible: float = 0.0,
        name_suffix: str = "Lights",
    ) -> str:
        """Create one Lights object, named '<zone>_<name_suffix>', per zone.

        Call it once per group of zones with the same values.

        Args:
            zone_names: Existing Zone names.
            schedule_name: Existing Schedule:Compact (Fraction).
            design_level_calculation_method: LightingLevel / Watts/Area / Watts/Person.
            lighting_level: Absolute watts (when method=LightingLevel).
            watts_per_floor_area: W/m^2 (when method=Watts/Area).
            watts_per_person: W/person (when method=Watts/Person).
            fraction_radiant: Radiant fraction (0-1).
            fraction_visible: Visible light fraction (0-1).
            name_suffix: Name suffix; give another one for a second Lights
                object in the same zones.
        """
        return create_in_zones(
            idf,
            "Lights",
            zone_names,
            lambda zone: Lights(
                name=f"{zone}_{name_suffix}",
                zone_or_zonelist_or_space_or_spacelist_name=zone,
                schedule_name=schedule_name,
                design_level_calculation_method=design_level_calculation_method,
                lighting_level=lighting_level,
                watts_per_floor_area=watts_per_floor_area,
                watts_per_person=watts_per_person,
                fraction_radiant=fraction_radiant,
                fraction_visible=fraction_visible,
            ),
        )

    @model_tool
    def delete_light(name: str) -> str:
        """Delete a Lights object."""
        objects.delete(idf, objects.get(idf, Lights, name), name)
        return ok(f"Lights '{name}' deleted.")

    return [
        create_lights,
        list_tool(
            idf,
            "list_lights",
            Lights,
            "List Lights objects: name, zone, schedule and level.",
            (
                "name",
                "zone_or_zonelist_or_space_or_spacelist_name",
                "schedule_name",
                "design_level_calculation_method",
                "lighting_level",
                "watts_per_floor_area",
                "watts_per_person",
            ),
        ),
        delete_light,
        list_zone_names_tool(idf, "List zone names a Lights load can use."),
        list_tool(
            idf,
            "list_schedules",
            ScheduleCompact,
            "List Schedule:Compact names and type limits.",
            ("name", "schedule_type_limits_name"),
        ),
    ]
