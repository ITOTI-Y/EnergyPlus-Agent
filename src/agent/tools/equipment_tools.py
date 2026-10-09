from typing import Literal

from idfpy.models.internal_gains import ElectricEquipment
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


def make_equipment_tools(config: ConfigState) -> list[BaseTool]:
    idf = config.idf

    @model_tool
    def create_equipment(
        zone_names: list[str],
        schedule_name: str,
        design_level_calculation_method: Literal[
            "EquipmentLevel", "Watts/Area", "Watts/Person"
        ] = "Watts/Area",
        design_level: float = 0.0,
        watts_per_floor_area: float = 0.0,
        watts_per_person: float = 0.0,
        fraction_latent: float = 0.0,
        fraction_radiant: float = 0.0,
        fraction_lost: float = 0.0,
        name_suffix: str = "Equipment",
    ) -> str:
        """Create one ElectricEquipment, named '<zone>_<name_suffix>', per zone.

        Call it once per group of zones with the same values.

        Args:
            zone_names: Existing Zone names.
            schedule_name: Existing Schedule:Compact (Fraction).
            design_level_calculation_method: EquipmentLevel / Watts/Area / Watts/Person.
            design_level: Absolute watts per zone (when method=EquipmentLevel).
            watts_per_floor_area: W/m^2 (when method=Watts/Area).
            watts_per_person: W/person (when method=Watts/Person).
            fraction_latent: Latent fraction of the heat gain (0-1).
            fraction_radiant: Radiant fraction of the heat gain (0-1).
            fraction_lost: Fraction leaving the zone, e.g. exhausted (0-1).
            name_suffix: Name suffix; give another one for a second
                ElectricEquipment in the same zones.
        """
        return create_in_zones(
            idf,
            "ElectricEquipment",
            zone_names,
            lambda zone: ElectricEquipment(
                name=f"{zone}_{name_suffix}",
                zone_or_zonelist_or_space_or_spacelist_name=zone,
                schedule_name=schedule_name,
                design_level_calculation_method=design_level_calculation_method,
                design_level=design_level,
                watts_per_floor_area=watts_per_floor_area,
                watts_per_person=watts_per_person,
                fraction_latent=fraction_latent,
                fraction_radiant=fraction_radiant,
                fraction_lost=fraction_lost,
            ),
        )

    @model_tool
    def delete_equipment(name: str) -> str:
        """Delete an ElectricEquipment object."""
        objects.delete(idf, objects.get(idf, ElectricEquipment, name), name)
        return ok(f"ElectricEquipment '{name}' deleted.")

    return [
        create_equipment,
        list_tool(
            idf,
            "list_equipment",
            ElectricEquipment,
            "List ElectricEquipment: name, zone, schedule and level.",
            (
                "name",
                "zone_or_zonelist_or_space_or_spacelist_name",
                "schedule_name",
                "design_level_calculation_method",
                "design_level",
                "watts_per_floor_area",
                "watts_per_person",
            ),
        ),
        delete_equipment,
        list_zone_names_tool(idf, "List zone names an equipment load can use."),
        list_tool(
            idf,
            "list_schedules",
            ScheduleCompact,
            "List Schedule:Compact names and type limits.",
            ("name", "schedule_type_limits_name"),
        ),
    ]
