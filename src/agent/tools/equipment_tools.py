from typing import Literal

from idfpy.models.internal_gains import ElectricEquipment
from idfpy.models.schedules import ScheduleCompact
from idfpy.models.thermal_zones import Zone
from langchain_core.tools import BaseTool

from src.agent.tools._share import list_tool, model_tool, ok
from src.modeling import objects
from src.state.config_state import ConfigState


def make_equipment_tools(config: ConfigState) -> list[BaseTool]:
    idf = config.idf

    @model_tool
    def create_equipment(
        name: str,
        zone_name: str,
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
    ) -> str:
        """Create an ElectricEquipment (plug load) object.

        Args:
            name: Unique equipment object name.
            zone_name: Existing Zone name.
            schedule_name: Existing Schedule:Compact (Fraction).
            design_level_calculation_method: EquipmentLevel / Watts/Area / Watts/Person.
            design_level: Absolute watts (when method=EquipmentLevel).
            watts_per_floor_area: W/m^2 (when method=Watts/Area).
            watts_per_person: W/person (when method=Watts/Person).
            fraction_latent: Latent fraction of the heat gain (0-1).
            fraction_radiant: Radiant fraction of the heat gain (0-1).
            fraction_lost: Fraction leaving the zone, e.g. exhausted (0-1).
        """
        equipment = objects.create(
            idf,
            ElectricEquipment(
                name=name,
                zone_or_zonelist_or_space_or_spacelist_name=zone_name,
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
        return ok(
            f"ElectricEquipment '{name}' created.",
            equipment.model_dump(exclude_none=True),
        )

    @model_tool
    def delete_equipment(name: str) -> str:
        """Delete an ElectricEquipment object."""
        objects.delete(idf, objects.get(idf, ElectricEquipment, name), name)
        return ok(f"ElectricEquipment '{name}' deleted.")

    return [
        create_equipment,
        list_tool(
            idf, "list_equipment", ElectricEquipment, "List all ElectricEquipment."
        ),
        delete_equipment,
        list_tool(idf, "list_zones", Zone, "List zones an equipment load can use."),
        list_tool(
            idf, "list_schedules", ScheduleCompact, "List Schedule:Compact objects."
        ),
    ]
