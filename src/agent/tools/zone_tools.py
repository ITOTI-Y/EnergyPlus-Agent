from idfpy.models.thermal_zones import Zone
from langchain_core.tools import BaseTool

from src.agent.tools._share import list_tool, model_tool, ok
from src.modeling import objects
from src.state.config_state import ConfigState


def make_zone_tools(config: ConfigState) -> list[BaseTool]:
    """Create Zone CRUD tools bound to `config`."""
    idf = config.idf

    @model_tool
    def create_zone(
        name: str,
        x_origin: float = 0.0,
        y_origin: float = 0.0,
        z_origin: float = 0.0,
        direction_of_relative_north: float = 0.0,
        multiplier: int = 1,
    ) -> str:
        """Create a thermal zone.

        Args:
            name: Unique zone name (e.g., 'F1_Office_North').
            x_origin: X of zone origin (meters).
            y_origin: Y of zone origin (meters).
            z_origin: Z of zone origin; use 0 for ground floor, floor height for higher floors.
            direction_of_relative_north: Zone rotation (degrees, 0-360).
            multiplier: Zone multiplier (>= 1) for repeated identical zones.
        """
        zone = objects.create(
            idf,
            Zone(
                name=name,
                x_origin=x_origin,
                y_origin=y_origin,
                z_origin=z_origin,
                direction_of_relative_north=direction_of_relative_north,
                multiplier=multiplier,
            ),
        )
        return ok(f"Zone '{name}' created.", zone.model_dump(exclude_none=True))

    @model_tool
    def get_zone(name: str) -> str:
        """Read a zone by name."""
        zone = objects.get(idf, Zone, name)
        return ok(f"Zone '{name}' read.", zone.model_dump(exclude_none=True))

    @model_tool
    def update_zone(
        name: str,
        x_origin: float | None = None,
        y_origin: float | None = None,
        z_origin: float | None = None,
        direction_of_relative_north: float | None = None,
        multiplier: int | None = None,
    ) -> str:
        """Update a zone's origin, rotation or multiplier; omitted fields stay."""
        changes = {
            "x_origin": x_origin,
            "y_origin": y_origin,
            "z_origin": z_origin,
            "direction_of_relative_north": direction_of_relative_north,
            "multiplier": multiplier,
        }
        zone = objects.update(
            idf,
            objects.get(idf, Zone, name),
            {k: v for k, v in changes.items() if v is not None},
        )
        return ok(f"Zone '{name}' updated.", zone.model_dump(exclude_none=True))

    @model_tool
    def delete_zone(name: str) -> str:
        """Delete a zone; refused while surfaces, loads or HVAC reference it."""
        objects.delete(idf, objects.get(idf, Zone, name), name)
        return ok(f"Zone '{name}' deleted.")

    return [
        create_zone,
        list_tool(idf, "list_zones", Zone, "List all existing thermal zones."),
        get_zone,
        update_zone,
        delete_zone,
    ]
