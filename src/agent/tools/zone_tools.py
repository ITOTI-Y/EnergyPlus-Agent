from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState
from src.mcp.tools.zone import ZoneTool


def make_zone_tools(config: ConfigState) -> list[BaseTool]:
    """Create Zone CRUD tools bound to `config`."""
    zt = ZoneTool(config)

    @tool
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
        return zt.create(
            {
                "Name": name,
                "X Origin": x_origin,
                "Y Origin": y_origin,
                "Z Origin": z_origin,
                "Direction of Relative North": direction_of_relative_north,
                "Multiplier": multiplier,
            }
        ).model_dump_json()

    @tool
    def list_zones() -> str:
        """List all existing thermal zones."""
        return zt.list_all().model_dump_json()

    @tool
    def get_zone(name: str) -> str:
        """Read a zone by name."""
        return zt.read(name).model_dump_json()

    @tool
    def update_zone(
        name: str,
        x_origin: float | None = None,
        y_origin: float | None = None,
        z_origin: float | None = None,
        direction_of_relative_north: float | None = None,
        multiplier: int | None = None,
    ) -> str:
        """Update a zone's origin coordinates."""
        payload: dict = {}
        if x_origin is not None:
            payload["X Origin"] = x_origin
        if y_origin is not None:
            payload["Y Origin"] = y_origin
        if z_origin is not None:
            payload["Z Origin"] = z_origin
        if direction_of_relative_north is not None:
            payload["Direction of Relative North"] = direction_of_relative_north
        if multiplier is not None:
            payload["Multiplier"] = multiplier
        return zt.update(name, payload).model_dump_json()

    @tool
    def delete_zone(name: str) -> str:
        """Delete a zone by name."""
        return zt.delete(name).model_dump_json()

    return [create_zone, list_zones, get_zone, update_zone, delete_zone]
