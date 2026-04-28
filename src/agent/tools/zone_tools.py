import json

from langchain_core.tools import BaseTool, tool

from idfpy.models.thermal_zones import Zone
from src.mcp.state import ConfigState


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def make_zone_tools(config: ConfigState) -> list[BaseTool]:
    """Create Zone CRUD tools bound to `config`."""
    idf = config._idf

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
        if idf.has("Zone", name):
            return _err(f"Zone '{name}' already exists.")
        try:
            idf.add(Zone(
                name=name,
                x_origin=x_origin,
                y_origin=y_origin,
                z_origin=z_origin,
                direction_of_relative_north=direction_of_relative_north,
                multiplier=multiplier,
            ))
            return _ok(
                f"Zone '{name}' created successfully.",
                idf.get("Zone", name).model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating zone '{name}': {e}")

    @tool
    def list_zones() -> str:
        """List all existing thermal zones."""
        items = [z.model_dump() for z in idf.all_of_type("Zone").values()]
        return _ok(f"Listed {len(items)} zones.", items)

    @tool
    def get_zone(name: str) -> str:
        """Read a zone by name."""
        obj = idf.get("Zone", name)
        if obj is None:
            return _err(f"Zone '{name}' not found.")
        return _ok(f"Zone '{name}' read successfully.", obj.model_dump())

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
        obj = idf.get("Zone", name)
        if obj is None:
            return _err(f"Zone '{name}' not found.")
        if x_origin is not None:
            obj.x_origin = x_origin
        if y_origin is not None:
            obj.y_origin = y_origin
        if z_origin is not None:
            obj.z_origin = z_origin
        if direction_of_relative_north is not None:
            obj.direction_of_relative_north = direction_of_relative_north
        if multiplier is not None:
            obj.multiplier = multiplier
        return _ok(f"Zone '{name}' updated successfully.", obj.model_dump())

    @tool
    def delete_zone(name: str) -> str:
        """Delete a zone by name."""
        refs = []
        for s in idf.all_of_type("BuildingSurface:Detailed").values():
            if s.zone_name == name:
                refs.append(f"Surface:{s.name}")
        for ils in idf.all_of_type("HVACTemplate:Zone:IdealLoadsAirSystem").values():
            if ils.zone_name == name:
                refs.append(f"IdealLoadsSystem:{ils.zone_name}")
        if refs:
            return _err(
                f"Zone '{name}' is referenced by other components.",
                {"references": refs},
            )
        removed = idf.remove("Zone", name)
        if removed is None:
            return _err(f"Zone '{name}' not found.")
        return _ok(f"Zone '{name}' deleted successfully.")

    return [create_zone, list_zones, get_zone, update_zone, delete_zone]
