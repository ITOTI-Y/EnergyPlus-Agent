from typing import Literal

from fastmcp import FastMCP
from idfpy.models.location import SiteLocation
from idfpy.models.simulation import Building
from idfpy.models.thermal_zones import Zone

from src.mcp.api.common import Outcome, dump, given, model_tool
from src.modeling import objects
from src.state.config_state import ConfigState

type Terrain = Literal["Country", "Suburbs", "City", "Ocean", "Urban"]


def register_core_tools(mcp: FastMCP, state: ConfigState) -> None:
    """Register Building, Site:Location and Zone tools."""
    idf = state.idf
    tool = model_tool(mcp)

    @tool
    def create_building(
        name: str, north_axis: float = 0.0, terrain: Terrain = "Suburbs"
    ) -> Outcome:
        """Create the Building object.

        Args:
            name: Building name.
            north_axis: Building north axis angle in degrees from true north.
            terrain: Surroundings used for wind and solar calculations.
        """
        building = Building(name=name, north_axis=north_axis, terrain=terrain)
        return f"Building '{name}' created.", dump(objects.create(idf, building))

    @tool
    def get_building(name: str) -> Outcome:
        """Read the Building object by name."""
        return f"Building '{name}' read.", dump(objects.get(idf, Building, name))

    @tool
    def update_building(
        name: str,
        new_name: str | None = None,
        north_axis: float | None = None,
        terrain: Terrain | None = None,
    ) -> Outcome:
        """Update the Building object; omitted fields stay unchanged."""
        building = objects.update(
            idf,
            objects.get(idf, Building, name),
            given(name=new_name, north_axis=north_axis, terrain=terrain),
        )
        return f"Building '{name}' updated.", dump(building)

    @tool
    def delete_building(name: str) -> Outcome:
        """Delete the Building object."""
        objects.delete(idf, objects.get(idf, Building, name), name)
        return f"Building '{name}' deleted.", None

    @tool
    def list_buildings() -> Outcome:
        """List Building objects."""
        return "Listed buildings.", objects.dumps(idf.all_of_type(Building))

    @tool
    def create_location(
        name: str, latitude: float, longitude: float, time_zone: float, elevation: float
    ) -> Outcome:
        """Create the Site:Location object.

        Args:
            name: Location name.
            latitude: Degrees, -90 to 90.
            longitude: Degrees, -180 to 180.
            time_zone: UTC offset in hours, -12 to 14.
            elevation: Meters above sea level.
        """
        location = SiteLocation(
            name=name,
            latitude=latitude,
            longitude=longitude,
            time_zone=time_zone,
            elevation=elevation,
        )
        return f"Location '{name}' created.", dump(objects.create(idf, location))

    @tool
    def get_location(name: str) -> Outcome:
        """Read the Site:Location object by name."""
        return f"Location '{name}' read.", dump(objects.get(idf, SiteLocation, name))

    @tool
    def update_location(
        name: str,
        new_name: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        time_zone: float | None = None,
        elevation: float | None = None,
    ) -> Outcome:
        """Update the Site:Location object; omitted fields stay unchanged."""
        location = objects.update(
            idf,
            objects.get(idf, SiteLocation, name),
            given(
                name=new_name,
                latitude=latitude,
                longitude=longitude,
                time_zone=time_zone,
                elevation=elevation,
            ),
        )
        return f"Location '{name}' updated.", dump(location)

    @tool
    def delete_location(name: str) -> Outcome:
        """Delete the Site:Location object."""
        objects.delete(idf, objects.get(idf, SiteLocation, name), name)
        return f"Location '{name}' deleted.", None

    @tool
    def list_locations() -> Outcome:
        """List Site:Location objects."""
        return "Listed locations.", objects.dumps(idf.all_of_type(SiteLocation))

    @tool
    def create_zone(
        name: str,
        x_origin: float = 0.0,
        y_origin: float = 0.0,
        z_origin: float = 0.0,
        direction_of_relative_north: float = 0.0,
        multiplier: int = 1,
    ) -> Outcome:
        """Create a thermal zone; add its surfaces with create_surface.

        Args:
            name: Unique zone name.
            x_origin: Zone origin X in meters.
            y_origin: Zone origin Y in meters.
            z_origin: Zone origin Z in meters.
            direction_of_relative_north: Zone rotation in degrees.
            multiplier: Count of identical zones represented (>= 1).
        """
        zone = Zone(
            name=name,
            x_origin=x_origin,
            y_origin=y_origin,
            z_origin=z_origin,
            direction_of_relative_north=direction_of_relative_north,
            multiplier=multiplier,
        )
        return f"Zone '{name}' created.", dump(objects.create(idf, zone))

    @tool
    def get_zone(name: str) -> Outcome:
        """Read a zone by name."""
        return f"Zone '{name}' read.", dump(objects.get(idf, Zone, name))

    @tool
    def update_zone(
        name: str,
        new_name: str | None = None,
        x_origin: float | None = None,
        y_origin: float | None = None,
        z_origin: float | None = None,
        direction_of_relative_north: float | None = None,
        multiplier: int | None = None,
    ) -> Outcome:
        """Update a zone; a new name is applied to every reference to the zone."""
        zone = objects.update(
            idf,
            objects.get(idf, Zone, name),
            given(
                name=new_name,
                x_origin=x_origin,
                y_origin=y_origin,
                z_origin=z_origin,
                direction_of_relative_north=direction_of_relative_north,
                multiplier=multiplier,
            ),
        )
        return f"Zone '{name}' updated.", dump(zone)

    @tool
    def delete_zone(name: str) -> Outcome:
        """Delete a zone; refused while surfaces, loads or HVAC reference it."""
        objects.delete(idf, objects.get(idf, Zone, name), name)
        return f"Zone '{name}' deleted.", None

    @tool
    def list_zones() -> Outcome:
        """List all zones."""
        return "Listed zones.", objects.dumps(idf.all_of_type(Zone))
