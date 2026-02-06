from fastmcp import FastMCP
from pydantic import Field

from src.mcp.api.common import ToolInput, to_payload
from src.mcp.tools import BuildingTool, LocationTool, ZoneTool


class BuildingCreateInput(ToolInput):
    name: str = Field(alias="Name")
    north_axis: float = Field(alias="North Axis")
    terrain: str = Field(alias="Terrain")


class BuildingUpdateInput(ToolInput):
    name: str = Field(alias="Name")
    north_axis: float = Field(alias="North Axis")
    terrain: str = Field(alias="Terrain")


class LocationCreateInput(ToolInput):
    name: str = Field(alias="Name")
    latitude: float = Field(alias="Latitude")
    longitude: float = Field(alias="Longitude")
    time_zone: float = Field(alias="Time Zone")
    elevation: float = Field(alias="Elevation")


class LocationUpdateInput(ToolInput):
    name: str = Field(alias="Name")
    latitude: float = Field(alias="Latitude")
    longitude: float = Field(alias="Longitude")
    time_zone: float = Field(alias="Time Zone")
    elevation: float = Field(alias="Elevation")


class ZoneCreateInput(ToolInput):
    name: str = Field(alias="Name")
    x_origin: float = Field(default=0.0, alias="X Origin")
    y_origin: float = Field(default=0.0, alias="Y Origin")
    z_origin: float = Field(default=0.0, alias="Z Origin")
    direction_of_relative_north: float | None = Field(
        default=0.0,
        alias="Direction of Relative North",
    )
    multiplier: int = Field(default=1, alias="Multiplier")
    ceiling_height: float | str = Field(default="autocalculate", alias="Ceiling Height")
    volume: float | str = Field(default="autocalculate", alias="Volume")
    floor_area: float | str = Field(default="autocalculate", alias="Floor Area")


class ZoneUpdateInput(ToolInput):
    name: str = Field(alias="Name")
    x_origin: float | None = Field(default=None, alias="X Origin")
    y_origin: float | None = Field(default=None, alias="Y Origin")
    z_origin: float | None = Field(default=None, alias="Z Origin")
    direction_of_relative_north: float | None = Field(
        default=None,
        alias="Direction of Relative North",
    )
    multiplier: int | None = Field(default=None, alias="Multiplier")
    ceiling_height: float | str | None = Field(default=None, alias="Ceiling Height")
    volume: float | str | None = Field(default=None, alias="Volume")
    floor_area: float | str | None = Field(default=None, alias="Floor Area")


def register_core_tools(
    mcp: FastMCP,
    building_tool: BuildingTool,
    location_tool: LocationTool,
    zone_tool: ZoneTool,
) -> None:
    @mcp.tool
    def create_building(
        name: str,
        north_axis: float,
        terrain: str,
    ) -> dict:
        payload = to_payload(
            BuildingCreateInput.model_validate(
                {
                    "name": name,
                    "north_axis": north_axis,
                    "terrain": terrain,
                }
            )
        )
        return building_tool.create(payload).to_mcp_response()

    @mcp.tool
    def get_building(name: str) -> dict:
        return building_tool.read(name).to_mcp_response()

    @mcp.tool
    def update_building(name: str, north_axis: float, terrain: str) -> dict:
        payload = to_payload(
            BuildingUpdateInput.model_validate(
                {
                    "name": name,
                    "north_axis": north_axis,
                    "terrain": terrain,
                }
            )
        )
        return building_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def delete_building(name: str) -> dict:
        return building_tool.delete(name).to_mcp_response()

    @mcp.tool
    def list_buildings() -> dict:
        return building_tool.list_all().to_mcp_response()

    @mcp.tool
    def create_location(
        name: str,
        latitude: float,
        longitude: float,
        time_zone: float,
        elevation: float,
    ) -> dict:
        payload = to_payload(
            LocationCreateInput.model_validate(
                {
                    "name": name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "time_zone": time_zone,
                    "elevation": elevation,
                }
            )
        )
        return location_tool.create(payload).to_mcp_response()

    @mcp.tool
    def get_location(name: str) -> dict:
        return location_tool.read(name).to_mcp_response()

    @mcp.tool
    def update_location(
        name: str,
        latitude: float,
        longitude: float,
        time_zone: float,
        elevation: float,
    ) -> dict:
        payload = to_payload(
            LocationUpdateInput.model_validate(
                {
                    "name": name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "time_zone": time_zone,
                    "elevation": elevation,
                }
            )
        )
        return location_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def delete_location(name: str) -> dict:
        return location_tool.delete(name).to_mcp_response()

    @mcp.tool
    def list_locations() -> dict:
        return location_tool.list_all().to_mcp_response()

    @mcp.tool
    def create_zone(
        name: str,
        x_origin: float = 0.0,
        y_origin: float = 0.0,
        z_origin: float = 0.0,
        direction_of_relative_north: float | None = 0.0,
        multiplier: int = 1,
        ceiling_height: float | str = "autocalculate",
        volume: float | str = "autocalculate",
        floor_area: float | str = "autocalculate",
    ) -> dict:
        payload = to_payload(
            ZoneCreateInput.model_validate(
                {
                    "name": name,
                    "x_origin": x_origin,
                    "y_origin": y_origin,
                    "z_origin": z_origin,
                    "direction_of_relative_north": direction_of_relative_north,
                    "multiplier": multiplier,
                    "ceiling_height": ceiling_height,
                    "volume": volume,
                    "floor_area": floor_area,
                }
            )
        )
        return zone_tool.create(payload).to_mcp_response()

    @mcp.tool
    def get_zone(name: str) -> dict:
        return zone_tool.read(name).to_mcp_response()

    @mcp.tool
    def update_zone(
        name: str,
        x_origin: float | None = None,
        y_origin: float | None = None,
        z_origin: float | None = None,
        direction_of_relative_north: float | None = None,
        multiplier: int | None = None,
        ceiling_height: float | str | None = None,
        volume: float | str | None = None,
        floor_area: float | str | None = None,
    ) -> dict:
        payload = to_payload(
            ZoneUpdateInput.model_validate(
                {
                    "name": name,
                    "x_origin": x_origin,
                    "y_origin": y_origin,
                    "z_origin": z_origin,
                    "direction_of_relative_north": direction_of_relative_north,
                    "multiplier": multiplier,
                    "ceiling_height": ceiling_height,
                    "volume": volume,
                    "floor_area": floor_area,
                }
            )
        )
        return zone_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def delete_zone(name: str) -> dict:
        return zone_tool.delete(name).to_mcp_response()

    @mcp.tool
    def list_zones() -> dict:
        return zone_tool.list_all().to_mcp_response()
