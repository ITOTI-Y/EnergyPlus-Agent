from fastmcp import FastMCP

from src.mcp.api.common import ToolInput, to_payload, validate_floor_vertices, convert_vertices_to_mcp_format, VertexValidationError
from src.mcp.tools import BuildingTool, LocationTool, ZoneTool, SurfaceTool
from pydantic import BaseModel, Field
from src.mcp.interface import ToolResponse


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

class FloorVertexInput(ToolInput):
    """Bottom-face vertex input model"""
    x: float = Field(..., alias="X", description="X coordinates")
    y: float = Field(..., alias="Y", description="Y coordinates")
    z: float = Field(..., alias="Z", description="Z coordinates(All points on the same base should be identical)")

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
    floor_vertices: list[FloorVertexInput] | None = Field(
        default=None, 
        alias="Floor Vertices",
        description="List of bottom vertices, arranged in clockwise order"
    )

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
    surface_tool: SurfaceTool,
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
        floor_vertices: list[dict], # [{"X": 0, "Y": 0, "Z": 0}, ...]
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
        zone_response = zone_tool.create(payload)

        if not zone_response.success:
            return zone_response.to_mcp_response()
        
        if floor_vertices is None:
            zone_tool.delete(name)
            return ToolResponse(
                success=False,
                message="Error: zone created without floor_vertices, please provide floor_vertices to create the zone.",
                data=zone_response.data
            ).to_mcp_response()
        else:
            try:
                vertices = convert_vertices_to_mcp_format(floor_vertices)
                is_valid, error = validate_floor_vertices(vertices)
            except (TypeError, ValueError) as e:
                zone_tool.delete(name)
                return ToolResponse(
                    success=False,
                    message=f"Invalid floor_vertices: {e}",
                ).to_mcp_response()

            if not is_valid:
                zone_tool.delete(name)
                return ToolResponse(
                    success=False,
                    message=f"Vertex validation failed: {error.message}",
                    data={"validation_error": error.model_dump()}
                ).to_mcp_response()
            if ceiling_height == "autocalculate":
                zone_tool.delete(name)
                return ToolResponse(
                    success=False,
                    message="When using the floor_vertices parameter, a specific ceiling_height value must be specified",
                ).to_mcp_response()
            try:
                height = float(ceiling_height)
            except (TypeError, ValueError):
                zone_tool.delete(name)
                return ToolResponse(
                    success=False,
                    message="ceiling_height must be a numeric value when floor_vertices is provided",
                ).to_mcp_response()
            if height <= 0:
                zone_tool.delete(name)
                return ToolResponse(
                    success=False,
                    message="ceiling_height must be greater than 0 when floor_vertices is provided",
                ).to_mcp_response()
            created_surfaces = []
            failed_surfaces = []
            n = len(vertices)
            z_floor = vertices[0]["Z"]
            z_ceiling = z_floor + height
            for i in range(n):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % n]
                wall_vertices = [
                    {"X": v1["X"], "Y": v1["Y"], "Z": z_floor},
                    {"X": v2["X"], "Y": v2["Y"], "Z": z_floor},
                    {"X": v2["X"], "Y": v2["Y"], "Z": z_ceiling},
                    {"X": v1["X"], "Y": v1["Y"], "Z": z_ceiling},
                ]
                surface_name = f"{name}_Wall_{i+1}"
                surface_data = {
                    "Name": surface_name,
                    "Surface Type": "Wall",
                    "Construction Name": "Default_Construction",  
                    "Zone Name": name,
                    "Outside Boundary Condition": "Outdoors",
                    "Sun Exposure": "SunExposed",
                    "Wind Exposure": "WindExposed",
                    "Vertices": wall_vertices,
                }
                surface_response = surface_tool.create(surface_data)

                if surface_response.success:
                    created_surfaces.append(surface_name)
                else:
                    failed_surfaces.append({"name": surface_name, "error": surface_response.message})

            floor_vertices_reversed = vertices[::-1]
            floor_surface_data = {
                "Name": f"{name}_Floor",
                "Surface Type": "Floor",
                "Construction Name": "Default_Construction",
                "Zone Name": name,
                "Outside Boundary Condition": "Ground",
                "Sun Exposure": "NoSun",
                "Wind Exposure": "NoWind",
                "Vertices": floor_vertices_reversed, 
            }
            floor_response = surface_tool.create(floor_surface_data)
            if floor_response.success:
                created_surfaces.append(f"{name}_Floor")
            else:
                failed_surfaces.append({"name": f"{name}_Floor", "error": surface_response.message})

            ceiling_vertices = [
                {"X": v["X"], "Y": v["Y"], "Z": z_ceiling}
                for v in vertices
            ]
            ceiling_surface_data = {
                "Name": f"{name}_Ceiling",
                "Surface Type": "Ceiling",
                "Construction Name": "Default_Construction",
                "Zone Name": name,
                "Outside Boundary Condition": "Adiabatic", 
                "Sun Exposure": "NoSun",
                "Wind Exposure": "NoWind",
                "Vertices": ceiling_vertices,
            }
            ceiling_response = surface_tool.create(ceiling_surface_data)
            if ceiling_response.success:
                created_surfaces.append(f"{name}_Ceiling")
            else:
                failed_surfaces.append({"name": f"{name}_Ceiling", "error": surface_response.message})
            
            return ToolResponse(
                success=len(failed_surfaces) == 0,
                message=(
                    f"Zone '{name}' created successfully with {len(created_surfaces)} surfaces."
                    if not failed_surfaces
                    else f"Zone '{name}' created with partial failures: "
                         f"{len(created_surfaces)} succeeded, {len(failed_surfaces)} failed."
                ),
                data={
                    "zone": zone_response.data,
                    "surfaces_created": created_surfaces,
                    "surfaces_failed": failed_surfaces if failed_surfaces else None,
                }
            ).to_mcp_response()

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
