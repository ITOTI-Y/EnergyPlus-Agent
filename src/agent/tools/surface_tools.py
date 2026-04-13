from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState
from src.mcp.tools.construction import ConstructionTool
from src.mcp.tools.surface import SurfaceTool
from src.mcp.tools.zone import ZoneTool


def make_surface_tools(config: ConfigState) -> list[BaseTool]:
    st = SurfaceTool(config)

    @tool
    def create_surface(
        name: str,
        surface_type: str,
        construction_name: str,
        zone_name: str,
        outside_boundary_condition: str,
        vertices: list[dict[str, float]],
        sun_exposure: str = "NoSun",
        wind_exposure: str = "NoWind",
        outside_boundary_condition_object: str | None = None,
    ) -> str:
        """Create a BuildingSurface:Detailed (wall/floor/roof/ceiling).

        Args:
            name: Unique surface name.
            surface_type: Wall / Floor / Roof / Ceiling.
            construction_name: Existing Construction name.
            zone_name: Existing Zone name the surface belongs to.
            outside_boundary_condition: Outdoors / Ground / Zone / Adiabatic / Surface.
            vertices: List of vertex dicts in meters. Each vertex is
                      `{"X": float, "Y": float, "Z": float}`. >= 3 points,
                      ordered counter-clockwise when viewed from OUTSIDE.
                      Example 4-vertex south wall (2m tall, 5m wide, at y=0):
                        [{"X": 0.0, "Y": 0.0, "Z": 0.0},
                         {"X": 5.0, "Y": 0.0, "Z": 0.0},
                         {"X": 5.0, "Y": 0.0, "Z": 2.0},
                         {"X": 0.0, "Y": 0.0, "Z": 2.0}]
            sun_exposure: SunExposed / NoSun (use SunExposed for outdoor-facing walls/roof).
            wind_exposure: WindExposed / NoWind.
            outside_boundary_condition_object: Matching surface name when
                                               outside_boundary_condition in {Surface, Zone}.
        """
        return st.create(
            {
                "Name": name,
                "Surface Type": surface_type,
                "Construction Name": construction_name,
                "Zone Name": zone_name,
                "Outside Boundary Condition": outside_boundary_condition,
                "Outside Boundary Condition Object": outside_boundary_condition_object,
                "Sun Exposure": sun_exposure,
                "Wind Exposure": wind_exposure,
                "Vertices": vertices,
            }
        ).model_dump_json()

    @tool
    def list_surfaces() -> str:
        """List all building surfaces."""
        return st.list_all().model_dump_json()

    @tool
    def get_surface(name: str) -> str:
        """Read a surface by name."""
        return st.read(name).model_dump_json()

    @tool
    def delete_surface(name: str) -> str:
        """Delete a surface. Fails if fenestration references it."""
        return st.delete(name).model_dump_json()

    @tool
    def list_zones() -> str:
        """Read-only: list zones a surface can be assigned to."""
        return ZoneTool(config).list_all().model_dump_json()

    @tool
    def list_constructions() -> str:
        """Read-only: list constructions a surface can reference."""
        return ConstructionTool(config).list_all().model_dump_json()

    return [
        create_surface,
        list_surfaces,
        get_surface,
        delete_surface,
        list_zones,
        list_constructions,
    ]
