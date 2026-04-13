from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState
from src.mcp.tools.construction import ConstructionTool
from src.mcp.tools.fenestration import FenestrationTool
from src.mcp.tools.surface import SurfaceTool


def make_fenestration_tools(config: ConfigState) -> list[BaseTool]:
    ft = FenestrationTool(config)

    @tool
    def create_fenestration(
        name: str,
        surface_type: str,
        construction_name: str,
        building_surface_name: str,
        vertices: list[dict[str, float]],
        multiplier: int = 1,
    ) -> str:
        """Create a FenestrationSurface:Detailed (window/door/skylight).

        Args:
            name: Unique fenestration name.
            surface_type: Window / Door / GlassDoor.
            construction_name: Existing Glazing construction name.
            building_surface_name: Existing parent Surface name.
            vertices: List of vertex dicts in meters. Each vertex is
                      `{"X": float, "Y": float, "Z": float}`. >= 3 points,
                      counter-clockwise from the outside, MUST lie on the
                      parent surface plane (coplanar).
                      Example 1.5x1.2m window centered on a south wall at
                      sill 0.8m (wall at y=0, spans x=0..5):
                        [{"X": 1.75, "Y": 0.0, "Z": 0.8},
                         {"X": 3.25, "Y": 0.0, "Z": 0.8},
                         {"X": 3.25, "Y": 0.0, "Z": 2.0},
                         {"X": 1.75, "Y": 0.0, "Z": 2.0}]
            multiplier: Number of identical copies (>= 1).
        """
        return ft.create(
            {
                "Name": name,
                "Surface Type": surface_type,
                "Construction Name": construction_name,
                "Building Surface Name": building_surface_name,
                "Multiplier": multiplier,
                "Number of Vertices": len(vertices),
                "Vertices": vertices,
            }
        ).model_dump_json()

    @tool
    def list_fenestrations() -> str:
        """List all fenestration surfaces."""
        return ft.list_all().model_dump_json()

    @tool
    def get_fenestration(name: str) -> str:
        """Read a fenestration by name."""
        return ft.read(name).model_dump_json()

    @tool
    def delete_fenestration(name: str) -> str:
        """Delete a fenestration."""
        return ft.delete(name).model_dump_json()

    @tool
    def list_surfaces() -> str:
        """Read-only: list parent surfaces a fenestration can attach to."""
        return SurfaceTool(config).list_all().model_dump_json()

    @tool
    def list_constructions() -> str:
        """Read-only: list constructions a fenestration can reference."""
        return ConstructionTool(config).list_all().model_dump_json()

    return [
        create_fenestration,
        list_fenestrations,
        get_fenestration,
        delete_fenestration,
        list_surfaces,
        list_constructions,
    ]
