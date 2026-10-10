from typing import Literal

from idfpy.models.constructions import Construction
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
)
from langchain_core.tools import BaseTool

from src.agent.tools._share import list_tool, model_tool, ok
from src.modeling import objects
from src.modeling.envelope import VertexSchema, fenestration_from_vertices
from src.state.config_state import ConfigState


def make_fenestration_tools(config: ConfigState) -> list[BaseTool]:
    idf = config.idf

    @model_tool
    def create_fenestration(
        name: str,
        surface_type: Literal["Window", "Door", "GlassDoor"],
        construction_name: str,
        building_surface_name: str,
        vertices: list[VertexSchema],
        multiplier: int = 1,
    ) -> str:
        """Create a FenestrationSurface:Detailed (window/door).

        Args:
            name: Unique fenestration name.
            surface_type: Window / Door / GlassDoor.
            construction_name: Existing construction name.
            building_surface_name: Existing parent surface name.
            vertices: 3 or 4 vertices in meters, counter-clockwise from the
                outside, lying on the parent surface plane. Example 1.5x1.2m
                window centered on a south wall at sill 0.8m (wall at y=0,
                spans x=0..5):
                  [{"X": 1.75, "Y": 0.0, "Z": 0.8},
                   {"X": 3.25, "Y": 0.0, "Z": 0.8},
                   {"X": 3.25, "Y": 0.0, "Z": 2.0},
                   {"X": 1.75, "Y": 0.0, "Z": 2.0}]
            multiplier: Number of identical copies (>= 1).
        """
        fenestration = objects.create(
            idf,
            fenestration_from_vertices(
                vertices,
                name=name,
                surface_type=surface_type,
                construction_name=construction_name,
                building_surface_name=building_surface_name,
                multiplier=float(multiplier),
            ),
        )
        return ok(
            f"Fenestration '{name}' created.",
            fenestration.model_dump(exclude_none=True),
        )

    @model_tool
    def get_fenestration(name: str) -> str:
        """Read a fenestration by name."""
        fenestration = objects.get(idf, FenestrationSurfaceDetailed, name)
        return ok(
            f"Fenestration '{name}' read.", fenestration.model_dump(exclude_none=True)
        )

    @model_tool
    def delete_fenestration(name: str) -> str:
        """Delete a fenestration."""
        objects.delete(idf, objects.get(idf, FenestrationSurfaceDetailed, name), name)
        return ok(f"Fenestration '{name}' deleted.")

    return [
        create_fenestration,
        list_tool(
            idf,
            "list_fenestrations",
            FenestrationSurfaceDetailed,
            "List all fenestration surfaces.",
        ),
        get_fenestration,
        delete_fenestration,
        list_tool(
            idf,
            "list_surfaces",
            BuildingSurfaceDetailed,
            "List parent surfaces a fenestration can attach to.",
        ),
        list_tool(
            idf,
            "list_constructions",
            Construction,
            "List constructions a fenestration can reference.",
        ),
    ]
