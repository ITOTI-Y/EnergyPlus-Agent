from typing import Literal

from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
)
from langchain_core.tools import BaseTool

from src.agent.tools._share import (
    list_constructions_tool,
    list_tool,
    model_tool,
    ok,
)
from src.modeling import objects
from src.modeling.envelope import VertexSchema, fenestration_from_vertices
from src.modeling.fenestration import add_fenestration, remove_fenestration
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
            construction_name: Existing construction: a window construction for
                Window and GlassDoor, an opaque one for Door.
            building_surface_name: Existing parent surface name. On a wall
                shared with another zone, the matching opening in that zone is
                created automatically.
            vertices: 3 or 4 vertices in meters on the parent surface plane
                and inside its outline; the order is corrected to match the
                parent surface. Example 1.5x1.2m
                window centered on a south wall at sill 0.8m (wall at y=0,
                spans x=0..5):
                  [{"X": 1.75, "Y": 0.0, "Z": 0.8},
                   {"X": 3.25, "Y": 0.0, "Z": 0.8},
                   {"X": 3.25, "Y": 0.0, "Z": 2.0},
                   {"X": 1.75, "Y": 0.0, "Z": 2.0}]
            multiplier: Number of identical copies (>= 1).
        """
        created, flipped = add_fenestration(
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
        notes = []
        if flipped:
            notes.append("vertex order reversed to face the same way as the surface")
        if len(created) > 1:
            notes.append(
                f"interzone opening, so '{created[1].name}' was added to "
                f"'{created[1].building_surface_name}'"
            )
        message = f"Fenestration '{name}' created" + (
            f" ({'; '.join(notes)})." if notes else "."
        )
        return ok(message, [f.model_dump(exclude_none=True) for f in created])

    @model_tool
    def get_fenestration(name: str) -> str:
        """Read a fenestration by name."""
        fenestration = objects.get(idf, FenestrationSurfaceDetailed, name)
        return ok(
            f"Fenestration '{name}' read.", fenestration.model_dump(exclude_none=True)
        )

    @model_tool
    def delete_fenestration(name: str) -> str:
        """Delete a fenestration, and its partner when it is an interzone opening."""
        deleted = remove_fenestration(idf, name)
        return ok(f"Deleted {', '.join(deleted)}.")

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
        list_constructions_tool(
            idf,
            "List constructions with their kind: window constructions for "
            "Window and GlassDoor, opaque ones for Door.",
            ("window", "opaque"),
        ),
    ]
