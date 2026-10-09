from typing import Literal

from idfpy.models.simulation import Building
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
from src.modeling.fenestration import (
    Facing,
    add_fenestration,
    add_windows_by_ratio,
    facing,
    remove_fenestration,
)
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
    def create_windows_by_ratio(
        window_to_wall_ratio: float,
        construction_name: str,
        zone_names: list[str] | None = None,
        facings: list[Facing] | None = None,
        sill_height: float = 0.8,
    ) -> str:
        """Put one strip window on every matching exterior wall, sized by ratio.

        The window spans the wall's width (5 cm clearance at each edge) at
        the given sill height, as high as the ratio needs. Prefer this to
        create_fenestration whenever the specification gives window-to-wall
        ratios; call it once per ratio, facing and zone group.

        Args:
            window_to_wall_ratio: Window area over wall area, e.g. 0.4.
            construction_name: Existing window construction.
            zone_names: Zones whose outdoor walls get windows; all when omitted.
            facings: Compass directions of the walls; all when omitted.
            sill_height: Height of the window bottom above the floor, in m.
        """
        created, problems = add_windows_by_ratio(
            idf,
            zone_names,
            facings,
            window_to_wall_ratio,
            construction_name,
            sill_height,
        )
        message = f"Created {len(created)} windows"
        if problems:
            message += f"; {len(problems)} walls got none"
        return ok(message + ".", {"created": created, "not_created": problems})

    @model_tool
    def list_surfaces(
        zone_names: list[str] | None = None,
        surface_type: Literal["Wall", "Floor", "Ceiling", "Roof"] | None = None,
        outside_boundary_condition: Literal["Outdoors", "Surface", "Ground"]
        | None = None,
    ) -> str:
        """List surfaces windows and doors can attach to, filtered.

        Each entry gives name, zone, type, boundary, construction, facing,
        area and corner coordinates. Filter by zone and type: a large
        building has hundreds of surfaces.

        Args:
            zone_names: Only these zones.
            surface_type: Only this surface type.
            outside_boundary_condition: Only surfaces with this boundary.
        """
        buildings = list(idf.all_of_type(Building).values())
        north = float(buildings[0].north_axis or 0.0) if buildings else 0.0
        items = [
            {
                "name": s.name,
                "zone": s.zone_name,
                "type": s.surface_type,
                "boundary": s.outside_boundary_condition,
                "construction": s.construction_name,
                "facing": facing(s, north) if s.surface_type == "Wall" else None,
                "area": round(s.area, 2),
                "corners": [[round(c, 3) for c in p] for p in s.vertices_as_tuples],
            }
            for s in idf.all_of_type(BuildingSurfaceDetailed).values()
            if (zone_names is None or s.zone_name in zone_names)
            and (surface_type is None or s.surface_type == surface_type)
            and (
                outside_boundary_condition is None
                or s.outside_boundary_condition == outside_boundary_condition
            )
        ]
        return ok(f"Listed {len(items)} surfaces.", items)

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
        create_windows_by_ratio,
        create_fenestration,
        list_tool(
            idf,
            "list_fenestrations",
            FenestrationSurfaceDetailed,
            "List all fenestration surfaces.",
        ),
        get_fenestration,
        delete_fenestration,
        list_surfaces,
        list_constructions_tool(
            idf,
            "List constructions with their kind: window constructions for "
            "Window and GlassDoor, opaque ones for Door.",
            ("window", "opaque"),
        ),
    ]
