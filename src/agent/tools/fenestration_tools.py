from typing import Literal

from idfpy.models import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
)
from langchain_core.tools import BaseTool

from src.agent.tools._share import (
    list_constructions_tool,
    list_surfaces_tool,
    list_tool,
    model_tool,
    ok,
    zones_of,
)
from src.modeling import objects
from src.modeling.envelope import VertexSchema, fenestration_from_vertices
from src.modeling.fenestration import (
    Facing,
    Point,
    add_fenestration,
    add_windows_by_ratio,
    punched_windows_for_ratio,
    remove_fenestration,
    window_for_ratio,
)
from src.state.config_state import ConfigState


def make_fenestration_tools(
    config: ConfigState, plan_zones: dict[str, list[str]]
) -> list[BaseTool]:
    """The fenestration phase's tools; ``plan_zones`` maps each zone plan's
    key to the zones made from it, so windows can be given per plan."""
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
        plans: list[str] | None = None,
        zone_names: list[str] | None = None,
        facings: list[Facing] | None = None,
        sill_height: float = 0.8,
        pattern: Literal["ribbon", "punched"] = "ribbon",
        window_width: float = 1.2,
        window_height: float = 1.5,
    ) -> str:
        """Put windows on every matching exterior wall, sized by ratio.

        A ribbon is one strip window spanning the wall's width (5 cm
        clearance at each edge), as high as the ratio needs. Punched windows
        are evenly spaced window_width x window_height windows, as many as
        the ratio holds, scaled alike to cover it exactly. Both start at
        the given sill height. Prefer this to create_fenestration whenever
        the specification gives window-to-wall ratios; call it once per
        ratio, pattern, facing and group of plans or zones.

        Args:
            window_to_wall_ratio: Window area over wall area, e.g. 0.4.
            construction_name: Existing window construction.
            plans: Zone plan keys: every zone made from these plans, on every
                storey, gets windows.
            zone_names: Zones whose outdoor walls get windows. Give plans,
                zone_names or both; for every zone, list every plan.
            facings: Compass directions of the walls; all when omitted.
            sill_height: Height of the window bottom above the floor, in m.
            pattern: 'ribbon' for continuous bands or curtain walls,
                'punched' for separate windows.
            window_width: Width of one punched window, in m.
            window_height: Height of one punched window, in m.
        """
        # Explicit zones: a call meant for one plan that names none would
        # put windows on every wall, which takes one deletion each to undo.
        zones = zones_of(plans, zone_names, plan_zones)
        if pattern == "ribbon":

            def layout(wall: BuildingSurfaceDetailed) -> list[list[Point]]:
                return [window_for_ratio(wall, window_to_wall_ratio, sill_height)]

        else:

            def layout(wall: BuildingSurfaceDetailed) -> list[list[Point]]:
                return punched_windows_for_ratio(
                    wall, window_to_wall_ratio, sill_height, window_width, window_height
                )

        created, problems = add_windows_by_ratio(
            idf, zones, facings, layout, construction_name
        )
        message = f"Created {len(created)} windows"
        if problems:
            message += f"; {len(problems)} walls got none"
        return ok(message + ".", {"created": created, "not_created": problems})

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
            "List fenestration surfaces: name, type, parent surface, "
            "construction, partner and multiplier.",
            (
                "name",
                "surface_type",
                "building_surface_name",
                "construction_name",
                "outside_boundary_condition_object",
                "multiplier",
            ),
        ),
        get_fenestration,
        delete_fenestration,
        list_surfaces_tool(
            idf,
            "List surfaces windows and doors can attach to, filtered by zone, "
            "type and boundary: a large building has hundreds of surfaces.",
        ),
        list_constructions_tool(
            idf,
            "List constructions with their kind: window constructions for "
            "Window and GlassDoor, opaque ones for Door.",
            ("window", "opaque"),
        ),
    ]
