from typing import Literal

from idfpy.models.thermal_zones import BuildingSurfaceDetailed, Zone
from langchain_core.tools import BaseTool

from src.agent.tools._share import (
    list_constructions_tool,
    list_tool,
    model_tool,
    ok,
)
from src.modeling import geometry, objects
from src.modeling.envelope import (
    VertexSchema,
    surface_geometry,
)
from src.modeling.geometry import PlanPointSchema, ZoneConstructions
from src.modeling.surfaces import SurfaceSpecSchema, add_surface, add_surfaces
from src.state.config_state import ConfigState


def make_surface_tools(config: ConfigState) -> list[BaseTool]:
    idf = config.idf

    @model_tool
    def create_surface(
        name: str,
        surface_type: Literal["Ceiling", "Floor", "Roof", "Wall"],
        construction_name: str,
        zone_name: str,
        outside_boundary_condition: Literal[
            "Outdoors", "Ground", "Zone", "Adiabatic", "Surface"
        ],
        vertices: list[VertexSchema],
        sun_exposure: Literal["SunExposed", "NoSun"] = "NoSun",
        wind_exposure: Literal["WindExposed", "NoWind"] = "NoWind",
        outside_boundary_condition_object: str | None = None,
    ) -> str:
        """Create a BuildingSurface:Detailed (wall/floor/roof/ceiling).

        Args:
            name: Unique surface name.
            surface_type: Wall / Floor / Roof / Ceiling.
            construction_name: Existing opaque construction name.
            zone_name: Existing Zone name the surface belongs to.
            outside_boundary_condition: Outdoors / Ground / Zone / Adiabatic / Surface.
            vertices: >= 3 vertices in meters, counter-clockwise when viewed
                from OUTSIDE. Example 4-vertex south wall (2m tall, 5m wide,
                at y=0):
                  [{"X": 0.0, "Y": 0.0, "Z": 0.0},
                   {"X": 5.0, "Y": 0.0, "Z": 0.0},
                   {"X": 5.0, "Y": 0.0, "Z": 2.0},
                   {"X": 0.0, "Y": 0.0, "Z": 2.0}]
            sun_exposure: SunExposed for outdoor-facing walls and roofs.
            wind_exposure: WindExposed for outdoor-facing walls and roofs.
            outside_boundary_condition_object: Name of the partner surface
                when outside_boundary_condition is Surface, or of the
                adjacent zone when it is Zone. The partner may be created
                after this surface; when it already exists it is linked back
                automatically.
        """
        created = add_surface(
            idf,
            BuildingSurfaceDetailed.model_validate(
                {
                    "name": name,
                    "surface_type": surface_type,
                    "construction_name": construction_name,
                    "zone_name": zone_name,
                    "outside_boundary_condition": outside_boundary_condition,
                    "outside_boundary_condition_object": outside_boundary_condition_object,
                    "sun_exposure": sun_exposure,
                    "wind_exposure": wind_exposure,
                    **surface_geometry(vertices),
                }
            ),
        )
        message = f"Surface '{name}' created."
        if len(created) > 1:
            message += f" '{created[1].name}' now faces it as its interzone partner."
        return ok(message, [c.model_dump(exclude_none=True) for c in created])

    @model_tool
    def get_surface(name: str) -> str:
        """Read a surface by name."""
        surface = objects.get(idf, BuildingSurfaceDetailed, name)
        return ok(f"Surface '{name}' read.", surface.model_dump(exclude_none=True))

    @model_tool
    def delete_surface(name: str) -> str:
        """Delete a surface; refused while fenestration or a partner uses it."""
        objects.delete(idf, objects.get(idf, BuildingSurfaceDetailed, name), name)
        return ok(f"Surface '{name}' deleted.")

    @model_tool
    def create_zone_geometry(
        zone_name: str,
        plan: list[PlanPointSchema],
        floor_z: float,
        height: float,
        exterior_wall_construction: str,
        roof_construction: str,
        ground_floor_construction: str,
        interior_wall_construction: str,
        interior_floor_construction: str,
    ) -> str:
        """Create all surfaces of a zone by extruding its floor plan.

        Prefer this over create_surface for any zone with vertical walls and
        a flat top. Faces touching another zone's faces become interzone
        pairs automatically, whatever the order zones are created in, also
        for zones of different height and for storeys whose plans differ.

        Args:
            zone_name: Existing zone.
            plan: Floor plan corners (X, Y in meters), in order around the zone.
            floor_z: Floor level in meters; 0 for the ground floor.
            height: Floor-to-ceiling height in meters.
            exterior_wall_construction: Opaque construction of outdoor walls.
            roof_construction: Opaque construction of the roof.
            ground_floor_construction: Opaque construction of a floor on the
                ground or above outdoor air.
            interior_wall_construction: Opaque construction of walls shared
                with another zone.
            interior_floor_construction: Opaque construction of floors and
                ceilings shared with another zone.
        """
        result = geometry.create_zone_geometry(
            idf,
            zone_name,
            plan,
            floor_z,
            height,
            ZoneConstructions(
                exterior_wall=exterior_wall_construction,
                roof=roof_construction,
                ground_floor=ground_floor_construction,
                interior_wall=interior_wall_construction,
                interior_floor=interior_floor_construction,
            ),
        )
        return ok(
            f"Zone '{zone_name}' extruded into {len(result.created)} surfaces."
            + " ".join(["", *result.notes]),
            {"created": result.created, "replaced": result.replaced},
        )

    @model_tool
    def create_surfaces(surfaces: list[SurfaceSpecSchema]) -> str:
        """Create several surfaces at once, e.g. sloped roofs and gable walls.

        Each surface succeeds or fails on its own; resend only the failed
        entries. Interzone partners may be later entries of the same list.
        """
        outcome = add_surfaces(idf, surfaces)
        message = f"Created {len(outcome.created)} surfaces"
        if outcome.failed:
            message += f"; {len(outcome.failed)} failed and must be resent"
        return ok(message + ".", {"created": outcome.created, "failed": outcome.failed})

    return [
        create_zone_geometry,
        create_surfaces,
        create_surface,
        list_tool(
            idf, "list_surfaces", BuildingSurfaceDetailed, "List all building surfaces."
        ),
        get_surface,
        delete_surface,
        list_tool(idf, "list_zones", Zone, "List zones a surface can belong to."),
        list_constructions_tool(
            idf, "List the opaque constructions a surface can use.", ("opaque",)
        ),
    ]
