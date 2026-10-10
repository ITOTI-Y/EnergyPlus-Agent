from typing import Literal

from idfpy.models.constructions import Construction
from idfpy.models.thermal_zones import BuildingSurfaceDetailed, Zone
from langchain_core.tools import BaseTool

from src.agent.tools._share import list_tool, model_tool, ok
from src.modeling import objects
from src.modeling.envelope import VertexSchema, surface_geometry
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
            construction_name: Existing Construction name.
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
                adjacent zone when it is Zone.
        """
        surface = objects.create(
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
        return ok(f"Surface '{name}' created.", surface.model_dump(exclude_none=True))

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

    return [
        create_surface,
        list_tool(
            idf, "list_surfaces", BuildingSurfaceDetailed, "List all building surfaces."
        ),
        get_surface,
        delete_surface,
        list_tool(idf, "list_zones", Zone, "List zones a surface can belong to."),
        list_tool(
            idf,
            "list_constructions",
            Construction,
            "List constructions a surface can reference.",
        ),
    ]
