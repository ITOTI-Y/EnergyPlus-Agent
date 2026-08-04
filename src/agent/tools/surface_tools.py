import json
from typing import Literal

from idfpy.models.constructions import Construction
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    BuildingSurfaceDetailedVerticesItem,
    FenestrationSurfaceDetailed,
    Zone,
)
from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def make_surface_tools(config: ConfigState) -> list[BaseTool]:

    @tool
    def create_surface(
        name: str,
        surface_type: Literal["Ceiling", "Floor", "Roof", "Wall"],
        construction_name: str,
        zone_name: str,
        outside_boundary_condition: Literal[
            "Outdoors", "Ground", "Zone", "Adiabatic", "Surface"
        ],
        vertices: list[dict[str, float]],
        sun_exposure: Literal["SunExposed", "NoSun"] = "NoSun",
        wind_exposure: Literal["WindExposed", "NoWind"] = "NoWind",
        outside_boundary_condition_object: Literal["Surface", "Zone"] | None = None,
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
        idf = config.idf
        if idf.has(BuildingSurfaceDetailed, name):
            return _err(f"Surface '{name}' already exists.")
        try:
            assert len(vertices) >= 3, "At least 3 vertices are required."
            vertex_items = [
                BuildingSurfaceDetailedVerticesItem(
                    vertex_x_coordinate=float(v["X"]),
                    vertex_y_coordinate=float(v["Y"]),
                    vertex_z_coordinate=float(v["Z"]),
                )
                for v in vertices
            ]
            surface = BuildingSurfaceDetailed(
                name=name,
                surface_type=surface_type,
                construction_name=construction_name,
                zone_name=zone_name,
                outside_boundary_condition=outside_boundary_condition,
                outside_boundary_condition_object=outside_boundary_condition_object,
                sun_exposure=sun_exposure,
                wind_exposure=wind_exposure,
                vertices=vertex_items,
            )
            idf.add(surface)
            return _ok(
                f"Surface '{name}' created successfully.",
                surface.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating surface '{name}': {e}")

    @tool
    def list_surfaces() -> str:
        """List all building surfaces."""
        idf = config.idf
        items = [
            s.model_dump() for s in idf.all_of_type(BuildingSurfaceDetailed).values()
        ]
        return _ok(f"Listed {len(items)} surfaces.", items)

    @tool
    def get_surface(name: str) -> str:
        """Read a surface by name."""
        idf = config.idf
        obj = idf.get(BuildingSurfaceDetailed, name)
        if obj is None:
            return _err(f"Surface '{name}' not found.")
        return _ok(f"Surface '{name}' read successfully.", obj.model_dump())

    @tool
    def delete_surface(name: str) -> str:
        """Delete a surface. Fails if fenestration references it."""
        idf = config.idf
        if not idf.has("BuildingSurface:Detailed", name):
            return _err(f"Surface '{name}' not found.")
        refs = []
        for f in idf.all_of_type(FenestrationSurfaceDetailed).values():
            if f.building_surface_name == name:
                refs.append(f"Fenestration:{f.name}")
        if refs:
            return _err(
                f"Surface '{name}' is referenced by fenestration.",
                {"references": refs},
            )
        idf.remove("BuildingSurface:Detailed", name)
        return _ok(f"Surface '{name}' deleted successfully.")

    @tool
    def list_zones() -> str:
        """Read-only: list zones a surface can be assigned to."""
        idf = config.idf
        items = [z.model_dump() for z in idf.all_of_type(Zone).values()]
        return _ok(f"Listed {len(items)} zones.", items)

    @tool
    def list_constructions() -> str:
        """Read-only: list constructions a surface can reference."""
        idf = config.idf
        items = [c.model_dump() for c in idf.all_of_type(Construction).values()]
        return _ok(f"Listed {len(items)} constructions.", items)

    return [
        create_surface,
        list_surfaces,
        get_surface,
        delete_surface,
        list_zones,
        list_constructions,
    ]
