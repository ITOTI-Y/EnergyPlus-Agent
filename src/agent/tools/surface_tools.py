import json

from langchain_core.tools import BaseTool, tool

from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    BuildingSurfaceDetailedVerticesItem,
)
from src.mcp.state import ConfigState


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def make_surface_tools(config: ConfigState) -> list[BaseTool]:
    idf = config._idf

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
        if idf.has("BuildingSurface:Detailed", name):
            return _err(f"Surface '{name}' already exists.")
        try:
            vertex_items = [
                BuildingSurfaceDetailedVerticesItem(
                    vertex_x_coordinate=float(v["X"]),
                    vertex_y_coordinate=float(v["Y"]),
                    vertex_z_coordinate=float(v["Z"]),
                )
                for v in vertices
            ]
            idf.add(BuildingSurfaceDetailed(
                name=name,
                surface_type=surface_type,
                construction_name=construction_name,
                zone_name=zone_name,
                outside_boundary_condition=outside_boundary_condition,
                outside_boundary_condition_object=outside_boundary_condition_object,
                sun_exposure=sun_exposure,
                wind_exposure=wind_exposure,
                vertices=vertex_items,
            ))
            return _ok(
                f"Surface '{name}' created successfully.",
                idf.get("BuildingSurface:Detailed", name).model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating surface '{name}': {e}")

    @tool
    def list_surfaces() -> str:
        """List all building surfaces."""
        items = [s.model_dump() for s in idf.all_of_type("BuildingSurface:Detailed").values()]
        return _ok(f"Listed {len(items)} surfaces.", items)

    @tool
    def get_surface(name: str) -> str:
        """Read a surface by name."""
        obj = idf.get("BuildingSurface:Detailed", name)
        if obj is None:
            return _err(f"Surface '{name}' not found.")
        return _ok(f"Surface '{name}' read successfully.", obj.model_dump())

    @tool
    def delete_surface(name: str) -> str:
        """Delete a surface. Fails if fenestration references it."""
        if not idf.has("BuildingSurface:Detailed", name):
            return _err(f"Surface '{name}' not found.")
        refs = []
        for f in idf.all_of_type("FenestrationSurface:Detailed").values():
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
        items = [z.model_dump() for z in idf.all_of_type("Zone").values()]
        return _ok(f"Listed {len(items)} zones.", items)

    @tool
    def list_constructions() -> str:
        """Read-only: list constructions a surface can reference."""
        items = [c.model_dump() for c in idf.all_of_type("Construction").values()]
        return _ok(f"Listed {len(items)} constructions.", items)

    return [
        create_surface,
        list_surfaces,
        get_surface,
        delete_surface,
        list_zones,
        list_constructions,
    ]
