import json
from typing import Literal

from idfpy.models import (
    BuildingSurfaceDetailed,
    BuildingSurfaceDetailedVerticesItem,
    Construction,
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
    idf = config._idf

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
        sun_exposure: Literal["NoSun", "SunExposed"] = "NoSun",
        wind_exposure: Literal["NoWind", "WindExposed"] = "NoWind",
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
        if idf is None:
            raise ValueError("IDF is None")
        if idf.has(BuildingSurfaceDetailed, name):
            return _err(f"Surface '{name}' already exists.")
        if not idf.has("Construction", construction_name):
            return _err(
                f"Construction '{construction_name}' not found. Create it in the construction phase first.",
                {"missing_ref": "Construction", "missing_name": construction_name},
            )
        if not idf.has("Zone", zone_name):
            return _err(
                f"Zone '{zone_name}' not found.",
                {"missing_ref": "Zone", "missing_name": zone_name},
            )
        try:
            vertex_items = [
                BuildingSurfaceDetailedVerticesItem(
                    vertex_x_coordinate=float(v["X"]),
                    vertex_y_coordinate=float(v["Y"]),
                    vertex_z_coordinate=float(v["Z"]),
                )
                for v in vertices
            ]
            idf.add(
                BuildingSurfaceDetailed(
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
            )
            data = idf.get(BuildingSurfaceDetailed, name)
            if data is None:
                raise ValueError("BuildingSurface:Detailed not found")
            return _ok(
                f"Surface '{name}' created successfully.",
                data.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating surface '{name}': {e}")

    @tool
    def list_surfaces() -> str:
        """List all building surfaces."""
        if idf is None:
            raise ValueError("IDF is None")
        items = [
            s.model_dump() for s in idf.all_of_type("BuildingSurface:Detailed").values()
        ]
        return _ok(f"Listed {len(items)} surfaces.", items)

    @tool
    def get_surface(name: str) -> str:
        """Read a surface by name."""
        if idf is None:
            raise ValueError("IDF is None")
        obj = idf.get(BuildingSurfaceDetailed, name)
        if obj is None:
            return _err(f"Surface '{name}' not found.")
        return _ok(f"Surface '{name}' read successfully.", obj.model_dump())

    @tool
    def update_surface(
        name: str,
        construction_name: str | None = None,
        zone_name: str | None = None,
        outside_boundary_condition: Literal[
            "Outdoors", "Ground", "Zone", "Adiabatic", "Surface"
        ]
        | None = None,
        outside_boundary_condition_object: str | None = None,
        sun_exposure: Literal["NoSun", "SunExposed"] | None = None,
        wind_exposure: Literal["NoWind", "WindExposed"] | None = None,
        vertices: list[dict[str, float]] | None = None,
    ) -> str:
        """Update fields of an existing surface by name.

        Only non-None fields are written. To change geometry, pass a full
        new ``vertices`` list (replaces all existing vertices).

        Args:
            name: Existing surface name.
            construction_name: New Construction name (must exist).
            zone_name: New Zone name (must exist).
            outside_boundary_condition / outside_boundary_condition_object:
                Boundary settings.
            sun_exposure / wind_exposure: SunExposed / NoSun, WindExposed / NoWind.
            vertices: New full vertex list ({"X","Y","Z"} dicts), >= 3 points.
        """
        if idf is None:
            raise ValueError("IDF is None")
        obj = idf.get(BuildingSurfaceDetailed, name)
        if obj is None:
            return _err(f"Surface '{name}' not found.")
        if construction_name is not None and not idf.has(
            Construction, construction_name
        ):
            return _err(
                f"Construction '{construction_name}' not found.",
                {"missing_ref": "Construction", "missing_name": construction_name},
            )
        if zone_name is not None and not idf.has("Zone", zone_name):
            return _err(
                f"Zone '{zone_name}' not found.",
                {"missing_ref": "Zone", "missing_name": zone_name},
            )
        try:
            if construction_name is not None:
                obj.construction_name = construction_name
            if zone_name is not None:
                obj.zone_name = zone_name
            if outside_boundary_condition is not None:
                obj.outside_boundary_condition = outside_boundary_condition
            if outside_boundary_condition_object is not None:
                obj.outside_boundary_condition_object = (
                    outside_boundary_condition_object
                )
            if sun_exposure is not None:
                obj.sun_exposure = sun_exposure
            if wind_exposure is not None:
                obj.wind_exposure = wind_exposure
            if vertices is not None:
                if len(vertices) < 3:
                    return _err("Surface needs >= 3 vertices.")
                obj.vertices = [
                    BuildingSurfaceDetailedVerticesItem(
                        vertex_x_coordinate=float(v["X"]),
                        vertex_y_coordinate=float(v["Y"]),
                        vertex_z_coordinate=float(v["Z"]),
                    )
                    for v in vertices
                ]
            return _ok(f"Surface '{name}' updated successfully.", obj.model_dump())
        except Exception as e:
            return _err(f"Error updating surface '{name}': {e}")

    @tool
    def delete_surface(name: str) -> str:
        """Delete a surface. Fails if fenestration references it."""
        if idf is None:
            raise ValueError("IDF is None")
        if not idf.has(BuildingSurfaceDetailed, name):
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
        if idf is None:
            raise ValueError("IDF is None")
        items = [z.model_dump() for z in idf.all_of_type(Zone).values()]
        return _ok(f"Listed {len(items)} zones.", items)

    @tool
    def list_constructions() -> str:
        """Read-only: list constructions a surface can reference."""
        if idf is None:
            raise ValueError("IDF is None")
        items = [c.model_dump() for c in idf.all_of_type(Construction).values()]
        return _ok(f"Listed {len(items)} constructions.", items)

    return [
        create_surface,
        list_surfaces,
        get_surface,
        update_surface,
        delete_surface,
        list_zones,
        list_constructions,
    ]
