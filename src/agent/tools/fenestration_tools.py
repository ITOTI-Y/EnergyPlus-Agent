import json
from typing import Literal

from idfpy.models.constructions import Construction
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
)
from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def make_fenestration_tools(config: ConfigState) -> list[BaseTool]:

    @tool
    def create_fenestration(
        name: str,
        surface_type: Literal["Window", "Door", "GlassDoor"],
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
        idf = config.idf
        if idf.has("FenestrationSurface:Detailed", name):
            return _err(f"Fenestration '{name}' already exists.")
        if not idf.has("Construction", construction_name):
            return _err(
                f"Construction '{construction_name}' not found. Create it in the construction phase first.",
                {"missing_ref": "Construction", "missing_name": construction_name},
            )
        if not idf.has("BuildingSurface:Detailed", building_surface_name):
            return _err(
                f"Parent surface '{building_surface_name}' not found.",
                {"missing_ref": "BuildingSurface:Detailed", "missing_name": building_surface_name},
            )
        try:
            assert 3 <= len(vertices) <= 4, (
                "At least 3 vertices are required, and at most 4 vertices are allowed."
            )
            fenestration = FenestrationSurfaceDetailed(
                name=name,
                surface_type=surface_type,
                construction_name=construction_name,
                building_surface_name=building_surface_name,
                multiplier=float(multiplier),
                number_of_vertices=len(vertices),
                vertex_1_x_coordinate=float(vertices[0]["X"]),
                vertex_1_y_coordinate=float(vertices[0]["Y"]),
                vertex_1_z_coordinate=float(vertices[0]["Z"]),
                vertex_2_x_coordinate=float(vertices[1]["X"]),
                vertex_2_y_coordinate=float(vertices[1]["Y"]),
                vertex_2_z_coordinate=float(vertices[1]["Z"]),
                vertex_3_x_coordinate=float(vertices[2]["X"]),
                vertex_3_y_coordinate=float(vertices[2]["Y"]),
                vertex_3_z_coordinate=float(vertices[2]["Z"]),
                vertex_4_x_coordinate=float(vertices[3]["X"])
                if len(vertices) > 3
                else None,
                vertex_4_y_coordinate=float(vertices[3]["Y"])
                if len(vertices) > 3
                else None,
                vertex_4_z_coordinate=float(vertices[3]["Z"])
                if len(vertices) > 3
                else None,
            )
            idf.add(fenestration)
            return _ok(
                f"Fenestration '{name}' created successfully.",
                fenestration.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating fenestration '{name}': {e}")

    @tool
    def list_fenestrations() -> str:
        """List all fenestration surfaces."""
        idf = config.idf
        items = [
            f.model_dump()
            for f in idf.all_of_type(FenestrationSurfaceDetailed).values()
        ]
        return _ok(f"Listed {len(items)} fenestrations.", items)

    @tool
    def get_fenestration(name: str) -> str:
        """Read a fenestration by name."""
        idf = config.idf
        obj = idf.get(FenestrationSurfaceDetailed, name)
        if obj is None:
            return _err(f"Fenestration '{name}' not found.")
        return _ok(f"Fenestration '{name}' read successfully.", obj.model_dump())

    @tool
    def update_fenestration(
        name: str,
        construction_name: str | None = None,
        building_surface_name: str | None = None,
        surface_type: str | None = None,
        multiplier: int | None = None,
        vertices: list[dict[str, float]] | None = None,
    ) -> str:
        """Update fields of an existing fenestration by name.

        Only non-None fields are written. To change geometry, pass a full
        new ``vertices`` list (replaces all existing vertices).

        Args:
            name: Existing fenestration name.
            construction_name: New glazing Construction name (must exist).
            building_surface_name: New parent Surface name (must exist).
            surface_type: Window / Door / GlassDoor.
            multiplier: Number of identical copies (>= 1).
            vertices: New full vertex list ({"X","Y","Z"} dicts), >= 3 points,
                      coplanar with the parent surface.
        """
        obj = idf.get("FenestrationSurface:Detailed", name)
        if obj is None:
            return _err(f"Fenestration '{name}' not found.")
        if construction_name is not None and not idf.has("Construction", construction_name):
            return _err(
                f"Construction '{construction_name}' not found.",
                {"missing_ref": "Construction", "missing_name": construction_name},
            )
        if building_surface_name is not None and not idf.has("BuildingSurface:Detailed", building_surface_name):
            return _err(
                f"Parent surface '{building_surface_name}' not found.",
                {"missing_ref": "BuildingSurface:Detailed", "missing_name": building_surface_name},
            )
        try:
            if construction_name is not None:
                obj.construction_name = construction_name
            if building_surface_name is not None:
                obj.building_surface_name = building_surface_name
            if surface_type is not None:
                obj.surface_type = surface_type
            if multiplier is not None:
                obj.multiplier = float(multiplier)
            if vertices is not None:
                if len(vertices) < 3:
                    return _err("Fenestration needs >= 3 vertices.")
                # Clear any existing vertex fields (up to 4), then set new ones
                for i in range(1, 5):
                    for axis in ("x", "y", "z"):
                        setattr(obj, f"vertex_{i}_{axis}_coordinate", None)
                for i, v in enumerate(vertices, start=1):
                    setattr(obj, f"vertex_{i}_x_coordinate", float(v["X"]))
                    setattr(obj, f"vertex_{i}_y_coordinate", float(v["Y"]))
                    setattr(obj, f"vertex_{i}_z_coordinate", float(v["Z"]))
                obj.number_of_vertices = len(vertices)
            return _ok(f"Fenestration '{name}' updated successfully.",
                       obj.model_dump())
        except Exception as e:
            return _err(f"Error updating fenestration '{name}': {e}")

    @tool
    def delete_fenestration(name: str) -> str:
        """Delete a fenestration."""
        idf = config.idf
        if not idf.has("FenestrationSurface:Detailed", name):
            return _err(f"Fenestration '{name}' not found.")
        idf.remove("FenestrationSurface:Detailed", name)
        return _ok(f"Fenestration '{name}' deleted successfully.")

    @tool
    def list_surfaces() -> str:
        """Read-only: list parent surfaces a fenestration can attach to."""
        idf = config.idf
        items = [
            s.model_dump() for s in idf.all_of_type(BuildingSurfaceDetailed).values()
        ]
        return _ok(f"Listed {len(items)} surfaces.", items)

    @tool
    def list_constructions() -> str:
        """Read-only: list constructions a fenestration can reference."""
        idf = config.idf
        items = [c.model_dump() for c in idf.all_of_type(Construction).values()]
        return _ok(f"Listed {len(items)} constructions.", items)

    return [
        create_fenestration,
        list_fenestrations,
        get_fenestration,
        update_fenestration,
        delete_fenestration,
        list_surfaces,
        list_constructions,
    ]
