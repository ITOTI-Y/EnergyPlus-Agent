import json

from langchain_core.tools import BaseTool, tool

from idfpy.models.thermal_zones import FenestrationSurfaceDetailed
from src.mcp.state import ConfigState


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def make_fenestration_tools(config: ConfigState) -> list[BaseTool]:
    idf = config._idf

    @tool
    def create_fenestration(
        name: str,
        surface_type: str,
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
        if idf.has("FenestrationSurface:Detailed", name):
            return _err(f"Fenestration '{name}' already exists.")
        try:
            kwargs: dict = {
                "name": name,
                "surface_type": surface_type,
                "construction_name": construction_name,
                "building_surface_name": building_surface_name,
                "multiplier": float(multiplier),
                "number_of_vertices": len(vertices),
            }
            for i, v in enumerate(vertices, start=1):
                kwargs[f"vertex_{i}_x_coordinate"] = float(v["X"])
                kwargs[f"vertex_{i}_y_coordinate"] = float(v["Y"])
                kwargs[f"vertex_{i}_z_coordinate"] = float(v["Z"])
            idf.add(FenestrationSurfaceDetailed(**kwargs))
            return _ok(
                f"Fenestration '{name}' created successfully.",
                idf.get("FenestrationSurface:Detailed", name).model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating fenestration '{name}': {e}")

    @tool
    def list_fenestrations() -> str:
        """List all fenestration surfaces."""
        items = [f.model_dump() for f in idf.all_of_type("FenestrationSurface:Detailed").values()]
        return _ok(f"Listed {len(items)} fenestrations.", items)

    @tool
    def get_fenestration(name: str) -> str:
        """Read a fenestration by name."""
        obj = idf.get("FenestrationSurface:Detailed", name)
        if obj is None:
            return _err(f"Fenestration '{name}' not found.")
        return _ok(f"Fenestration '{name}' read successfully.", obj.model_dump())

    @tool
    def delete_fenestration(name: str) -> str:
        """Delete a fenestration."""
        if not idf.has("FenestrationSurface:Detailed", name):
            return _err(f"Fenestration '{name}' not found.")
        idf.remove("FenestrationSurface:Detailed", name)
        return _ok(f"Fenestration '{name}' deleted successfully.")

    @tool
    def list_surfaces() -> str:
        """Read-only: list parent surfaces a fenestration can attach to."""
        items = [s.model_dump() for s in idf.all_of_type("BuildingSurface:Detailed").values()]
        return _ok(f"Listed {len(items)} surfaces.", items)

    @tool
    def list_constructions() -> str:
        """Read-only: list constructions a fenestration can reference."""
        items = [c.model_dump() for c in idf.all_of_type("Construction").values()]
        return _ok(f"Listed {len(items)} constructions.", items)

    return [
        create_fenestration,
        list_fenestrations,
        get_fenestration,
        delete_fenestration,
        list_surfaces,
        list_constructions,
    ]
