import json

from idfpy.models import (
    BuildingSurfaceDetailed,
    Construction,
    ConstructionAirBoundary,
    FenestrationSurfaceDetailed,
)
from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState

_LAYER_FIELDS = [
    "outside_layer",
    "layer_2",
    "layer_3",
    "layer_4",
    "layer_5",
    "layer_6",
    "layer_7",
    "layer_8",
    "layer_9",
    "layer_10",
]

_ALL_MATERIAL_TYPES = [
    "Material",
    "Material:NoMass",
    "Material:AirGap",
    "WindowMaterial:SimpleGlazingSystem",
    "WindowMaterial:Glazing",
]

# idfpy object-type strings for each construction variant. The plain
# `Construction` is a layered assembly; `Construction:AirBoundary` is a
# layer-less open-air boundary used on interior subsurfaces (doors/windows
# between zones) to connect them without a paired subsurface in the other
# zone. Both are valid `construction_name` targets for surfaces/fenestrations.
_ALL_CONSTRUCTION_TYPES = ["Construction", "Construction:AirBoundary"]


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def _material_exists(idf, name: str) -> bool:
    return any(idf.has(t, name) for t in _ALL_MATERIAL_TYPES)


def _find_construction(idf, name: str):
    """Return (object_type, obj) for a construction with the given name, or
    (None, None). Looks across both `Construction` and `Construction:AirBoundary`
    so the read/list/delete tools treat them uniformly."""
    for t in _ALL_CONSTRUCTION_TYPES:
        obj = idf.get(t, name)
        if obj is not None:
            return t, obj
    return None, None


def _is_airboundary_construction(idf, name: str) -> bool:
    """True if *name* is a Construction:AirBoundary (open-air boundary)."""
    return idf.has("Construction:AirBoundary", name)


def make_construction_tools(config: ConfigState, rag=None) -> list[BaseTool]:
    idf = config._idf

    @tool
    def create_construction(name: str, layers: list[str]) -> str:
        """Create a Construction as an ordered list of material layers.

        Args:
            name: Unique construction name (e.g., 'ExtWall_Brick').
            layers: Material names from outside to inside. All names must
                    already exist in the materials list. >= 1 layer.
        """
        if idf is None:
            raise ValueError("IDF is None")
        if idf.has("Construction", name):
            return _err(f"Construction '{name}' already exists.")
        if idf.has("Construction:AirBoundary", name):
            return _err(f"Construction '{name}' already exists (as an AirBoundary).")
        if not layers:
            return _err("Construction must have at least one layer.")
        if len(layers) > 10:
            return _err("Construction cannot have more than 10 layers.")
        missing = [lyr for lyr in layers if not _material_exists(idf, lyr)]
        if missing:
            return _err(
                f"Materials not found: {missing}. Create them first.",
                {
                    "missing": missing,
                    "missing_ref": "Material",
                    "missing_name": missing[0],
                },
            )
        try:
            kwargs: dict = {"name": name}
            for i, layer_name in enumerate(layers):
                kwargs[_LAYER_FIELDS[i]] = layer_name
            idf.add(Construction(**kwargs))
            data = idf.get(Construction, name)
            if data is None:
                raise ValueError("Construction not found")
            return _ok(
                f"Construction '{name}' created successfully.",
                data.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating construction '{name}': {e}")

    @tool
    def create_airboundary_construction(
        name: str,
        air_exchange_method: str = "None",
        simple_mixing_air_changes_per_hour: float | None = None,
        simple_mixing_schedule_name: str | None = None,
    ) -> str:
        """Create a Construction:AirBoundary — a layer-less open-air boundary.

        Used on INTERIOR subsurfaces (doors/windows/glass-doors hosted on a
        zone-separating wall whose Outside Boundary Condition is 'Surface') to
        model an open passage between zones. It has no material layers and
        does NOT require a paired subsurface in the adjacent zone, unlike a
        regular interior construction. Assign it as the construction_name of
        interior doors/windows.

        Args:
            name: Unique construction name (e.g., 'Interior_Door_Open').
            air_exchange_method: 'None' (radiant/daylighting only, default) or
                                 'SimpleMixing' (adds an air-mixing rate).
            simple_mixing_air_changes_per_hour: Required when method is
                'SimpleMixing'; air changes per hour based on the smaller
                zone's volume. Ignored otherwise.
            simple_mixing_schedule_name: Optional Schedule:Compact name for the
                SimpleMixing rate. Only valid with method 'SimpleMixing'.
        """
        if idf is None:
            raise ValueError("IDF is None")
        if idf.has("Construction:AirBoundary", name):
            return _err(f"Construction:AirBoundary '{name}' already exists.")
        if idf.has("Construction", name):
            return _err(
                f"Construction '{name}' already exists (as a layered Construction)."
            )
        try:
            kwargs: dict = {"name": name, "air_exchange_method": air_exchange_method}
            if air_exchange_method == "SimpleMixing":
                if simple_mixing_air_changes_per_hour is None:
                    return _err(
                        "simple_mixing_air_changes_per_hour is required when "
                        "air_exchange_method is 'SimpleMixing'."
                    )
                kwargs["simple_mixing_air_changes_per_hour"] = (
                    simple_mixing_air_changes_per_hour
                )
                if simple_mixing_schedule_name is not None:
                    kwargs["simple_mixing_schedule_name"] = simple_mixing_schedule_name
            idf.add(ConstructionAirBoundary(**kwargs))
            data = idf.get(ConstructionAirBoundary, name)
            if data is None:
                raise ValueError("Construction:AirBoundary not found")
            return _ok(
                f"Construction:AirBoundary '{name}' created successfully.",
                data.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating Construction:AirBoundary '{name}': {e}")

    @tool
    def list_constructions() -> str:
        """List all constructions (layered Construction + Construction:AirBoundary)."""
        items = []
        if idf is None:
            raise ValueError("IDF is None")
        for t in _ALL_CONSTRUCTION_TYPES:
            for obj in idf.all_of_type(t).values():
                items.append({"type": t, **obj.model_dump()})
        return _ok(f"Listed {len(items)} constructions.", items)

    @tool
    def get_construction(name: str) -> str:
        """Read a construction by name (layered or AirBoundary)."""
        const_type, obj = _find_construction(idf, name)
        if obj is None:
            return _err(f"Construction '{name}' not found.")
        return _ok(
            f"Construction '{name}' read successfully.",
            {"type": const_type, **obj.model_dump()},
        )

    @tool
    def update_construction(name: str, layers: list[str]) -> str:
        """Replace the entire layer sequence of an existing layered Construction.

        Only applies to layered `Construction` objects — NOT to
        `Construction:AirBoundary` (which has no layers). To change an
        AirBoundary, delete and recreate it.

        Args:
            name: Existing construction name.
            layers: New ordered list of material names (outside → inside).
                    All names must already exist. 1-10 layers.
        """
        if idf is None:
            raise ValueError("IDF is None")
        obj = idf.get(Construction, name)
        if obj is None:
            # Distinguish "not found" from "is an AirBoundary (wrong tool)".
            if idf.has(ConstructionAirBoundary, name):
                return _err(
                    f"'{name}' is a Construction:AirBoundary and has no layers; "
                    f"use delete_construction + create_airboundary_construction."
                )
            return _err(f"Construction '{name}' not found.")
        if not layers or len(layers) > 10:
            return _err("Construction must have 1-10 layers.")
        missing = [lyr for lyr in layers if not _material_exists(idf, lyr)]
        if missing:
            return _err(
                f"Materials not found: {missing}.",
                {
                    "missing": missing,
                    "missing_ref": "Material",
                    "missing_name": missing[0],
                },
            )
        try:
            # Clear existing layers then set new ones
            for lf in _LAYER_FIELDS:
                setattr(obj, lf, None)
            for i, layer_name in enumerate(layers):
                setattr(obj, _LAYER_FIELDS[i], layer_name)
            return _ok(f"Construction '{name}' updated successfully.", obj.model_dump())
        except Exception as e:
            return _err(f"Error updating construction '{name}': {e}")

    @tool
    def delete_construction(name: str) -> str:
        """Delete a construction (layered or AirBoundary).

        Fails if referenced by surfaces/fenestration.
        """
        const_type, _obj = _find_construction(idf, name)
        if const_type is None:
            return _err(f"Construction '{name}' not found.")
        refs = []
        if idf is None:
            raise ValueError("IDF is None")
        for s in idf.all_of_type(BuildingSurfaceDetailed).values():
            if s.construction_name == name:
                refs.append(f"Surface:{s.name}")
        for f in idf.all_of_type(FenestrationSurfaceDetailed).values():
            if f.construction_name == name:
                refs.append(f"Fenestration:{f.name}")
        if refs:
            return _err(
                f"Construction '{name}' is referenced by other components.",
                {"references": refs},
            )
        idf.remove(const_type, name)
        return _ok(f"Construction '{name}' deleted successfully.")

    @tool
    def list_materials() -> str:
        """Read-only: list all materials available for use as construction layers."""
        items = []
        if idf is None:
            raise ValueError("IDF is None")
        for t in _ALL_MATERIAL_TYPES:
            for obj in idf.all_of_type(t).values():
                items.append({"type": t, **obj.model_dump()})
        return _ok(f"Listed {len(items)} materials.", items)

    tools = [
        create_construction,
        create_airboundary_construction,
        list_constructions,
        get_construction,
        update_construction,
        delete_construction,
        list_materials,
    ]
    if rag is not None:
        from src.agent.tools.rag_tools import (
            TABLE_ALL_MATERIALS,
            TABLE_CONSTRUCTIONS,
            make_rag_tool,
        )

        tools.append(make_rag_tool([TABLE_CONSTRUCTIONS, TABLE_ALL_MATERIALS], rag=rag))
    return tools
