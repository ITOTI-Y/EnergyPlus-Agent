import json

from langchain_core.tools import BaseTool, tool

from idfpy.models.constructions import (
    Construction,
    Material,
    MaterialAirGap,
    MaterialNoMass,
    WindowMaterialSimpleGlazingSystem,
)
from src.mcp.state import ConfigState

_LAYER_FIELDS = [
    "outside_layer", "layer_2", "layer_3", "layer_4", "layer_5",
    "layer_6", "layer_7", "layer_8", "layer_9", "layer_10",
]

_ALL_MATERIAL_TYPES = [
    "Material",
    "Material:NoMass",
    "Material:AirGap",
    "WindowMaterial:SimpleGlazingSystem",
]


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def _material_exists(idf, name: str) -> bool:
    return any(idf.has(t, name) for t in _ALL_MATERIAL_TYPES)


def make_construction_tools(config: ConfigState) -> list[BaseTool]:
    idf = config._idf

    @tool
    def create_construction(name: str, layers: list[str]) -> str:
        """Create a Construction as an ordered list of material layers.

        Args:
            name: Unique construction name (e.g., 'ExtWall_Brick').
            layers: Material names from outside to inside. All names must
                    already exist in the materials list. >= 1 layer.
        """
        if idf.has("Construction", name):
            return _err(f"Construction '{name}' already exists.")
        if not layers:
            return _err("Construction must have at least one layer.")
        if len(layers) > 10:
            return _err("Construction cannot have more than 10 layers.")
        missing = [lyr for lyr in layers if not _material_exists(idf, lyr)]
        if missing:
            return _err(
                f"Materials not found: {missing}. Create them first.",
                {"missing": missing},
            )
        try:
            kwargs: dict = {"name": name}
            for i, layer_name in enumerate(layers):
                kwargs[_LAYER_FIELDS[i]] = layer_name
            idf.add(Construction(**kwargs))
            return _ok(
                f"Construction '{name}' created successfully.",
                idf.get("Construction", name).model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating construction '{name}': {e}")

    @tool
    def list_constructions() -> str:
        """List all constructions."""
        items = [c.model_dump() for c in idf.all_of_type("Construction").values()]
        return _ok(f"Listed {len(items)} constructions.", items)

    @tool
    def get_construction(name: str) -> str:
        """Read a construction by name."""
        obj = idf.get("Construction", name)
        if obj is None:
            return _err(f"Construction '{name}' not found.")
        return _ok(f"Construction '{name}' read successfully.", obj.model_dump())

    @tool
    def delete_construction(name: str) -> str:
        """Delete a construction. Fails if referenced by surfaces/fenestration."""
        if not idf.has("Construction", name):
            return _err(f"Construction '{name}' not found.")
        refs = []
        for s in idf.all_of_type("BuildingSurface:Detailed").values():
            if s.construction_name == name:
                refs.append(f"Surface:{s.name}")
        for f in idf.all_of_type("FenestrationSurface:Detailed").values():
            if f.construction_name == name:
                refs.append(f"Fenestration:{f.name}")
        if refs:
            return _err(
                f"Construction '{name}' is referenced by other components.",
                {"references": refs},
            )
        idf.remove("Construction", name)
        return _ok(f"Construction '{name}' deleted successfully.")

    @tool
    def list_materials() -> str:
        """Read-only: list all materials available for use as construction layers."""
        items = []
        for t in _ALL_MATERIAL_TYPES:
            for obj in idf.all_of_type(t).values():
                items.append({"type": t, **obj.model_dump()})
        return _ok(f"Listed {len(items)} materials.", items)

    return [
        create_construction,
        list_constructions,
        get_construction,
        delete_construction,
        list_materials,
    ]
