import json

from langchain_core.tools import BaseTool, tool

from idfpy.models.constructions import (
    Material,
    MaterialAirGap,
    MaterialNoMass,
    WindowMaterialSimpleGlazingSystem,
)
from src.mcp.state import ConfigState

# idfpy object-type strings for each material variant
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


def _find_material(idf, name: str):
    """Return (object_type, obj) for a material with the given name, or (None, None)."""
    for t in _ALL_MATERIAL_TYPES:
        obj = idf.get(t, name)
        if obj is not None:
            return t, obj
    return None, None


def make_material_tools(config: ConfigState) -> list[BaseTool]:
    idf = config._idf

    @tool
    def create_standard_material(
        name: str,
        roughness: str,
        thickness: float,
        conductivity: float,
        density: float,
        specific_heat: float,
    ) -> str:
        """Create a Standard material (solid layer with thermal mass).

        Args:
            name: Unique material name.
            roughness: One of VeryRough / Rough / MediumRough / MediumSmooth / Smooth / VerySmooth.
            thickness: Meters, > 0.
            conductivity: W/(m*K), > 0.
            density: kg/m^3, > 0.
            specific_heat: J/(kg*K), > 0.
        """
        if idf.has("Material", name):
            return _err(f"Material '{name}' already exists.")
        try:
            idf.add(Material(
                name=name,
                roughness=roughness,
                thickness=thickness,
                conductivity=conductivity,
                density=density,
                specific_heat=specific_heat,
            ))
            return _ok(
                f"Material '{name}' created successfully.",
                idf.get("Material", name).model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating material '{name}': {e}")

    @tool
    def create_nomass_material(
        name: str,
        roughness: str,
        thermal_resistance: float,
    ) -> str:
        """Create a NoMass material (R-value only).

        Args:
            name: Unique material name.
            roughness: Same options as create_standard_material.
            thermal_resistance: R-value, m^2*K/W, > 0.
        """
        if idf.has("Material:NoMass", name):
            return _err(f"Material:NoMass '{name}' already exists.")
        try:
            idf.add(MaterialNoMass(
                name=name,
                roughness=roughness,
                thermal_resistance=thermal_resistance,
            ))
            return _ok(
                f"Material:NoMass '{name}' created successfully.",
                idf.get("Material:NoMass", name).model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating NoMass material '{name}': {e}")

    @tool
    def create_airgap_material(name: str, thermal_resistance: float) -> str:
        """Create an AirGap material (air cavity resistance)."""
        if idf.has("Material:AirGap", name):
            return _err(f"Material:AirGap '{name}' already exists.")
        try:
            idf.add(MaterialAirGap(name=name, thermal_resistance=thermal_resistance))
            return _ok(
                f"Material:AirGap '{name}' created successfully.",
                idf.get("Material:AirGap", name).model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating AirGap material '{name}': {e}")

    @tool
    def create_glazing_material(
        name: str,
        u_factor: float,
        solar_heat_gain_coefficient: float,
        visible_transmittance: float | None = None,
    ) -> str:
        """Create a Glazing material (simplified window).

        Args:
            name: Unique material name.
            u_factor: Overall U-value, W/(m^2*K), > 0.
            solar_heat_gain_coefficient: SHGC, 0-1.
            visible_transmittance: Optional VT, 0-1.
        """
        if idf.has("WindowMaterial:SimpleGlazingSystem", name):
            return _err(f"WindowMaterial:SimpleGlazingSystem '{name}' already exists.")
        try:
            idf.add(WindowMaterialSimpleGlazingSystem(
                name=name,
                u_factor=u_factor,
                solar_heat_gain_coefficient=solar_heat_gain_coefficient,
                visible_transmittance=visible_transmittance,
            ))
            return _ok(
                f"WindowMaterial:SimpleGlazingSystem '{name}' created successfully.",
                idf.get("WindowMaterial:SimpleGlazingSystem", name).model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating glazing material '{name}': {e}")

    @tool
    def list_materials() -> str:
        """List all materials."""
        items = []
        for t in _ALL_MATERIAL_TYPES:
            for obj in idf.all_of_type(t).values():
                items.append({"type": t, **obj.model_dump()})
        return _ok(f"Listed {len(items)} materials.", items)

    @tool
    def get_material(name: str) -> str:
        """Read a material by name."""
        mat_type, obj = _find_material(idf, name)
        if obj is None:
            return _err(f"Material '{name}' not found.")
        return _ok(f"Material '{name}' read successfully.", {"type": mat_type, **obj.model_dump()})

    @tool
    def delete_material(name: str) -> str:
        """Delete a material. Fails if referenced by a construction."""
        mat_type, obj = _find_material(idf, name)
        if obj is None:
            return _err(f"Material '{name}' not found.")
        # Check if any construction references this material
        refs = []
        layer_fields = [
            "outside_layer", "layer_2", "layer_3", "layer_4", "layer_5",
            "layer_6", "layer_7", "layer_8", "layer_9", "layer_10",
        ]
        for c in idf.all_of_type("Construction").values():
            for lf in layer_fields:
                if getattr(c, lf, None) == name:
                    refs.append(f"Construction:{c.name}")
                    break
        if refs:
            return _err(
                f"Material '{name}' is referenced by constructions.",
                {"references": refs},
            )
        idf.remove(mat_type, name)
        return _ok(f"Material '{name}' deleted successfully.")

    return [
        create_standard_material,
        create_nomass_material,
        create_airgap_material,
        create_glazing_material,
        list_materials,
        get_material,
        delete_material,
    ]
