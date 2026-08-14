import json
from typing import Final

from idfpy.idf import IDF
from idfpy.models._base import IDFBaseModel
from idfpy.models.constructions import (
    Construction,
    Material,
    MaterialAirGap,
    MaterialNoMass,
    WindowMaterialSimpleGlazingSystem,
)
from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState

MATERIAL_CLASSES: Final = (
    Material,
    MaterialNoMass,
    MaterialAirGap,
    WindowMaterialSimpleGlazingSystem,
)


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def _find_material(idf: IDF, name: str) -> IDFBaseModel | None:
    """Return the material object with the given name, or None."""
    for t in MATERIAL_CLASSES:
        obj = idf.get(t, name)
        if obj is not None:
            return obj
    return None


def make_material_tools(config: ConfigState) -> list[BaseTool]:

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
        idf = config.idf
        existing = _find_material(idf, name)
        if existing is not None:
            return _err(
                f"Material '{name}' already exists as a {existing.idf_object_type()}."
            )
        try:
            material = Material.model_validate(
                {
                    "name": name,
                    "roughness": roughness,
                    "thickness": thickness,
                    "conductivity": conductivity,
                    "density": density,
                    "specific_heat": specific_heat,
                }
            )
            idf.add(material)
            return _ok(
                f"Material '{name}' created successfully.",
                material.model_dump(),
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
        idf = config.idf
        if idf.has("Material:NoMass", name):
            return _err(f"Material:NoMass '{name}' already exists.")
        try:
            material = MaterialNoMass.model_validate(
                {
                    "name": name,
                    "roughness": roughness,
                    "thermal_resistance": thermal_resistance,
                }
            )
            idf.add(material)
            return _ok(
                f"Material:NoMass '{name}' created successfully.",
                material.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating NoMass material '{name}': {e}")

    @tool
    def create_airgap_material(name: str, thermal_resistance: float) -> str:
        """Create an AirGap material (air cavity resistance)."""
        idf = config.idf
        if idf.has("Material:AirGap", name):
            return _err(f"Material:AirGap '{name}' already exists.")
        try:
            material = MaterialAirGap(name=name, thermal_resistance=thermal_resistance)
            idf.add(material)
            return _ok(
                f"Material:AirGap '{name}' created successfully.",
                material.model_dump(),
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
        idf = config.idf
        if idf.has("WindowMaterial:SimpleGlazingSystem", name):
            return _err(f"WindowMaterial:SimpleGlazingSystem '{name}' already exists.")
        try:
            material = WindowMaterialSimpleGlazingSystem(
                name=name,
                u_factor=u_factor,
                solar_heat_gain_coefficient=solar_heat_gain_coefficient,
                visible_transmittance=visible_transmittance,
            )
            idf.add(material)
            return _ok(
                f"WindowMaterial:SimpleGlazingSystem '{name}' created successfully.",
                material.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating glazing material '{name}': {e}")

    @tool
    def list_materials() -> str:
        """List all materials."""
        idf = config.idf
        items = []
        for t in MATERIAL_CLASSES:
            for obj in idf.all_of_type(t).values():
                items.append({"type": obj.idf_object_type(), **obj.model_dump()})
        return _ok(f"Listed {len(items)} materials.", items)

    @tool
    def get_material(name: str) -> str:
        """Read a material by name."""
        idf = config.idf
        obj = _find_material(idf, name)
        if obj is None:
            return _err(f"Material '{name}' not found.")
        return _ok(
            f"Material '{name}' read successfully.",
            {"type": obj.idf_object_type(), **obj.model_dump()},
        )

    @tool
    def update_material(
        name: str,
        roughness: str | None = None,
        thickness: float | None = None,
        conductivity: float | None = None,
        density: float | None = None,
        specific_heat: float | None = None,
        thermal_resistance: float | None = None,
        u_factor: float | None = None,
        solar_heat_gain_coefficient: float | None = None,
        visible_transmittance: float | None = None,
    ) -> str:
        """Update fields of an existing material by name.

        Only non-None fields are written; the rest stay unchanged. The
        material variant (standard / nomass / airgap / glazing) is detected
        automatically — pass only the fields relevant to that variant.

        Args:
            name: Existing material name.
            roughness / thickness / conductivity / density / specific_heat:
                Standard Material fields.
            thermal_resistance: NoMass or AirGap R-value.
            u_factor / solar_heat_gain_coefficient / visible_transmittance:
                Glazing (SimpleGlazingSystem) fields.
        """
        idf = config.idf
        obj = _find_material(idf, name)
        if obj is None:
            return _err(f"Material '{name}' not found.")
        try:
            if roughness is not None and hasattr(obj, "roughness"):
                obj.roughness = roughness
            if thermal_resistance is not None and hasattr(obj, "thermal_resistance"):
                obj.thermal_resistance = thermal_resistance
            if isinstance(obj, Material):
                if thickness is not None:
                    obj.thickness = thickness
                if conductivity is not None:
                    obj.conductivity = conductivity
                if density is not None:
                    obj.density = density
                if specific_heat is not None:
                    obj.specific_heat = specific_heat
            if isinstance(obj, WindowMaterialSimpleGlazingSystem):
                if u_factor is not None:
                    obj.u_factor = u_factor
                if solar_heat_gain_coefficient is not None:
                    obj.solar_heat_gain_coefficient = solar_heat_gain_coefficient
                if visible_transmittance is not None:
                    obj.visible_transmittance = visible_transmittance
            return _ok(
                f"Material '{name}' updated successfully.",
                {"type": type(obj).__name__, **obj.model_dump()},
            )
        except Exception as e:
            return _err(f"Error updating material '{name}': {e}")

    @tool
    def delete_material(name: str) -> str:
        """Delete a material. Fails if referenced by a construction."""
        idf = config.idf
        obj = _find_material(idf, name)
        if obj is None:
            return _err(f"Material '{name}' not found.")
        refs = []
        layer_fields = [
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
        for c in idf.all_of_type(Construction).values():
            for lf in layer_fields:
                if getattr(c, lf, None) == name:
                    refs.append(f"Construction:{c.name}")
                    break
        if refs:
            return _err(
                f"Material '{name}' is referenced by constructions.",
                {"references": refs},
            )
        idf.remove(obj.idf_object_type(), name)
        return _ok(f"Material '{name}' deleted successfully.")

    return [
        create_standard_material,
        create_nomass_material,
        create_airgap_material,
        create_glazing_material,
        list_materials,
        get_material,
        update_material,
        delete_material,
    ]
