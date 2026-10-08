from typing import Any, Literal

from idfpy import IDFBaseModel
from idfpy.models.constructions import (
    Material,
    MaterialAirGap,
    MaterialNoMass,
    WindowMaterialGas,
    WindowMaterialGlazing,
    WindowMaterialSimpleGlazingSystem,
)
from langchain_core.tools import BaseTool, tool

from src.agent.tools._share import model_tool, ok
from src.modeling import objects
from src.modeling.envelope import Roughness, all_materials, find_material
from src.state.config_state import ConfigState


def _material_dump(material: IDFBaseModel) -> dict[str, Any]:
    return {
        "type": material.idf_object_type(),
        **material.model_dump(exclude_none=True),
    }


def list_materials_tool(config: ConfigState) -> BaseTool:
    @tool
    def list_materials() -> str:
        """List all materials, with their EnergyPlus type."""
        items = [_material_dump(m) for m in all_materials(config.idf)]
        return ok(f"Listed {len(items)} materials.", items)

    return list_materials


def make_material_tools(config: ConfigState) -> list[BaseTool]:
    idf = config.idf

    @model_tool
    def create_standard_material(
        name: str,
        roughness: Roughness,
        thickness: float,
        conductivity: float,
        density: float,
        specific_heat: float,
    ) -> str:
        """Create a Standard material (solid layer with thermal mass).

        Args:
            name: Unique material name.
            roughness: Surface roughness.
            thickness: Meters, > 0.
            conductivity: W/(m*K), > 0.
            density: kg/m^3, > 0.
            specific_heat: J/(kg*K), > 0.
        """
        material = objects.create(
            idf,
            Material(
                name=name,
                roughness=roughness,
                thickness=thickness,
                conductivity=conductivity,
                density=density,
                specific_heat=specific_heat,
            ),
        )
        return ok(f"Material '{name}' created.", _material_dump(material))

    @model_tool
    def create_nomass_material(
        name: str, roughness: Roughness, thermal_resistance: float
    ) -> str:
        """Create a NoMass material (R-value only).

        Args:
            name: Unique material name.
            roughness: Surface roughness.
            thermal_resistance: R-value, m^2*K/W, > 0.
        """
        material = objects.create(
            idf,
            MaterialNoMass(
                name=name, roughness=roughness, thermal_resistance=thermal_resistance
            ),
        )
        return ok(f"Material:NoMass '{name}' created.", _material_dump(material))

    @model_tool
    def create_airgap_material(name: str, thermal_resistance: float) -> str:
        """Create an AirGap material for opaque walls, roofs and floors.

        Not for windows: separate glass panes with create_window_gas_material.

        Args:
            name: Unique material name.
            thermal_resistance: Air cavity resistance, m^2*K/W.
        """
        material = objects.create(
            idf, MaterialAirGap(name=name, thermal_resistance=thermal_resistance)
        )
        return ok(f"Material:AirGap '{name}' created.", _material_dump(material))

    @model_tool
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
        material = objects.create(
            idf,
            WindowMaterialSimpleGlazingSystem(
                name=name,
                u_factor=u_factor,
                solar_heat_gain_coefficient=solar_heat_gain_coefficient,
                visible_transmittance=visible_transmittance,
            ),
        )
        return ok(
            f"WindowMaterial:SimpleGlazingSystem '{name}' created.",
            _material_dump(material),
        )

    @model_tool
    def create_window_glazing_material(
        name: str,
        thickness: float,
        solar_transmittance: float = 0.775,
        solar_reflectance: float = 0.071,
        visible_transmittance: float = 0.881,
        visible_reflectance: float = 0.080,
        conductivity: float = 0.9,
    ) -> str:
        """Create one glass pane for a multi-pane window (WindowMaterial:Glazing).

        Defaults describe clear float glass; both faces share the reflectances.

        Args:
            name: Unique material name.
            thickness: Pane thickness in meters, e.g. 0.006.
            solar_transmittance: At normal incidence, 0-1.
            solar_reflectance: At normal incidence, 0-1.
            visible_transmittance: At normal incidence, 0-1.
            visible_reflectance: At normal incidence, 0-1.
            conductivity: W/(m*K).
        """
        material = objects.create(
            idf,
            WindowMaterialGlazing(
                name=name,
                optical_data_type="SpectralAverage",
                thickness=thickness,
                solar_transmittance_at_normal_incidence=solar_transmittance,
                front_side_solar_reflectance_at_normal_incidence=solar_reflectance,
                back_side_solar_reflectance_at_normal_incidence=solar_reflectance,
                visible_transmittance_at_normal_incidence=visible_transmittance,
                front_side_visible_reflectance_at_normal_incidence=visible_reflectance,
                back_side_visible_reflectance_at_normal_incidence=visible_reflectance,
                conductivity=conductivity,
            ),
        )
        return ok(f"WindowMaterial:Glazing '{name}' created.", _material_dump(material))

    @model_tool
    def create_window_gas_material(
        name: str,
        thickness: float,
        gas_type: Literal["Air", "Argon", "Krypton", "Xenon"] = "Air",
    ) -> str:
        """Create the gas layer between two glass panes (WindowMaterial:Gas).

        Args:
            name: Unique material name.
            thickness: Gap width in meters, e.g. 0.012.
            gas_type: Fill gas.
        """
        material = objects.create(
            idf, WindowMaterialGas(name=name, gas_type=gas_type, thickness=thickness)
        )
        return ok(f"WindowMaterial:Gas '{name}' created.", _material_dump(material))

    @model_tool
    def get_material(name: str) -> str:
        """Read a material by name."""
        return ok(f"Material '{name}' read.", _material_dump(find_material(idf, name)))

    @model_tool
    def delete_material(name: str) -> str:
        """Delete a material; refused while a construction uses it."""
        objects.delete(idf, find_material(idf, name), name)
        return ok(f"Material '{name}' deleted.")

    return [
        create_standard_material,
        create_nomass_material,
        create_airgap_material,
        create_glazing_material,
        create_window_glazing_material,
        create_window_gas_material,
        list_materials_tool(config),
        get_material,
        delete_material,
    ]
