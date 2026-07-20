import json
from typing import Literal

from idfpy.models import (
    Construction,
    Material,
    MaterialAirGap,
    MaterialNoMass,
    WindowMaterialGlazing,
    WindowMaterialSimpleGlazingSystem,
)
from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState

# idfpy object-type strings for each material variant
_ALL_MATERIAL_TYPES = [
    "Material",
    "Material:NoMass",
    "Material:AirGap",
    "WindowMaterial:SimpleGlazingSystem",
    "WindowMaterial:Glazing",
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


def make_material_tools(config: ConfigState, rag=None) -> list[BaseTool]:
    idf = config._idf

    @tool
    def create_standard_material(
        name: str,
        roughness: Literal[
            "VeryRough", "Rough", "MediumRough", "MediumSmooth", "Smooth", "VerySmooth"
        ],
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
        if idf is None:
            raise ValueError("IDF is None")
        if idf.has("Material", name):
            return _err(f"Material '{name}' already exists.")
        try:
            idf.add(
                Material(
                    name=name,
                    roughness=roughness,
                    thickness=thickness,
                    conductivity=conductivity,
                    density=density,
                    specific_heat=specific_heat,
                )
            )
            data = idf.get(Material, name)
            if data is None:
                raise ValueError("Material not found")
            return _ok(
                f"Material '{name}' created successfully.",
                data.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating material '{name}': {e}")

    @tool
    def create_nomass_material(
        name: str,
        roughness: Literal[
            "VeryRough", "Rough", "MediumRough", "MediumSmooth", "Smooth", "VerySmooth"
        ],
        thermal_resistance: float,
    ) -> str:
        """Create a NoMass material (R-value only).

        Args:
            name: Unique material name.
            roughness: Same options as create_standard_material.
            thermal_resistance: R-value, m^2*K/W, > 0.
        """
        if idf is None:
            raise ValueError("IDF is None")
        if idf.has(MaterialNoMass, name):
            return _err(f"Material:NoMass '{name}' already exists.")
        try:
            idf.add(
                MaterialNoMass(
                    name=name,
                    roughness=roughness,
                    thermal_resistance=thermal_resistance,
                )
            )
            data = idf.get(MaterialNoMass, name)
            if data is None:
                raise ValueError("Material:NoMass not found")
            return _ok(
                f"Material:NoMass '{name}' created successfully.",
                data.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating NoMass material '{name}': {e}")

    @tool
    def create_airgap_material(name: str, thermal_resistance: float) -> str:
        """Create an AirGap material (air cavity resistance)."""
        if idf is None:
            raise ValueError("IDF is None")
        if idf.has(MaterialAirGap, name):
            return _err(f"Material:AirGap '{name}' already exists.")
        try:
            idf.add(MaterialAirGap(name=name, thermal_resistance=thermal_resistance))
            data = idf.get(MaterialAirGap, name)
            if data is None:
                raise ValueError("Material:AirGap not found")
            return _ok(
                f"Material:AirGap '{name}' created successfully.",
                data.model_dump(),
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
        """Create a Glazing material (WindowMaterial:SimpleGlazingSystem).

        THIS IS THE PREFERRED AND DEFAULT WAY TO MODEL WINDOWS. Use it for
        all windows, including "double pane", "triple pane", "low-e", and
        "clear glass" specs — just provide the whole-window equivalent
        U-factor / SHGC / VT. Typical values:
          single pane:  U~5.8  SHGC~0.78
          double pane:  U~1.8-2.8  SHGC~0.4-0.6
          triple pane:  U~0.8-1.4  SHGC~0.3-0.5
        This simplified model is numerically robust and never triggers
        EnergyPlus convergence errors.

        Args:
            name: Unique material name.
            u_factor: Overall U-value, W/(m^2*K), > 0.
            solar_heat_gain_coefficient: SHGC, 0-1.
            visible_transmittance: Optional VT, 0-1.
        """
        if idf is None:
            raise ValueError("IDF is None")
        if idf.has(WindowMaterialSimpleGlazingSystem, name):
            return _err(f"WindowMaterial:SimpleGlazingSystem '{name}' already exists.")
        try:
            idf.add(
                WindowMaterialSimpleGlazingSystem(
                    name=name,
                    u_factor=u_factor,
                    solar_heat_gain_coefficient=solar_heat_gain_coefficient,
                    visible_transmittance=visible_transmittance,
                )
            )
            data = idf.get(WindowMaterialSimpleGlazingSystem, name)
            if data is None:
                raise ValueError("WindowMaterial:SimpleGlazingSystem not found")
            return _ok(
                f"WindowMaterial:SimpleGlazingSystem '{name}' created successfully.",
                data.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating glazing material '{name}': {e}")

    @tool
    def create_glazing_layer_material(
        name: str,
        thickness: float,
        solar_transmittance: float,
        visible_transmittance: float,
        conductivity: float,
        front_solar_reflectance: float = 0.07,
        back_solar_reflectance: float = 0.07,
        front_visible_reflectance: float = 0.08,
        back_visible_reflectance: float = 0.08,
        infrared_transmittance: float = 0.0,
        front_emissivity: float = 0.84,
        back_emissivity: float = 0.84,
        dirt_correction_factor: float = 1.0,
        solar_diffusing: str = "No",
    ) -> str:
        """Create a TRUE per-pane glass layer (WindowMaterial:Glazing).

        DEPRECATED — prefer create_glazing_material instead.

        This tool creates a per-pane glass layer whose 13+ optical fields
        frequently cause EnergyPlus to crash with a Fatal "Convergence error
        in SolveForWindowTemperatures". For virtually all window specs you
        should use create_glazing_material (a WindowMaterial:SimpleGlazingSystem
        that takes only U-factor / SHGC / VT and never triggers convergence
        failures).

        Use this tool ONLY when the spec explicitly provides per-pane optical
        data (glass thickness, solar transmittance, visible transmittance,
        reflectance, AND emissivity for each individual pane). For "double
        pane" / "triple pane" / "low-e" / "clear glass" specs that give only a
        whole-window U-factor or qualitative description, use
        create_glazing_material instead.

        Args:
            name: Unique material name.
            thickness: Pane thickness, meters, > 0 (e.g. 0.003 for 3mm).
            solar_transmittance: Solar transmittance at normal incidence, 0-1.
            visible_transmittance: Visible transmittance at normal incidence, 0-1.
            conductivity: Glass conductivity, W/(m*K) (~1.0 for clear glass).
            front_solar_reflectance / back_solar_reflectance: 0-1.
            front_visible_reflectance / back_visible_reflectance: 0-1.
            infrared_transmittance: IR transmittance at normal incidence, 0-1.
            front_emissivity / back_emissivity: IR hemispherical emissivity, 0-1.
            dirt_correction_factor: 0.5-1.0 (1.0 = clean).
            solar_diffusing: "Yes" or "No".
        """
        if idf is None:
            raise ValueError("IDF is None")
        if idf.has(WindowMaterialGlazing, name):
            return _err(f"WindowMaterial:Glazing '{name}' already exists.")
        # Hard guard: per-pane WindowMaterial:Glazing with LLM-generated
        # optical parameters frequently makes EnergyPlus's
        # SolveForWindowTemperatures fail to converge (Fatal abort). The
        # simplified create_glazing_material (WindowMaterial:SimpleGlazingSystem)
        # has the same thermal behavior at the whole-window level and never
        # triggers this. Force the LLM onto the safe path — it must call
        # create_glazing_material with an equivalent U-factor/SHGC instead.
        return _err(
            "create_glazing_layer_material is disabled to prevent EnergyPlus "
            "convergence failures. The per-pane WindowMaterial:Glazing model "
            "requires optical parameters that are very hard to get right and "
            "frequently crash EnergyPlus with 'Convergence error in "
            "SolveForWindowTemperatures'. Instead use create_glazing_material "
            "(WindowMaterial:SimpleGlazingSystem) with an equivalent whole-"
            "window U-factor and SHGC. Typical values: single pane U~5.8 "
            "SHGC~0.78; double pane U~1.8-2.8 SHGC~0.4-0.6; triple pane "
            "U~0.8-1.4 SHGC~0.3-0.5. Then make the window construction use "
            "this SimpleGlazingSystem as its SOLE layer.",
            data={
                "hint": "use create_glazing_material instead"
            },  # FIXME: data is not used
        )
        try:  # pragma: no cover - unreachable while the guard above is active
            idf.add(
                WindowMaterialGlazing(
                    name=name,
                    optical_data_type="SpectralAverage",
                    thickness=thickness,
                    solar_transmittance_at_normal_incidence=solar_transmittance,
                    front_side_solar_reflectance_at_normal_incidence=front_solar_reflectance,
                    back_side_solar_reflectance_at_normal_incidence=back_solar_reflectance,
                    visible_transmittance_at_normal_incidence=visible_transmittance,
                    front_side_visible_reflectance_at_normal_incidence=front_visible_reflectance,
                    back_side_visible_reflectance_at_normal_incidence=back_visible_reflectance,
                    infrared_transmittance_at_normal_incidence=infrared_transmittance,
                    front_side_infrared_hemispherical_emissivity=front_emissivity,
                    back_side_infrared_hemispherical_emissivity=back_emissivity,
                    conductivity=conductivity,
                    dirt_correction_factor_for_solar_and_visible_transmittance=dirt_correction_factor,
                    solar_diffusing=solar_diffusing,
                )
            )
            data = idf.get(WindowMaterialGlazing, name)
            if data is None:
                raise ValueError("WindowMaterial:Glazing not found")
            return _ok(
                f"WindowMaterial:Glazing '{name}' created successfully.",
                idf.get("WindowMaterial:Glazing", name).model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating glazing layer material '{name}': {e}")

    @tool
    def list_materials() -> str:
        """List all materials."""
        if idf is None:
            raise ValueError("IDF is None")
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
        return _ok(
            f"Material '{name}' read successfully.",
            {"type": mat_type, **obj.model_dump()},
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
        mat_type, obj = _find_material(idf, name)
        if obj is None:
            return _err(f"Material '{name}' not found.")
        try:
            # Fields common to standard/nomass/airgap
            if roughness is not None and hasattr(obj, "roughness"):
                obj.roughness = roughness
            if thermal_resistance is not None and hasattr(obj, "thermal_resistance"):
                obj.thermal_resistance = thermal_resistance
            # Standard-only fields
            if mat_type == "Material":
                if thickness is not None:
                    obj.thickness = thickness
                if conductivity is not None:
                    obj.conductivity = conductivity
                if density is not None:
                    obj.density = density
                if specific_heat is not None:
                    obj.specific_heat = specific_heat
            # Glazing-only fields
            if mat_type == "WindowMaterial:SimpleGlazingSystem":
                if u_factor is not None:
                    obj.u_factor = u_factor
                if solar_heat_gain_coefficient is not None:
                    obj.solar_heat_gain_coefficient = solar_heat_gain_coefficient
                if visible_transmittance is not None:
                    obj.visible_transmittance = visible_transmittance
            return _ok(
                f"Material '{name}' updated successfully.",
                {"type": mat_type, **obj.model_dump()},
            )
        except Exception as e:
            return _err(f"Error updating material '{name}': {e}")

    @tool
    def delete_material(name: str) -> str:
        """Delete a material. Fails if referenced by a construction."""
        mat_type, obj = _find_material(idf, name)
        if obj is None:
            return _err(f"Material '{name}' not found.")
        # Check if any construction references this material
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
        if idf is None:
            raise ValueError("IDF is None")
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
        idf.remove(mat_type, name)
        return _ok(f"Material '{name}' deleted successfully.")

    tools = [
        create_standard_material,
        create_nomass_material,
        create_airgap_material,
        create_glazing_material,
        create_glazing_layer_material,
        list_materials,
        get_material,
        update_material,
        delete_material,
    ]
    if rag is not None:
        from src.agent.tools.rag_tools import (
            TABLE_ALL_MATERIALS,
            TABLE_NO_MASS_MATERIALS,
            TABLE_STANDARD_MATERIALS,
            make_rag_tool,
        )

        tools.append(
            make_rag_tool(
                [
                    TABLE_STANDARD_MATERIALS,
                    TABLE_NO_MASS_MATERIALS,
                    TABLE_ALL_MATERIALS,
                ],
                rag=rag,
            )
        )
    return tools
