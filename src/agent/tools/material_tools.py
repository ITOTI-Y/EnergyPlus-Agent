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
from pydantic import BaseModel, Field

from src.agent.tools._share import model_tool, ok
from src.modeling import objects
from src.modeling.envelope import (
    MaterialObject,
    Roughness,
    all_materials,
    check_material_name_free,
    find_material,
)
from src.modeling.errors import ModelingError, describe_error
from src.state.config_state import ConfigState


def _material_dump(material: IDFBaseModel) -> dict[str, Any]:
    return {
        "type": material.idf_object_type(),
        **material.model_dump(exclude_none=True),
    }


class AbsorptanceSchema(BaseModel):
    """Surface absorptances; left out, EnergyPlus uses 0.9 / 0.7 / 0.7."""

    thermal_absorptance: float | None = Field(
        default=None, gt=0, le=0.99999, description="Long-wave emittance"
    )
    solar_absorptance: float | None = Field(default=None, ge=0, le=1)
    visible_absorptance: float | None = Field(default=None, ge=0, le=1)


class StandardMaterialSchema(AbsorptanceSchema):
    """Solid layer with thermal mass."""

    name: str
    roughness: Roughness
    thickness: float = Field(gt=0, description="m")
    conductivity: float = Field(gt=0, description="W/(m*K)")
    density: float = Field(gt=0, description="kg/m^3")
    specific_heat: float = Field(gt=0, description="J/(kg*K)")


class NoMassMaterialSchema(AbsorptanceSchema):
    """Layer known by its R-value only."""

    name: str
    roughness: Roughness
    thermal_resistance: float = Field(gt=0, description="m^2*K/W")


class AirGapMaterialSchema(BaseModel):
    """Air cavity of an opaque wall or roof; never in a window."""

    name: str
    thermal_resistance: float = Field(gt=0, description="m^2*K/W")


class SimpleGlazingSchema(BaseModel):
    """A whole window as one layer (WindowMaterial:SimpleGlazingSystem)."""

    name: str
    u_factor: float = Field(gt=0, description="W/(m^2*K)")
    solar_heat_gain_coefficient: float = Field(gt=0, lt=1)
    visible_transmittance: float | None = Field(default=None, gt=0, lt=1)


class GlassPaneSchema(BaseModel):
    """One glass pane of a multi-pane window; defaults are clear float glass."""

    name: str
    thickness: float = Field(gt=0, description="m, e.g. 0.006")
    solar_transmittance: float = Field(default=0.775, description="at normal incidence")
    solar_reflectance: float = Field(default=0.071, description="both faces")
    visible_transmittance: float = Field(default=0.881)
    visible_reflectance: float = Field(default=0.080, description="both faces")
    conductivity: float = Field(default=0.9, description="W/(m*K)")


class WindowGasSchema(BaseModel):
    """Gas layer between two panes."""

    name: str
    thickness: float = Field(gt=0, description="m, e.g. 0.012")
    gas_type: Literal["Air", "Argon", "Krypton", "Xenon"] = "Air"


def _pane(pane: GlassPaneSchema) -> WindowMaterialGlazing:
    return WindowMaterialGlazing(
        name=pane.name,
        optical_data_type="SpectralAverage",
        thickness=pane.thickness,
        solar_transmittance_at_normal_incidence=pane.solar_transmittance,
        front_side_solar_reflectance_at_normal_incidence=pane.solar_reflectance,
        back_side_solar_reflectance_at_normal_incidence=pane.solar_reflectance,
        visible_transmittance_at_normal_incidence=pane.visible_transmittance,
        front_side_visible_reflectance_at_normal_incidence=pane.visible_reflectance,
        back_side_visible_reflectance_at_normal_incidence=pane.visible_reflectance,
        conductivity=pane.conductivity,
    )


def list_materials_tool(config: ConfigState) -> BaseTool:
    @tool
    def list_materials() -> str:
        """List material names with their EnergyPlus type."""
        items = [
            {"name": m.name, "type": m.idf_object_type()}
            for m in all_materials(config.idf)
        ]
        return ok(f"Listed {len(items)} materials.", items)

    return list_materials


def make_material_tools(config: ConfigState) -> list[BaseTool]:
    idf = config.idf

    @model_tool
    def create_materials(
        standard: list[StandardMaterialSchema] | None = None,
        nomass: list[NoMassMaterialSchema] | None = None,
        airgap: list[AirGapMaterialSchema] | None = None,
        simple_glazing: list[SimpleGlazingSchema] | None = None,
        glass_panes: list[GlassPaneSchema] | None = None,
        window_gases: list[WindowGasSchema] | None = None,
    ) -> str:
        """Create materials of any types in one call; give all you need at once.

        Each material succeeds or fails alone. One that exists with the same
        values is kept; one that exists with other values fails. The reply
        is final: it names only the failures, so there is no need to list
        materials afterwards.

        Args:
            standard: Solid layers with thermal mass (Material). Give the
                absorptances a reference result has; leave them out otherwise.
            nomass: Layers known by R-value only (Material:NoMass), with
                absorptances as for standard.
            airgap: Air cavities in opaque walls or roofs (Material:AirGap).
            simple_glazing: Whole windows as one layer
                (WindowMaterial:SimpleGlazingSystem).
            glass_panes: Panes of multi-pane windows (WindowMaterial:Glazing).
            window_gases: Gas layers between panes (WindowMaterial:Gas).
        """
        materials: list[MaterialObject] = [
            *(Material(**m.model_dump(exclude_none=True)) for m in standard or []),
            *(MaterialNoMass(**m.model_dump(exclude_none=True)) for m in nomass or []),
            *(MaterialAirGap(**m.model_dump()) for m in airgap or []),
            *(
                WindowMaterialSimpleGlazingSystem(**m.model_dump())
                for m in simple_glazing or []
            ),
            *(_pane(m) for m in glass_panes or []),
            *(WindowMaterialGas(**m.model_dump()) for m in window_gases or []),
        ]
        if not materials:
            # Haiku sent empty calls to finish; as errors they tripped the
            # failure-loop guard and the phase lost its final answer.
            return ok(
                "No materials given. If every material exists, give your final "
                "answer now."
            )
        created = kept = 0
        failed = []
        for material in materials:
            try:
                check_material_name_free(idf, material)
                _, new = objects.create_or_same(idf, material)
            except (ModelingError, ValueError) as e:
                failed.append(
                    f"{getattr(material, 'name', '?')}: {describe_error(e)[0]}"
                )
                continue
            created += new
            kept += not new
        if failed and not created + kept:
            raise ModelingError("No material created.", {"failed": failed})
        message = f"Created {created} materials"
        if kept:
            message += f"; {kept} already existed with the same values"
        return ok(message + ".", {"failed": failed} if failed else None)

    @model_tool
    def get_material(name: str) -> str:
        """Read a material by name, with all its values."""
        return ok(f"Material '{name}' read.", _material_dump(find_material(idf, name)))

    @model_tool
    def delete_material(name: str) -> str:
        """Delete a material; refused while a construction uses it."""
        objects.delete(idf, find_material(idf, name), name)
        return ok(f"Material '{name}' deleted.")

    return [
        create_materials,
        list_materials_tool(config),
        get_material,
        delete_material,
    ]
