from fastmcp import FastMCP
from pydantic import Field

from src.mcp.api.common import ToolInput, to_payload
from src.mcp.tools import (
    ConstructionTool,
    FenestrationTool,
    MaterialTool,
    SurfaceTool,
)


class StandardMaterialCreateInput(ToolInput):
    name: str = Field(alias="Name")
    material_type: str = Field(default="Standard", alias="Type")
    roughness: str = Field(alias="Roughness")
    thickness: float = Field(alias="Thickness")
    conductivity: float = Field(alias="Conductivity")
    density: float = Field(alias="Density")
    specific_heat: float = Field(alias="Specific_Heat")


class StandardMaterialUpdateInput(ToolInput):
    name: str = Field(alias="Name")
    material_type: str = Field(default="Standard", alias="Type")
    roughness: str | None = Field(default=None, alias="Roughness")
    thickness: float | None = Field(default=None, alias="Thickness")
    conductivity: float | None = Field(default=None, alias="Conductivity")
    density: float | None = Field(default=None, alias="Density")
    specific_heat: float | None = Field(default=None, alias="Specific_Heat")


class NoMassMaterialCreateInput(ToolInput):
    name: str = Field(alias="Name")
    material_type: str = Field(default="NoMass", alias="Type")
    roughness: str = Field(alias="Roughness")
    thermal_resistance: float = Field(alias="Thermal_Resistance")


class NoMassMaterialUpdateInput(ToolInput):
    name: str = Field(alias="Name")
    material_type: str = Field(default="NoMass", alias="Type")
    roughness: str | None = Field(default=None, alias="Roughness")
    thermal_resistance: float | None = Field(default=None, alias="Thermal_Resistance")


class AirGapMaterialCreateInput(ToolInput):
    name: str = Field(alias="Name")
    material_type: str = Field(default="AirGap", alias="Type")
    thermal_resistance: float = Field(alias="Thermal_Resistance")


class AirGapMaterialUpdateInput(ToolInput):
    name: str = Field(alias="Name")
    material_type: str = Field(default="AirGap", alias="Type")
    thermal_resistance: float | None = Field(default=None, alias="Thermal_Resistance")


class GlazingMaterialCreateInput(ToolInput):
    name: str = Field(alias="Name")
    material_type: str = Field(default="Glazing", alias="Type")
    u_factor: float = Field(alias="U-Factor")
    solar_heat_gain_coefficient: float = Field(alias="Solar_Heat_Gain_Coefficient")
    visible_transmittance: float = Field(alias="Visible_Transmittance")


class GlazingMaterialUpdateInput(ToolInput):
    name: str = Field(alias="Name")
    material_type: str = Field(default="Glazing", alias="Type")
    u_factor: float | None = Field(default=None, alias="U-Factor")
    solar_heat_gain_coefficient: float | None = Field(
        default=None,
        alias="Solar_Heat_Gain_Coefficient",
    )
    visible_transmittance: float | None = Field(default=None, alias="Visible_Transmittance")


class ConstructionCreateInput(ToolInput):
    name: str = Field(alias="Name")
    layers: list[str] = Field(alias="Layers")


class ConstructionUpdateInput(ToolInput):
    name: str = Field(alias="Name")
    layers: list[str] | None = Field(default=None, alias="Layers")


class SurfaceCreateInput(ToolInput):
    name: str = Field(alias="Name")
    surface_type: str = Field(alias="Surface Type")
    construction_name: str = Field(alias="Construction Name")
    zone_name: str = Field(alias="Zone Name")
    outside_boundary_condition: str = Field(alias="Outside Boundary Condition")
    sun_exposure: str = Field(alias="Sun Exposure")
    wind_exposure: str = Field(alias="Wind Exposure")
    vertices: list[dict] = Field(alias="Vertices")
    outside_boundary_condition_object: str | None = Field(
        default=None,
        alias="Outside Boundary Condition Object",
    )
    space_name: str | None = Field(default=None, alias="Space Name")
    view_factor_to_ground: float | str = Field(
        default="autocalculate",
        alias="View Factor to Ground",
    )
    number_of_vertices: int | str = Field(default="autocalculate", alias="Number of Vertices")


class SurfaceUpdateInput(ToolInput):
    name: str = Field(alias="Name")
    surface_type: str | None = Field(default=None, alias="Surface Type")
    construction_name: str | None = Field(default=None, alias="Construction Name")
    zone_name: str | None = Field(default=None, alias="Zone Name")
    space_name: str | None = Field(default=None, alias="Space Name")
    outside_boundary_condition: str | None = Field(
        default=None,
        alias="Outside Boundary Condition",
    )
    outside_boundary_condition_object: str | None = Field(
        default=None,
        alias="Outside Boundary Condition Object",
    )
    sun_exposure: str | None = Field(default=None, alias="Sun Exposure")
    wind_exposure: str | None = Field(default=None, alias="Wind Exposure")
    view_factor_to_ground: float | str | None = Field(
        default=None,
        alias="View Factor to Ground",
    )
    number_of_vertices: int | str | None = Field(default=None, alias="Number of Vertices")
    vertices: list[dict] | None = Field(default=None, alias="Vertices")


class FenestrationCreateInput(ToolInput):
    name: str = Field(alias="Name")
    surface_type: str = Field(alias="Surface Type")
    construction_name: str = Field(alias="Construction Name")
    building_surface_name: str = Field(alias="Building Surface Name")
    vertices: list[dict] = Field(alias="Vertices")
    outside_boundary_condition_object: str | None = Field(
        default=None,
        alias="Outside Boundary Condition Object",
    )
    view_factor_to_ground: float | str = Field(
        default="autocalculate",
        alias="View Factor to Ground",
    )
    frame_and_divider_name: str | None = Field(default=None, alias="Frame and Divider Name")
    multiplier: int = Field(default=1, alias="Multiplier")
    number_of_vertices: int | str = Field(default="autocalculate", alias="Number of Vertices")


class FenestrationUpdateInput(ToolInput):
    name: str = Field(alias="Name")
    surface_type: str | None = Field(default=None, alias="Surface Type")
    construction_name: str | None = Field(default=None, alias="Construction Name")
    building_surface_name: str | None = Field(default=None, alias="Building Surface Name")
    outside_boundary_condition_object: str | None = Field(
        default=None,
        alias="Outside Boundary Condition Object",
    )
    view_factor_to_ground: float | str | None = Field(
        default=None,
        alias="View Factor to Ground",
    )
    frame_and_divider_name: str | None = Field(default=None, alias="Frame and Divider Name")
    multiplier: int | None = Field(default=None, alias="Multiplier")
    number_of_vertices: int | str | None = Field(default=None, alias="Number of Vertices")
    vertices: list[dict] | None = Field(default=None, alias="Vertices")


def register_envelope_tools(
    mcp: FastMCP,
    material_tool: MaterialTool,
    construction_tool: ConstructionTool,
    surface_tool: SurfaceTool,
    fenestration_tool: FenestrationTool,
) -> None:
    @mcp.tool
    def create_standard_material(
        name: str,
        roughness: str,
        thickness: float,
        conductivity: float,
        density: float,
        specific_heat: float,
    ) -> dict:
        payload = to_payload(
            StandardMaterialCreateInput.model_validate(
                {
                    "name": name,
                    "roughness": roughness,
                    "thickness": thickness,
                    "conductivity": conductivity,
                    "density": density,
                    "specific_heat": specific_heat,
                }
            )
        )
        return material_tool.create(payload).to_mcp_response()

    @mcp.tool
    def create_no_mass_material(
        name: str,
        roughness: str,
        thermal_resistance: float,
    ) -> dict:
        payload = to_payload(
            NoMassMaterialCreateInput.model_validate(
                {
                    "name": name,
                    "roughness": roughness,
                    "thermal_resistance": thermal_resistance,
                }
            )
        )
        return material_tool.create(payload).to_mcp_response()

    @mcp.tool
    def create_air_gap_material(
        name: str,
        thermal_resistance: float,
    ) -> dict:
        payload = to_payload(
            AirGapMaterialCreateInput.model_validate(
                {
                    "name": name,
                    "thermal_resistance": thermal_resistance,
                }
            )
        )
        return material_tool.create(payload).to_mcp_response()

    @mcp.tool
    def create_glazing_material(
        name: str,
        u_factor: float,
        solar_heat_gain_coefficient: float,
        visible_transmittance: float,
    ) -> dict:
        payload = to_payload(
            GlazingMaterialCreateInput.model_validate(
                {
                    "name": name,
                    "u_factor": u_factor,
                    "solar_heat_gain_coefficient": solar_heat_gain_coefficient,
                    "visible_transmittance": visible_transmittance,
                }
            )
        )
        return material_tool.create(payload).to_mcp_response()

    @mcp.tool
    def get_material(name: str) -> dict:
        return material_tool.read(name).to_mcp_response()

    @mcp.tool
    def update_standard_material(
        name: str,
        roughness: str | None = None,
        thickness: float | None = None,
        conductivity: float | None = None,
        density: float | None = None,
        specific_heat: float | None = None,
    ) -> dict:
        payload = to_payload(
            StandardMaterialUpdateInput.model_validate(
                {
                    "name": name,
                    "roughness": roughness,
                    "thickness": thickness,
                    "conductivity": conductivity,
                    "density": density,
                    "specific_heat": specific_heat,
                }
            )
        )
        return material_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def update_no_mass_material(
        name: str,
        roughness: str | None = None,
        thermal_resistance: float | None = None,
    ) -> dict:
        payload = to_payload(
            NoMassMaterialUpdateInput.model_validate(
                {
                    "name": name,
                    "roughness": roughness,
                    "thermal_resistance": thermal_resistance,
                }
            )
        )
        return material_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def update_air_gap_material(
        name: str,
        thermal_resistance: float | None = None,
    ) -> dict:
        payload = to_payload(
            AirGapMaterialUpdateInput.model_validate(
                {
                    "name": name,
                    "thermal_resistance": thermal_resistance,
                }
            )
        )
        return material_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def update_glazing_material(
        name: str,
        u_factor: float | None = None,
        solar_heat_gain_coefficient: float | None = None,
        visible_transmittance: float | None = None,
    ) -> dict:
        payload = to_payload(
            GlazingMaterialUpdateInput.model_validate(
                {
                    "name": name,
                    "u_factor": u_factor,
                    "solar_heat_gain_coefficient": solar_heat_gain_coefficient,
                    "visible_transmittance": visible_transmittance,
                }
            )
        )
        return material_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def delete_material(name: str) -> dict:
        return material_tool.delete(name).to_mcp_response()

    @mcp.tool
    def list_materials() -> dict:
        return material_tool.list_all().to_mcp_response()

    @mcp.tool
    def create_construction(
        name: str,
        layers: list[str],
    ) -> dict:
        payload = to_payload(
            ConstructionCreateInput.model_validate(
                {
                    "name": name,
                    "layers": layers,
                }
            )
        )
        return construction_tool.create(payload).to_mcp_response()

    @mcp.tool
    def get_construction(name: str) -> dict:
        return construction_tool.read(name).to_mcp_response()

    @mcp.tool
    def update_construction(
        name: str,
        layers: list[str] | None = None,
    ) -> dict:
        payload = to_payload(
            ConstructionUpdateInput.model_validate(
                {
                    "name": name,
                    "layers": layers,
                }
            )
        )
        return construction_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def delete_construction(name: str) -> dict:
        return construction_tool.delete(name).to_mcp_response()

    @mcp.tool
    def list_constructions() -> dict:
        return construction_tool.list_all().to_mcp_response()

    @mcp.tool
    def create_surface(
        name: str,
        surface_type: str,
        construction_name: str,
        zone_name: str,
        outside_boundary_condition: str,
        sun_exposure: str,
        wind_exposure: str,
        vertices: list[dict],
        outside_boundary_condition_object: str | None = None,
        space_name: str | None = None,
        view_factor_to_ground: float | str = "autocalculate",
        number_of_vertices: int | str = "autocalculate",
    ) -> dict:
        payload = to_payload(
            SurfaceCreateInput.model_validate(
                {
                    "name": name,
                    "surface_type": surface_type,
                    "construction_name": construction_name,
                    "zone_name": zone_name,
                    "outside_boundary_condition": outside_boundary_condition,
                    "sun_exposure": sun_exposure,
                    "wind_exposure": wind_exposure,
                    "vertices": vertices,
                    "outside_boundary_condition_object": outside_boundary_condition_object,
                    "space_name": space_name,
                    "view_factor_to_ground": view_factor_to_ground,
                    "number_of_vertices": number_of_vertices,
                }
            )
        )
        return surface_tool.create(payload).to_mcp_response()

    @mcp.tool
    def get_surface(name: str) -> dict:
        return surface_tool.read(name).to_mcp_response()

    @mcp.tool
    def update_surface(
        name: str,
        surface_type: str | None = None,
        construction_name: str | None = None,
        zone_name: str | None = None,
        space_name: str | None = None,
        outside_boundary_condition: str | None = None,
        outside_boundary_condition_object: str | None = None,
        sun_exposure: str | None = None,
        wind_exposure: str | None = None,
        view_factor_to_ground: float | str | None = None,
        number_of_vertices: int | str | None = None,
        vertices: list[dict] | None = None,
    ) -> dict:
        payload = to_payload(
            SurfaceUpdateInput.model_validate(
                {
                    "name": name,
                    "surface_type": surface_type,
                    "construction_name": construction_name,
                    "zone_name": zone_name,
                    "space_name": space_name,
                    "outside_boundary_condition": outside_boundary_condition,
                    "outside_boundary_condition_object": outside_boundary_condition_object,
                    "sun_exposure": sun_exposure,
                    "wind_exposure": wind_exposure,
                    "view_factor_to_ground": view_factor_to_ground,
                    "number_of_vertices": number_of_vertices,
                    "vertices": vertices,
                }
            )
        )
        return surface_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def delete_surface(name: str) -> dict:
        return surface_tool.delete(name).to_mcp_response()

    @mcp.tool
    def list_surfaces() -> dict:
        return surface_tool.list_all().to_mcp_response()

    @mcp.tool
    def create_fenestration_surface(
        name: str,
        surface_type: str,
        construction_name: str,
        building_surface_name: str,
        vertices: list[dict],
        outside_boundary_condition_object: str | None = None,
        view_factor_to_ground: float | str = "autocalculate",
        frame_and_divider_name: str | None = None,
        multiplier: int = 1,
        number_of_vertices: int | str = "autocalculate",
    ) -> dict:
        payload = to_payload(
            FenestrationCreateInput.model_validate(
                {
                    "name": name,
                    "surface_type": surface_type,
                    "construction_name": construction_name,
                    "building_surface_name": building_surface_name,
                    "vertices": vertices,
                    "outside_boundary_condition_object": outside_boundary_condition_object,
                    "view_factor_to_ground": view_factor_to_ground,
                    "frame_and_divider_name": frame_and_divider_name,
                    "multiplier": multiplier,
                    "number_of_vertices": number_of_vertices,
                }
            )
        )
        return fenestration_tool.create(payload).to_mcp_response()

    @mcp.tool
    def get_fenestration_surface(name: str) -> dict:
        return fenestration_tool.read(name).to_mcp_response()

    @mcp.tool
    def update_fenestration_surface(
        name: str,
        surface_type: str | None = None,
        construction_name: str | None = None,
        building_surface_name: str | None = None,
        outside_boundary_condition_object: str | None = None,
        view_factor_to_ground: float | str | None = None,
        frame_and_divider_name: str | None = None,
        multiplier: int | None = None,
        number_of_vertices: int | str | None = None,
        vertices: list[dict] | None = None,
    ) -> dict:
        payload = to_payload(
            FenestrationUpdateInput.model_validate(
                {
                    "name": name,
                    "surface_type": surface_type,
                    "construction_name": construction_name,
                    "building_surface_name": building_surface_name,
                    "outside_boundary_condition_object": outside_boundary_condition_object,
                    "view_factor_to_ground": view_factor_to_ground,
                    "frame_and_divider_name": frame_and_divider_name,
                    "multiplier": multiplier,
                    "number_of_vertices": number_of_vertices,
                    "vertices": vertices,
                }
            )
        )
        return fenestration_tool.update(name, payload).to_mcp_response()

    @mcp.tool
    def delete_fenestration_surface(name: str) -> dict:
        return fenestration_tool.delete(name).to_mcp_response()

    @mcp.tool
    def list_fenestration_surfaces() -> dict:
        return fenestration_tool.list_all().to_mcp_response()
