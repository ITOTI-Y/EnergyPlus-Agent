from fastmcp import FastMCP
from omegaconf import OmegaConf

from src.mcp.state import ConfigState
from src.mcp.tools import (
    WorkflowTool,
    ZoneTool,
    MaterialTool,
    SurfaceTool,
    FenestrationTool,
    ConstructionTool,
)

mcp = FastMCP(
    name="EnergyPlus Agent",
    version="0.1.0",
    instructions="EnergyPlus Agent is a tool for building energy simulation.",
)

state = ConfigState()

zone_tool = ZoneTool(state)
workflow_tool = WorkflowTool(state)
material_tool = MaterialTool(state)
Construction_tool = ConstructionTool(state)
surface_tool = SurfaceTool(state)
fenestration_tool = FenestrationTool(state)

@mcp.tool
def create_zone(
    name: str,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
    z_origin: float = 0.0,
    direction_of_relative_north: float | None = 0.0,
    multiplier: int = 1,
    ceiling_height: float | str = "autocalculate",
    volume: float | str = "autocalculate",
    floor_area: float | str = "autocalculate",
) -> dict:
    data = {
        "Name": name,
        "X Origin": x_origin,
        "Y Origin": y_origin,
        "Z Origin": z_origin,
        "Direction of Relative North": direction_of_relative_north,
        "Multiplier": multiplier,
        "Ceiling Height": ceiling_height,
        "Volume": volume,
        "Floor Area": floor_area,
    }

    return zone_tool.create(data).to_mcp_response()


@mcp.tool
def get_zone(name: str) -> dict:
    return zone_tool.read(name).to_mcp_response()


@mcp.tool
def update_zone(
    name: str,
    x_origin: float | None = None,
    y_origin: float | None = None,
    z_origin: float | None = None,
    direction_of_relative_north: float | None = None,
    multiplier: int | None = None,
    ceiling_height: float | str | None = None,
    volume: float | str | None = None,
    floor_area: float | str | None = None,
) -> dict:
    data = {
        "Name": name,
        "X Origin": x_origin,
        "Y Origin": y_origin,
        "Z Origin": z_origin,
        "Direction of Relative North": direction_of_relative_north,
        "Multiplier": multiplier,
        "Ceiling Height": ceiling_height,
        "Volume": volume,
        "Floor Area": floor_area,
    }
    return zone_tool.update(name, data).to_mcp_response()


@mcp.tool
def delete_zone(name: str) -> dict:
    return zone_tool.delete(name).to_mcp_response()


@mcp.tool
def list_zones() -> dict:
    return zone_tool.list_all().to_mcp_response()

@mcp.tool
def create_standard_material(
    name: str,
    roughness: str,
    thickness: float,
    conductivity: float,
    density: float,
    specific_heat: float,
) -> dict:
    data = {
        "Name": name,
        "Roughness": roughness,
        "Thickness": thickness,
        "Conductivity": conductivity,
        "Density": density,
        "Specific Heat": specific_heat,
    }
    return material_tool.create(data).to_mcp_response()

@mcp.tool
def create_no_mass_material(
    name: str,
    roughness: str,
    thermal_resistance: float,
) -> dict:
    data = {
        "Name": name,
        "Roughness": roughness,
        "Thermal Resistance": thermal_resistance,
    }
    return material_tool.create(data).to_mcp_response()

@mcp.tool
def create_air_gap_material(
    name: str,
    thermal_resistance: float,
) -> dict:
    data = {
        "Name": name,
        "Thermal Resistance": thermal_resistance,
    }
    return material_tool.create(data).to_mcp_response()

@mcp.tool
def create_glazing_material(
    name: str,
    u_factor: float,
    solar_heat_gain_coefficient: float,
    visible_transmittance: float,
) -> dict:
    data = {
        "Name": name,
        "U-Factor": u_factor,
        "Solar Heat Gain Coefficient": solar_heat_gain_coefficient,
        "Visible Transmittance": visible_transmittance,
    }
    return material_tool.create(data).to_mcp_response()

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
    data = {
        "Name": name,
        "Roughness": roughness,
        "Thickness": thickness,
        "Conductivity": conductivity,
        "Density": density,
        "Specific Heat": specific_heat,
    }
    return material_tool.update(name, data).to_mcp_response()

@mcp.tool
def update_no_mass_material(
    name: str,
    roughness: str | None = None,
    thermal_resistance: float | None = None,
) -> dict:
    data = {
        "Name": name,
        "Roughness": roughness,
        "Thermal Resistance": thermal_resistance,
    }
    return material_tool.update(name, data).to_mcp_response()

@mcp.tool
def update_air_gap_material(
    name: str,
    thermal_resistance: float | None = None,
) -> dict:
    data = {
        "Name": name,
        "Thermal Resistance": thermal_resistance,
    }
    return material_tool.update(name, data).to_mcp_response()

@mcp.tool
def update_glazing_material(
    name: str,
    u_factor: float | None = None,
    solar_heat_gain_coefficient: float | None = None,
    visible_transmittance: float | None = None,
) -> dict:
    data = {
        "Name": name,
        "U-Factor": u_factor,
        "Solar Heat Gain Coefficient": solar_heat_gain_coefficient,
        "Visible Transmittance": visible_transmittance,
    }
    return material_tool.update(name, data).to_mcp_response()

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
    data = {
        "Name": name,
        "Layers": layers,
    }
    return Construction_tool.create(data).to_mcp_response()

@mcp.tool
def get_construction(name: str) -> dict:
    return Construction_tool.read(name).to_mcp_response()

@mcp.tool
def update_construction(
    name: str,
    layers: list[str] | None = None,
) -> dict:
    data = {
        "Name": name,
        "Layers": layers,
    }
    return Construction_tool.update(name, data).to_mcp_response()

@mcp.tool
def delete_construction(name: str) -> dict:
    return Construction_tool.delete(name).to_mcp_response()

@mcp.tool
def list_constructions() -> dict:
    return Construction_tool.list_all().to_mcp_response()

@mcp.tool
def create_surface(
    name: str,
    surface_type: str,
    construction_name: str,
    zone_name: str,
    space_name: str | None = None,
    outside_boundary_condition: str,
    outside_boundary_condition_object: str | None = None,
    sun_exposure: str,
    wind_exposure: str,
    view_factor_to_ground: float | str = "autocalculate",
    number_of_vertices: int | str = "autocalculate",
    vertices: list[dict],
) -> dict:
    data = {
        "Name": name,
        "Surface Type": surface_type,
        "Construction Name": construction_name,
        "Zone Name": zone_name,
        "Space Name": space_name,
        "Outside Boundary Condition": outside_boundary_condition,
        "Outside Boundary Condition Object": outside_boundary_condition_object,
        "Sun Exposure": sun_exposure,
        "Wind Exposure": wind_exposure,
        "View Factor to Ground": view_factor_to_ground,
        "Number of Vertices": number_of_vertices,
        "Vertices": vertices,
    }
    return surface_tool.create(data).to_mcp_response()

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
    data = {
        "Name": name,
        "Surface Type": surface_type,
        "Construction Name": construction_name,
        "Zone Name": zone_name,
        "Space Name": space_name,
        "Outside Boundary Condition": outside_boundary_condition,
        "Outside Boundary Condition Object": outside_boundary_condition_object,
        "Sun Exposure": sun_exposure,
        "Wind Exposure": wind_exposure,
        "View Factor to Ground": view_factor_to_ground,
        "Number of Vertices": number_of_vertices,
        "Vertices": vertices,
    }
    return surface_tool.update(name, data).to_mcp_response()

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
    outside_boundary_condition_object: str | None = None,
    view_factor_to_ground: float | str = "autocalculate",
    frame_and_divider_name: str | None = None,
    multiplier: int = 1,
    number_of_vertices: int | str = "autocalculate",
    vertices: list[dict],
) -> dict:
    data = {
        "Name": name,
        "Surface Type": surface_type,
        "Construction Name": construction_name,
        "Building Surface Name": building_surface_name,
        "Outside Boundary Condition Object": outside_boundary_condition_object,
        "View Factor to Ground": view_factor_to_ground,
        "Frame and Divider Name": frame_and_divider_name,
        "Multiplier": multiplier,
        "Number of Vertices": number_of_vertices,
        "Vertices": vertices,
    }
    return fenestration_tool.create(data).to_mcp_response()

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
    data = {
        "Name": name,
        "Surface Type": surface_type,
        "Construction Name": construction_name,
        "Building Surface Name": building_surface_name,
        "Outside Boundary Condition Object": outside_boundary_condition_object,
        "View Factor to Ground": view_factor_to_ground,
        "Frame and Divider Name": frame_and_divider_name,
        "Multiplier": multiplier,
        "Number of Vertices": number_of_vertices,
        "Vertices": vertices,
    }
    return fenestration_tool.update(name, data).to_mcp_response()

@mcp.tool
def delete_fenestration_surface(name: str) -> dict:
    return fenestration_tool.delete(name).to_mcp_response()

@mcp.tool
def list_fenestration_surfaces() -> dict:
    return fenestration_tool.list_all().to_mcp_response()

@mcp.tool
def export_yaml(output_path: str = "./output/yaml/output.yaml") -> dict:
    return workflow_tool.export_yaml(output_path).to_mcp_response()


@mcp.tool
def load_yaml(input_path: str = "data/schemas/building_schema.yaml") -> dict:
    return workflow_tool.load_yaml(input_path).to_mcp_response()


@mcp.tool
def validate_config() -> dict:
    return workflow_tool.validate_config().to_mcp_response()


@mcp.tool
def run_simulation(
    epw_path: str = "data/weather/Shenzhen.epw", output_dir: str = "./output"
) -> dict:
    return workflow_tool.run_simulation(epw_path, output_dir).to_mcp_response()


@mcp.tool
def get_summary() -> dict:
    return workflow_tool.get_summary().to_mcp_response()


@mcp.tool
def clear_all() -> dict:
    return workflow_tool.clear_all().to_mcp_response()


@mcp.resource("config://current")
def get_current_config() -> str:
    return OmegaConf.to_yaml(state.to_yaml_dict())


@mcp.resource("config://summary")
def get_summary_resource() -> str:
    return OmegaConf.to_yaml(state.get_summary().model_dump())


if __name__ == "__main__":
    mcp.run()
