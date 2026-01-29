from fastmcp import FastMCP
from omegaconf import OmegaConf

from src.mcp.state import ConfigState
from src.mcp.tools import (
    ConstructionTool,
    FenestrationTool,
    MaterialTool,
    SurfaceTool,
    WorkflowTool,
    ZoneTool,
    SettingTool,
    BuildingTool,
    ThermostatTool,
    IdealLoadsSystemTool,
    ScheduleTool,
    ScheduleTypeLimitsTool,
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
construction_tool = ConstructionTool(state)
surface_tool = SurfaceTool(state)
fenestration_tool = FenestrationTool(state)
setting_tool = SettingTool(state)
building_tool = BuildingTool(state)
schedule_tool = ScheduleTool(state)
schedule_limits_tool = ScheduleTypeLimitsTool(state)
hvac_thermostat_tool = ThermostatTool(state)
hvac_ideal_tool = IdealLoadsSystemTool(state)


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
        "Type": "Standard",
        "Roughness": roughness,
        "Thickness": thickness,
        "Conductivity": conductivity,
        "Density": density,
        "Specific_Heat": specific_heat,
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
        "Type": "NoMass",
        "Roughness": roughness,
        "Thermal_Resistance": thermal_resistance,
    }
    return material_tool.create(data).to_mcp_response()


@mcp.tool
def create_air_gap_material(
    name: str,
    thermal_resistance: float,
) -> dict:
    data = {
        "Name": name,
        "Type": "AirGap",
        "Thermal_Resistance": thermal_resistance,
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
        "Type": "Glazing",
        "U-Factor": u_factor,
        "Solar_Heat_Gain_Coefficient": solar_heat_gain_coefficient,
        "Visible_Transmittance": visible_transmittance,
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
        "Type": "Standard",
        "Roughness": roughness,
        "Thickness": thickness,
        "Conductivity": conductivity,
        "Density": density,
        "Specific_Heat": specific_heat,
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
        "Type": "NoMass",
        "Roughness": roughness,
        "Thermal_Resistance": thermal_resistance,
    }
    return material_tool.update(name, data).to_mcp_response()


@mcp.tool
def update_air_gap_material(
    name: str,
    thermal_resistance: float | None = None,
) -> dict:
    data = {
        "Name": name,
        "Type": "AirGap",
        "Thermal_Resistance": thermal_resistance,
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
        "Type": "Glazing",
        "U-Factor": u_factor,
        "Solar_Heat_Gain_Coefficient": solar_heat_gain_coefficient,
        "Visible_Transmittance": visible_transmittance,
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
    return construction_tool.create(data).to_mcp_response()


@mcp.tool
def get_construction(name: str) -> dict:
    return construction_tool.read(name).to_mcp_response()


@mcp.tool
def update_construction(
    name: str,
    layers: list[str] | None = None,
) -> dict:
    data = {
        "Name": name,
        "Layers": layers,
    }
    return construction_tool.update(name, data).to_mcp_response()


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
    vertices: list[dict],
    outside_boundary_condition_object: str | None = None,
    view_factor_to_ground: float | str = "autocalculate",
    frame_and_divider_name: str | None = None,
    multiplier: int = 1,
    number_of_vertices: int | str = "autocalculate",
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


@mcp.tool
def update_building(
    north_axis: float | None = None,
    terrain: str | None = None,
    solar_distribution: str | None = None,
    loads_convergence_tolerance_value: float | None = None,
    temperature_convergence_tolerance_value: float | None = None,
    maximum_number_of_warmup_days: int | None = None,
    minimum_number_of_warmup_days: int | None = None
) -> dict:
    """Update global Building parameters."""
    data = {
        "Name": "Building",
        "North Axis": north_axis,
        "Terrain": terrain,
        "Solar Distribution": solar_distribution,
        "Loads Convergence Tolerance Value": loads_convergence_tolerance_value,
        "Temperature Convergence Tolerance Value": temperature_convergence_tolerance_value,
        "Maximum Number of Warmup Days": maximum_number_of_warmup_days,
        "Minimum Number of Warmup Days": minimum_number_of_warmup_days
    }
    if building_tool.storage:
        return building_tool.update("Building", data).to_mcp_response()
    else:
        return building_tool.create(data).to_mcp_response()

@mcp.tool
def get_building() -> dict:
    """Get current Building parameters."""
    if not building_tool.storage:
         return {"success": False, "message": "Building not initialized yet."}
    return building_tool.read("Building").to_mcp_response()


@mcp.tool
def list_settings() -> dict:
    """List all configured global settings (e.g., SimulationControl, Site:Location)."""
    return setting_tool.list_all().to_mcp_response()

@mcp.tool
def get_setting(setting_type: str) -> dict:
    """
    Get a specific global setting.
    Args:
        setting_type: One of 'SimulationControl', 'Site:Location', 'RunPeriod', 'GlobalGeometryRules'.
    """
    return setting_tool.read(setting_type).to_mcp_response()

@mcp.tool
def update_setting(setting_type: str, parameters: dict) -> dict:
    """
    Update or create a global setting.
    Args:
        setting_type: The type name (e.g., 'SimulationControl', 'Site:Location', 'RunPeriod').
        parameters: A dictionary of fields to update (e.g., {"do_zone_sizing_calculation": true}).
    """
    data = parameters.copy()
    data["Name"] = setting_type
    if setting_type in setting_tool.storage:
        return setting_tool.update(setting_type, data).to_mcp_response()
    else:
        return setting_tool.create(data).to_mcp_response()

@mcp.tool
def delete_setting(setting_type: str) -> dict:
    """Reset a setting to default or remove it if optional."""
    return setting_tool.delete(setting_type).to_mcp_response()
@mcp.tool
def create_schedule_type_limits(
    name: str,
    lower_limit_value: float | str | None = None,
    upper_limit_value: float | str | None = None,
    numeric_type: str = "CONTINUOUS",
    unit_type: str = "Dimensionless"
) -> dict:
    """Create ScheduleTypeLimits."""
    data = {
        "Name": name,
        "Lower Limit Value": lower_limit_value if lower_limit_value is not None else "",
        "Upper Limit Value": upper_limit_value if upper_limit_value is not None else "",
        "Numeric Type": numeric_type,
        "Unit Type": unit_type
    }
    return schedule_limits_tool.create(data).to_mcp_response()

@mcp.tool
def list_schedule_type_limits() -> dict:
    return schedule_limits_tool.list_all().to_mcp_response()

@mcp.tool
def delete_schedule_type_limits(name: str) -> dict:
    return schedule_limits_tool.delete(name).to_mcp_response()

@mcp.tool
def create_schedule_compact(
    name: str,
    schedule_type_limits_name: str,
    data_points: list[dict]
) -> dict:
    """
    Create a Schedule:Compact.
    Args:
        data_points: List of nested dicts structure.
        Example: [{"Through": "12/31", "Days": [{"For": "AllDays", "Times": [{"Until": "24:00", "Value": 1.0}]}]}]
    """
    data = {
        "Name": name,
        "Schedule Type Limits Name": schedule_type_limits_name,
        "Data": data_points
    }
    return schedule_tool.create(data).to_mcp_response()

@mcp.tool
def list_schedules() -> dict:
    return schedule_tool.list_all().to_mcp_response()

@mcp.tool
def delete_schedule(name: str) -> dict:
    return schedule_tool.delete(name).to_mcp_response()
@mcp.tool
def create_thermostat(
    name: str,
    heating_setpoint_schedule_name: str,
    cooling_setpoint_schedule_name: str
) -> dict:
    """Create a HVACTemplate:Thermostat."""
    data = {
        "Name": name,
        "Heating Setpoint Schedule Name": heating_setpoint_schedule_name,
        "Cooling Setpoint Schedule Name": cooling_setpoint_schedule_name
    }
    return hvac_thermostat_tool.create(data).to_mcp_response()

@mcp.tool
def delete_thermostat(name: str) -> dict:
    return hvac_thermostat_tool.delete(name).to_mcp_response()

@mcp.tool
def create_ideal_loads_system(
    zone_name: str,
    template_thermostat_name: str,
    system_availability_schedule_name: str | None = None
) -> dict:
    """
    Create an Ideal Loads Air System for a specific Zone.
    Note: The Zone Name serves as the unique identifier for this system.
    """
    data = {
        "Zone Name": zone_name,
        "Template Thermostat Name": template_thermostat_name,
        "System Availability Schedule Name": system_availability_schedule_name
    }
    return hvac_ideal_tool.create(data).to_mcp_response()

@mcp.tool
def delete_ideal_loads_system(zone_name: str) -> dict:
    """Delete Ideal Loads System by Zone Name."""
    return hvac_ideal_tool.delete(zone_name).to_mcp_response()

@mcp.tool
def list_hvac_components() -> dict:
    """List all Thermostats and Ideal Loads Systems."""
    thermostats = hvac_thermostat_tool.list_all().to_mcp_response()
    systems = hvac_ideal_tool.list_all().to_mcp_response()
    return {
        "thermostats": thermostats,
        "ideal_loads_systems": systems
    }

if __name__ == "__main__":
    mcp.run()
