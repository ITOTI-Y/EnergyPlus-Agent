from fastmcp import FastMCP
from omegaconf import OmegaConf

from src.mcp.state import ConfigState
from src.mcp.tools import (
    BuildingTool,
    ConstructionTool,
    FenestrationTool,
    IdealLoadsSystemTool,
    LightTool,
    LocationTool,
    MaterialTool,
    PeopleTool,
    ScheduleTool,
    SurfaceTool,
    ThermostatTool,
    WorkflowTool,
    ZoneTool,
)

mcp = FastMCP(
    name="EnergyPlus Agent",
    version="0.1.0",
    instructions="EnergyPlus Agent is a tool for building energy simulation.",
)

state = ConfigState()

zone_tool = ZoneTool(state)
building_tool = BuildingTool(state)
location_tool = LocationTool(state)
workflow_tool = WorkflowTool(state)
material_tool = MaterialTool(state)
construction_tool = ConstructionTool(state)
surface_tool = SurfaceTool(state)
fenestration_tool = FenestrationTool(state)
schedule_tool = ScheduleTool(state)
thermostat_tool = ThermostatTool(state)
ideal_loads_system_tool = IdealLoadsSystemTool(state)
people_tool = PeopleTool(state)
light_tool = LightTool(state)


@mcp.tool
def create_building(
    name: str,
    north_axis: float,
    terrain: str,
) -> dict:
    data = {
        "Name": name,
        "North Axis": north_axis,
        "Terrain": terrain,
    }
    return building_tool.create(data).to_mcp_response()


@mcp.tool
def get_building(name: str) -> dict:
    return building_tool.read(name).to_mcp_response()


@mcp.tool
def update_building(name: str, north_axis: float, terrain: str) -> dict:
    data = {
        "Name": name,
        "North Axis": north_axis,
        "Terrain": terrain,
    }
    return building_tool.update(name, data).to_mcp_response()


@mcp.tool
def delete_building(name: str) -> dict:
    return building_tool.delete(name).to_mcp_response()


@mcp.tool
def list_buildings() -> dict:
    return building_tool.list_all().to_mcp_response()


@mcp.tool
def create_location(
    name: str,
    latitude: float,
    longitude: float,
    time_zone: float,
    elevation: float,
) -> dict:
    data = {
        "Name": name,
        "Latitude": latitude,
        "Longitude": longitude,
        "Time Zone": time_zone,
        "Elevation": elevation,
    }
    return location_tool.create(data).to_mcp_response()


@mcp.tool
def get_location(name: str) -> dict:
    return location_tool.read(name).to_mcp_response()


@mcp.tool
def update_location(
    name: str, latitude: float, longitude: float, time_zone: float, elevation: float
) -> dict:
    data = {
        "Name": name,
        "Latitude": latitude,
        "Longitude": longitude,
        "Time Zone": time_zone,
        "Elevation": elevation,
    }
    return location_tool.update(name, data).to_mcp_response()


@mcp.tool
def delete_location(name: str) -> dict:
    return location_tool.delete(name).to_mcp_response()


@mcp.tool
def list_locations() -> dict:
    return location_tool.list_all().to_mcp_response()


@mcp.tool
def create_schedule_type_limits(
    name: str,
    lower_limit_value: float,
    upper_limit_value: float,
    numeric_type: str,
    unit_type: str,
) -> dict:
    data = {
        "ScheduleTypeLimits": {
            "Name": name,
            "Lower Limit Value": lower_limit_value,
            "Upper Limit Value": upper_limit_value,
            "Numeric Type": numeric_type,
            "Unit Type": unit_type,
        }
    }
    return schedule_tool.create(data).to_mcp_response()


@mcp.tool
def get_schedule_type_limits(name: str) -> dict:
    return schedule_tool.read(name).to_mcp_response()


@mcp.tool
def update_schedule_type_limits(
    name: str,
    lower_limit_value: float,
    upper_limit_value: float,
    numeric_type: str,
    unit_type: str,
) -> dict:
    data = {
        "ScheduleTypeLimits": {
            "Name": name,
            "Lower Limit Value": lower_limit_value,
            "Upper Limit Value": upper_limit_value,
            "Numeric Type": numeric_type,
            "Unit Type": unit_type,
        }
    }
    return schedule_tool.update(name, data).to_mcp_response()


@mcp.tool
def delete_schedule_type_limits(name: str) -> dict:
    return schedule_tool.delete(name).to_mcp_response()


@mcp.tool
def list_schedule_type_limits() -> dict:
    return schedule_tool.list_all().to_mcp_response()


@mcp.tool
def create_schedule_compact(
    name: str,
    schedule_type_limits_name: str,
    times: list[dict],
) -> dict:
    data = {
        "Schedule:Compact": {
            "Name": name,
            "Schedule Type Limits Name": schedule_type_limits_name,
            "Data": times,
        }
    }
    return schedule_tool.create(data).to_mcp_response()


@mcp.tool
def get_schedule_compact(name: str) -> dict:
    return schedule_tool.read(name).to_mcp_response()


@mcp.tool
def update_schedule_compact(
    name: str, schedule_type_limits_name: str, times: list[dict]
) -> dict:
    data = {
        "Schedule:Compact": {
            "Name": name,
            "Schedule Type Limits Name": schedule_type_limits_name,
            "Data": times,
        }
    }
    return schedule_tool.update(name, data).to_mcp_response()


@mcp.tool
def delete_schedule_compact(name: str) -> dict:
    return schedule_tool.delete(name).to_mcp_response()


@mcp.tool
def list_schedule_compacts() -> dict:
    return schedule_tool.list_all().to_mcp_response()


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
def create_hvac_thermostat(
    name: str,
    heating_setpoint_schedule_name: str,
    cooling_setpoint_schedule_name: str,
) -> dict:
    data = {
        "Name": name,
        "Heating Setpoint Schedule Name": heating_setpoint_schedule_name,
        "Cooling Setpoint Schedule Name": cooling_setpoint_schedule_name,
    }
    return thermostat_tool.create(data).to_mcp_response()


@mcp.tool
def get_hvac_thermostat(name: str) -> dict:
    return thermostat_tool.read(name).to_mcp_response()


@mcp.tool
def update_hvac_thermostat(
    name: str,
    heating_setpoint_schedule_name: str | None = None,
    cooling_setpoint_schedule_name: str | None = None,
) -> dict:
    data = {
        "Name": name,
        "Heating Setpoint Schedule Name": heating_setpoint_schedule_name,
        "Cooling Setpoint Schedule Name": cooling_setpoint_schedule_name,
    }
    return thermostat_tool.update(name, data).to_mcp_response()


@mcp.tool
def delete_hvac_thermostat(name: str) -> dict:
    return thermostat_tool.delete(name).to_mcp_response()


@mcp.tool
def list_hvac_thermostats() -> dict:
    return thermostat_tool.list_all().to_mcp_response()


@mcp.tool
def create_hvac_ideal_loads_system(
    zone_name: str,
    template_thermostat_name: str,
    system_availability_schedule_name: str | None = None,
) -> dict:
    data = {
        "Zone Name": zone_name,
        "Template Thermostat Name": template_thermostat_name,
        "System Availability Schedule Name": system_availability_schedule_name,
    }
    return ideal_loads_system_tool.create(data).to_mcp_response()


@mcp.tool
def get_hvac_ideal_loads_system(zone_name: str) -> dict:
    return ideal_loads_system_tool.read(zone_name).to_mcp_response()


@mcp.tool
def update_hvac_ideal_loads_system(
    zone_name: str,
    template_thermostat_name: str | None = None,
    system_availability_schedule_name: str | None = None,
) -> dict:
    data = {
        "Zone Name": zone_name,
        "Template Thermostat Name": template_thermostat_name,
        "System Availability Schedule Name": system_availability_schedule_name,
    }
    return ideal_loads_system_tool.update(zone_name, data).to_mcp_response()


@mcp.tool
def delete_hvac_ideal_loads_system(zone_name: str) -> dict:
    return ideal_loads_system_tool.delete(zone_name).to_mcp_response()


@mcp.tool
def list_hvac_ideal_loads_systems() -> dict:
    return ideal_loads_system_tool.list_all().to_mcp_response()


@mcp.tool
def create_people(
    name: str,
    zone_or_zonelist_or_space_or_spacelist_name: str,
    number_of_people_schedule_name: str,
    activity_level_schedule_name: str,
    number_of_people_calculation_method: str = "People",
    number_of_people: float | None = None,
    people_per_floor_area: float | None = None,
    floor_area_per_person: float | None = None,
    fraction_radiant: float = 0.3,
    sensible_heat_fraction: float | str = "Autocalculate",
) -> dict:
    data = {
        "Name": name,
        "Zone or ZoneList or Space or SpaceList Name": zone_or_zonelist_or_space_or_spacelist_name,
        "Number of People Schedule Name": number_of_people_schedule_name,
        "Number of People Calculation Method": number_of_people_calculation_method,
        "Number of People": number_of_people,
        "People per Floor Area": people_per_floor_area,
        "Floor Area per Person": floor_area_per_person,
        "Fraction Radiant": fraction_radiant,
        "Sensible Heat Fraction": sensible_heat_fraction,
        "Activity Level Schedule Name": activity_level_schedule_name,
    }
    return people_tool.create(data).to_mcp_response()


@mcp.tool
def get_people(name: str) -> dict:
    return people_tool.read(name).to_mcp_response()


@mcp.tool
def update_people(
    name: str,
    zone_or_zonelist_or_space_or_spacelist_name: str | None = None,
    number_of_people_schedule_name: str | None = None,
    activity_level_schedule_name: str | None = None,
    number_of_people_calculation_method: str | None = None,
    number_of_people: float | None = None,
    people_per_floor_area: float | None = None,
    floor_area_per_person: float | None = None,
    fraction_radiant: float | None = None,
    sensible_heat_fraction: float | str | None = None,
) -> dict:
    data = {
        "Name": name,
        "Zone or ZoneList or Space or SpaceList Name": zone_or_zonelist_or_space_or_spacelist_name,
        "Number of People Schedule Name": number_of_people_schedule_name,
        "Number of People Calculation Method": number_of_people_calculation_method,
        "Number of People": number_of_people,
        "People per Floor Area": people_per_floor_area,
        "Floor Area per Person": floor_area_per_person,
        "Fraction Radiant": fraction_radiant,
        "Sensible Heat Fraction": sensible_heat_fraction,
        "Activity Level Schedule Name": activity_level_schedule_name,
    }
    return people_tool.update(name, data).to_mcp_response()


@mcp.tool
def delete_people(name: str) -> dict:
    return people_tool.delete(name).to_mcp_response()


@mcp.tool
def list_people() -> dict:
    return people_tool.list_all().to_mcp_response()


@mcp.tool
def create_light(
    name: str,
    zone_or_zone_list_or_space_or_space_list_name: str,
    schedule_name: str,
    design_level_calculation_method: str = "LightingLevel",
    lighting_level: float | None = None,
    watts_per_floor_area: float | None = None,
    watts_per_person: float | None = None,
    return_air_fraction: float = 0.0,
    fraction_radiant: float = 0.0,
    fraction_visible: float = 0.0,
    fraction_replaceable: float = 1.0,
    end_use_subcategory: str = "General",
) -> dict:
    data = {
        "Name": name,
        "Zone or ZoneList or Space or SpaceList Name": zone_or_zone_list_or_space_or_space_list_name,
        "Schedule Name": schedule_name,
        "Design Level Calculation Method": design_level_calculation_method,
        "Lighting Level": lighting_level,
        "Watts per Floor Area": watts_per_floor_area,
        "Watts per Person": watts_per_person,
        "Return Air Fraction": return_air_fraction,
        "Fraction Radiant": fraction_radiant,
        "Fraction Visible": fraction_visible,
        "Fraction Replaceable": fraction_replaceable,
        "End Use Subcategory": end_use_subcategory,
    }
    return light_tool.create(data).to_mcp_response()


@mcp.tool
def get_light(name: str) -> dict:
    return light_tool.read(name).to_mcp_response()


@mcp.tool
def update_light(
    name: str,
    zone_or_zone_list_or_space_or_space_list_name: str | None = None,
    schedule_name: str | None = None,
    design_level_calculation_method: str | None = None,
    lighting_level: float | None = None,
    watts_per_floor_area: float | None = None,
    watts_per_person: float | None = None,
    return_air_fraction: float | None = None,
    fraction_radiant: float | None = None,
    fraction_visible: float | None = None,
    fraction_replaceable: float | None = None,
    end_use_subcategory: str | None = None,
) -> dict:
    data = {
        "Name": name,
        "Zone or ZoneList or Space or SpaceList Name": zone_or_zone_list_or_space_or_space_list_name,
        "Schedule Name": schedule_name,
        "Design Level Calculation Method": design_level_calculation_method,
        "Lighting Level": lighting_level,
        "Watts per Floor Area": watts_per_floor_area,
        "Watts per Person": watts_per_person,
        "Return Air Fraction": return_air_fraction,
        "Fraction Radiant": fraction_radiant,
        "Fraction Visible": fraction_visible,
        "Fraction Replaceable": fraction_replaceable,
        "End Use Subcategory": end_use_subcategory,
    }
    return light_tool.update(name, data).to_mcp_response()


@mcp.tool
def delete_light(name: str) -> dict:
    return light_tool.delete(name).to_mcp_response()


@mcp.tool
def list_lights() -> dict:
    return light_tool.list_all().to_mcp_response()


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
