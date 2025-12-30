from fastmcp import FastMCP
from omegaconf import OmegaConf

from src.mcp.state import ConfigState
from src.mcp.tools import (
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
workflow_tool = WorkflowTool(state)


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
