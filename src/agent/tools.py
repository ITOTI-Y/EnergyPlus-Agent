from langchain_core.tools import tool

from src.mcp.state import ConfigState
from src.mcp.tools import ZoneTool


def create_geometry_tools(config_state: ConfigState) -> list:

    zone_tool = ZoneTool(config_state)

    @tool
    def create_zone(
        name: str,
        x_origin: float,
        y_origin: float,
        z_origin: float,
        direction_of_relative_north: float | None = None,
    ):
        result = zone_tool.create(
            {
                "name": name,
                "x_origin": x_origin,
                "y_origin": y_origin,
                "z_origin": z_origin,
                "direction_of_relative_north": direction_of_relative_north,
            }
        )
        return result.to_mcp_response()

    return [create_zone]
