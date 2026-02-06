from fastmcp import FastMCP
from omegaconf import OmegaConf

from src.mcp.state import ConfigState


def register_resources(mcp: FastMCP, state: ConfigState) -> None:
    @mcp.resource("config://current")
    def get_current_config() -> str:
        return OmegaConf.to_yaml(state.to_yaml_dict())

    @mcp.resource("config://summary")
    def get_summary_resource() -> str:
        return OmegaConf.to_yaml(state.get_summary().model_dump())
