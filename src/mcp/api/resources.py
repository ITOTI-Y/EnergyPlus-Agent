import json

from fastmcp import FastMCP

from src.state.config_state import ConfigState


def register_resources(mcp: FastMCP, state: ConfigState) -> None:
    """Register MCP resource endpoints for configuration access.

    Args:
        mcp: FastMCP server instance.
        state: Shared ConfigState instance.
    """

    @mcp.resource("config://current")
    def get_current_config() -> str:
        """Get the full current model as epJSON."""
        return json.dumps(state.idf.to_dict(), indent=2)

    @mcp.resource("config://summary")
    def get_summary_resource() -> str:
        """Get component counts and key settings of the current model as JSON."""
        return state.get_summary().model_dump_json(exclude_none=True, indent=2)
