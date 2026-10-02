from fastmcp import FastMCP

from src.mcp.api import (
    register_core_tools,
    register_envelope_tools,
    register_hvac_tools,
    register_load_tools,
    register_resources,
    register_schedule_tools,
    register_workflow_tools,
)
from src.mcp.tools import WorkflowTool
from src.state.config_state import ConfigState


def create_mcp_server() -> FastMCP:
    """Create the MCP server; every tool edits one shared IDF-backed state."""
    mcp = FastMCP(
        name="EnergyPlus Agent",
        version="0.1.0",
        instructions="EnergyPlus Agent is a tool for building energy simulation.",
    )
    state = ConfigState()
    register_core_tools(mcp, state)
    register_schedule_tools(mcp, state)
    register_envelope_tools(mcp, state)
    register_hvac_tools(mcp, state)
    register_load_tools(mcp, state)
    register_workflow_tools(mcp=mcp, workflow_tool=WorkflowTool(state))
    register_resources(mcp=mcp, state=state)
    return mcp


mcp = create_mcp_server()


if __name__ == "__main__":
    mcp.run()
