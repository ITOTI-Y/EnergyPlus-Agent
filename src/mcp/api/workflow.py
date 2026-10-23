from typing import Any

from fastmcp import FastMCP

from src.mcp.tools import WorkflowTool


def register_workflow_tools(mcp: FastMCP, workflow_tool: WorkflowTool) -> None:
    """Register workflow tools (model export/load, validate, simulate) with the MCP server.

    Args:
        mcp: FastMCP server instance.
        workflow_tool: WorkflowTool instance for workflow operations.
    """

    @mcp.tool
    def export_model(
        output_path: str = "./output/model/output.epJSON",
    ) -> dict[str, Any]:
        """Export the current model to an IDF or epJSON file.

        Args:
            output_path: Output file; the suffix (.idf or .epJSON) selects
                the format.

        Returns:
            MCP response with the exported file path.
        """
        return workflow_tool.export_model(output_path).to_mcp_response()

    @mcp.tool
    def load_model(
        input_path: str = "data/schemas/building_schema.epJSON",
    ) -> dict[str, Any]:
        """Replace the current model with an IDF or epJSON file.

        Args:
            input_path: Model file; the suffix (.idf or .epJSON) selects
                the format.

        Returns:
            MCP response with the configuration summary after loading.
        """
        return workflow_tool.load_model(input_path).to_mcp_response()

    @mcp.tool
    def validate_config() -> dict[str, Any]:
        """Check references, fenestration placement and completeness of the model.

        Returns:
            MCP response with the summary, or the problems found, each tied to
            an object type, name and field.
        """
        return workflow_tool.validate_config().to_mcp_response()

    @mcp.tool
    def run_simulation(
        epw_path: str = "data/weather/Shenzhen.epw",
        output_dir: str = "./output",
    ) -> dict[str, Any]:
        """Run an EnergyPlus simulation with the current configuration.

        Args:
            epw_path: Path to the EPW weather data file; design days are read
                from the .ddy file with the same stem when the model has none.
            output_dir: Directory for simulation output files.

        Returns:
            MCP response with IDF path and output directory.
        """
        return workflow_tool.run_simulation(epw_path, output_dir).to_mcp_response()

    @mcp.tool
    def get_summary() -> dict[str, Any]:
        """Get a summary of the current configuration state.

        Returns:
            MCP response with component counts and key settings.
        """
        return workflow_tool.get_summary().to_mcp_response()

    @mcp.tool
    def clear_all() -> dict[str, Any]:
        """Clear all configuration state, resetting to defaults.

        Returns:
            MCP response confirming the state has been cleared.
        """
        return workflow_tool.clear_all().to_mcp_response()
