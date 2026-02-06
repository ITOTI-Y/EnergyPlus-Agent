from fastmcp import FastMCP

from src.mcp.tools import WorkflowTool


def register_workflow_tools(mcp: FastMCP, workflow_tool: WorkflowTool) -> None:
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
        epw_path: str = "data/weather/Shenzhen.epw",
        output_dir: str = "./output",
    ) -> dict:
        return workflow_tool.run_simulation(epw_path, output_dir).to_mcp_response()

    @mcp.tool
    def get_summary() -> dict:
        return workflow_tool.get_summary().to_mcp_response()

    @mcp.tool
    def clear_all() -> dict:
        return workflow_tool.clear_all().to_mcp_response()
