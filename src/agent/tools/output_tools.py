import json

from langchain_core.tools import BaseTool, tool

from idfpy.models.outputs import OutputVariable
from src.mcp.state import ConfigState


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def make_output_tools(config: ConfigState) -> list[BaseTool]:
    idf = config._idf

    @tool
    def add_output_variable(
        variable_name: str,
        key_value: str = "*",
        reporting_frequency: str = "Hourly",
    ) -> str:
        """Add an Output:Variable request.

        Args:
            variable_name: Report variable name (e.g., 'Zone Air Temperature').
            key_value: Object key filter; '*' means all matching objects.
            reporting_frequency: Detailed / Timestep / Hourly / Daily / Monthly / RunPeriod.
        """
        try:
            idf.add(OutputVariable(
                key_value=key_value,
                variable_name=variable_name,
                reporting_frequency=reporting_frequency,
            ))
            return _ok(f"Output:Variable '{variable_name}' added successfully.")
        except Exception as e:
            return _err(f"Error adding Output:Variable '{variable_name}': {e}")

    @tool
    def list_output_variables() -> str:
        """List all registered Output:Variable entries."""
        items = [ov.model_dump() for ov in idf.all_of_type("Output:Variable").values()]
        return _ok(f"Listed {len(items)} Output:Variable entries.", items)

    return [add_output_variable, list_output_variables]
