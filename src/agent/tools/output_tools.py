import json
from typing import Literal

from idfpy.models.outputs import OutputVariable
from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def make_output_tools(config: ConfigState) -> list[BaseTool]:

    @tool
    def add_output_variable(
        variable_name: str,
        key_value: str = "*",
        reporting_frequency: Literal[
            "Detailed", "Timestep", "Hourly", "Daily", "Monthly", "RunPeriod"
        ] = "Hourly",
    ) -> str:
        """Add an Output:Variable request.

        Args:
            variable_name: Report variable name (e.g., 'Zone Air Temperature').
            key_value: Object key filter; '*' means all matching objects.
            reporting_frequency: Detailed / Timestep / Hourly / Daily / Monthly / RunPeriod.
        """
        idf = config.idf
        try:
            output_variable = OutputVariable(
                key_value=key_value,
                variable_name=variable_name,
                reporting_frequency=reporting_frequency,
            )
            idf.add(output_variable)
            return _ok(f"Output:Variable '{variable_name}' added successfully.")
        except Exception as e:
            return _err(f"Error adding Output:Variable '{variable_name}': {e}")

    @tool
    def list_output_variables() -> str:
        """List all registered Output:Variable entries."""
        idf = config.idf
        items = [ov.model_dump() for ov in idf.all_of_type(OutputVariable).values()]
        return _ok(f"Listed {len(items)} Output:Variable entries.", items)

    return [add_output_variable, list_output_variables]
