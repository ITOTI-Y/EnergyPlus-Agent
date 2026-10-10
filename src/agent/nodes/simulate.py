from typing import Final

from idfpy.models.outputs import OutputVariable
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.agent.state import AgentState, AgentStateUpdate, SimContext
from src.mcp.tools.workflow import WorkflowTool
from src.state.config_state import ConfigState

# Without at least one Output:Variable, EnergyPlus runs the full RunPeriod
# but `eplusout.eso` stays 0 bytes. Applied only when the model has none.
_DEFAULT_OUTPUT_VARIABLES: Final = (
    ("*", "Zone Mean Air Temperature"),
    ("*", "Zone Air Relative Humidity"),
    ("*", "Zone Ideal Loads Supply Air Total Heating Energy"),
    ("*", "Zone Ideal Loads Supply Air Total Cooling Energy"),
    ("*", "Zone Lights Electricity Energy"),
    ("*", "Zone People Total Heating Energy"),
    ("", "Facility Total HVAC Electricity Demand Rate"),
)


def _ensure_default_output_variables(config: ConfigState) -> None:
    if config.idf.all_of_type(OutputVariable):
        return
    for key, name in _DEFAULT_OUTPUT_VARIABLES:
        config.idf.add(
            OutputVariable(
                key_value=key, variable_name=name, reporting_frequency="Hourly"
            )
        )


def simulate_node(state: AgentState, runtime: Runtime[SimContext]) -> AgentStateUpdate:
    """Run EnergyPlus on a copy of the model through `WorkflowTool`."""
    ctx = runtime.context

    config = state.config_state.model_copy(deep=True)
    _ensure_default_output_variables(config)

    workflow = WorkflowTool(config)
    response = workflow.run_simulation(
        epw_path=str(ctx.epw_path.resolve().absolute()),
        output_dir=str(ctx.output_dir.resolve().absolute()),
    )

    message = f"[simulate] {response.message}"
    if response.success and isinstance(response.data, dict):
        message += f" idf={response.data.get('idf_path')}"

    return AgentStateUpdate(messages=[AIMessage(content=message)])
