from typing import Final, Literal

from idfpy.models.outputs import OutputVariable
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from src.agent.state import AgentState, SimContext
from src.mcp.tools.workflow import WorkflowTool
from src.modeling.validation import ModelIssue
from src.state.config_state import ConfigState

# Without at least one Output:Variable, EnergyPlus runs the full RunPeriod
# but `eplusout.eso` stays 0 bytes. Applied only when the model has none.
_DEFAULT_OUTPUT_VARIABLES: Final = (
    ("*", "Zone Mean Air Temperature"),
    ("*", "Zone Air Relative Humidity"),
    ("*", "Zone Ideal Loads Supply Air Total Heating Energy"),
    ("*", "Zone Ideal Loads Supply Air Total Cooling Energy"),
    ("*", "Zone Lights Electricity Energy"),
    ("*", "Zone Electric Equipment Electricity Energy"),
    ("*", "Zone People Total Heating Energy"),
    ("", "Facility Total HVAC Electricity Demand Rate"),
)


SimulateCommand = Command[Literal["validate", "__end__"]]


def _ensure_default_output_variables(config: ConfigState) -> None:
    if config.idf.all_of_type(OutputVariable):
        return
    for key, name in _DEFAULT_OUTPUT_VARIABLES:
        config.idf.add(
            OutputVariable(
                key_value=key, variable_name=name, reporting_frequency="Hourly"
            )
        )


def simulate_node(state: AgentState, runtime: Runtime[SimContext]) -> SimulateCommand:
    """Run EnergyPlus on a copy of the model through `WorkflowTool`.

    Severe and Fatal messages go back to validate as problems tied to the
    objects they name; a run that cannot start ends the graph with the
    reason.
    """
    ctx = runtime.context

    config = state.config_state.model_copy(deep=True)
    _ensure_default_output_variables(config)

    workflow = WorkflowTool(config)
    response = workflow.run_simulation(
        epw_path=str(ctx.epw_path.resolve().absolute()),
        output_dir=str(ctx.output_dir.resolve().absolute()),
    )

    message = f"[simulate] {response.message}"
    data = response.data if isinstance(response.data, dict) else {}
    if response.success:
        message += f" idf={data.get('idf_path')}"
    issues = [ModelIssue(**e) for e in data.get("errors", [])]
    if issues:
        return SimulateCommand(
            goto="validate",
            update={
                "validation_errors": issues,
                "messages": [AIMessage(content=message)],
            },
        )
    return SimulateCommand(
        goto="__end__", update={"messages": [AIMessage(content=message)]}
    )
