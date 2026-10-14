from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from src.agent.state import AgentState, SimContext
from src.mcp.tools.workflow import WorkflowTool
from src.modeling.validation import ModelIssue

SimulateCommand = Command[Literal["validate", "__end__"]]


def simulate_node(state: AgentState, runtime: Runtime[SimContext]) -> SimulateCommand:
    """Run EnergyPlus on the model through `WorkflowTool`.

    Severe and Fatal messages go back to validate as problems tied to the
    objects they name, and so does a failed exit without such messages, as
    a problem of the whole model; a run that cannot start ends the graph
    with the reason.
    """
    ctx = runtime.context

    workflow = WorkflowTool(state.config_state)
    response = workflow.run_simulation(
        epw_path=str(ctx.epw_path.resolve().absolute()),
        output_dir=str(ctx.output_dir.resolve().absolute()),
    )

    message = f"[simulate] {response.message}"
    data = response.data if isinstance(response.data, dict) else {}
    if response.success:
        message += f" idf={data.get('idf_path')}"
    issues = [ModelIssue(**e) for e in data.get("errors", [])]
    if not response.success and not issues and "return_code" in data:
        # EnergyPlus ran but failed without a Severe or Fatal message: still
        # a problem to fix, not a finished run.
        issues = [
            ModelIssue(
                None,
                None,
                None,
                f"EnergyPlus exited with code {data['return_code']} without "
                f"Severe or Fatal messages; see {data.get('output_dir')}"
                "/eplusout.err.",
            )
        ]
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
