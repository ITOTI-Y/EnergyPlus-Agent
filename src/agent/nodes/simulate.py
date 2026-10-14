from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.agent.state import AgentState, AgentStateUpdate, SimContext
from src.mcp.tools.workflow import WorkflowTool


def simulate_node(state: AgentState, runtime: Runtime[SimContext]) -> AgentStateUpdate:
    """Run EnergyPlus on the model through `WorkflowTool`."""
    ctx = runtime.context

    workflow = WorkflowTool(state.config_state)
    response = workflow.run_simulation(
        epw_path=str(ctx.epw_path.resolve().absolute()),
        output_dir=str(ctx.output_dir.resolve().absolute()),
    )

    message = f"[simulate] {response.message}"
    if response.success and isinstance(response.data, dict):
        message += f" idf={response.data.get('idf_path')}"

    return AgentStateUpdate(messages=[AIMessage(content=message)])
