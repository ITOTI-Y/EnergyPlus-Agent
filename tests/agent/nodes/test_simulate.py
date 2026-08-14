from types import SimpleNamespace
from typing import cast

from langgraph.runtime import Runtime

from src.agent.nodes import simulate as simulate_module
from src.agent.state import AgentState, SimContext
from src.mcp.interface import ToolResponse


def test_simulate_node_does_not_reuse_stale_energyplus_errors(
    monkeypatch, tmp_path
) -> None:
    err_path = tmp_path / "eplusout.err"
    err_path.write_text("** Severe ** stale failure\n", encoding="utf-8")

    class SuccessfulWorkflow:
        def __init__(self, config) -> None:
            self.config = config

        def run_simulation(self, *, epw_path: str, output_dir: str) -> ToolResponse:
            assert not err_path.exists()
            return ToolResponse(success=True, message="current run succeeded", data={})

    monkeypatch.setattr(simulate_module, "WorkflowTool", SuccessfulWorkflow)
    runtime = cast(
        Runtime[SimContext],
        SimpleNamespace(
            context=SimContext(epw_path=tmp_path / "weather.epw", output_dir=tmp_path)
        ),
    )
    result = simulate_module.simulate_node(AgentState(), runtime)

    assert "current run succeeded" in result["messages"][0].content
