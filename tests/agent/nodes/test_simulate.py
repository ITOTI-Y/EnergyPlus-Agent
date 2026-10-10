from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from langgraph.runtime import Runtime

from src.agent.nodes import simulate as simulate_module
from src.agent.state import AgentState, SimContext
from src.mcp.interface import ToolResponse


def _simulate(monkeypatch: pytest.MonkeyPatch, response: ToolResponse):
    monkeypatch.setattr(
        simulate_module.WorkflowTool, "run_simulation", lambda self, **_: response
    )
    runtime = SimpleNamespace(context=SimContext(epw_path=Path("w.epw")))
    return simulate_module.simulate_node(
        AgentState(user_input="brief"), cast(Runtime[SimContext], runtime)
    )


def test_a_failed_exit_without_severe_messages_goes_back_to_validate(monkeypatch):
    response = ToolResponse(
        success=False,
        message="EnergyPlus reported 0 error message(s).",
        data={"return_code": 1, "output_dir": "run_1", "errors": []},
    )

    command = _simulate(monkeypatch, response)

    assert command.goto == "validate"
    [issue] = command.update["validation_errors"]
    assert issue.object_type is None
    assert "exited with code 1" in issue.message
    assert "run_1/eplusout.err" in issue.message


def test_a_run_that_cannot_start_ends_the_graph(monkeypatch):
    response = ToolResponse(success=False, message="Cannot add design days: x")

    command = _simulate(monkeypatch, response)

    assert command.goto == "__end__"
