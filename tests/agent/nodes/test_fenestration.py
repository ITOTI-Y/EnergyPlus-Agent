from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.agent.nodes import fenestration as fenestration_module
from src.agent.state import AgentState, IntakeOutput
from src.mcp.state import ConfigState


def _intake() -> IntakeOutput:
    return IntakeOutput.model_validate(
        {
            "building": {"Name": "Test_Building"},
            "site_location": {
                "Name": "Test_Location",
                "Latitude": 22.5,
                "Longitude": 114.0,
                "Time Zone": 8.0,
                "Elevation": 10.0,
            },
            "zone_specs": "",
            "material_specs": "",
            "schedule_specs": "",
            "construction_specs": "Create Wall_Const and Win_Const.",
            "surface_specs": "Create South_Wall.",
            "fenestration_specs": "Create one south window on South_Wall.",
            "hvac_specs": "",
            "people_specs": "",
            "lights_specs": "",
        }
    )


class _FakeTraceCollector:
    def __init__(self, phase: str) -> None:
        self.phase = phase

    def export(self) -> dict[str, str]:
        return {"phase": self.phase}


def _patch_fenestration_dependencies(monkeypatch, result, hop=None):
    local = ConfigState()
    agent = object()
    tools = [object()]
    captured = {
        "build_kwargs": None,
        "invoke": None,
        "backhop": None,
        "trace": None,
    }

    def fake_build_react_agent(**kwargs):
        captured["build_kwargs"] = kwargs
        return agent

    def fake_invoke_with_self_repair(agent_arg, local_config, specs, **kwargs):
        captured["invoke"] = {
            "agent": agent_arg,
            "local_config": local_config,
            "specs": specs,
            "kwargs": kwargs,
        }
        return result

    def fake_maybe_backhop(result_arg, state_arg, local_arg, phase):
        captured["backhop"] = {
            "result": result_arg,
            "state": state_arg,
            "local": local_arg,
            "phase": phase,
        }
        return hop

    def fake_record_phase_trace(phase, trace):
        captured["trace"] = {"phase": phase, "trace": trace}

    monkeypatch.setattr(fenestration_module, "clone_for_phase", lambda state: local)
    monkeypatch.setattr(
        fenestration_module, "make_fenestration_tools", lambda config: tools
    )
    monkeypatch.setattr(fenestration_module, "create_llm", lambda: "llm")
    monkeypatch.setattr(
        fenestration_module, "build_react_agent", fake_build_react_agent
    )
    monkeypatch.setattr(
        fenestration_module,
        "invoke_with_self_repair",
        fake_invoke_with_self_repair,
    )
    monkeypatch.setattr(fenestration_module, "maybe_backhop", fake_maybe_backhop)
    monkeypatch.setattr(fenestration_module, "TraceCollector", _FakeTraceCollector)
    monkeypatch.setattr(
        fenestration_module, "record_phase_trace", fake_record_phase_trace
    )
    return local, tools, captured


def test_fenestration_agent_uses_intake_specs_and_returns_summary(monkeypatch):
    result = {
        "messages": [
            AIMessage(content="created draft"),
            AIMessage(content="created one window"),
        ]
    }
    local, tools, captured = _patch_fenestration_dependencies(monkeypatch, result)
    state = AgentState(
        intake_output=_intake(),
        user_input="fallback user prompt",
        is_revision=True,
        validation_errors=["previous validation failure"],
    )

    output = fenestration_module.fenestration_agent(state)

    build_kwargs = captured["build_kwargs"]
    assert build_kwargs["llm"] == "llm"
    assert build_kwargs["tools"] is tools
    assert (
        build_kwargs["system_prompt"] == fenestration_module.FENESTRATION_SYSTEM_PROMPT
    )
    assert build_kwargs["trace_collector"].phase == "fenestration"

    invoke_call = captured["invoke"]
    assert invoke_call["local_config"] is local
    assert invoke_call["specs"] == "Create one south window on South_Wall."
    assert invoke_call["kwargs"] == {
        "phase": "fenestration",
        "is_revision": True,
        "validation_errors": ["previous validation failure"],
    }
    assert captured["backhop"] == {
        "result": result,
        "state": state,
        "local": local,
        "phase": "fenestration",
    }
    assert captured["trace"] == {
        "phase": "fenestration",
        "trace": {"phase": "fenestration"},
    }
    assert output["config_state"] is local
    assert output["upstream_request"] == {}
    assert [message.content for message in output["messages"]] == [
        "[fenestration] created one window"
    ]


def test_fenestration_agent_falls_back_to_user_input_without_intake(monkeypatch):
    result = {"messages": [HumanMessage(content="no final assistant message")]}
    _, _, captured = _patch_fenestration_dependencies(monkeypatch, result)
    state = AgentState(user_input="Create windows from the direct user prompt.")

    output = fenestration_module.fenestration_agent(state)

    assert captured["invoke"]["specs"] == "Create windows from the direct user prompt."
    assert [message.content for message in output["messages"]] == [
        "[fenestration] fenestration done"
    ]


def test_fenestration_agent_returns_backhop_command(monkeypatch):
    result = {
        "messages": [AIMessage(content="missing construction")],
        "hop_request": {
            "target": "construction",
            "missing_ref": "Construction",
            "missing_name": "Win_Const",
        },
    }
    hop = Command(
        goto="construction",
        update={
            "upstream_request": {
                "target": "construction",
                "specs": "Create Win_Const.",
            }
        },
    )
    local, _, captured = _patch_fenestration_dependencies(
        monkeypatch,
        result,
        hop=hop,
    )
    state = AgentState(intake_output=_intake())

    output = fenestration_module.fenestration_agent(state)

    assert output is hop
    assert captured["backhop"] == {
        "result": result,
        "state": state,
        "local": local,
        "phase": "fenestration",
    }
    assert captured["trace"] == {
        "phase": "fenestration",
        "trace": {"phase": "fenestration"},
    }
