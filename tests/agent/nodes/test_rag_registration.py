from types import ModuleType
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pytest import MonkeyPatch

from src.agent.nodes import construction, hvac, material, schedule
from src.agent.state import AgentState
from src.mcp.state import ConfigState


class _FakeTraceCollector:
    def __init__(self, phase: str) -> None:
        self.phase = phase

    def export(self) -> dict[str, str]:
        return {"phase": self.phase}


def _assert_agent_registers_rag(
    monkeypatch: MonkeyPatch,
    module: ModuleType,
    agent_name: str,
    factory_name: str,
) -> None:
    local = ConfigState()
    fake_rag = object()
    captured: dict[str, Any] = {}

    def fake_make_tools(config: ConfigState, rag: object | None = None) -> list[Any]:
        captured["factory"] = {"config": config, "rag": rag}
        return []

    def fake_build_react_agent(**kwargs: Any) -> object:
        captured["build"] = kwargs
        return object()

    monkeypatch.setattr(module, "clone_for_phase", lambda state: local)
    monkeypatch.setattr(module, "_get_rag", lambda: fake_rag, raising=False)
    monkeypatch.setattr(module, factory_name, fake_make_tools)
    monkeypatch.setattr(module, "create_llm", lambda: object())
    monkeypatch.setattr(module, "build_react_agent", fake_build_react_agent)
    monkeypatch.setattr(module, "TraceCollector", _FakeTraceCollector)
    monkeypatch.setattr(module, "record_phase_trace", lambda *args: None)
    monkeypatch.setattr(
        module,
        "invoke_with_self_repair",
        lambda *args, **kwargs: {"messages": [AIMessage(content="done")]},
    )
    if hasattr(module, "maybe_backhop"):
        monkeypatch.setattr(module, "maybe_backhop", lambda *args: None)

    getattr(module, agent_name)(AgentState(user_input="spec"))

    assert captured["factory"] == {"config": local, "rag": fake_rag}
    assert "search_energyplus_reference" in captured["build"]["system_prompt"]


@pytest.mark.parametrize(
    ("module", "agent_name", "factory_name"),
    [
        (material, "material_agent", "make_material_tools"),
        (construction, "construction_agent", "make_construction_tools"),
        (schedule, "schedule_agent", "make_schedule_tools"),
        (hvac, "hvac_agent", "make_hvac_tools"),
    ],
)
def test_phase_agents_register_rag_search(
    monkeypatch: MonkeyPatch,
    module: ModuleType,
    agent_name: str,
    factory_name: str,
) -> None:
    _assert_agent_registers_rag(monkeypatch, module, agent_name, factory_name)
