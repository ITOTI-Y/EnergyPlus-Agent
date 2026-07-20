import json

from idfpy.models import Zone
from langchain_core.messages import ToolMessage

from src.agent.nodes import zone_validator as zv
from src.mcp.state import ConfigState
from src.validator import ZoneSchema


class _FakeValidatorAgent:
    def invoke(self, state, config=None):
        content = state.messages[0].content
        payload = json.loads(
            content.split("ZONES ACTUALLY CREATED (read from the model):\n", 1)[1]
        )
        names = {z.get("name") for z in payload}
        if not payload:
            result = {
                "success": False,
                "message": "zones rejected",
                "data": {"reasons": ["0 zones created but specs require zones."]},
            }
        elif "F1_Corridor" not in names:
            result = {
                "success": False,
                "message": "zones rejected",
                "data": {
                    "reasons": [
                        "Specs require zone 'F1_Corridor' but it was not created."
                    ]
                },
            }
        else:
            result = {
                "success": True,
                "message": "zones approved",
                "data": {"reason": "all required zones were created"},
            }
        return {
            "messages": [
                ToolMessage(content=json.dumps(result), tool_call_id="verdict")
            ]
        }


class _NoVerdictAgent:
    def invoke(self, state, config=None):
        return {"messages": []}


def _state_with_zones(*names: str) -> ConfigState:
    state = ConfigState()
    for name in names:
        state.zones.append(ZoneSchema.model_validate({"Name": name}))
        state.idf.add(Zone(name=name))
    return state


def test_zone_validator_rejects_zero_zones(monkeypatch):
    monkeypatch.setattr(zv, "build_react_agent", lambda **kwargs: _FakeValidatorAgent())

    decision, reasons = zv.run_zone_validator(
        "Create two zones: F1_Office and F1_Corridor",
        ConfigState(),
        llm=object(),
    )

    assert decision == "rejected"
    assert any("0 zones" in r for r in reasons)


def test_zone_validator_rejects_missing_named_zone(monkeypatch):
    monkeypatch.setattr(zv, "build_react_agent", lambda **kwargs: _FakeValidatorAgent())

    decision, reasons = zv.run_zone_validator(
        "Create two zones: F1_Office and F1_Corridor",
        _state_with_zones("F1_Office"),
        llm=object(),
    )

    assert decision == "rejected"
    assert any("F1_Corridor" in r for r in reasons)


def test_zone_validator_approves_matching_zones(monkeypatch):
    monkeypatch.setattr(zv, "build_react_agent", lambda **kwargs: _FakeValidatorAgent())

    decision, reasons = zv.run_zone_validator(
        "Create two zones: F1_Office and F1_Corridor",
        _state_with_zones("F1_Office", "F1_Corridor"),
        llm=object(),
    )

    assert decision == "approved"
    assert reasons is None


def test_zone_validator_fails_closed_without_verdict(monkeypatch):
    monkeypatch.setattr(zv, "build_react_agent", lambda **kwargs: _NoVerdictAgent())

    decision, reasons = zv.run_zone_validator(
        "Create one zone: F1_Office",
        _state_with_zones("F1_Office"),
        llm=object(),
    )

    assert decision == "rejected"
    assert "Validator did not issue" in reasons[0]
