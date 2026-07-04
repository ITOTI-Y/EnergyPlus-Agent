import json

from langchain_core.messages import AIMessage, ToolMessage

from src.agent.nodes._share import MAX_SELF_REPAIR_ROUNDS, invoke_with_self_repair
from src.mcp.state import ConfigState


def _missing_zone_result(name: str = "Office_North") -> dict:
    return {
        "messages": [
            ToolMessage(
                content=json.dumps(
                    {
                        "success": False,
                        "message": f"Zone '{name}' not found.",
                        "data": {"missing_ref": "Zone", "missing_name": name},
                    }
                ),
                tool_call_id="missing-zone",
            )
        ]
    }


class _AlwaysMissingAgent:
    def __init__(self):
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        return _missing_zone_result()


class _MissingThenDoneAgent:
    def __init__(self):
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        if self.calls == 1:
            return _missing_zone_result()
        return {"messages": [AIMessage(content="used an existing zone name")]}


def test_missing_ref_delays_backhop_until_repair_budget_exhausted():
    agent = _AlwaysMissingAgent()

    result = invoke_with_self_repair(
        agent,
        ConfigState(),
        "Create HVAC for Office_North",
        phase="hvac",
    )

    assert agent.calls == MAX_SELF_REPAIR_ROUNDS + 1
    assert result["hop_request"]["target"] == "zone"
    assert result["hop_request"]["missing_name"] == "Office_North"


def test_missing_ref_can_self_repair_without_backhop():
    agent = _MissingThenDoneAgent()

    result = invoke_with_self_repair(
        agent,
        ConfigState(),
        "Create HVAC for Office_North",
        phase="hvac",
    )

    assert agent.calls == 2
    assert "hop_request" not in result
