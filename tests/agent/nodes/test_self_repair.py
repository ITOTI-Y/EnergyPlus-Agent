from typing import Any, cast

from idfpy.models.constructions import Construction
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.nodes._share import MAX_SELF_REPAIR_ROUNDS, invoke_with_self_repair
from src.mcp.state import ConfigState


class _StubAgent:
    """Records invoke payloads and echoes messages back like create_agent."""

    def __init__(self) -> None:
        self.payloads: list[Any] = []

    def invoke(self, payload: Any) -> dict[str, Any]:
        self.payloads.append(payload)
        messages = (
            payload["messages"] if isinstance(payload, dict) else payload.messages
        )
        return {"messages": [*messages, AIMessage(content="done")]}


def _broken_config() -> ConfigState:
    config = ConfigState()
    config.idf.add(Construction(name="C1", outside_layer="Missing_Mat"))
    return config


def test_self_repair_invokes_with_message_dict():
    stub = _StubAgent()

    invoke_with_self_repair(cast(Any, stub), ConfigState(), "specs", phase="test")

    assert len(stub.payloads) == 1
    payload = stub.payloads[0]
    assert isinstance(payload, dict)
    assert isinstance(payload["messages"][0], HumanMessage)
    assert payload["messages"][0].content == "specs"


def test_self_repair_exhausts_rounds_on_persistent_errors():
    stub = _StubAgent()

    result = invoke_with_self_repair(
        cast(Any, stub), _broken_config(), "specs", phase="test"
    )

    assert len(stub.payloads) == MAX_SELF_REPAIR_ROUNDS + 1
    feedback = stub.payloads[1]["messages"][-1]
    assert isinstance(feedback, HumanMessage)
    assert "Cross-reference validation failed" in feedback.content
    assert isinstance(result["messages"][-1], AIMessage)
