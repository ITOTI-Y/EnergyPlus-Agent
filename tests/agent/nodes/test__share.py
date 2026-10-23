from typing import Any, cast, override

import httpx
from idfpy.models import (
    Construction,
    ScheduleCompact,
    ScheduleCompactDataItem,
    ScheduleTypeLimits,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from openai import APIError
from pydantic import BaseModel

from src.agent.nodes._share import (
    MAX_SELF_REPAIR_ROUNDS,
    invoke_with_self_repair,
    structured,
)
from src.state.config_state import ConfigState


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

    invoke_with_self_repair(
        cast(Any, stub), ConfigState(), "specs", phase="construction"
    )

    assert len(stub.payloads) == 1
    payload = stub.payloads[0]
    assert isinstance(payload, dict)
    assert isinstance(payload["messages"][0], HumanMessage)
    assert payload["messages"][0].content == "specs"


def test_self_repair_fixes_missing_schedule_type_limits():
    config = ConfigState()
    config.idf.add(
        ScheduleCompact(
            name="Office_Occupancy",
            schedule_type_limits_name="Fraction",
            data=[
                ScheduleCompactDataItem(field="Through: 12/31"),
                ScheduleCompactDataItem(field="For: AllDays"),
                ScheduleCompactDataItem(field="Until: 24:00"),
                ScheduleCompactDataItem(field="1.0"),
            ],
        )
    )

    class _RepairingAgent(_StubAgent):
        """Creates the missing ScheduleTypeLimits when re-invoked."""

        @override
        def invoke(self, payload: Any) -> dict[str, Any]:
            if self.payloads:
                config.idf.add(
                    ScheduleTypeLimits(
                        name="Fraction",
                        lower_limit_value=0.0,
                        upper_limit_value=1.0,
                        numeric_type="Continuous",
                    )
                )
            return super().invoke(payload)

    stub = _RepairingAgent()

    invoke_with_self_repair(cast(Any, stub), config, "specs", phase="schedule")

    assert len(stub.payloads) == 2
    feedback = stub.payloads[1]["messages"][-1]
    assert isinstance(feedback, HumanMessage)
    assert "Fraction" in feedback.content


def test_self_repair_exhausts_rounds_on_persistent_errors():
    stub = _StubAgent()

    result = invoke_with_self_repair(
        cast(Any, stub), _broken_config(), "specs", phase="construction"
    )

    assert len(stub.payloads) == MAX_SELF_REPAIR_ROUNDS + 1
    feedback = stub.payloads[1]["messages"][-1]
    assert isinstance(feedback, HumanMessage)
    assert "Construction 'C1' field outside_layer" in feedback.content
    assert isinstance(result["messages"][-1], AIMessage)


def test_self_repair_ignores_problems_owned_by_other_phases():
    stub = _StubAgent()

    invoke_with_self_repair(cast(Any, stub), _broken_config(), "specs", phase="surface")

    assert len(stub.payloads) == 1


class _Answer(BaseModel):
    text: str


class _CutOffThenAnswering:
    """A chat model whose first structured reply was stopped at the limit."""

    def __init__(self) -> None:
        self.sent: list[list[Any]] = []

    def with_structured_output(
        self, *args: Any, **kwargs: Any
    ) -> "_CutOffThenAnswering":
        return self

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        self.sent.append(messages)
        if len(self.sent) == 1:
            cut = AIMessage(content="", response_metadata={"finish_reason": "length"})
            return {"raw": cut, "parsed": None, "parsing_error": None}
        return {
            "raw": AIMessage(content=""),
            "parsed": {"text": "ok"},
            "parsing_error": None,
        }


def test_a_reply_cut_off_at_the_token_limit_is_asked_again_more_briefly():
    llm = _CutOffThenAnswering()

    answer, _ = structured(
        cast(Any, llm), _Answer, [HumanMessage(content="q")], lambda a: a
    )

    assert answer.text == "ok"
    retry = llm.sent[1][-1].content
    assert (
        "reached the output token limit before the _Answer tool call was complete"
        in retry
    )


class _FailingMidStreamOnce(_CutOffThenAnswering):
    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        self.sent.append(messages)
        if len(self.sent) == 1:
            request = httpx.Request("POST", "https://llm.invalid/v1/chat/completions")
            raise APIError("The operation was aborted", request, body=None)
        return {"raw": AIMessage(content=""), "parsed": {"text": "ok"}}


def test_a_structured_call_failing_mid_stream_is_made_again_unchanged():
    llm = _FailingMidStreamOnce()
    messages: list[BaseMessage] = [HumanMessage(content="q")]

    answer, _ = structured(cast(Any, llm), _Answer, messages, lambda a: a)

    assert answer.text == "ok"
    assert llm.sent == [messages, messages]
