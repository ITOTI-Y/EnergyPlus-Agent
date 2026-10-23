import time
from collections.abc import Callable, Iterator
from itertools import count
from typing import Any, Final, override

import httpx
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import APIError, BadRequestError
from pydantic import BaseModel

from src.agent import llm as llm_module
from src.agent.llm import (
    MAX_CONSECUTIVE_FAILURES,
    MAX_REPEATED_CALLS,
    MAX_REPEATED_FAILURES,
    MAX_TOTAL_FAILURES,
    REPEAT_NOTICE_AFTER,
    FailureLoopGuard,
    build_agent,
    create_llm,
    transient,
)
from src.agent.tools import make_surface_tools
from src.configs.config import LLMConfig
from src.state.config_state import ConfigState


def _config(**overrides) -> LLMConfig:
    defaults = {
        "provider": "openai",
        "model_name": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 1000,
        "api_key": "test-key",
    }
    return LLMConfig.model_validate({**defaults, **overrides})


def test_create_llm_sends_no_thinking_budget_without_one():
    llm = create_llm(_config(reasoning_max_tokens=None))
    assert isinstance(llm, ChatOpenAI)
    assert llm.extra_body is None


def test_create_llm_sends_the_thinking_budget_as_reasoning_max_tokens():
    llm = create_llm(_config(reasoning_max_tokens=8192))
    assert isinstance(llm, ChatOpenAI)
    assert llm.extra_body == {"reasoning": {"max_tokens": 8192}}


def test_create_llm_streams_replies():
    llm = create_llm(_config())
    assert isinstance(llm, ChatOpenAI)
    assert llm.streaming


def test_create_llm_passes_configured_retries_and_timeout():
    llm = create_llm(_config(max_retries=5, timeout=30.0))
    assert isinstance(llm, ChatOpenAI)
    assert llm.max_retries == 5
    assert llm.request_timeout == 30.0


class _ScriptedModel(GenericFakeChatModel):
    """Fake chat model that accepts tools, as create_agent requires."""

    @override
    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self


class _FailingOnce:
    """Message iterator whose first model call fails with ``error``."""

    def __init__(self, error: Exception, reply: AIMessage) -> None:
        self.error, self.reply, self.calls = error, reply, 0

    def __iter__(self) -> "_FailingOnce":
        return self

    def __next__(self) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            raise self.error
        return self.reply


class Done(BaseModel):
    text: str


DONE: Final = AIMessage(
    content="",
    tool_calls=[{"name": "Done", "id": "call_done", "args": {"text": "done"}}],
)


def _stream_error() -> APIError:
    request = httpx.Request("POST", "https://llm.invalid/v1/chat/completions")
    return APIError("The service is currently unavailable.", request, body=None)


def test_a_call_failing_mid_stream_is_made_again(monkeypatch):
    replies = _FailingOnce(_stream_error(), DONE)
    model = _ScriptedModel(messages=replies)
    monkeypatch.setattr(llm_module, "create_llm", lambda config=None: model)

    agent = build_agent("", [], Done)
    result = agent.invoke({"messages": [("user", "go")]})

    assert result["structured_response"].text == "done"
    assert replies.calls == 2


def test_a_refused_request_is_not_made_again():
    request = httpx.Request("POST", "https://llm.invalid/v1/chat/completions")
    refused = BadRequestError(
        "bad", response=httpx.Response(400, request=request), body=None
    )

    assert transient(_stream_error())
    assert not transient(refused)


def test_tool_calls_of_one_round_run_in_the_model_order(monkeypatch):
    # The first call sleeps longest: only calls run one at a time in the
    # model's order finish as 0, 1, 2.
    ran: list[int] = []

    @tool
    def create_step(i: int) -> str:
        """Record a step."""
        time.sleep(0.05 * (3 - i))
        ran.append(i)
        return "ok"

    round_ = AIMessage(
        content="",
        tool_calls=[
            {"name": "create_step", "id": f"call_{i}", "args": {"i": i}}
            for i in range(3)
        ],
    )
    model = _ScriptedModel(messages=iter([round_, DONE]))
    monkeypatch.setattr(llm_module, "create_llm", lambda config=None: model)

    build_agent("", [create_step], Done).invoke({"messages": [("user", "go")]})

    assert ran == [0, 1, 2]


def _repeating_call(name: str, args_for: Callable[[int], dict]) -> Iterator[AIMessage]:
    for i in count():
        yield AIMessage(
            content="",
            tool_calls=[{"name": name, "id": f"call_{i}", "args": args_for(i)}],
        )


def _run_guarded(calls: Iterator[AIMessage]) -> tuple[FailureLoopGuard, list]:
    guard = FailureLoopGuard()
    agent = create_agent(
        model=_ScriptedModel(messages=calls),
        tools=make_surface_tools(ConfigState()),
        middleware=[guard],
    )
    return guard, agent.invoke({"messages": [("user", "build")]})["messages"]


def _empty_vertex_surface(i: int) -> dict:
    return {
        "name": f"S{i}",
        "surface_type": "Wall",
        "construction_name": "C",
        "zone_name": "Z",
        "outside_boundary_condition": "Outdoors",
        "vertices": [{}, {}, {}],
    }


def test_guard_stops_identical_failing_calls():
    # The same invalid call, resent unchanged.
    guard, messages = _run_guarded(
        _repeating_call("create_surface", lambda i: _empty_vertex_surface(0))
    )

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == MAX_REPEATED_FAILURES
    assert all(m.status == "error" for m in tool_messages)
    assert "vertices.0.X" in str(tool_messages[0].content)
    # The call's arguments are already in the model's own message.
    assert "with kwargs" not in str(tool_messages[0].content)
    assert guard.reason is not None
    assert messages[-1].content.startswith("Stopped: create_surface failed")


def test_guard_notices_then_stops_a_repeated_successful_call():
    # A listing repeated unchanged, as while a needed object is missing.
    guard, messages = _run_guarded(_repeating_call("list_surfaces", lambda i: {}))

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == MAX_REPEATED_CALLS
    noticed = ["will not change" in str(m.content) for m in tool_messages]
    assert noticed == [i + 1 >= REPEAT_NOTICE_AFTER for i in range(len(noticed))]
    assert guard.reason is not None
    assert messages[-1].content.startswith("Stopped: list_surfaces was called 6")


def test_guard_stops_consecutive_failures_with_varying_arguments():
    guard, messages = _run_guarded(
        _repeating_call("create_surface", _empty_vertex_surface)
    )

    assert sum(isinstance(m, ToolMessage) for m in messages) == MAX_CONSECUTIVE_FAILURES
    assert guard.reason is not None
    assert "in a row" in guard.reason


def _failures_between_successes(i: int) -> AIMessage:
    call = (
        {"name": "list_surfaces", "args": {}}
        if i % 2
        else {"name": "create_surface", "args": _empty_vertex_surface(i)}
    )
    return AIMessage(content="", tool_calls=[{**call, "id": f"call_{i}"}])


def test_guard_stops_failures_interleaved_with_successes():
    guard, messages = _run_guarded(_failures_between_successes(i) for i in count())

    failed = [m for m in messages if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(failed) == MAX_TOTAL_FAILURES
    assert guard.reason is not None
    assert "failed in this phase" in guard.reason
