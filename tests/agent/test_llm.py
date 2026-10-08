import threading
import time
from collections.abc import Callable, Iterator
from itertools import count
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.agent.llm import (
    MAX_CONSECUTIVE_FAILURES,
    MAX_REPEATED_FAILURES,
    MAX_TOTAL_FAILURES,
    FailureLoopGuard,
    create_llm,
    serial_write_tools_middleware,
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
    llm = create_llm(_config())
    assert isinstance(llm, ChatOpenAI)
    assert llm.extra_body is None


def test_create_llm_sends_the_thinking_budget_as_reasoning_max_tokens():
    llm = create_llm(_config(reasoning_max_tokens=8192))
    assert isinstance(llm, ChatOpenAI)
    assert llm.extra_body == {"reasoning": {"max_tokens": 8192}}


def test_create_llm_passes_configured_retries_and_timeout():
    llm = create_llm(_config(max_retries=5, timeout=30.0))
    assert isinstance(llm, ChatOpenAI)
    assert llm.max_retries == 5
    assert llm.request_timeout == 30.0


def _request(tool_name: str) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": tool_name, "args": {}, "id": "tc1"},
        tool=None,
        state=None,
        runtime=cast(Any, None),
    )


class _OverlapProbe:
    """Handler that records whether two executions ever overlapped."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def __call__(self, _request: ToolCallRequest) -> ToolMessage:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        time.sleep(0.05)
        with self._lock:
            self._active -= 1
        return ToolMessage(content="ok", tool_call_id="tc1")


def _run_concurrently(middleware, tool_name: str) -> int:
    probe = _OverlapProbe()
    threads = [
        threading.Thread(
            target=middleware.wrap_tool_call, args=(_request(tool_name), probe)
        )
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return probe.max_active


def test_write_tools_serialized_within_one_agent():
    middleware = serial_write_tools_middleware()
    assert _run_concurrently(middleware, "create_zone") == 1


def test_read_tools_not_serialized():
    middleware = serial_write_tools_middleware()
    assert _run_concurrently(middleware, "list_zones") == 2


class _ScriptedModel(GenericFakeChatModel):
    """Fake chat model that accepts tools, as create_agent requires."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self


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
    # The reported loop: the same surface with empty vertices, resent forever.
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
