import threading
import time
from typing import Any, cast

from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.agent.llm import create_llm, serial_write_tools_middleware
from src.configs.config import LLMConfig


def _config(**overrides) -> LLMConfig:
    defaults = {
        "provider": "openai",
        "model_name": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 1000,
        "api_key": "test-key",
    }
    return LLMConfig.model_validate({**defaults, **overrides})


def test_create_llm_sets_default_thinking_budget():
    llm = create_llm(_config())
    assert isinstance(llm, ChatOpenAI)
    assert llm.extra_body == {"max_thinking_budget": 4096}


def test_create_llm_respects_configured_thinking_budget():
    llm = create_llm(_config(max_thinking_budget=8192))
    assert isinstance(llm, ChatOpenAI)
    assert llm.extra_body == {"max_thinking_budget": 8192}


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
