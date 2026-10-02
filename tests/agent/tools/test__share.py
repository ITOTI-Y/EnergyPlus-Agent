import json
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

import src.agent.tools as agent_tools
from src.state.config_state import ConfigState


def _all_tools() -> list[BaseTool]:
    config = ConfigState()
    return [
        tool
        for name in agent_tools.__all__
        for tool in getattr(agent_tools, name)(config)
    ]


def _bare_objects(schema: Any, path: str) -> Iterator[str]:
    if isinstance(schema, dict):
        if schema.get("type") == "object" and not schema.get("properties"):
            yield path
        for key, value in schema.items():
            yield from _bare_objects(value, f"{path}.{key}")
    elif isinstance(schema, list):
        for i, value in enumerate(schema):
            yield from _bare_objects(value, f"{path}[{i}]")


@pytest.mark.parametrize("tool", _all_tools(), ids=lambda t: t.name)
def test_tool_arguments_declare_object_properties(tool: BaseTool):
    # Gemini fills an object without declared properties with {}, which
    # turned a surface agent into an endless empty-vertex retry loop.
    parameters = convert_to_openai_tool(tool)["function"]["parameters"]

    assert list(_bare_objects(parameters.get("properties", {}), tool.name)) == []


def test_rejected_operation_returns_error_status_with_details():
    tools = {t.name: t for t in agent_tools.make_zone_tools(ConfigState())}
    call = ToolCall(
        name="delete_zone", args={"name": "Nowhere"}, id="1", type="tool_call"
    )

    message = tools["delete_zone"].invoke(call)

    assert message.status == "error"
    assert json.loads(message.content)["message"] == "Zone 'Nowhere' not found."
