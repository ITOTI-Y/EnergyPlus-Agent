import json
from typing import Any, cast

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.agent.trace import TraceCollector, _tool_success


def _msg(content: str, status: str = "success") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="tc1", status=status)


def test_tool_success_reads_json_envelope():
    ok = json.dumps({"success": True, "message": "Zone 'Z1' created successfully."})
    dup = json.dumps({"success": False, "message": "Zone 'Z1' already exists."})

    assert _tool_success(_msg(ok)) is True
    assert _tool_success(_msg(dup)) is False


def test_tool_success_error_status_wins():
    assert _tool_success(_msg("boom", status="error")) is False


def test_tool_success_non_json_content_defaults_to_success():
    assert _tool_success(_msg("plain text mentioning error handling")) is True


def test_collector_wrap_records_duplicate_creation_as_failure():
    collector = TraceCollector(phase="test")
    request = ToolCallRequest(
        tool_call={"name": "create_zone", "args": {"name": "Z1"}, "id": "tc1"},
        tool=None,
        state=None,
        runtime=cast(Any, None),
    )
    payload = json.dumps({"success": False, "message": "Zone 'Z1' already exists."})

    collector.wrap(request, lambda _: _msg(payload))

    assert collector.traces[0]["success"] is False
