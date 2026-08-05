from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, wrap_tool_call
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from loguru import logger


def _tool_success(message: ToolMessage) -> bool:
    """Judge tool success from message status and the tools' JSON envelope.

    Phase tools return `{"success": bool, "message": ..., "data": ...}`;
    failures like duplicate creation carry `success: false` without the
    word "error", so substring matching would misreport them.
    """
    if message.status == "error":
        return False
    try:
        payload = json.loads(str(message.content))
    except (TypeError, ValueError):
        return True
    if isinstance(payload, dict) and isinstance(payload.get("success"), bool):
        return payload["success"]
    return True


class TraceCollector:
    """Collect tool-call traces for one phase agent.

    Plugged into `create_agent` graphs via `trace_middleware(collector)`,
    whose `wrap_tool_call` hook delegates to `self.wrap`.

    Each phase agent should instantiate its own collector to avoid
    cross-phase contamination under parallel execution.
    """

    def __init__(self, phase: str = "unknown") -> None:
        self.phase = phase
        self.traces: list[dict[str, Any]] = []

    def wrap(
        self,
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool execution, record trace, pass through result."""
        tool_name = request.tool_call.get("name", "<unknown>")
        entry: dict[str, Any] = {
            "phase": self.phase,
            "tool_name": tool_name,
            "tool_args": request.tool_call.get("args", {}),
        }

        result = execute(request)

        if isinstance(result, ToolMessage):
            entry["result"] = str(result.content)
            entry["success"] = _tool_success(result)
        else:
            entry["result"] = str(result)
            entry["success"] = True

        self.traces.append(entry)
        logger.debug(
            "Tool trace[{}]: {} -> {}", self.phase, tool_name, entry["success"]
        )
        return result

    def export(self) -> list[dict[str, Any]]:
        """Return a copy of all collected traces."""
        return list(self.traces)

    def clear(self) -> None:
        self.traces.clear()


def trace_middleware(collector: TraceCollector) -> AgentMiddleware:
    """Adapt a collector to a `create_agent` middleware.

    `create_agent` builds its own ToolNode, so `collector.wrap` cannot be
    passed as `wrap_tool_call=`. The middleware `wrap_tool_call` hook takes
    the same `(request, handler)` shape, so the collector plugs in directly.
    """
    factory = wrap_tool_call(name=f"TraceMiddleware_{collector.phase}")
    return factory(collector.wrap)


_trace_store: dict[str, list[dict[str, Any]]] = {}


def record_phase_trace(phase: str, entries: list[dict[str, Any]]) -> None:
    """Append one phase's trace entries to the session-scoped store."""
    _trace_store.setdefault(phase, []).extend(entries)


def export_traces() -> dict[str, list[dict[str, Any]]]:
    """Snapshot the current session's traces, phase -> entries."""
    return {phase: list(entries) for phase, entries in _trace_store.items()}


def reset_traces() -> None:
    """Clear the trace store. Called at the start of every `run_session`."""
    _trace_store.clear()
