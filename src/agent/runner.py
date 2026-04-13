from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from src.agent.state import AgentState, SimContext

type InterruptDecision = dict[str, Any]
type InterruptHandler = Callable[[dict[str, Any]], InterruptDecision]
type NodeEventHandler = Callable[[str, dict[str, Any]], None]
type AgentGraph = CompiledStateGraph[AgentState, SimContext, AgentState, AgentState]


def run_session(
    graph: AgentGraph,
    initial: AgentState,
    context: SimContext,
    config: RunnableConfig,
    on_interrupt: InterruptHandler,
    on_event: NodeEventHandler | None = None,
) -> dict[str, Any]:
    """Drive `graph` from `initial` until END, handling validate interrupts.

    Args:
        graph: Compiled agent graph from `build_graph()`.
        initial: Seed AgentState (user_input + optional image_paths).
        context: Runtime context (epw_path, output_dir).
        config: RunnableConfig carrying thread_id for checkpointer.
        on_interrupt: Called at each validate interrupt. Receives the
            interrupt payload `{"summary": ..., "errors": [...], "message": ...}`,
            returns `{"approved": bool, ...}`.
        on_event: Optional debug hook called as `on_event(node_name, update)`
            after every non-interrupt node. `update` is the partial state
            dict that node returned. Default None = silent.

    Returns:
        Final state dict after END (pulled from the checkpointer).
    """
    payload: Any = initial
    while True:
        for event in graph.stream(
            payload, config=config, context=context, stream_mode="updates"
        ):
            if on_event is None:
                continue
            for node, update in event.items():
                if node == "__interrupt__":
                    continue
                on_event(node, update)

        snapshot = graph.get_state(config)
        pending = [t.interrupts[0] for t in snapshot.tasks if t.interrupts]
        if not pending:
            return dict(snapshot.values)

        decision = on_interrupt(pending[0].value)
        payload = Command(resume=decision)


def interactive_approval(payload: dict[str, Any]) -> InterruptDecision:
    """Console approval prompt for the validate interrupt.

    Prints a one-line ConfigSummary + any cross-ref errors, then reads
    y / yes to approve or any other text as rejection feedback.
    """

    summary = payload["summary"]
    errors = payload["errors"]

    print("\n===== VALIDATE INTERRUPT =====")
    print(
        f"  zones={summary['zones_count']} materials={summary['materials_count']} "
        f"surfaces={summary['surfaces_count']} "
        f"fenestrations={summary['fenestrations_count']}"
    )
    if errors:
        print("  errors:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  no cross-reference errors")

    answer = input("Approve? [y/N or feedback text] > ").strip()
    if answer.lower() in ["y", "yes"]:
        return {"approved": True}
    else:
        return {"approved": False, "feedback": answer}


def auto_approval(payload: dict[str, Any]) -> InterruptDecision:
    """Auto-approve the validate interrupt.

    Args:
        payload: Payload from the validate interrupt.

    Returns:
        InterruptDecision with "approved": True.
    """
    errors = payload["errors"]
    if not errors:
        return {"approved": True}
    return {
        "approved": False,
        "feedback": "please address errors: " + "; ".join(errors),
    }


def print_final_messages(state: dict[str, Any], n: int = 5) -> None:
    print("\n===== FINAL =====")
    for m in state.get("messages", [])[-n:]:
        content = m.content if hasattr(m, "content") else str(m)
        print(f"  {getattr(m, 'type', 'msg')}: {str(content)[:200]}")
