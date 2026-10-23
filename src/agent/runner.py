from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from loguru import logger

from src.agent.nodes.validate import Abort, Approve, ReviewDecision, Revise
from src.agent.state import AgentState, SimContext
from src.agent.trace import reset_traces

type InterruptHandler = Callable[[dict[str, Any]], ReviewDecision]
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
            interrupt payload `{"summary": ..., "errors": [...], "message": ...}`
            and returns `Approve()`, `Revise(feedback)` or `Abort()`.
        on_event: Optional debug hook called as `on_event(node_name, update)`
            after every non-interrupt node. `update` is the partial state
            dict that node returned. Default None = silent.

    Returns:
        Final state dict after END (pulled from the checkpointer).
    """
    reset_traces()
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


def interactive_approval(payload: dict[str, Any]) -> ReviewDecision:
    """Console approval prompt for the validate interrupt.

    Prints a one-line ConfigSummary + any validation problems, then reads
    y / yes to approve, q / quit to abort, or any other text as feedback.
    """

    summary = payload["summary"]
    errors = payload["errors"]

    logger.info(
        "validate interrupt: zones={} materials={} surfaces={} fenestrations={}",
        summary["zones_count"],
        summary["materials_count"],
        summary["surfaces_count"],
        summary["fenestrations_count"],
    )
    if errors:
        for e in errors:
            logger.warning("validation problem: {}", e)
    else:
        logger.info("no validation problems")

    answer = input("Approve? [y = approve, q = abort, or feedback text] > ").strip()
    match answer.lower():
        case "y" | "yes":
            return Approve()
        case "q" | "quit":
            return Abort()
        case _:
            return Revise(answer)


def auto_approval(payload: dict[str, Any]) -> ReviewDecision:
    """Approve a model without errors; abort when errors remain.

    Validate interrupts with errors only after its retries are spent, so
    another revision would repeat them.
    """
    return Abort() if payload["errors"] else Approve()


def print_final_messages(state: dict[str, Any], n: int = 5) -> None:
    for m in state.get("messages", [])[-n:]:
        logger.info("final {}: {}", m.type, str(m.content)[:200])
