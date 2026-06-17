"""Shared helpers for phase-agent nodes.

Kept here (rather than `src/agent/_share.py`) because the scope is
nodes-internal — no other part of the agent package uses these.
"""

from __future__ import annotations

from typing import Any, Final

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from src.agent._share import language_directive
from src.agent.react import ReactState
from src.mcp.state import ConfigState

MAX_SELF_REPAIR_ROUNDS: Final = 2
"""Max extra invokes per phase for cross-ref self-repair.

Two rounds is enough for the LLM to see its own error feedback and
react; repeated failures beyond that point usually mean the intake
specs are broken, which the outer validate loop handles better.
"""

REVISION_PREFIX: Final[str] = (
    "IMPORTANT — REVISION MODE. You are MODIFYING an existing model that was "
    "built in a previous turn. The model already contains objects.\n\n"
    "Before doing anything, call the `list_*` tool to see what already "
    "exists (by exact name). Then:\n"
    "- To change an existing object: use `update_*` (PREFERRED — never "
    "delete+recreate an object that only needs a field change).\n"
    "- To remove an object: use `delete_*`.\n"
    "- To add a brand-new object: use `create_*`.\n"
    "- DO NOT recreate objects that already exist and need no change.\n"
    "- If the spec says 'no changes needed', call `list_*` once to confirm "
    "then return immediately without creating/updating/deleting anything.\n\n"
    "Apply only the modifications described below:\n\n"
)
"""Prefix prepended to phase specs when ``is_revision`` is True, steering
phase agents toward incremental update/delete over full recreation."""


def apply_revision_prefix(specs: str, is_revision: bool) -> str:
    """Prepend the revision-mode instruction prefix when applicable."""
    return REVISION_PREFIX + specs if is_revision else specs


def invoke_with_self_repair(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    local_config: ConfigState,
    specs: str,
    *,
    phase: str,
    is_revision: bool = False,
) -> dict[str, Any]:
    """Run a phase ReAct agent and force cross-reference self-repair.

    After each `agent.invoke`, call `local_config.validate_references()`
    in code (not tool — cannot be skipped by the LLM). If errors exist,
    push them back as a HumanMessage and invoke again. Loop up to
    MAX_SELF_REPAIR_ROUNDS.

    Since phase agents only see objects they created + upstream phases'
    outputs (no cross-phase bleed through LangGraph's deep-copy model),
    any error surfaced here is either the LLM referencing a bad name
    (self-repairable) or an upstream resource truly missing (LLM should
    report in summary; outer validate loop handles the recovery).

    Args:
        agent: Compiled ReAct subgraph from `build_react_agent`.
        local_config: The deep-copied ConfigState the phase mutates.
        specs: Natural-language task for the phase (from intake_output).
        phase: Name used in logs ("construction", "surface", ...).
        is_revision: When True, prepend REVISION_PREFIX to steer the agent
            toward update/delete over full recreation.

    Returns:
        The final ReAct result dict (shape {"messages": [...]}).
    """
    full_specs = apply_revision_prefix(specs, is_revision)
    messages: list[AnyMessage] = [HumanMessage(content=full_specs)]

    for attempt in range(MAX_SELF_REPAIR_ROUNDS + 1):
        result = agent.invoke(ReactState(messages=messages))
        errors = local_config.validate_references()

        if not errors:
            if attempt > 0:
                logger.info("[{}] self-repair succeeded on round {}", phase, attempt)
            return result

        if attempt == MAX_SELF_REPAIR_ROUNDS:
            logger.warning(
                "[{}] self-repair exhausted after {} rounds, {} errors remain "
                "— escalating to outer validate loop",
                phase,
                MAX_SELF_REPAIR_ROUNDS,
                len(errors),
            )
            return result

        logger.info(
            "[{}] self-repair round {}: {} cross-ref errors",
            phase,
            attempt + 1,
            len(errors),
        )
        feedback = HumanMessage(
            content=(
                "Cross-reference validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\n\nFix the objects YOU just created: use `update_<x>` to "
                "rename references, or `delete_<x>` + `create_<x>` to "
                "rebuild. If the broken reference names an upstream "
                "resource (zone / schedule / material / construction / "
                "surface) that truly does not exist, report it in your "
                "final message and do NOT fabricate a replacement — "
                "upstream phases own those objects." + language_directive()
            )
        )
        messages = [*list(result["messages"]), feedback]

    return result
