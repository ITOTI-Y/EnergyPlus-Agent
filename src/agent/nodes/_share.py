"""Shared helpers for phase-agent nodes.

Kept here (rather than `src/agent/_share.py`) because the scope is
nodes-internal — no other part of the agent package uses these.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from src.agent._share import language_directive
from src.agent.phases import Phase, owner
from src.modeling.validation import model_issues
from src.state.config_state import ConfigState

if TYPE_CHECKING:
    from src.agent.state import AgentState

MAX_SELF_REPAIR_ROUNDS: Final = 2
"""Max extra invokes per phase for cross-ref self-repair.

Two rounds is enough for the LLM to see its own error feedback and
react; repeated failures beyond that point usually mean the intake
specs are broken, which the outer validate loop handles better.
"""


def skipped(state: AgentState, phase: Phase) -> bool:
    """Whether the phase sits out this pass; its objects stay as they are."""
    return phase not in state.pending_phases


def with_feedback(specs: str, state: AgentState, phase: Phase) -> str:
    """The phase task, plus the problems its objects had in the previous pass."""
    problems = state.phase_feedback.get(phase)
    if not problems:
        return specs
    listed = "\n".join(f"  - {p}" for p in problems)
    return (
        f"{specs}\n\nThe objects this phase built in the previous attempt had "
        f"these problems, and were removed. Build them again without them:\n"
        f"{listed}"
    )


def last_message_text(result: dict[str, Any]) -> str:
    """Final AI message of an agent run, used when it gave no structured answer."""
    message = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None
    )
    return message.text if message is not None else "no response"


def invoke_with_self_repair(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    local_config: ConfigState,
    specs: str,
    *,
    phase: Phase,
) -> dict[str, Any]:
    """Run a phase agent and make it repair problems in its own objects.

    After each `agent.invoke`, check the model in code (the LLM cannot skip
    it) and send back the problems blamed on object types this phase owns.
    Loop up to MAX_SELF_REPAIR_ROUNDS.

    Problems in other phases' objects are left to the cross-reference nodes:
    this phase has no tools to fix them. A problem in its own object may
    still name a missing upstream object, which the LLM should report rather
    than fabricate.

    Args:
        agent: Compiled agent graph from `build_agent`.
        local_config: The deep-copied ConfigState the phase mutates.
        specs: Natural-language task for the phase (from intake_output).
        phase: Phase whose objects are checked; also used in logs.

    Returns:
        The final agent result dict (shape {"messages": [...]}, plus
        "structured_response" when the agent declares a response_format).
    """
    messages: list[AnyMessage] = [HumanMessage(content=specs)]

    for attempt in range(MAX_SELF_REPAIR_ROUNDS + 1):
        result = agent.invoke({"messages": messages})
        errors = [i for i in model_issues(local_config.idf) if owner(i) == phase]

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
                "Validation found problems in the objects you created:\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\n\nFix the objects YOU just created: `delete_<x>` the "
                "broken object, then `create_<x>` it again. If the broken "
                "reference names an upstream "
                "resource (zone / schedule / material / construction / "
                "surface) that truly does not exist, report it in your "
                "final message and do NOT fabricate a replacement — "
                "upstream phases own those objects." + language_directive()
            )
        )
        messages = [*list(result["messages"]), feedback]

    return result
