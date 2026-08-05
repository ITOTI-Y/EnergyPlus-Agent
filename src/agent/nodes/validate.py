"""Global cross-reference validation + directed-rollback router.

When validate_references() reports errors, classify each by the phase
that owns the broken reference (the phase whose object holds the bad
pointer) and route back to the *earliest* such phase via Command(goto=...).
That phase re-runs with is_revision=True and the global errors injected
into its specs, so it can fix its objects via `update_*` tools. After
max_retries rounds, fall through to human-in-the-loop review.
"""

from typing import Literal

from langchain_core.messages import RemoveMessage
from langgraph.types import Command, interrupt

from src.agent.nodes._share import classify_errors, earliest_phase
from src.agent.state import AgentState

# Every node validate can route to. The phase names must stay in sync with
# PIPELINE_ORDER; spelled out because Literal takes only literal arguments.
_RollbackTarget = Literal[
    "simulate",
    "intake",
    "revise",
    "zone",
    "material",
    "schedule",
    "construction",
    "surface",
    "fenestration",
    "hvac",
    "people",
    "lights",
]
RollbackCommand = Command[_RollbackTarget]


def validate_node(state: AgentState) -> RollbackCommand:
    """Validate full config; directed-rollback on error up to max_retries.

    Routing strategy:
    - errors + retries remaining + classifiable -> goto the earliest
      owning phase (directed rollback). is_revision is forced True and
      validation_errors are surfaced so the phase fixes its own objects
      via update_* rather than recreating them.
    - errors + retries remaining + unclassifiable -> goto the entry node
      (intake for first-run, revise for revision turns) for a full rebuild.
    - clean OR retries exhausted -> interrupt() for human review.
        - approved  -> goto simulate
        - rejected  -> goto intake/revise with human feedback
    """
    errors = state.config_state.validate_references()

    if errors and state.retry_count < state.max_retries:
        grouped = classify_errors(errors)
        target = earliest_phase(set(grouped.keys()))

        clear_messages = [
            RemoveMessage(id=m.id) for m in state.messages if m.id is not None
        ]

        if target:
            return RollbackCommand(
                goto=target,
                update={
                    "validation_errors": errors,
                    "retry_count": state.retry_count + 1,
                    "is_revision": True,
                    "messages": clear_messages,
                },
            )

        entry = "revise" if state.is_revision else "intake"
        return RollbackCommand(
            goto=entry,
            update={
                "validation_errors": errors,
                "retry_count": state.retry_count + 1,
                "messages": clear_messages,
            },
        )

    summary = state.config_state.get_summary()
    decision = interrupt(
        {
            "summary": summary.model_dump(),
            "errors": errors,
            "message": "Review configuration before simulation. "
            "Respond with {'approved': True} or "
            "{'approved': False, 'feedback': '...', 'errors': [...]}.",
        }
    )

    if decision.get("approved"):
        return RollbackCommand(goto="simulate")

    entry = "revise" if state.is_revision else "intake"
    return RollbackCommand(
        goto=entry,
        update={
            "user_input": decision.get("feedback", state.user_input),
            "validation_errors": decision.get("errors", []),
            "retry_count": 0,
            "messages": [
                RemoveMessage(id=m.id) for m in state.messages if m.id is not None
            ],
        },
    )
