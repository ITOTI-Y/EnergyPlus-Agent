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

from src.agent.nodes._share import classify_errors, earliest_phase, PIPELINE_ORDER
from src.agent.state import AgentState

# Every node name validate can route to. Kept explicit (instead of just
# ["simulate", "intake", "revise"]) so LangGraph recognizes directed
# rollback targets in the Command return type.
_RollbackTarget = Literal[
    "simulate",
    "intake",
    "revise",
    *PIPELINE_ORDER,
]


def validate_node(state: AgentState) -> Command[_RollbackTarget]:
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
            # Directed rollback: hop straight to the phase that owns the
            # broken reference. is_revision=True steers its agent toward
            # update_* / delete_* over full recreation.
            return Command(
                goto=target,
                update={
                    "validation_errors": errors,
                    "retry_count": state.retry_count + 1,
                    "is_revision": True,
                    "messages": clear_messages,
                },
            )

        # Unclassifiable errors: fall back to the entry node for a full
        # re-intake / re-revise pass.
        entry = "revise" if state.is_revision else "intake"
        return Command(
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
        return Command(goto="simulate")

    # On rejection, route back to the entry node matching the current mode:
    # revision turns loop back to revise, first-run turns loop back to intake.
    entry = "revise" if state.is_revision else "intake"
    return Command(
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
