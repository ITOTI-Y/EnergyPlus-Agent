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

Destination = Literal["simulate", "intake"]
ValidateCommand = Command[Destination]


def validate_node(state: AgentState) -> ValidateCommand:
    """Validate full config; auto-retry on error up to max_retries; else HITL.

    Return behavior:
    - errors + retries remaining -> goto intake with error feedback
    - clean or retries exhausted  -> interrupt() for human review
        - approved -> goto simulate
        - rejected -> goto intake with human feedback
    """
    errors = state.config_state.validate_references()

    if errors and state.retry_count < state.max_retries:
        return ValidateCommand(
            goto="intake",
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
        return ValidateCommand(goto="simulate")

    return ValidateCommand(
        goto="intake",
        update={
            "user_input": decision.get("feedback", state.user_input),
            "validation_errors": decision.get("errors", []),
            "retry_count": 0,
            "messages": [
                RemoveMessage(id=m.id) for m in state.messages if m.id is not None
            ],
        },
    )
