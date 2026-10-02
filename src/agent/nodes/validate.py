from typing import Literal

from langchain_core.messages import RemoveMessage
from langgraph.types import Command, interrupt

from src.agent.state import AgentState

Destination = Literal["simulate", "intake"]
ValidateCommand = Command[Destination]


def validate_node(state: AgentState) -> ValidateCommand:
    """Act on the issues found by the preceding cross-reference node.

    Auto-retry on error up to max_retries; else HITL.

    Return behavior:
    - errors + retries remaining -> goto intake with error feedback
    - clean or retries exhausted  -> interrupt() for human review
        - approved -> goto simulate
        - rejected -> goto intake with human feedback
    """
    errors = state.validation_errors

    if errors and state.retry_count < state.max_retries:
        return ValidateCommand(
            goto="intake",
            update={
                "validation_errors": errors,
                "retry_count": state.retry_count + 1,
                "messages": [
                    RemoveMessage(id=m.id) for m in state.messages if m.id is not None
                ],
            },
        )

    summary = state.config_state.get_summary()
    decision = interrupt(
        {
            "summary": summary.model_dump(),
            "errors": [str(e) for e in errors],
            "message": "Review configuration before simulation. "
            "Respond with {'approved': True} or "
            "{'approved': False, 'feedback': '...'}.",
        }
    )

    if decision.get("approved"):
        return ValidateCommand(goto="simulate")

    return ValidateCommand(
        goto="intake",
        update={
            "user_input": decision.get("feedback", state.user_input),
            "validation_errors": [],
            "retry_count": 0,
            "messages": [
                RemoveMessage(id=m.id) for m in state.messages if m.id is not None
            ],
        },
    )
