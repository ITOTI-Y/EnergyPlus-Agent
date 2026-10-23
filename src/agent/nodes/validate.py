from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.types import Command, interrupt
from loguru import logger

from src.agent.phases import DEPENDS_ON, Phase, owner, rerun_closure
from src.agent.state import AgentState, IntakeOutput
from src.modeling.validation import ModelIssue

Destination = Literal["simulate", "intake", "plan_rerun", "__end__"]
ValidateCommand = Command[Destination]


@dataclass(frozen=True)
class Approve:
    """Run the simulation."""


@dataclass(frozen=True)
class Revise:
    """Send the model back to intake with the reviewer's correction."""

    feedback: str


@dataclass(frozen=True)
class Abort:
    """End the session with the problems left in the state."""


type ReviewDecision = Approve | Revise | Abort


def _feedback(issues: list[ModelIssue]) -> dict[str, list[str]]:
    by_phase: dict[str, list[str]] = defaultdict(list)
    for issue in issues:
        if (phase := owner(issue)) is not None:
            by_phase[phase].append(str(issue))
    return dict(by_phase)


def _phase_fixable(issue: ModelIssue, intake: IntakeOutput | None) -> bool:
    """Whether rerunning the owning phase, with the same task, can fix it.

    Zones are built from the intake output and surfaces from the zone
    prisms, so their problems need a new intake output; only sloped
    surfaces come from the surface phase's own LLM.
    """
    match owner(issue):
        case None | "zone":
            return False
        case "surface":
            return intake is not None and bool(intake.surface_specs.strip())
        case _:
            return True


def ordered(phases: set[Phase]) -> list[Phase]:
    return [p for p in DEPENDS_ON if p in phases]


def _clear_messages(state: AgentState) -> list[RemoveMessage]:
    return [RemoveMessage(id=m.id) for m in state.messages if m.id is not None]


def validate_node(state: AgentState) -> ValidateCommand:
    """Choose the next step from the problems found in the model.

    1. Problems the owning phases can fix -> rerun those phases and their
       dependants with the problems as feedback (once per intake output).
    2. Otherwise, while global retries remain -> intake revises its output.
    3. Otherwise, or with no problems -> a reviewer decides: `Approve`
       runs the simulation, `Revise` sends its feedback to intake as a
       correction with the retry budget restored, and `Abort` ends the
       session, keeping `validation_errors` for the caller.
    """
    errors = state.validation_errors
    feedback = _feedback(errors)
    if errors:
        logger.warning(
            "validate: {} problems (subgroup retried {}, intake revisions {}):\n{}",
            len(errors),
            state.subgroup_retried,
            state.global_retries,
            "\n".join(f"  - {e}" for e in errors),
        )
        if not state.subgroup_retried and all(
            _phase_fixable(e, state.intake_output) for e in errors
        ):
            owners = {p for e in errors if (p := owner(e)) is not None}
            rerun = rerun_closure(owners) | set(state.unfinished_phases)
            return ValidateCommand(
                goto="plan_rerun",
                update={
                    "pending_phases": ordered(rerun),
                    "phase_feedback": feedback,
                    "subgroup_retried": True,
                },
            )
        if state.global_retries < state.max_global_retries:
            return ValidateCommand(
                goto="intake",
                update={
                    "phase_feedback": feedback,
                    "global_retries": state.global_retries + 1,
                    "messages": _clear_messages(state),
                },
            )

    summary = state.config_state.get_summary()
    decision: ReviewDecision = interrupt(
        {
            "summary": summary.model_dump(),
            "errors": [str(e) for e in errors],
            "message": "Review configuration before simulation. "
            "Resume with Approve(), Revise(feedback) or Abort().",
        }
    )

    match decision:
        case Approve():
            return ValidateCommand(goto="simulate")
        case Revise(feedback=text):
            return ValidateCommand(
                goto="intake",
                update={
                    "review_feedback": text,
                    "phase_feedback": feedback,
                    "global_retries": 0,
                    "messages": _clear_messages(state),
                },
            )
        case Abort():
            return ValidateCommand(
                goto="__end__",
                update={
                    "messages": [
                        AIMessage(
                            content=f"[validate] aborted with {len(errors)} "
                            "unresolved problems"
                        )
                    ]
                },
            )
