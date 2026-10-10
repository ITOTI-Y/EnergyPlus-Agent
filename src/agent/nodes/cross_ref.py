from src.agent.phases import (
    FOUNDATION_PHASES,
    PHASE_TYPES,
    Phase,
    missing_output_issues,
)
from src.agent.state import AgentState, AgentStateUpdate
from src.modeling.validation import ModelIssue, completeness_issues, model_issues


def check_state(state: AgentState, phases: tuple[Phase, ...]) -> list[ModelIssue]:
    """Problems in the model, plus ``phases`` that created nothing."""
    idf = state.config_state.idf
    issues = model_issues(idf)
    if state.intake_output is not None:
        issues += missing_output_issues(idf, state.intake_output, phases)
    return issues


def cross_ref_foundations_node(state: AgentState) -> AgentStateUpdate:
    """Check after zone, material and schedule; surfaces do not exist yet."""
    return AgentStateUpdate(validation_errors=check_state(state, FOUNDATION_PHASES))


def cross_ref_complete_node(state: AgentState) -> AgentStateUpdate:
    """Full check after every phase has run."""
    idf = state.config_state.idf
    issues = check_state(state, tuple(PHASE_TYPES)) + completeness_issues(idf)
    return AgentStateUpdate(validation_errors=issues)
