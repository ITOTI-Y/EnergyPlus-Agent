from src.agent.state import AgentState, AgentStateUpdate


def cross_ref_foundations_node(state: AgentState) -> AgentStateUpdate:
    """Cross-ref check after phase 1 (zone + material + schedule).

    Most checks are moot at this stage (no constructions, surfaces, HVAC yet),
    but any early violation of shared identity still surfaces here.
    """
    return AgentStateUpdate(validation_errors=state.config_state.validate_references())


def cross_ref_complete_node(state: AgentState) -> AgentStateUpdate:
    """Full cross-ref check after phase 3 (hvac + people + lights)."""
    return AgentStateUpdate(validation_errors=state.config_state.validate_references())
