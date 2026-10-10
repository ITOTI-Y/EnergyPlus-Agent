from typing import Any

from langgraph.types import Overwrite
from loguru import logger

from src.agent.phases import remove_phase_objects
from src.agent.state import AgentState


def plan_rerun_node(state: AgentState) -> dict[str, Any]:
    """Remove the objects of the phases about to run, before they run.

    Runs alone between intake or validate and the phase fan-out. The model
    is written with ``Overwrite`` because the reducer merges parallel
    branches as a union, which would bring the removed objects back.
    """
    config = state.config_state.model_copy(deep=True)
    removed = remove_phase_objects(config.idf, set(state.pending_phases))
    logger.info(
        "Rerunning phases {}; removed {} objects", state.pending_phases, len(removed)
    )
    # A plain dict: AgentStateUpdate types the fields by value, not Overwrite.
    return {
        "config_state": Overwrite(config),
        "build_issues": Overwrite([]),
        "validation_errors": [],
    }
