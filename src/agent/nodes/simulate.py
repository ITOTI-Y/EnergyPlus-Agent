"""Export YAML -> IDF, run EnergyPlus, and route on failure.

On success: route to ``analyze`` (the normal path).
On failure (``response.success`` is False OR ``eplusout.err`` has Fatal/Severe
lines) with retries remaining: route back to ``revise`` with the extracted
error text so the agent can fix the model and re-run. Once
``sim_retry_count`` reaches ``max_sim_retries``, fall through to ``analyze``
so the failure is still recorded (the run is not left dangling).
"""

from typing import Literal

from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from src.agent.nodes._share import clone_for_phase
from src.agent.state import AgentState, SimContext
from src.mcp.tools.workflow import WorkflowTool
from src.results.err_parser import extract_errors, format_errors_for_llm

# Default Output:Variable set. Without at least one entry, EnergyPlus
# runs the full RunPeriod but `eplusout.eso` stays 0 bytes — nothing is
# recorded. Applied only when `config.output_variable` is empty; users
# / LLM can override by populating it themselves.
_DEFAULT_OUTPUT_VARIABLES: tuple[tuple[str, str, str], ...] = (
    ("*", "Zone Mean Air Temperature", "Hourly"),
    ("*", "Zone Air Relative Humidity", "Hourly"),
    ("*", "Zone Ideal Loads Supply Air Total Heating Energy", "Hourly"),
    ("*", "Zone Ideal Loads Supply Air Total Cooling Energy", "Hourly"),
    ("*", "Zone Lights Electricity Energy", "Hourly"),
    ("*", "Zone People Total Heating Energy", "Hourly"),
    ("", "Facility Total HVAC Electricity Demand Rate", "Hourly"),
    ("*", "Surface Outside Face Incident Solar Radiation Rate per Area", "Hourly"),
)

# Where simulate may route. Kept explicit so LangGraph recognizes the
# Command targets (analyze = normal path, revise = failure-rollback path).
_SimRoute = Literal["analyze", "revise"]


def _ensure_default_output_variables(config) -> None:
    """Populate `config.output_variable` with office-default monitoring set
    if the user / LLM has not specified any."""
    if config.output_variable:
        return
    from src.validator import OutputVariableSchema

    for key, name, freq in _DEFAULT_OUTPUT_VARIABLES:
        config.output_variable.append(
            OutputVariableSchema.model_validate(
                {
                    "Key Value": key,
                    "Variable Name": name,
                    "Reporting Frequency": freq,
                }
            )
        )


def simulate_node(
    state: AgentState, runtime: Runtime[SimContext]
) -> Command[_SimRoute]:
    """Export YAML -> IDF, run EnergyPlus, route on success/failure.

    Failure detection uses two independent signals (either triggers a
    rollback): (1) ``response.success`` is False (EnergyPlus exit != 0 or
    validation error), and (2) ``eplusout.err`` has Fatal/Severe lines —
    because EnergyPlus can exit 0 while still reporting Severe errors that
    invalidate the results.
    """
    ctx = runtime.context

    config = clone_for_phase(state)
    _ensure_default_output_variables(config)

    workflow = WorkflowTool(config)
    response = workflow.run_simulation(
        epw_path=str(ctx.epw_path.resolve().absolute()),
        output_dir=str(ctx.output_dir.resolve().absolute()),
    )

    # --- failure detection ---
    err_path = ctx.output_dir / "eplusout.err"
    err_info = extract_errors(err_path)
    has_error_level = err_info["has_error_level"]
    sim_failed = (not response.success) or has_error_level

    # Clear conversation messages on any rollback to bound context growth.
    clear_messages = [
        RemoveMessage(id=m.id) for m in state.messages if m.id is not None
    ]

    if sim_failed and state.sim_retry_count < state.max_sim_retries:
        # Roll back to revise with the concrete error text so the LLM can
        # fix the model. The error text goes ONLY into simulation_errors
        # (read by revise_node, which injects it into the *_specs prompt so
        # downstream phase agents see the fix instruction). We deliberately
        # do NOT also write validation_errors here: validation_errors has no
        # reducer, and the parallel phase-1 nodes (zone/material/schedule)
        # each write it too, which triggers LangGraph's
        # InvalidUpdateError ("Can receive only one value per step").
        error_block = format_errors_for_llm(err_info)
        if not response.success:
            # Surface BOTH the workflow's own message AND any structured
            # errors it carried. The geometry preflight in particular
            # returns a generic message + specific errors in data["errors"]
            # (e.g. "Zone 'X' has 0 BuildingSurface:Detailed objects") —
            # without surfacing them the LLM only sees "cannot run
            # simulation" and has no idea what to fix.
            if response.message:
                error_block = (error_block + "\n" if error_block else "") + response.message
            if isinstance(response.data, dict):
                preflight_errors = response.data.get("errors") or []
                if preflight_errors:
                    bullet = "\n".join(f"  - {e}" for e in preflight_errors)
                    error_block = (
                        (error_block + "\n" if error_block else "")
                        + "Geometry completeness errors:\n" + bullet
                    )
        errors_for_phase = (
            [error_block] if error_block else ["EnergyPlus simulation failed."]
        )
        return Command(
            goto="revise",
            update={
                "simulation_errors": errors_for_phase,
                "sim_retry_count": state.sim_retry_count + 1,
                "is_revision": True,
                "messages": clear_messages,
            },
        )

    # Success path, OR failure with retries exhausted (fall through so the
    # run completes and the harness can record the failure verdict).
    if sim_failed:
        message = (
            f"[simulate] EnergyPlus simulation failed; sim-retry budget "
            f"exhausted ({state.sim_retry_count}/{state.max_sim_retries}). "
            f"{format_errors_for_llm(err_info)}"
        ).strip()
    else:
        message = f"[simulate] {response.message}"
        if response.success and isinstance(response.data, dict):
            message += f" idf={response.data.get('idf_path')}"

    return Command(
        goto="analyze",
        update={"messages": [AIMessage(content=message)]},
    )
