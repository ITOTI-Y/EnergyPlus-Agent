from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger
from pydantic import BaseModel, Field

from src.agent._share import language_directive
from src.agent.llm import build_agent
from src.agent.nodes._share import clone_for_phase, invoke_with_self_repair
from src.agent.nodes.zone_validator import run_zone_validator
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_zone_tools
from src.agent.trace import TraceCollector, record_phase_trace, trace_middleware

# Rebuild rounds driven by validator reject reasons. After exhaustion the
# current zones are kept: downstream hvac back-hop and simulate integrity
# checks remain as safety nets, so the pipeline is never blocked here.
MAX_ZONE_VALIDATION_ROUNDS = 3

ZONE_SYSTEM_PROMPT = """You are a thermal zone creation expert for EnergyPlus.
Given zone specifications, create all required zones using create_zone tool.

Rules:
- Zone names must be unique and descriptive, typically '{floor}_{usage}_{direction}'
  (e.g., 'F1_Office_North', 'F2_Corridor').
- Set z_origin to the floor's lower elevation: ground floor = 0,
  floor 2 = first-floor height (e.g., 3.0), etc.
- direction_of_relative_north is 0 unless the description specifies a rotation.
- multiplier is 1 unless the description explicitly duplicates a typical floor.
- After creating all zones, call list_zones once to verify, then stop with
  a one-line summary of zone count and names.
"""


class ZoneResponse(BaseModel):
    """Structured summary returned by the zone phase agent."""

    zone_names: list[str] = Field(description="Names of all zones created")
    summary: str = Field(description="One-line summary of the zone creation result")


def zone_agent(state: AgentState) -> AgentStateUpdate:
    local = clone_for_phase(state)
    tools = make_zone_tools(local)
    collector = TraceCollector(phase="zone")

    agent = build_agent(
        tools=tools,
        system_prompt=ZONE_SYSTEM_PROMPT,
        response_format=ZoneResponse,
        middleware=[trace_middleware(collector)],
    )

    specs = state.intake_output.zone_specs if state.intake_output else state.user_input
    upstream = state.upstream_request
    consumed_upstream = bool(upstream and upstream.get("target") == "zone")
    if consumed_upstream:
        specs = f"{specs}\n\n{upstream['specs']}"

    result = invoke_with_self_repair(
        agent,
        local,
        specs,
        phase="zone",
        is_revision=state.is_revision,
        validation_errors=state.validation_errors,
    )

    # The main agent can silently emit zero tool calls when the LLM gateway
    # misbehaves, leaving 0 zones and no error. The validator compares the
    # created zones against the specs and drives a rebuild on reject.
    final_validation_errors: list[str] = []
    for v_round in range(MAX_ZONE_VALIDATION_ROUNDS):
        decision, reasons = run_zone_validator(specs, local)
        if decision == "approved":
            if v_round > 0:
                logger.info(
                    "[zone] validator approved on round {}/{}",
                    v_round + 1,
                    MAX_ZONE_VALIDATION_ROUNDS,
                )
            break
        logger.info(
            "[zone] validator rejected (round {}/{}): {}",
            v_round + 1,
            MAX_ZONE_VALIDATION_ROUNDS,
            reasons,
        )
        final_validation_errors = list(reasons or [])
        feedback = HumanMessage(
            content=(
                "Zone completeness validation FAILED. The zones you created do "
                "NOT satisfy the specs. Fix these specific problems using "
                "update_zone / delete_zone + create_zone, then call list_zones "
                "to verify:\n"
                + "\n".join(f"  - {r}" for r in (reasons or []))
                + "\n\nDo NOT just acknowledge — actually create/fix the zones."
                + language_directive()
            )
        )
        result = agent.invoke({"messages": [*result["messages"], feedback]})
    else:
        logger.warning(
            "[zone] validation still not approved after {} rounds; proceeding "
            "with current zones",
            MAX_ZONE_VALIDATION_ROUNDS,
        )
        final_validation_errors = [
            f"Zone validation failed after {MAX_ZONE_VALIDATION_ROUNDS} rounds: "
            + "; ".join(final_validation_errors or ["validator did not approve"])
        ]

    response: ZoneResponse | None = result.get("structured_response")
    summary = response.summary if response else "zone done"

    record_phase_trace("zone", collector.export())

    update = AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[zone] {summary}")],
    )
    if final_validation_errors:
        update["validation_errors"] = final_validation_errors
        update["messages"] = [
            *update["messages"],
            AIMessage(content="[zone-validator] " + " ".join(final_validation_errors)),
        ]
    # Drop the consumed back-hop request so it can't be re-injected on retry.
    # An empty dict is the reducer's explicit-clear sentinel (a bare None would
    # be treated as "field omitted" by sibling branches and leave the value).
    if consumed_upstream:
        update["upstream_request"] = {}
    return update
