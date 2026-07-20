from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

from src.agent._share import language_directive
from src.agent.llm import create_llm
from src.agent.nodes._share import clone_for_phase, invoke_with_self_repair
from src.agent.nodes.zone_validator import run_zone_validator
from src.agent.react import ReactState, build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_zone_tools
from src.agent.trace import TraceCollector, record_phase_trace

# Max number of times the main zone agent is re-invoked after the validator
# rejects. Round 0 = the initial build (validated once); rounds 1..N are
# rebuilds driven by reject reasons. After exhaustion the current zones are
# kept and a warning is logged (the pipeline is not blocked — downstream
# hvac back-hop + simulate integrity checks remain as safety nets).
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


def zone_agent(state: AgentState) -> AgentStateUpdate:
    local = clone_for_phase(state)
    tools = make_zone_tools(local)
    collector = TraceCollector(phase="zone")

    # One LLM instance shared by the main zone agent and the validator, so
    # the LLM config YAML is parsed once.
    llm = create_llm()
    agent = build_react_agent(
        llm=llm,
        tools=tools,
        system_prompt=ZONE_SYSTEM_PROMPT,
        trace_collector=collector,
    )

    specs = state.intake_output.zone_specs if state.intake_output else state.user_input
    # If reached via a back-hop from surface (needed a zone), append.
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

    # Post-build completeness validation. The validator compares the zones
    # actually created (read from `local`) against `specs` and either approves
    # or returns concrete reasons. On reject, the reasons are fed back to the
    # main zone agent as a HumanMessage and it is re-invoked, up to
    # MAX_ZONE_VALIDATION_ROUNDS rounds. This catches the failure mode where
    # the main agent's LLM silently produced zero tool calls (zero zones)
    # without raising any error.
    final_validation_errors: list[str] = []
    for v_round in range(MAX_ZONE_VALIDATION_ROUNDS):
        decision, reasons = run_zone_validator(specs, local, llm)
        if decision == "approved":
            if v_round > 0:
                logger.info(
                    "[zone] validator approved on round {}/{}",
                    v_round + 1,
                    MAX_ZONE_VALIDATION_ROUNDS,
                )
            break
        # Rejected — feed reasons back to the main agent and rebuild.
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
        result = agent.invoke(ReactState(messages=[*result["messages"], feedback]))
    else:
        # Exhausted all rounds and still not approved — proceed with whatever
        # zones exist (do not block the pipeline). Downstream hvac back-hop
        # and simulate integrity checks remain as safety nets.
        logger.warning(
            "[zone] validation still not approved after {} rounds; proceeding "
            "with current zones",
            MAX_ZONE_VALIDATION_ROUNDS,
        )
        final_validation_errors = [
            "Zone validation failed after "
            f"{MAX_ZONE_VALIDATION_ROUNDS} rounds: "
            + "; ".join(final_validation_errors or ["validator did not approve"])
        ]

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "zone done"

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
