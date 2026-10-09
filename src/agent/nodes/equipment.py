from langchain_core.messages import AIMessage
from pydantic import Field

from src.agent.llm import build_agent
from src.agent.nodes._share import (
    PhaseReport,
    invoke_with_self_repair,
    last_message_text,
    missing_input_issues,
    skipped,
    with_feedback,
)
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_equipment_tools
from src.agent.trace import TraceCollector, record_phase_trace, trace_middleware

EQUIPMENT_SYSTEM_PROMPT = """You are a plug-load expert for EnergyPlus.
For each specified zone, create an ElectricEquipment object via
create_equipment.

Workflow:
1. FIRST call `list_zones` to see the exact zone names.
2. FIRST call `list_schedules` to see the exact Schedule:Compact names
   (you need an equipment fraction schedule).
3. Create one ElectricEquipment object per zone via `create_equipment`.
4. Call `list_equipment` once at the end to confirm.

Rules:
- `zone_name` and `schedule_name` MUST appear verbatim in the list_zones /
  list_schedules results.
- If a needed zone or schedule is missing, do NOT invent a name and do NOT
  list again: give your final answer at once, with it in `missing_inputs`.
- Use the names the specification gives; otherwise '{zone}_Equipment'.
- design_level_calculation_method:
    * 'EquipmentLevel' -> supply design_level (W, absolute)
    * 'Watts/Area' -> supply watts_per_floor_area (W/m^2)
    * 'Watts/Person' -> supply watts_per_person (W/person)
- Typical office plug load: 8-15 W/m^2 (Watts/Area). Use 10 when unspecified.
- Office equipment: fraction_radiant 0.5, fraction_latent 0, fraction_lost 0,
  unless the specification says otherwise.
"""


class EquipmentResponse(PhaseReport):
    """Structured summary returned by the equipment phase agent."""

    equipment_names: list[str] = Field(
        description="Names of all ElectricEquipment objects created"
    )
    summary: str = Field(
        description="One-line summary of the equipment creation result"
    )


def equipment_agent(state: AgentState) -> AgentStateUpdate:
    if skipped(state, "equipment"):
        return AgentStateUpdate()
    local = state.config_state.model_copy(deep=True)
    tools = make_equipment_tools(local)
    collector = TraceCollector(phase="equipment")

    agent = build_agent(
        tools=tools,
        system_prompt=EQUIPMENT_SYSTEM_PROMPT,
        response_format=EquipmentResponse,
        middleware=[trace_middleware(collector)],
    )

    specs = (
        state.intake_output.equipment_specs if state.intake_output else state.user_input
    )
    result = invoke_with_self_repair(
        agent, local, with_feedback(specs, state, "equipment"), phase="equipment"
    )

    response: EquipmentResponse | None = result.get("structured_response")
    summary = response.summary if response else last_message_text(result)

    record_phase_trace("equipment", collector.export())
    return AgentStateUpdate(
        config_state=local,
        build_issues=missing_input_issues("equipment", response),
        messages=[AIMessage(content=f"[equipment] {summary}")],
    )
