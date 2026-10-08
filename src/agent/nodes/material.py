from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from src.agent.llm import build_agent
from src.agent.nodes._share import last_message_text, skipped, with_feedback
from src.agent.state import AgentState, AgentStateUpdate, SimContext
from src.agent.tools import make_material_tools
from src.agent.tools.reference_tools import make_reference_tools, reference_prompt
from src.agent.trace import TraceCollector, record_phase_trace, trace_middleware

MATERIAL_SYSTEM_PROMPT = """You are a building material expert for EnergyPlus.
Given material specifications, create all required materials.

Choose the correct material type:
- create_standard_material for solid opaque layers with thermal mass
  (brick, concrete, insulation board, gypsum). Requires thickness,
  conductivity (W/m-K), density (kg/m^3), specific heat (J/kg-K).
- create_nomass_material when only R-value is known (thin finishes, membranes).
- create_airgap_material for enclosed air cavities in OPAQUE wall/roof
  assemblies only. Never use it in a window.
- Windows, either:
  * create_glazing_material for a simplified window given as a whole:
    u_factor (W/m^2-K), solar_heat_gain_coefficient (0-1), optional
    visible_transmittance (0-1). It is used as the only layer.
  * or, for an explicit multi-pane window, create_window_glazing_material
    for each glass pane (thickness in m) and create_window_gas_material for
    the gap between panes (Air / Argon / Krypton / Xenon, thickness in m).

Rules:
- Use the material names the specification gives, verbatim; constructions
  reference them. Name other materials uniquely and self-describing with
  letters, digits and '_' only (e.g., 'Brick_100mm', 'EPS_Insulation_R5',
  'Window_U1p8_SHGC0p4').
- Roughness options: VeryRough, Rough, MediumRough, MediumSmooth, Smooth, VerySmooth.
- Use typical ASHRAE values when the description is vague.
- Call list_materials once at the end to verify.
"""


def material_agent(state: AgentState, runtime: Runtime[SimContext]) -> AgentStateUpdate:
    if skipped(state, "material"):
        return AgentStateUpdate()
    local = state.config_state.model_copy(deep=True)
    tools = make_material_tools(local)
    prompt = MATERIAL_SYSTEM_PROMPT
    if (reference := runtime.context.reference) is not None:
        tools += make_reference_tools(reference, ("material", "construction"))
        prompt += reference_prompt(reference, ("material", "construction"))
    collector = TraceCollector(phase="material")

    agent = build_agent(
        tools=tools,
        system_prompt=prompt,
        middleware=[trace_middleware(collector)],
    )

    specs = (
        state.intake_output.material_specs if state.intake_output else state.user_input
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content=with_feedback(specs, state, "material"))]}
    )

    summary = last_message_text(result)

    record_phase_trace("material", collector.export())
    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[material] {summary}")],
    )
