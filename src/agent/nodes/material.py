from langchain_core.messages import AIMessage

from src.agent.llm import create_llm
from src.agent.nodes._share import clone_for_phase, invoke_with_self_repair
from src.agent.react import build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_material_tools
from src.agent.trace import TraceCollector, record_phase_trace

MATERIAL_SYSTEM_PROMPT = """You are a building material expert for EnergyPlus.
Given material specifications, create all required materials.

Choose the correct material type:
- create_standard_material for solid opaque layers with thermal mass
  (brick, concrete, insulation board, gypsum). Requires thickness,
  conductivity (W/m-K), density (kg/m^3), specific heat (J/kg-K).
- create_nomass_material when only R-value is known (thin finishes, membranes).
- create_airgap_material for enclosed air cavities in wall/roof assemblies.
- create_glazing_material for ALL transparent/translucent glazing: windows,
  skylights, glass doors, and any "glass" / "glazing" material. Supply
  u_factor (W/m^2-K), solar_heat_gain_coefficient (0-1), optional
  visible_transmittance (0-1).

CRITICAL — GLASS MUST BE A WINDOWMATERIAL, NEVER AN OPAQUE MATERIAL:
- Any material described as glass / glazing / window-pane / clear or tinted
  transparent layer (e.g. 'Float_Glass', 'Single_Glazing', 'Double_Pane')
  MUST be created with create_glazing_material. This produces a
  WindowMaterial:SimpleGlazingSystem object.
- DO NOT create glass with create_standard_material or create_nomass_material.
  An opaque Material named 'Float_Glass' will make EnergyPlus abort with
  "FenestrationSurface has an opaque surface construction; it should have a
  window construction". A glazing construction REQUIRES a WindowMaterial layer.
- Rule of thumb: if the material is meant to be seen through or admits
  daylight, it is glazing -> create_glazing_material.

Rules:
- Material names must be unique and self-describing (e.g., 'Brick_100mm',
  'EPS_Insulation_R5', 'Window_U1.8_SHGC0.4').
- Roughness options: VeryRough, Rough, MediumRough, MediumSmooth, Smooth, VerySmooth.
- Use typical ASHRAE values when the description is vague.
- Call list_materials once at the end to verify.
"""


def material_agent(state: AgentState) -> AgentStateUpdate:
    local = clone_for_phase(state)
    tools = make_material_tools(local)
    collector = TraceCollector(phase="material")

    agent = build_react_agent(
        llm=create_llm(),
        tools=tools,
        system_prompt=MATERIAL_SYSTEM_PROMPT,
        trace_collector=collector,
    )

    specs = (
        state.intake_output.material_specs if state.intake_output else state.user_input
    )
    # If reached via a back-hop from construction (needed a material), append.
    upstream = state.upstream_request
    if upstream and upstream.get("target") == "material":
        specs = f"{specs}\n\n{upstream['specs']}"
    result = invoke_with_self_repair(
        agent,
        local,
        specs,
        phase="material",
        is_revision=state.is_revision,
        validation_errors=state.validation_errors,
    )

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "material done"

    record_phase_trace("material", collector.export())
    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[material] {summary}")],
    )
