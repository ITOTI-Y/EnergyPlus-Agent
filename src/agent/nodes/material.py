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
- create_glazing_material for a SIMPLIFIED whole-window model: use this
  when the spec gives an overall U-factor / SHGC / VT for the whole
  window (single pane OR a pre-computed assembly equivalent). This
  produces a WindowMaterial:SimpleGlazingSystem that MUST be the ONLY
  layer of its construction — it cannot be combined with gas gaps or
  other panes (see CRITICAL rule below).
- create_glazing_layer_material for a TRUE per-pane glass layer
  (WindowMaterial:Glazing): use this when you need to assemble a
  multi-pane (double/triple) window from individual glass panes + air
  gaps. It carries thickness + per-pane optical/thermal data, so it CAN
  be composed with create_airgap_material in a layered construction.
  Supply thickness (m), solar_transmittance, visible_transmittance,
  conductivity (W/m-K); other optical fields have sensible defaults.

CRITICAL — GLASS MUST BE A WINDOWMATERIAL, NEVER AN OPAQUE MATERIAL:
- Any material described as glass / glazing / window-pane / clear or tinted
  transparent layer (e.g. 'Float_Glass', 'Single_Glazing', 'Double_Pane')
  MUST be created with create_glazing_material (whole-window) OR
  create_glazing_layer_material (per-pane). NEVER use create_standard_material
  or create_nomass_material for glass.
- An opaque Material named 'Float_Glass' will make EnergyPlus abort with
  "FenestrationSurface has an opaque surface construction; it should have a
  window construction". A glazing construction REQUIRES a WindowMaterial layer.
- Rule of thumb: if the material is meant to be seen through or admits
  daylight, it is glazing.

CRITICAL — SimpleGlazingSystem IS A WHOLE-WINDOW MODEL, NOT A PANE:
- WindowMaterial:SimpleGlazingSystem (from create_glazing_material) collapses
  the ENTIRE window into one U-factor/SHGC/VT object. It MUST be the SOLE
  layer of its construction. Combining it with other layers — e.g.
  [SimpleGlazingSystem + AirGap + SimpleGlazingSystem] for a "double pane" —
  is INVALID: EnergyPlus has no per-pane optical data and aborts with a Fatal
  "Convergence error in SolveForWindowTemperatures" (NaN glass temperatures).
- To build a REAL double/triple-pane window, use per-pane glass layers:
    1. create_glazing_layer_material for each pane (e.g. 'Clear_Glass_3mm'),
    2. create_airgap_material for the gap (e.g. 'Air_Gap_13mm'),
    3. construction phase: create_construction(layers=['Clear_Glass_3mm',
       'Air_Gap_13mm', 'Clear_Glass_3mm']).
- Quick decision: if the spec gives a single U-factor/SHGC for the window,
  use create_glazing_material (one layer). If it describes pane thicknesses
  or a multi-pane makeup, use create_glazing_layer_material (layered).

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
