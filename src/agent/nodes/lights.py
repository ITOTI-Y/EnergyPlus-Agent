from langchain_core.messages import AIMessage, HumanMessage

from src.agent.llm import create_llm
from src.agent.react import ReactState, build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_lights_tools
from src.agent.trace import TRACE_STORE, TraceCollector

LIGHTS_SYSTEM_PROMPT = """You are a lighting-load expert for EnergyPlus.
For each specified zone, create a Lights object via create_light.

Rules:
- name convention: '{zone}_Lights'.
- zone_name must reference an existing Zone.
- schedule_name must reference an existing Schedule:Compact (Fraction).
- design_level_calculation_method:
    * 'LightingLevel' -> supply lighting_level (W, absolute)
    * 'Watts/Area' -> supply watts_per_floor_area (W/m^2)
    * 'Watts/Person' -> supply watts_per_person (W/person)
- Typical office LPD: 8-12 W/m^2 (Watts/Area). Use 10 when unspecified.
- fraction_radiant ~ 0.7 for recessed fluorescent/LED, 0.42 for pendant.
- fraction_visible ~ 0.18 for LED.
- Call list_lights once at the end.
"""


def lights_agent(state: AgentState) -> AgentStateUpdate:
    local = state.config_state.model_copy(deep=True)
    tools = make_lights_tools(local)
    collector = TraceCollector(phase="lights")

    agent = build_react_agent(
        llm=create_llm(),
        tools=tools,
        system_prompt=LIGHTS_SYSTEM_PROMPT,
        trace_collector=collector,
    )

    specs = (
        state.intake_output.lights_specs if state.intake_output else state.user_input
    )
    result = agent.invoke(ReactState(messages=[HumanMessage(content=specs)]))

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "lights done"

    TRACE_STORE.setdefault("lights", []).extend(collector.export())
    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[lights] {summary}")],
    )
