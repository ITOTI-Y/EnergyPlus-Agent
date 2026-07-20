from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from src.agent.llm import create_llm
from src.agent.react import ReactState, build_react_agent
from src.agent.state import AgentState, AgentStateUpdate, SimContext
from src.agent.tools.analysis_tools import make_analysis_tools
from src.agent.trace import TraceCollector, record_phase_trace

ANALYZE_SYSTEM_PROMPT = """You are a building energy performance analyst for EnergyPlus.

## Mandatory Analysis Workflow
1. Call get_simulation_status FIRST. If it reports failure, write a short error report and STOP.
2. Call get_available_variables to discover what data was recorded.
3. Thermal comfort: call get_variable_statistics("Zone Mean Air Temperature"),
   then get_comfort_statistics().
4. Energy: call get_energy_summary() first (HTML table); if unavailable, fall back to
   get_variable_statistics for heating/cooling/lighting energy variables from the CSV.
5. Peak loads: call get_peak_hours for temperature (mode="max" and mode="min") and
   any significant demand variables.
6. Write the final structured report.

## Report Structure
### Simulation Overview
Status, elapsed time, number of warnings/severe errors (list up to 5 severe errors if present).

### Thermal Environment
Per zone: annual mean temperature, comfort hours (%), hot hours (%), cold hours (%).
Comfort band: 20-26 °C unless the building description specifies otherwise.

### Energy Consumption
Total electricity (kWh), total natural gas (kWh), EUI (MJ/m²).
End-use breakdown table: Heating / Cooling / Lighting / People / Equipment / …

### Peak Conditions
Top 3 hottest hours (date, zone, temperature °C).
Top 3 coldest hours (date, zone, temperature °C).
Peak HVAC demand hour if available.

### Key Findings and Recommendations
3-5 concise bullet points highlighting the most important results and actionable suggestions.

## Rules
- Use only data returned by the tools — do NOT invent or estimate values.
- If a variable is missing, note it briefly and skip that section.
- Round all numbers to 2 decimal places in the report.
- Energy values must be reported in kWh (the tools convert automatically).
- Temperature values are in °C.
"""


def analyze_node(state: AgentState, runtime: Runtime[SimContext]) -> AgentStateUpdate:
    """Read EnergyPlus output files and produce an energy performance report."""
    ctx = runtime.context
    tools = make_analysis_tools(ctx.output_dir)
    collector = TraceCollector(phase="analyze")

    agent = build_react_agent(
        llm=create_llm(),
        tools=tools,
        system_prompt=ANALYZE_SYSTEM_PROMPT,
        trace_collector=collector,
    )

    # Provide the simulation outcome message as context (if available)
    sim_msg = next(
        (
            str(getattr(m, "content", ""))
            for m in reversed(state.messages)
            if "[simulate]" in str(getattr(m, "content", ""))
        ),
        "",
    )

    specs = (
        f"Output directory: {ctx.output_dir}\n"
        f"Simulation result: {sim_msg}\n"
        f"Building description: {state.user_input[:500]}\n\n"
        "Analyze the simulation results following the mandatory workflow above."
    )

    result = agent.invoke(ReactState(messages=[HumanMessage(content=specs)]))

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "Analysis complete."

    record_phase_trace("analyze", collector.export())
    return AgentStateUpdate(messages=[AIMessage(content=f"[analyze] {summary}")])
