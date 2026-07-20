from langchain_core.messages import AIMessage

from src.agent.llm import create_llm
from src.agent.nodes._share import clone_for_phase, invoke_with_self_repair
from src.agent.react import build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_schedule_tools
from src.agent.tools.rag_tools import _get_rag
from src.agent.trace import TraceCollector, record_phase_trace

SCHEDULE_SYSTEM_PROMPT = """You are a scheduling expert for EnergyPlus.
Given schedule specifications, create all ScheduleTypeLimits and
Schedule:Compact objects required by later phases (HVAC, People, Lights).

Required type limits to create first (if referenced):
- 'Fraction' (0.0 to 1.0, CONTINUOUS, Dimensionless)
- 'Temperature' (-100 to 100, CONTINUOUS, Temperature)
- 'Activity Level' (0 to 1000, CONTINUOUS, Dimensionless)
- 'OnOff' (0 to 1, DISCRETE, Dimensionless)

Then create Schedule:Compact entries. The `data` argument is a NESTED LIST
OF DICTS (not a flat string list). Shape:

    [
      {
        "Through": "MM/DD",              // last block MUST be "12/31"
        "Days": [
          {
            "For": "<DayType>",           // Weekdays / Weekends / Saturday / Sunday /
                                          // AllDays / AllOtherDays / Holidays /
                                          // SummerDesignDay / WinterDesignDay /
                                          // Monday..Friday / CustomDay1 / CustomDay2
            "Times": [
              {"Until": {"Time": "HH:MM", "Value": <float>}},
              ...
              {"Until": {"Time": "24:00", "Value": <float>}}   // last MUST be 24:00
            ]
          },
          ...   // additional day-type blocks under the same Through
        ]
      },
      ...   // additional Through blocks for seasonal variation
    ]

Example (medium-office lighting fraction schedule):

    [
      {
        "Through": "12/31",
        "Days": [
          {"For": "Weekdays", "Times": [
            {"Until": {"Time": "05:00", "Value": 0.05}},
            {"Until": {"Time": "07:00", "Value": 0.10}},
            {"Until": {"Time": "08:00", "Value": 0.30}},
            {"Until": {"Time": "17:00", "Value": 0.90}},
            {"Until": {"Time": "18:00", "Value": 0.70}},
            {"Until": {"Time": "20:00", "Value": 0.50}},
            {"Until": {"Time": "22:00", "Value": 0.30}},
            {"Until": {"Time": "23:00", "Value": 0.10}},
            {"Until": {"Time": "24:00", "Value": 0.05}}
          ]},
          {"For": "Saturday", "Times": [
            {"Until": {"Time": "06:00", "Value": 0.05}},
            {"Until": {"Time": "08:00", "Value": 0.10}},
            {"Until": {"Time": "14:00", "Value": 0.50}},
            {"Until": {"Time": "17:00", "Value": 0.15}},
            {"Until": {"Time": "24:00", "Value": 0.05}}
          ]},
          {"For": "AllOtherDays", "Times": [
            {"Until": {"Time": "24:00", "Value": 0.05}}
          ]}
        ]
      }
    ]

Downstream completeness checklist — BEFORE finishing, re-read the spec
and ensure every schedule the downstream phases will reference exists.
Typical required schedules for a conditioned occupied zone:

  Downstream field                              | Type           | Typical values
  ----------------------------------------------|----------------|----------------
  thermostat.heating_setpoint_schedule_name     | Temperature    | 20 occupied / 15 setback
  thermostat.cooling_setpoint_schedule_name     | Temperature    | 24 occupied / 28 setback
  ideal_loads.system_availability_schedule_name | Fraction/OnOff | 1 during hours, else 0
  people.number_of_people_schedule_name         | Fraction       | occupancy pattern
  people.activity_level_schedule_name           | Activity Level | ~120 W/person seated
  lights.schedule_name                          | Fraction       | lighting pattern

If the spec implies occupancy but does not explicitly name an activity-
level schedule, CREATE ONE anyway (e.g. "Office_Activity_Level" at
120 W/person constant). People objects cannot be built without it.

Rules:
- Create type limits BEFORE the schedules that reference them.
- Use the EXACT schedule names the spec states; otherwise downstream
  phases will reference non-existent schedules.
- The LAST "Through" block must be "12/31" (full-year coverage).
- Within each "For" block, the LAST "Until.Time" must be "24:00".
- Cover every day type: either use "AllDays", or use specific day types
  followed by "AllOtherDays" to catch the rest.
- Call list_schedules once at the end.

Reference database:
- Call search_energyplus_reference to look up standard schedule type limit
  bounds (e.g. 'temperature type limits heating cooling') or reference compact
  schedule profiles for your building type (e.g. 'medium office occupancy
  weekday fraction schedule'). Use the returned time-value data as a starting
  point, then adjust to match the spec.
"""


def schedule_agent(state: AgentState) -> AgentStateUpdate:
    local = clone_for_phase(state)
    tools = make_schedule_tools(local, rag=_get_rag())
    collector = TraceCollector(phase="schedule")

    agent = build_react_agent(
        llm=create_llm(),
        tools=tools,
        system_prompt=SCHEDULE_SYSTEM_PROMPT,
        trace_collector=collector,
    )

    if state.intake_output:
        io = state.intake_output
        specs = (
            f"--- Schedule specifications (primary task) ---\n{io.schedule_specs}\n\n"
            "--- Downstream specs (reference only; do NOT create non-schedule "
            "objects here, but USE these to infer which schedules the later "
            "phases will reference) ---\n"
            f"[hvac_specs]\n{io.hvac_specs}\n\n"
            f"[people_specs]\n{io.people_specs}\n\n"
            f"[lights_specs]\n{io.lights_specs}\n"
        )
    else:
        specs = state.user_input
    # If reached via a back-hop (downstream needed a schedule), append.
    upstream = state.upstream_request
    consumed_upstream = bool(upstream and upstream.get("target") == "schedule")
    if consumed_upstream:
        specs = f"{specs}\n\n{upstream['specs']}"
    result = invoke_with_self_repair(
        agent,
        local,
        specs,
        phase="schedule",
        is_revision=state.is_revision,
        validation_errors=state.validation_errors,
    )

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "schedule done"

    record_phase_trace("schedule", collector.export())
    update = AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[schedule] {summary}")],
    )
    # Drop the consumed back-hop request so it can't be re-injected on retry.
    # An empty dict is the reducer's explicit-clear sentinel (a bare None would
    # be treated as "field omitted" by sibling branches and leave the value).
    if consumed_upstream:
        update["upstream_request"] = {}
    return update
