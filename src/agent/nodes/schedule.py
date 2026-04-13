from langchain_core.messages import AIMessage, HumanMessage

from src.agent.llm import create_llm
from src.agent.react import ReactState, build_react_agent
from src.agent.state import AgentState, AgentStateUpdate
from src.agent.tools import make_schedule_tools
from src.agent.trace import TRACE_STORE, TraceCollector

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

Rules:
- Create type limits BEFORE the schedules that reference them.
- Schedule names are referenced elsewhere (thermostat setpoints, people, lights);
  use predictable names like 'OfficeOccupancy', 'OfficeLighting',
  'HeatingSetpoint', 'CoolingSetpoint'.
- The LAST "Through" block must be "12/31" (full-year coverage).
- Within each "For" block, the LAST "Until.Time" must be "24:00".
- Cover every day type: either use "AllDays", or use specific day types
  followed by "AllOtherDays" to catch the rest.
- Call list_schedules once at the end.
"""


def schedule_agent(state: AgentState) -> AgentStateUpdate:
    local = state.config_state.model_copy(deep=True)
    tools = make_schedule_tools(local)
    collector = TraceCollector(phase="schedule")

    agent = build_react_agent(
        llm=create_llm(),
        tools=tools,
        system_prompt=SCHEDULE_SYSTEM_PROMPT,
        trace_collector=collector,
    )

    specs = (
        state.intake_output.schedule_specs if state.intake_output else state.user_input
    )
    result = agent.invoke(ReactState(messages=[HumanMessage(content=specs)]))

    final = [
        m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls
    ]
    summary = final[-1].content if final else "schedule done"

    TRACE_STORE.setdefault("schedule", []).extend(collector.export())
    return AgentStateUpdate(
        config_state=local,
        messages=[AIMessage(content=f"[schedule] {summary}")],
    )
