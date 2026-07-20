from src.agent.nodes.zone import zone_agent
from src.agent.state import AgentState, IntakeOutput
from src.mcp.state import _idf_values


def test_zone_agent_creates_two_zones():
    intake = IntakeOutput.model_validate(
        {
            "building": {"Name": "Test"},
            "site_location": {
                "Name": "Test",
                "Latitude": 22.5,
                "Longitude": 114.0,
                "Time Zone": 8.0,
                "Elevation": 10.0,
            },
            "zone_specs": "Create two zones: F1_Office (6x6m, ground floor) and F1_Corridor (6x2m, ground floor).",
            "material_specs": "",
            "schedule_specs": "",
            "construction_specs": "",
            "surface_specs": "",
            "fenestration_specs": "",
            "hvac_specs": "",
            "people_specs": "",
            "lights_specs": "",
        }
    )
    out = zone_agent(AgentState(intake_output=intake))
    # Read zones from the IDF (the source of truth since the idfpy refactor
    # in commit 3092c07). The legacy ConfigState.zones list is no longer
    # synced by create_zone and would always read empty here.
    zones = _idf_values(out["config_state"].idf, "Zone")
    assert len(zones) == 2
    assert {z.name for z in zones} == {"F1_Office", "F1_Corridor"}
