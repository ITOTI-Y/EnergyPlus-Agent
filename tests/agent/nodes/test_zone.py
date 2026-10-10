from idfpy.models.thermal_zones import Zone

from src.agent.nodes.zone import zone_node
from src.agent.state import AgentState
from tests.agent.intake_data import intake, zone_spec

SQUARE = [(0, 0), (5, 0), (5, 5), (0, 5)]


def test_zones_come_from_the_intake_output():
    output = intake(zones=[zone_spec("Office", SQUARE), zone_spec("Store", SQUARE)])

    out = zone_node(AgentState(pending_phases=["zone"], intake_output=output))

    assert set(out["config_state"].idf.all_of_type(Zone)) == {"Office", "Store"}


def test_zone_phase_not_pending_leaves_the_model_alone():
    output = intake(zones=[zone_spec("Office", SQUARE)])

    assert (
        zone_node(AgentState(pending_phases=["material"], intake_output=output)) == {}
    )
