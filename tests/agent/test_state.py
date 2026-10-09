from idfpy.models.thermal_zones import Zone

from src.agent import build_graph
from src.agent.state import merge_config_state
from src.mcp.state import ConfigState


def test_config_states_stay_isolated_after_graph_build():
    build_graph()
    first, second = ConfigState(), ConfigState()

    first.idf.add(Zone(name="Z1"))

    assert list(second.idf.all_of_type(Zone)) == []


def test_parallel_branches_merge_into_union():
    parent = ConfigState()
    branch_a = parent.model_copy(deep=True)
    branch_b = parent.model_copy(deep=True)
    branch_a.idf.add(Zone(name="A"))
    branch_b.idf.add(Zone(name="B"))

    merged = merge_config_state(merge_config_state(parent, branch_a), branch_b)

    assert list(branch_a.idf.all_of_type(Zone)) == ["A"]
    assert sorted(merged.idf.all_of_type(Zone)) == ["A", "B"]
