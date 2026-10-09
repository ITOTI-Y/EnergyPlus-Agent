import pickle

import pytest
from idfpy.models.constructions import Construction, Material
from idfpy.models.outputs import OutputVariable
from idfpy.models.thermal_zones import Zone
from pydantic import ValidationError

from src.agent import build_graph
from src.agent.state import IntakePatch, merge_config_state
from src.state.config_state import ConfigState
from tests.agent.intake_data import intake, layout, zone_spec


def test_config_states_stay_isolated_after_graph_build():
    build_graph()
    first, second = ConfigState(), ConfigState()

    first.idf.add(Zone(name="Z1"))

    assert list(second.idf.all_of_type(Zone)) == []


def test_merge_idf_parallel_branches_union():
    parent = ConfigState()
    parent.idf.add(Zone(name="Z0"))
    branch_a = parent.model_copy(deep=True)
    branch_a.idf.add(Zone(name="Z_A"))
    branch_b = parent.model_copy(deep=True)
    branch_b.idf.add(
        Material(
            name="Concrete",
            roughness="Rough",
            thickness=0.1,
            conductivity=1.4,
            density=2100.0,
            specific_heat=900.0,
        )
    )

    merged = merge_config_state(merge_config_state(parent, branch_a), branch_b)

    assert set(merged.idf.all_of_type(Zone)) == {"Z0", "Z_A"}
    assert "Concrete" in merged.idf.all_of_type(Material)


def test_merge_idf_new_wins_on_conflict():
    old = ConfigState()
    old.idf.add(Zone(name="Z", x_origin=1.0))
    new = ConfigState()
    new.idf.add(Zone(name="Z", x_origin=9.0))

    merged = merge_config_state(old, new)

    zone = merged.idf.get(Zone, "Z")
    assert zone is not None
    assert zone.x_origin == 9.0


def test_merge_idf_nameless_objects_not_duplicated():
    parent = ConfigState()
    parent.idf.add(OutputVariable(key_value="*", variable_name="Zone Air Temperature"))
    branch = parent.model_copy(deep=True)
    branch.idf.add(
        OutputVariable(key_value="*", variable_name="Zone Mean Radiant Temperature")
    )

    merged = merge_config_state(parent, branch)

    assert len(merged.idf.all_of_type(OutputVariable)) == 2


def test_merge_idf_nameless_objects_from_parallel_branches_both_retained():
    parent = ConfigState()
    parent.idf.add(OutputVariable(key_value="*", variable_name="Zone Air Temperature"))
    branch_a = parent.model_copy(deep=True)
    branch_a.idf.add(
        OutputVariable(key_value="*", variable_name="Zone Mean Radiant Temperature")
    )
    branch_b = parent.model_copy(deep=True)
    branch_b.idf.add(
        OutputVariable(
            key_value="*", variable_name="Site Outdoor Air Drybulb Temperature"
        )
    )

    merged = merge_config_state(merge_config_state(parent, branch_a), branch_b)

    variable_names = {
        v.variable_name for v in merged.idf.all_of_type(OutputVariable).values()
    }
    assert variable_names == {
        "Zone Air Temperature",
        "Zone Mean Radiant Temperature",
        "Site Outdoor Air Drybulb Temperature",
    }


def test_merge_does_not_mutate_inputs():
    old = ConfigState()
    old.idf.add(Zone(name="Z_OLD"))
    new = ConfigState()
    new.idf.add(Zone(name="Z_NEW"))

    merge_config_state(old, new)

    assert set(old.idf.all_of_type(Zone)) == {"Z_OLD"}
    assert set(new.idf.all_of_type(Zone)) == {"Z_NEW"}


def test_patch_reports_phases_whose_input_changed():
    before = intake(
        zones=[zone_spec("A", [(0, 0), (5, 0), (5, 5), (0, 5)])], hvac_specs="ideal"
    )
    patch = IntakePatch.model_validate(
        {
            "reason": "r",
            # Layout given but unchanged.
            "zone_plans": before.model_dump()["zone_plans"],
            "storeys": before.model_dump()["storeys"],
            "hvac_specs": "ideal loads, 20/26 C",
        }
    )

    after, changed = patch.apply(before)

    assert changed == {"hvac"}
    assert after.hvac_specs == "ideal loads, 20/26 C"
    assert after.zones == before.zones


def test_patch_repeating_a_plan_key_is_rejected():
    square = [(0, 0), (5, 0), (5, 5), (0, 5)]
    patch = IntakePatch.model_validate(
        {"reason": "r", **layout(zone_spec("A", square), zone_spec("A", square))}
    )

    with pytest.raises(ValidationError, match="repeated"):
        patch.apply(intake())


def test_storeys_name_zones_and_carry_level_height_and_multiplier():
    square = [{"X": x, "Y": y} for x, y in [(0, 0), (5, 0), (5, 5), (0, 5)]]
    plan = {
        "plan": square,
        "exterior_wall_construction": "Wall",
        "roof_construction": "Wall",
        "ground_floor_construction": "Slab",
        "interior_wall_construction": "Wall",
        "interior_floor_construction": "Slab",
    }
    output = intake(
        zone_plans=[plan | {"key": "Office"}, plan | {"key": "Lobby"}],
        storeys=[
            {"name": "G", "height": 4, "zones": [
                {"plan": "Office"}, {"plan": "Lobby", "height": 7.5}]},
            {"name": "T", "height": 3.5, "multiplier": 18,
             "zones": [{"plan": "Office"}]},
            {"name": "Top", "height": 3.5, "zones": [{"plan": "Office"}]},
        ],
    )  # fmt: skip

    assert [(z.name, z.floor_z, z.height, z.multiplier) for z in output.zones] == [
        ("G_Office", 0, 4, 1),
        ("G_Lobby", 0, 7.5, 1),
        ("T_Office", 4, 3.5, 18),
        ("Top_Office", 4 + 18 * 3.5, 3.5, 1),
    ]
    with pytest.raises(ValidationError, match="not defined"):
        intake(zone_plans=[], storeys=[{"name": "G", "height": 3,
                                        "zones": [{"plan": "Office"}]}])  # fmt: skip


@pytest.mark.parametrize(
    ("second", "phases"),
    [
        ("B", {"surface"}),  # moved: only its geometry changes
        ("C", {"zone", "surface"}),  # renamed: objects naming it change too
    ],
)
def test_zone_patch_rebuilds_zones_only_when_names_change(second, phases):
    square = [(0, 0), (5, 0), (5, 5), (0, 5)]
    before = intake(
        zones=[zone_spec("A", square), zone_spec("B", [(4, 0), (9, 0), (9, 5), (4, 5)])]
    )
    moved = zone_spec(second, [(5, 0), (10, 0), (10, 5), (5, 5)])
    patch = IntakePatch.model_validate(
        {"reason": "r", **layout(zone_spec("A", square), moved)}
    )

    _, changed = patch.apply(before)

    assert changed == phases


def test_copied_and_checkpointed_models_resolve_references_in_themselves():
    original = ConfigState()
    original.idf.add(
        Material(
            name="Brick",
            roughness="Rough",
            thickness=0.1,
            conductivity=0.9,
            density=1900.0,
            specific_heat=800.0,
        )
    )
    original.idf.add(Construction(name="Wall", outside_layer="Brick"))

    for copy in (
        original.model_copy(deep=True),
        pickle.loads(pickle.dumps(original)),
    ):
        wall = copy.idf.get(Construction, "Wall")
        assert wall is not None
        # Phase nodes work on copies, and checkpoints pickle the state; a
        # lookup must not reach back into the original model.
        assert wall.outside_layer_ref is copy.idf.get(Material, "Brick")


def test_a_plan_too_narrow_for_perimeter_core_is_sent_back_to_intake():
    narrow = zone_spec("Wing", [(0, 0), (40, 0), (40, 10), (0, 10)])
    plans = layout(narrow)
    plans["zone_plans"][0]["zoning"] = "perimeter_core"

    # A validation error is what intake's structured call returns to the LLM.
    with pytest.raises(ValidationError, match="too short for perimeter_core"):
        intake(**plans)
