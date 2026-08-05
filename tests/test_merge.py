from idfpy.models.constructions import Material
from idfpy.models.outputs import OutputVariable
from idfpy.models.thermal_zones import Zone

from src.agent.state import merge_config_state
from src.mcp.state import ConfigState
from src.validator import ScheduleCollectionSchema, ZoneSchema


def test_merge_named_list_union():
    a = ConfigState.model_validate(
        {"zones": [ZoneSchema.model_validate({"Name": "A"})]}
    )
    b = ConfigState.model_validate(
        {"zones": [ZoneSchema.model_validate({"Name": "B"})]}
    )
    merged = merge_config_state(a, b)
    assert {z.name for z in merged.zones} == {"A", "B"}


def test_merge_new_wins_on_conflict():
    a = ConfigState.model_validate(
        {"zones": [ZoneSchema.model_validate({"Name": "A", "X Origin": 1.0})]}
    )
    b = ConfigState.model_validate(
        {"zones": [ZoneSchema.model_validate({"Name": "A", "X Origin": 9.0})]}
    )
    merged = merge_config_state(a, b)
    assert merged.zones[0].x_origin == 9.0


_SCHEDULE_DATA = [
    {
        "Through": "12/31",
        "Days": [
            {
                "For": "AllDays",
                "Times": [{"Until": {"Time": "24:00", "Value": 1.0}}],
            }
        ],
    }
]


def test_merge_schedules_nested():
    a = ConfigState.model_validate(
        {
            "schedules": ScheduleCollectionSchema.model_validate(
                {
                    "schedules": [
                        {
                            "Name": "S1",
                            "Schedule Type Limits Name": "F",
                            "Data": _SCHEDULE_DATA,
                        }
                    ]
                }
            )
        }
    )
    b = ConfigState.model_validate(
        {
            "schedules": ScheduleCollectionSchema.model_validate(
                {
                    "schedules": [
                        {
                            "Name": "S2",
                            "Schedule Type Limits Name": "F",
                            "Data": _SCHEDULE_DATA,
                        }
                    ]
                }
            )
        }
    )
    merged = merge_config_state(a, b)
    assert {s.name for s in merged.schedules.schedules} == {"S1", "S2"}


def test_merge_preserves_idf_objects():
    parent = ConfigState()
    branch = parent.model_copy(deep=True)
    branch.idf.add(Zone(name="F1_Office"))

    merged = merge_config_state(parent, branch)

    assert "F1_Office" in merged.idf.all_of_type(Zone)


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


def test_merge_does_not_mutate_inputs():
    old = ConfigState()
    old.idf.add(Zone(name="Z_OLD"))
    new = ConfigState()
    new.idf.add(Zone(name="Z_NEW"))

    merge_config_state(old, new)

    assert set(old.idf.all_of_type(Zone)) == {"Z_OLD"}
    assert set(new.idf.all_of_type(Zone)) == {"Z_NEW"}
