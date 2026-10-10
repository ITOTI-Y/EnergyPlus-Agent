from idfpy import IDF
from idfpy.models.constructions import Construction, Material
from idfpy.models.thermal_zones import Zone

from src.agent.phases import (
    missing_output_issues,
    owner,
    remove_phase_objects,
    rerun_closure,
)
from src.modeling.validation import ModelIssue
from tests.agent.intake_data import intake, zone_spec


def test_issues_route_to_the_phase_owning_the_object_type():
    def issue(object_type: str | None) -> ModelIssue:
        return ModelIssue(object_type, "X", None, "problem")

    assert owner(issue("Material:NoMass")) == "material"
    assert owner(issue("HVACTemplate:Zone:IdealLoadsAirSystem")) == "hvac"
    assert owner(issue(None)) is None


def test_phase_with_a_task_but_no_objects_is_reported():
    idf = IDF()
    idf.add(Zone(name="Office"))
    idf.add(
        Material(
            name="Brick",
            roughness="Rough",
            thickness=0.1,
            conductivity=0.9,
            density=1900.0,
            specific_heat=800.0,
        )
    )
    output = intake(
        zones=[zone_spec("Office", [(0, 0), (5, 0), (5, 5), (0, 5)])],
        material_specs="brick",
        schedule_specs="office hours",
    )

    issues = missing_output_issues(
        idf, output, ("zone", "material", "schedule", "lights")
    )

    assert [(i.object_type, owner(i)) for i in issues] == [
        ("ScheduleTypeLimits", "schedule")
    ]


def test_rerun_closure_follows_references_downstream():
    assert rerun_closure({"material"}) == {
        "material",
        "construction",
        "surface",
        "fenestration",
    }
    assert rerun_closure({"lights"}) == {"lights"}


def test_removal_takes_only_the_given_phases():
    idf = IDF()
    idf.add(Zone(name="Office"))
    idf.add(
        Material(
            name="Brick",
            roughness="Rough",
            thickness=0.1,
            conductivity=0.9,
            density=1900.0,
            specific_heat=800.0,
        )
    )
    idf.add(Construction(name="Wall", outside_layer="Brick"))

    removed = remove_phase_objects(idf, rerun_closure({"material"}))

    assert removed == ["Construction 'Wall'", "Material 'Brick'"]
    assert list(idf.all_of_type(Zone)) == ["Office"]
