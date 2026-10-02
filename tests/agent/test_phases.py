from idfpy import IDF
from idfpy.models.constructions import Material
from idfpy.models.thermal_zones import Zone

from src.agent.phases import missing_output_issues, owner
from src.agent.state import IntakeOutput
from src.modeling.validation import ModelIssue


def _intake(**specs: str) -> IntakeOutput:
    fields = {
        f"{phase}_specs": ""
        for phase in (
            "zone",
            "material",
            "schedule",
            "construction",
            "surface",
            "fenestration",
            "hvac",
            "people",
            "lights",
        )
    }
    return IntakeOutput.model_validate(
        {"building": {"name": "B"}, "site_location": {"name": "S"}, **fields, **specs}
    )


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
    intake = _intake(
        zone_specs="one zone", material_specs="brick", schedule_specs="office hours"
    )

    issues = missing_output_issues(
        idf, intake, ("zone", "material", "schedule", "lights")
    )

    assert [(i.object_type, owner(i)) for i in issues] == [
        ("ScheduleTypeLimits", "schedule")
    ]
