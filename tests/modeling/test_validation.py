import pytest
from idfpy import IDF
from idfpy.models.constructions import Construction, MaterialNoMass
from idfpy.models.hvac_templates import HVACTemplateZoneIdealLoadsAirSystem
from idfpy.models.internal_gains import Lights, People
from idfpy.models.location import SiteGroundTemperatureBuildingSurface
from idfpy.models.schedules import ScheduleCompact
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
    Zone,
)

from src.modeling.validation import (
    completeness_issues,
    fenestration_issues,
    foundation_issues,
    reference_issues,
    simulation_issues,
)
from src.runner.runner import EnergyPlusMessage

type Point = tuple[float, float, float]

# South wall at y=0, 10 m long and 3 m high, vertices counter-clockwise
# seen from outside (from -y).
WALL: list[Point] = [(0, 0, 3), (0, 0, 0), (10, 0, 0), (10, 0, 3)]
WINDOW: list[Point] = [(2, 0, 2), (2, 0, 1), (4, 0, 1), (4, 0, 2)]


def _model(window: list[Point]) -> IDF:
    idf = IDF()
    idf.add(
        BuildingSurfaceDetailed.model_validate(
            {
                "name": "Wall",
                "surface_type": "Wall",
                "construction_name": "C",
                "zone_name": "Z",
                "outside_boundary_condition": "Outdoors",
                "vertices": [
                    {
                        "vertex_x_coordinate": x,
                        "vertex_y_coordinate": y,
                        "vertex_z_coordinate": z,
                    }
                    for x, y, z in WALL
                ],
            }
        )
    )
    fields = {
        f"vertex_{i}_{axis}_coordinate": value
        for i, point in enumerate(window, start=1)
        for axis, value in zip("xyz", point, strict=True)
    }
    idf.add(
        FenestrationSurfaceDetailed.model_validate(
            {
                "name": "Window",
                "surface_type": "Window",
                "construction_name": "Glass",
                "building_surface_name": "Wall",
                **fields,
            }
        )
    )
    return idf


def test_reference_issues_name_object_field_and_target():
    idf = IDF()
    idf.add(ScheduleCompact(name="Occupancy", schedule_type_limits_name="Fraction"))
    idf.add(
        People(
            name="Office_People",
            zone_or_zonelist_or_space_or_spacelist_name="Office",
            number_of_people_schedule_name="Occupancy",
            activity_level_schedule_name="Activity",
        )
    )
    idf.add(
        Lights(
            name="Office_Lights",
            zone_or_zonelist_or_space_or_spacelist_name="Office",
            schedule_name="Occupancy",
        )
    )

    issues = {(i.object_type, i.object_name, i.field) for i in reference_issues(idf)}

    assert issues == {
        ("Schedule:Compact", "Occupancy", "schedule_type_limits_name"),
        ("People", "Office_People", "zone_or_zonelist_or_space_or_spacelist_name"),
        ("People", "Office_People", "activity_level_schedule_name"),
        ("Lights", "Office_Lights", "zone_or_zonelist_or_space_or_spacelist_name"),
    }


def test_nameless_object_is_named_by_its_zone():
    idf = IDF()
    idf.add(
        HVACTemplateZoneIdealLoadsAirSystem(
            zone_name="Office", template_thermostat_name="Thermostat"
        )
    )

    assert {i.object_name for i in reference_issues(idf)} == {"Office"}


def test_window_on_its_wall_has_no_issue():
    assert fenestration_issues(_model(WINDOW)) == []


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        ([(x, 0.5, z) for x, _, z in WINDOW], "off the plane"),
        ([(x + 9, y, z) for x, y, z in WINDOW], "outside surface"),
        (WINDOW[::-1], "faces opposite"),
    ],
    ids=["off_plane", "outside", "reversed"],
)
def test_misplaced_window_is_reported(window: list[Point], expected: str):
    [issue] = fenestration_issues(_model(window))

    assert issue.object_type == "FenestrationSurface:Detailed"
    assert expected in issue.message


def test_empty_model_is_incomplete():
    idf = IDF()
    assert [i.message for i in completeness_issues(idf)] == ["The model has no zones."]
    idf.add(Zone(name="Office"))
    assert [i.message for i in completeness_issues(idf)] == [
        "The model has no surfaces."
    ]


def test_energyplus_messages_are_tied_to_quoted_objects():
    # Texts as reported by EnergyPlus 26.1 for an opaque window construction,
    # a reversed window and a schedule day that stops before 24:00.
    idf = _model(WINDOW)
    idf.add(ScheduleCompact(name="Always On", schedule_type_limits_name="Fraction"))
    messages = [
        EnergyPlusMessage(
            "Severe",
            'FenestrationSurface:Detailed="WINDOW" has an opaque surface construction; it should have a window construction.',
        ),
        EnergyPlusMessage(
            "Severe",
            'checkSubSurfAzTiltNorm: Outward facing angle of subsurface differs more than 90.0 degrees from base surface.\nSubsurface="WINDOW" Tilt = 90.0  Azimuth = 180.0\nBase surface="WALL" Tilt = 90.0  Azimuth = 0.0',
        ),
        EnergyPlusMessage(
            "Severe",
            'ProcessScheduleInput: ProcessIntervalFields, Processing time fields, incomplete day detected, Schedule:Compact DaySchedule Fields=ALWAYS ON_dy_1\nref Schedule:Compact="ALWAYS ON"',
        ),
        EnergyPlusMessage(
            "Fatal", "GetSurfaceData: Errors discovered, program terminates."
        ),
        EnergyPlusMessage("Warning", 'Zone="WALL" ignored'),
    ]

    issues = [(i.object_type, i.object_name) for i in simulation_issues(idf, messages)]

    assert issues == [
        ("FenestrationSurface:Detailed", "Window"),
        ("FenestrationSurface:Detailed", "Window"),
        ("Schedule:Compact", "Always On"),
        (None, None),
    ]


def test_ground_floor_with_no_mass_layer_is_reported_for_kiva():
    idf = IDF()
    idf.add(Zone(name="Office"))
    idf.add(
        MaterialNoMass(name="Insulation", roughness="Smooth", thermal_resistance=2.0)
    )
    idf.add(Construction(name="Slab", outside_layer="Insulation"))
    floor = BuildingSurfaceDetailed.model_validate(
        {
            "name": "Floor",
            "surface_type": "Floor",
            "construction_name": "Slab",
            "zone_name": "Office",
            "outside_boundary_condition": "Ground",
            "vertices": [
                {
                    "vertex_x_coordinate": x,
                    "vertex_y_coordinate": y,
                    "vertex_z_coordinate": 0,
                }
                for x, y in [(0, 0), (0, 5), (5, 5), (5, 0)]
            ],
        }
    )
    idf.add(floor)

    [issue] = foundation_issues(idf)

    assert (issue.object_type, issue.object_name) == ("Construction", "Slab")
    assert "Insulation" in issue.message
    idf.add(SiteGroundTemperatureBuildingSurface())
    assert foundation_issues(idf) == []
