from idfpy.models.internal_gains import Lights

from src.mcp.state import ConfigState


def test_validate_references_empty_state_returns_no_errors():
    assert ConfigState().validate_references() == []


def test_validate_references_reports_lights_missing_zone():
    config = ConfigState()
    config.idf.add(
        Lights(
            name="Z1_Lights",
            zone_or_zonelist_or_space_or_spacelist_name="Missing_Zone",
            schedule_name="Missing_Schedule",
            design_level_calculation_method="Watts/Area",
            watts_per_floor_area=10.0,
        )
    )

    errors = config.validate_references()

    assert any("Missing_Zone" in e for e in errors)
    assert any("Missing_Schedule" in e for e in errors)
