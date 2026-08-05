from idfpy.models.hvac_templates import HVACTemplateThermostat
from idfpy.models.schedules import ScheduleCompact
from idfpy.models.thermal_zones import Zone

from src.mcp.state import ConfigState
from src.mcp.tools.hvac import IdealLoadsSystemTool, ThermostatTool


def _schedule(name: str) -> ScheduleCompact:
    return ScheduleCompact(name=name, schedule_type_limits_name="Fraction")


def test_create_thermostat_fails_on_missing_schedules():
    state = ConfigState()
    result = ThermostatTool(state).create(
        {
            "name": "Office_Thermostat",
            "heating_setpoint_schedule_name": "Heating_SP",
            "cooling_setpoint_schedule_name": "Cooling_SP",
        }
    )
    assert not result.success
    assert not state.idf.has("HVACTemplate:Thermostat", "Office_Thermostat")
    assert isinstance(result.data, dict)
    missing = result.data["missing_references"]
    assert len(missing) == 2
    assert any("heating_setpoint_schedule_name" in m for m in missing)
    assert any("cooling_setpoint_schedule_name" in m for m in missing)


def test_create_thermostat_succeeds_with_existing_schedules():
    state = ConfigState()
    state.idf.add(_schedule("Heating_SP"))
    state.idf.add(_schedule("Cooling_SP"))
    result = ThermostatTool(state).create(
        {
            "name": "Office_Thermostat",
            "heating_setpoint_schedule_name": "Heating_SP",
            "cooling_setpoint_schedule_name": "Cooling_SP",
        }
    )
    assert result.success
    assert state.idf.has("HVACTemplate:Thermostat", "Office_Thermostat")


def test_create_ideal_loads_fails_on_missing_references():
    state = ConfigState()
    result = IdealLoadsSystemTool(state).create(
        {
            "zone_name": "Zone_1",
            "template_thermostat_name": "Office_Thermostat",
            "system_availability_schedule_name": "HVAC_Avail",
        }
    )
    assert not result.success
    assert isinstance(result.data, dict)
    missing = result.data["missing_references"]
    assert len(missing) == 3
    assert any("zone_name" in m for m in missing)
    assert any("template_thermostat_name" in m for m in missing)
    assert any("system_availability_schedule_name" in m for m in missing)
    assert not state.idf.all_of_type("HVACTemplate:Zone:IdealLoadsAirSystem")


def test_create_ideal_loads_succeeds_with_existing_references():
    state = ConfigState()
    state.idf.add(Zone(name="Zone_1"))
    state.idf.add(_schedule("Heating_SP"))
    state.idf.add(_schedule("Cooling_SP"))
    state.idf.add(
        HVACTemplateThermostat(
            name="Office_Thermostat",
            heating_setpoint_schedule_name="Heating_SP",
            cooling_setpoint_schedule_name="Cooling_SP",
        )
    )
    result = IdealLoadsSystemTool(state).create(
        {
            "zone_name": "Zone_1",
            "template_thermostat_name": "Office_Thermostat",
        }
    )
    assert result.success
    assert state.idf.all_of_type("HVACTemplate:Zone:IdealLoadsAirSystem")


def test_update_thermostat_fails_on_missing_schedule_and_keeps_original():
    state = ConfigState()
    state.idf.add(_schedule("Heating_SP"))
    state.idf.add(_schedule("Cooling_SP"))
    tool = ThermostatTool(state)
    tool.create(
        {
            "name": "Office_Thermostat",
            "heating_setpoint_schedule_name": "Heating_SP",
            "cooling_setpoint_schedule_name": "Cooling_SP",
        }
    )

    result = tool.update(
        "Office_Thermostat", {"heating_setpoint_schedule_name": "Missing_SP"}
    )
    assert not result.success
    existing = state.idf.get(HVACTemplateThermostat, "Office_Thermostat")
    assert existing is not None
    assert existing.heating_setpoint_schedule_name == "Heating_SP"
