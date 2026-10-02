FRACTION_SCHEDULE = {
    "schedule_type_limits_name": "Any",
    "times": [
        {
            "Through": "12/31",
            "Days": [
                {
                    "For": "AllDays",
                    "Times": [{"Until": {"Time": "24:00", "Value": 1.0}}],
                }
            ],
        }
    ],
}

THERMOSTAT = {
    "name": "Office_Thermostat",
    "heating_setpoint_schedule_name": "Heating_SP",
    "cooling_setpoint_schedule_name": "Cooling_SP",
}


def _seed_schedules(call_tool) -> None:
    call_tool("create_schedule_type_limits", {"name": "Any"})
    for name in ("Heating_SP", "Cooling_SP"):
        assert call_tool(
            "create_schedule_compact", {"name": name, **FRACTION_SCHEDULE}
        )["success"]


def test_create_thermostat_reports_each_missing_schedule(call_tool):
    result = call_tool("create_hvac_thermostat", THERMOSTAT)

    assert not result["success"]
    missing = result["data"]["missing_references"]
    assert len(missing) == 2
    assert any("heating_setpoint_schedule_name" in m for m in missing)
    assert not call_tool("list_hvac_thermostats", {})["data"]


def test_create_ideal_loads_reports_each_missing_reference(call_tool):
    result = call_tool(
        "create_hvac_ideal_loads_system",
        {
            "zone_name": "Zone_1",
            "template_thermostat_name": "Office_Thermostat",
            "system_availability_schedule_name": "HVAC_Avail",
        },
    )

    assert not result["success"]
    assert len(result["data"]["missing_references"]) == 3


def test_ideal_loads_system_is_keyed_by_zone(call_tool):
    _seed_schedules(call_tool)
    call_tool("create_zone", {"name": "Zone_1"})
    call_tool("create_hvac_thermostat", THERMOSTAT)
    system = {"zone_name": "Zone_1", "template_thermostat_name": "Office_Thermostat"}

    assert call_tool("create_hvac_ideal_loads_system", system)["success"]
    duplicate = call_tool("create_hvac_ideal_loads_system", system)
    deleted = call_tool("delete_hvac_ideal_loads_system", {"zone_name": "Zone_1"})

    assert not duplicate["success"]
    assert deleted["success"]


def test_failed_update_keeps_thermostat_unchanged(call_tool):
    _seed_schedules(call_tool)
    call_tool("create_hvac_thermostat", THERMOSTAT)

    result = call_tool(
        "update_hvac_thermostat",
        {"name": "Office_Thermostat", "heating_setpoint_schedule_name": "Missing_SP"},
    )
    thermostat = call_tool("get_hvac_thermostat", {"name": "Office_Thermostat"})

    assert not result["success"]
    assert thermostat["data"]["heating_setpoint_schedule_name"] == "Heating_SP"
