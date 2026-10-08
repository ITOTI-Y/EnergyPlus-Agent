FRACTION_SCHEDULE = {
    "name": "Office_Equipment",
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

EQUIPMENT = {
    "name": "Office_Equipment",
    "zone_name": "Office",
    "schedule_name": "Office_Equipment",
    "watts_per_floor_area": 12.0,
}


def test_equipment_needs_existing_zone_and_schedule(call_tool):
    result = call_tool("create_equipment", EQUIPMENT)

    assert not result["success"]
    assert len(result["data"]["missing_references"]) == 2
    assert not call_tool("list_equipment", {})["data"]


def test_equipment_created_then_updated(call_tool):
    call_tool("create_zone", {"name": "Office"})
    call_tool("create_schedule_type_limits", {"name": "Any"})
    call_tool("create_schedule_compact", FRACTION_SCHEDULE)

    assert call_tool("create_equipment", EQUIPMENT)["success"]
    updated = call_tool(
        "update_equipment", {"name": "Office_Equipment", "fraction_radiant": 0.5}
    )

    assert updated["success"]
    assert updated["data"]["watts_per_floor_area"] == 12.0
    assert updated["data"]["fraction_radiant"] == 0.5
