BRICK = {
    "name": "Brick",
    "roughness": "Rough",
    "thickness": 0.1,
    "conductivity": 0.9,
    "density": 1900.0,
    "specific_heat": 800.0,
}


def test_a_material_name_is_refused_in_another_type(call_tool):
    call_tool("create_standard_material", BRICK)

    clash = call_tool(
        "create_no_mass_material",
        {"name": "Brick", "roughness": "Rough", "thermal_resistance": 1.0},
    )

    # Constructions name their layers; two materials named Brick are ambiguous.
    assert not clash["success"]
    assert "Material 'Brick' already exists" in clash["message"]


def test_absorptances_given_are_stored_and_others_default(call_tool):
    call_tool("create_standard_material", {**BRICK, "solar_absorptance": 0.45})
    call_tool(
        "create_no_mass_material",
        {
            "name": "Roofing",
            "roughness": "Rough",
            "thermal_resistance": 0.06,
            "thermal_absorptance": 0.75,
        },
    )

    brick = call_tool("get_material", {"name": "Brick"})["data"]
    roofing = call_tool("get_material", {"name": "Roofing"})["data"]

    assert (brick["solar_absorptance"], brick["thermal_absorptance"]) == (0.45, 0.9)
    assert (roofing["thermal_absorptance"], roofing["solar_absorptance"]) == (0.75, 0.7)
