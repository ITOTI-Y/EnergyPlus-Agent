import json

from src.agent.nodes.intake import _tool, unpack_leaked_fields
from src.agent.state import IntakeOutput, IntakePatch


def test_intake_tools_offer_no_empty_choices():
    # Gemini rejects "" in tool enums; idfpy uses it for blank fields.
    for schema in (IntakeOutput, IntakePatch):
        tool = json.dumps(_tool(schema))

        assert '"enum"' in tool
        assert '""' not in tool


def test_fields_written_inside_another_are_moved_back():
    # Shapes of Haiku 5.5 replies on the 21-storey tower brief.
    args = {
        "lights_specs": "10 W/m2 in offices.</lights_specs>\n"
        '<parameter name="equipment_specs">12 W/m2 plug loads.</equipment_specs>\n'
        "<people_specs>0.1 people/m2.</people_specs>\n"
        "<equipment_specs>x</equipment_specs>\n",
        "people_specs": "kept",
        "hvac_specs": "ideal loads",
    }

    filled = unpack_leaked_fields(
        args, ["lights_specs", "equipment_specs", "people_specs", "hvac_specs"]
    )

    assert filled == ["equipment_specs"]
    assert args == {
        "lights_specs": "10 W/m2 in offices.",
        "people_specs": "kept",
        "hvac_specs": "ideal loads",
        "equipment_specs": "12 W/m2 plug loads.",
    }
