import json

from idfpy.models.constructions import Material, MaterialNoMass
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.tools import BaseTool

from src.agent.tools import make_material_tools
from src.state.config_state import ConfigState

BRICK = {
    "name": "Brick_100mm",
    "roughness": "MediumRough",
    "thickness": 0.1,
    "conductivity": 0.89,
    "density": 1920.0,
    "specific_heat": 790.0,
}


def _tools() -> tuple[ConfigState, dict[str, BaseTool]]:
    config = ConfigState()
    return config, {t.name: t for t in make_material_tools(config)}


def _create(tools: dict[str, BaseTool], **lists: list[dict]) -> ToolMessage:
    return tools["create_materials"].invoke(
        ToolCall(name="create_materials", args=lists, id="1", type="tool_call")
    )


def test_materials_of_several_types_are_created_in_one_call():
    _, tools = _tools()

    reply = _create(
        tools,
        standard=[BRICK],
        nomass=[{"name": "Pad", "roughness": "Smooth", "thermal_resistance": 0.2}],
        simple_glazing=[
            {"name": "Glass", "u_factor": 2.8, "solar_heat_gain_coefficient": 0.25}
        ],
        glass_panes=[{"name": "Clear_6mm", "thickness": 0.006}],
        window_gases=[{"name": "Air_12mm", "thickness": 0.012}],
    )
    listed = json.loads(tools["list_materials"].invoke({}))

    assert json.loads(str(reply.content)) == {
        "success": True,
        "message": "Created 5 materials.",
        "data": None,
    }
    # Names and types only: the list is resent with every later call.
    assert {"name": "Brick_100mm", "type": "Material"} in listed["data"]
    assert all(set(item) == {"name", "type"} for item in listed["data"])


def test_get_material_reads_all_values():
    _, tools = _tools()
    _create(tools, standard=[BRICK])

    got = json.loads(tools["get_material"].invoke({"name": "Brick_100mm"}))

    assert got["data"]["type"] == "Material"
    assert got["data"]["conductivity"] == 0.89


def test_get_material_missing_returns_error():
    _, tools = _tools()

    got = json.loads(tools["get_material"].invoke({"name": "Nope"}))

    assert not got["success"]


def test_same_material_again_is_kept_and_name_clashes_are_reported():
    config, tools = _tools()
    _create(tools, standard=[BRICK])

    reply = json.loads(
        _create(
            tools,
            standard=[BRICK, {**BRICK, "name": "Brick_200mm"}],
            # Constructions name their layers: one name, one material of any type.
            nomass=[
                {"name": "Brick_100mm", "roughness": "Rough", "thermal_resistance": 1}
            ],
        ).text
    )

    assert reply["message"] == (
        "Created 1 materials; 1 already existed with the same values."
    )
    [failed] = reply["data"]["failed"]
    assert failed.startswith("Brick_100mm: Material 'Brick_100mm' already exists")
    assert set(config.idf.all_of_type(Material)) == {"Brick_100mm", "Brick_200mm"}
    assert not config.idf.all_of_type(MaterialNoMass)


def test_a_call_where_every_material_fails_is_an_error():
    _, tools = _tools()
    _create(tools, standard=[BRICK])

    reply = _create(tools, standard=[{**BRICK, "thickness": 0.2}])

    assert reply.status == "error"
    assert "Brick_100mm" in str(reply.content)
