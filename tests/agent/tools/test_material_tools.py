import json

from idfpy.models.constructions import Material
from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool

from src.agent.tools import make_material_tools
from src.state.config_state import ConfigState


def _tools() -> tuple[ConfigState, dict[str, BaseTool]]:
    config = ConfigState()
    return config, {t.name: t for t in make_material_tools(config)}


def _create_brick(tools: dict[str, BaseTool]) -> dict:
    return json.loads(
        tools["create_standard_material"].invoke(
            {
                "name": "Brick_100mm",
                "roughness": "MediumRough",
                "thickness": 0.1,
                "conductivity": 0.89,
                "density": 1920.0,
                "specific_heat": 790.0,
            }
        )
    )


def test_list_materials_returns_created_material():
    _, tools = _tools()
    assert _create_brick(tools)["success"]

    listed = json.loads(tools["list_materials"].invoke({}))

    assert listed["success"]
    assert len(listed["data"]) == 1
    assert listed["data"][0]["type"] == "Material"
    assert listed["data"][0]["name"] == "Brick_100mm"


def test_get_material_finds_created_material():
    _, tools = _tools()
    assert _create_brick(tools)["success"]

    got = json.loads(tools["get_material"].invoke({"name": "Brick_100mm"}))

    assert got["success"]
    assert got["data"]["type"] == "Material"


def test_get_material_missing_returns_error():
    _, tools = _tools()

    got = json.loads(tools["get_material"].invoke({"name": "Nope"}))

    assert not got["success"]


def test_same_material_twice_is_accepted_and_a_different_one_refused():
    config, tools = _tools()
    brick = {
        "name": "Brick",
        "roughness": "Rough",
        "thickness": 0.1,
        "conductivity": 0.9,
        "density": 1900.0,
        "specific_heat": 800.0,
    }
    tools["create_standard_material"].invoke(brick)

    again = json.loads(tools["create_standard_material"].invoke(brick))
    different = tools["create_standard_material"].invoke(
        ToolCall(
            name="create_standard_material",
            args={**brick, "thickness": 0.2},
            id="1",
            type="tool_call",
        )
    )

    assert "already exists, unchanged" in again["message"]
    assert different.status == "error"
    assert len(config.idf.all_of_type(Material)) == 1
