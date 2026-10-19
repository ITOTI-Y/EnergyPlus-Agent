import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from idfpy import IDF
from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

import src.agent.tools as agent_tools
from src.agent.tools._share import list_surfaces_tool
from src.state.config_state import ConfigState


def _all_tools() -> list[BaseTool]:
    config = ConfigState()
    tools = []
    for name in agent_tools.__all__:
        factory = getattr(agent_tools, name)
        # Factories of tools that take zones by plan also take the plans.
        takes_plans = "plan_zones" in inspect.signature(factory).parameters
        tools += factory(config, {}) if takes_plans else factory(config)
    return tools


def _bare_objects(schema: Any, path: str) -> Iterator[str]:
    if isinstance(schema, dict):
        if schema.get("type") == "object" and not schema.get("properties"):
            yield path
        for key, value in schema.items():
            yield from _bare_objects(value, f"{path}.{key}")
    elif isinstance(schema, list):
        for i, value in enumerate(schema):
            yield from _bare_objects(value, f"{path}[{i}]")


@pytest.mark.parametrize("tool", _all_tools(), ids=lambda t: t.name)
def test_tool_arguments_declare_object_properties(tool: BaseTool):
    # A model fills an object schema without declared properties with {}.
    parameters = convert_to_openai_tool(tool)["function"]["parameters"]

    assert list(_bare_objects(parameters.get("properties", {}), tool.name)) == []


def test_rejected_operation_returns_error_status_with_details():
    tools = {t.name: t for t in agent_tools.make_lights_tools(ConfigState(), {})}
    call = ToolCall(
        name="delete_light", args={"name": "Nowhere"}, id="1", type="tool_call"
    )

    message = tools["delete_light"].invoke(call)

    assert message.status == "error"
    assert json.loads(message.content)["message"] == "Lights 'Nowhere' not found."


def test_surface_list_is_filtered_and_compact():
    idf = IDF.from_dict(
        json.loads(
            (
                Path(__file__).parents[3] / "data/schemas/building_schema.epJSON"
            ).read_text()
        )
    )
    list_surfaces = list_surfaces_tool(idf, "List surfaces.")

    listed = json.loads(
        list_surfaces.invoke({"zone_names": ["Zone_East"], "surface_type": "Wall"})
    )["data"]

    assert listed and {s["zone"] for s in listed} == {"Zone_East"}
    assert {s["type"] for s in listed} == {"Wall"}
    assert set(listed[0]) == {
        "name", "zone", "type", "boundary", "construction", "facing", "area",
        "corners",
    }  # fmt: skip


def test_misspelt_argument_is_refused_not_dropped():
    tools = {t.name: t for t in agent_tools.make_lights_tools(ConfigState(), {})}
    call = ToolCall(
        name="delete_light", args={"name-": "Nowhere"}, id="1", type="tool_call"
    )

    message = tools["delete_light"].invoke(call)

    assert "name-: Extra inputs are not permitted" in message.content


def test_heat_gain_fractions_over_one_are_refused():
    tools = {
        t.name: t for t in agent_tools.make_equipment_tools(ConfigState(), {"A": []})
    }
    call = ToolCall(
        name="create_equipment",
        args={
            "schedule_name": "Always",
            "plans": ["A"],
            "design_level_calculation_method": "Watts/Area",
            "watts_per_floor_area": 10,
            "fraction_radiant": 0.5,
            "fraction_latent": 0.6,
        },
        id="1",
        type="tool_call",
    )

    message = tools["create_equipment"].invoke(call)

    assert message.status == "error" and "add up to more than 1" in message.content
