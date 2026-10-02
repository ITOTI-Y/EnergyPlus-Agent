import asyncio
from collections.abc import Iterator
from typing import Any

from src.mcp.server import create_mcp_server


def _bare_objects(schema: Any, path: str) -> Iterator[str]:
    if isinstance(schema, dict):
        if schema.get("type") == "object" and not schema.get("properties"):
            yield path
        for key, value in schema.items():
            yield from _bare_objects(value, f"{path}.{key}")
    elif isinstance(schema, list):
        for i, value in enumerate(schema):
            yield from _bare_objects(value, f"{path}[{i}]")


def test_tool_arguments_declare_object_properties():
    tools = asyncio.run(create_mcp_server().list_tools())

    bare = [p for t in tools for p in _bare_objects(t.parameters["properties"], t.name)]

    assert bare == []


def test_rejected_operation_returns_failure_with_field_details(call_tool):
    result = call_tool(
        "create_standard_material",
        {
            "name": "Brick",
            "roughness": "Rough",
            "thickness": -0.1,
            "conductivity": 0.9,
            "density": 1900.0,
            "specific_heat": 800.0,
        },
    )

    assert not result["success"]
    assert [e["field"] for e in result["data"]["errors"]] == ["thickness"]


def test_rename_reaches_referencing_objects(call_tool):
    call_tool(
        "create_no_mass_material",
        {"name": "Board", "roughness": "Smooth", "thermal_resistance": 0.5},
    )
    call_tool("create_construction", {"name": "Wall", "layers": ["Board"]})

    renamed = call_tool(
        "update_no_mass_material", {"name": "Board", "new_name": "Gypsum"}
    )
    wall = call_tool("get_construction", {"name": "Wall"})

    assert renamed["success"]
    assert wall["data"]["outside_layer"] == "Gypsum"
