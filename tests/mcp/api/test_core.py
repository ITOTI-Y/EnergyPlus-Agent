from pathlib import Path
from typing import Final

SCHEMA: Final = (
    Path(__file__).parents[3] / "data" / "schemas" / "building_schema.epJSON"
)


def test_tools_edit_the_model_that_clear_all_starts(call_tool):
    assert call_tool("clear_all", {})["success"]

    assert call_tool("create_zone", {"name": "Office"})["success"]

    assert call_tool("get_summary", {})["data"]["zones_count"] == 1


def test_tools_edit_a_loaded_model(call_tool):
    assert call_tool("load_model", {"input_path": str(SCHEMA)})["success"]
    before = call_tool("get_summary", {})["data"]["zones_count"]

    assert call_tool("create_zone", {"name": "Annex"})["success"]

    assert call_tool("get_summary", {})["data"]["zones_count"] == before + 1
