import json
from dataclasses import dataclass
from typing import Any, cast

import pytest
from _pytest.monkeypatch import MonkeyPatch

from src.agent.tools import (
    make_construction_tools,
    make_hvac_tools,
    make_material_tools,
    make_schedule_tools,
    rag_tools,
)
from src.mcp.state import ConfigState


@dataclass
class _SearchResult:
    description: str
    table_name: str
    record_id: str
    score: float
    full_data: dict[str, Any]


class _FakeRag:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None, float]] = []

    def search(
        self,
        *,
        query: str,
        top_k: int,
        chunk_type: str | None,
        score_threshold: float,
    ) -> list[_SearchResult]:
        self.calls.append((query, top_k, chunk_type, score_threshold))
        results = {
            "materials": [
                _SearchResult(
                    description="Concrete",
                    table_name="materials",
                    record_id="material-low",
                    score=0.61,
                    full_data={"conductivity": 1.4},
                )
            ],
            "constructions": [
                _SearchResult(
                    description="Insulated wall",
                    table_name="constructions",
                    record_id="construction-high",
                    score=0.93456,
                    full_data={"layers": ["brick", "insulation"]},
                ),
                _SearchResult(
                    description="Mass wall",
                    table_name="constructions",
                    record_id="construction-mid",
                    score=0.72,
                    full_data={"layers": ["concrete"]},
                ),
            ],
        }
        return results.get(chunk_type or "", [])


def test_rag_tool_reports_unavailable_without_credentials(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("QDRANT_ENDPOINT", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(rag_tools, "_rag", None)
    monkeypatch.setattr(rag_tools, "_rag_attempted", False)

    result = json.loads(rag_tools.make_rag_tool().invoke({"query": "wall assembly"}))

    assert result == {
        "success": False,
        "message": (
            "EnergyPlus reference database is unavailable. "
            "Use ASHRAE default values and proceed."
        ),
        "data": None,
    }


def test_rag_tool_merges_allowed_tables_by_score() -> None:
    fake_rag = _FakeRag()
    tool = rag_tools.make_rag_tool(
        allowed_tables=["materials", "constructions"],
        top_k=2,
        score_threshold=0.5,
        rag=cast(Any, fake_rag),
    )

    result = json.loads(tool.invoke({"query": "exterior wall"}))

    assert result["success"] is True
    assert [record["record_id"] for record in result["data"]] == [
        "construction-high",
        "construction-mid",
    ]
    assert result["data"][0]["score"] == 0.9346
    assert fake_rag.calls == [
        ("exterior wall", 2, "materials", 0.5),
        ("exterior wall", 2, "constructions", 0.5),
    ]


@pytest.mark.parametrize(
    ("factory", "expected_tables"),
    [
        (
            make_material_tools,
            ["standard_materials", "no_mass_materials", "all_materials"],
        ),
        (make_construction_tools, ["constructions"]),
        (make_schedule_tools, ["schedule_type_limits", "schedule_compact"]),
        (make_hvac_tools, ["sizingperiod_designday"]),
    ],
)
def test_phase_tool_factories_scope_rag_search(
    factory: Any,
    expected_tables: list[str],
) -> None:
    fake_rag = _FakeRag()

    tools = factory(ConfigState(), rag=cast(Any, fake_rag))

    assert [tool.name for tool in tools].count("search_energyplus_reference") == 1
    result = json.loads(tools[-1].invoke({"query": "reference query"}))
    assert result["success"] is True
    assert [call[2] for call in fake_rag.calls] == expected_tables


@pytest.mark.parametrize(
    "factory",
    [
        make_material_tools,
        make_construction_tools,
        make_schedule_tools,
        make_hvac_tools,
    ],
)
def test_phase_tool_factories_omit_rag_search_when_unavailable(factory: Any) -> None:
    tools = factory(ConfigState(), rag=None)

    assert "search_energyplus_reference" not in [tool.name for tool in tools]
