import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient

from src.agent.tools.reference_tools import (
    compact_hit,
    make_reference_tools,
    reference_prompt,
)
from src.reference.climate import Climate
from src.reference.index import Embedder, Hit
from src.reference.search import ReferenceSearch
from src.reference.settings import ReferenceSettings


class FakeQdrant:
    def __init__(self) -> None:
        self.filters: list[Any] = []

    def query_points(self, collection: str, **kwargs: Any) -> SimpleNamespace:
        self.filters.append(kwargs["query_filter"])
        payload = {
            "kind": "material",
            "object_type": "Material:NoMass",
            "name": "Roof Insulation",
            "data": {"name": "Roof Insulation", "thermal_resistance": 3.5},
            "climate_zones": ["1A"],
            "building_types": ["OfficeMedium"],
            "standards": ["ASHRAE 90.1-2022"],
            "roles": ["Roof, Outdoors"],
        }
        return SimpleNamespace(points=[SimpleNamespace(score=0.7, payload=payload)])


class FakeEmbedder:
    def query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_search_tool_filters_by_the_site_zones_and_returns_data():
    qdrant = FakeQdrant()
    search = ReferenceSearch(
        cast(QdrantClient, qdrant),
        "reference",
        cast(Embedder, FakeEmbedder()),
        Climate(5153.0, 136.0, "1A", ["1A", "2A"]),
    )
    [find] = make_reference_tools(search, ("material",))

    result = json.loads(find.invoke({"query": "office roof insulation"}))

    assert find.name == "find_reference_materials"
    assert result[0]["data"]["thermal_resistance"] == 3.5
    zone_filter = qdrant.filters[0].must[1]
    assert (zone_filter.key, zone_filter.match.any) == ("climate_zones", ["1A", "2A"])
    assert "1A, 2A" in reference_prompt(search, ("material",))


def test_settings_need_both_services_or_neither(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REFERENCE_QDRANT_URL", "http://localhost:6333")
    monkeypatch.delenv("REFERENCE_EMBEDDING_URL", raising=False)

    with pytest.raises(ValidationError, match="neither"):
        ReferenceSettings(_env_file=None)


WALL = Hit(
    score=0.63218,
    kind="construction",
    object_type="Construction",
    name="nonres_ext_wall_mid",
    data={
        "name": "nonres_ext_wall_mid",
        "layers": ["F07 Stucco", "Wall Insulation"],
        "materials": [
            {"object_type": "Material", "name": "F07 Stucco", "thickness": 0.025,
             "thermal_absorptance": 0.9, "solar_absorptance": 0.92,
             "visible_absorptance": 0.7},
            {"object_type": "Material:NoMass", "name": "Wall Insulation",
             "thermal_resistance": 0.53, "thermal_absorptance": 0.9,
             "solar_absorptance": 0.7, "visible_absorptance": 0.7},
        ],
    },
    climate_zones=["1A"],
    building_types=["OfficeLarge"],
    standards=["ASHRAE 90.1-2022"],
    roles=["Wall, Outdoors"],
)  # fmt: skip


def test_construction_result_keeps_only_non_default_absorptances():
    entry = compact_hit(WALL, with_material_values=True)

    assert entry["score"] == 0.63
    assert "kind" not in entry and "object_type" not in entry
    assert entry["data"] == {
        "layers": ["F07 Stucco", "Wall Insulation"],
        "materials": [
            {"object_type": "Material", "name": "F07 Stucco", "thickness": 0.025,
             "solar_absorptance": 0.92},
            {"object_type": "Material:NoMass", "name": "Wall Insulation",
             "thermal_resistance": 0.53},
        ],
    }  # fmt: skip
    assert entry["building_types"] == ["OfficeLarge"]
    assert entry["roles"] == ["Wall, Outdoors"]


def test_construction_phase_gets_layer_names_without_material_values():
    entry = compact_hit(WALL, with_material_values=False)

    assert entry["data"] == {"layers": ["F07 Stucco", "Wall Insulation"]}
