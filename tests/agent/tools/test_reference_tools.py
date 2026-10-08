import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from src.agent.tools.reference_tools import make_reference_tools, reference_prompt
from src.reference.climate import Climate
from src.reference.index import Embedder
from src.reference.search import ReferenceSearch, ReferenceSearchError
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


class FlakyQdrant(FakeQdrant):
    """Times out on the first ``failures`` searches."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    def query_points(self, collection: str, **kwargs: Any) -> SimpleNamespace:
        if self.failures:
            self.failures -= 1
            raise ResponseHandlingException(TimeoutError("timed out"))
        return super().query_points(collection, **kwargs)


def _search(qdrant: FakeQdrant) -> ReferenceSearch:
    return ReferenceSearch(
        cast(QdrantClient, qdrant),
        "reference",
        cast(Embedder, FakeEmbedder()),
        Climate(5153.0, 136.0, "1A", ["1A", "2A"]),
        qdrant_url="http://localhost:6333",
    )


def test_a_failed_search_is_tried_once_more():
    hits = _search(FlakyQdrant(failures=1)).find("roof", "material", by_climate=True)

    assert [h.name for h in hits] == ["Roof Insulation"]


def test_a_search_failing_twice_stops_the_run_with_the_reason():
    [find] = make_reference_tools(_search(FlakyQdrant(failures=2)), ("material",))

    # Raised through the tool: the run stops instead of inventing values.
    with pytest.raises(ReferenceSearchError, match="Qdrant at http://localhost:6333"):
        find.invoke({"query": "roof insulation"})
