"""The whole graph, from a brief and a street photo to an EnergyPlus run.

A five-storey office goes through every phase with the reference library
on and the photo's massing from SAM 3 and image-to-3D, replayed from one
cassette. Record with `pytest tests/agent/test_graph_e2e.py
--record-mode=once` and every service in SERVICES set in the local .env.

Phase 1 and phase 3 nodes run in parallel, so requests are matched on their
JSON body rather than their order.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import pytest
from idfpy.models import (
    BuildingSurfaceDetailed,
    ElectricEquipment,
    FenestrationSurfaceDetailed,
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
    Lights,
    People,
    Zone,
)
from langchain_core.runnables import RunnableConfig
from vcr.cassette import Cassette
from vcr.request import Request

from src.agent import AgentState, SimContext, build_graph, mesh_massing
from src.agent import _share as agent_share
from src.agent.nodes import observe
from src.agent.runner import InterruptDecision, auto_approval, run_session
from src.reference.search import ReferenceSearch
from src.reference.settings import ReferenceSettings
from tests.agent.recording import CASSETTE_ORIGIN, RECORDED_MODEL

EPW: Final = Path("data/weather/Shenzhen.epw")
PHOTO: Final = Path(__file__).parent / "data" / "office_street_view.jpg"
MAX_REVIEWS: Final = 3

RECORDED_SETTINGS: Final = {
    "LLM_VISION_MODEL": RECORDED_MODEL,
    "LLM_TEMPERATURE": "1.0",
    "LLM_REASONING_MAX_TOKENS": "null",
    "LLM_MAX_TOKENS": "32768",
    "REFERENCE_EMBEDDING_MODEL": "qwen/qwen3-embedding-8b",
    "REFERENCE_COLLECTION": "ep_reference_qwen3_embedding_8b",
}

SERVICES: Final = {
    "LLM_BASE_URL": f"{CASSETTE_ORIGIN}/v1",
    "REFERENCE_EMBEDDING_URL": f"{CASSETTE_ORIGIN}/v1",
    "REFERENCE_QDRANT_URL": CASSETTE_ORIGIN,
    "IMAGE_TO_3D_URL": f"{CASSETTE_ORIGIN}/hunyuan3d",
    "SEGMENT_URL": f"{CASSETTE_ORIGIN}/sam3",
}
BRIEF: Final = """Design a 5-story office building in Shenzhen with a central atrium; the photo shows it from the street.
Overall dimensions: 40m x 30m footprint.
Ground floor height: 5.0m; standard floors (2F-5F): 4.0m each.
Central atrium: 16m x 12m, open from ground to roof with a skylight (U=2.0, SHGC=0.35).
Opaque envelope (exterior walls, roof, ground floor, interior partitions and floors): not specified;
use typical medium-office assemblies for the site's climate zone.
Windows: Low-E double glazing (U=1.6, SHGC=0.35), 40% WWR on south/east facades, 25% on north/west.
Floor layout per typical floor (2F-5F):
  - South wing: 4 open-plan offices (~80 m² each), 10 people per office, 120 W/person.
  - North wing: 4 meeting rooms (~30 m² each), 12 people capacity, 100 W/person.
  - East wing: server room (~40 m²), 500 W/m² equipment load, 24/7 cooling to 22°C.
  - West wing: break room (~35 m²) and restrooms (~25 m²).
  - Core: elevator lobby and stairwell (traffic box, ~85 m², unconditioned).
Ground floor (1F):
  - Lobby: 200 m², 5m ceiling, 15 W/m² lighting, conditioned 20-26°C.
  - Retail/café space: 120 m², 20 people, 150 W/person, 12 W/m² lighting.
  - Mechanical/electrical room: 80 m², unconditioned, 200 W/m² equipment.
  - Loading dock: 60 m², unconditioned.
Lighting: 12 W/m² offices, 15 W/m² lobby, 8 W/m² corridors.
Schedules: not specified, except that the server room runs 24/7; use typical
medium-office occupancy, lighting and equipment schedules.
HVAC:
  - Offices/meeting rooms: VAV with reheat, cooling 24°C / heating 21°C occupied, setback 28°C / 15°C.
  - Server room: dedicated split DX system, 22°C year-round.
  - Lobby: fan coil units, cooling 26°C / heating 20°C.
  - Atrium: natural ventilation when outdoor temp 18-28°C, otherwise mechanical.
Infiltration: 0.5 ACH for all conditioned zones.
Ventilation: per ASHRAE 62.1 (office 8.5 L/s/person, meeting 5 L/s/person)."""


def _real_services() -> dict[str, str]:
    missing = [name for name in SERVICES if not os.environ.get(name)]
    if missing:
        pytest.fail(f"recording needs {', '.join(missing)} in the local .env")
    return {os.environ[name].rstrip("/"): SERVICES[name] for name in SERVICES}


@pytest.fixture
def vcr_config(
    vcr_config: dict[str, Any], tmp_path: Path, record_mode: str
) -> dict[str, Any]:
    # Longest first, so a base URL that extends another one wins.
    bases = sorted(
        _real_services().items() if record_mode != "none" else (),
        key=lambda item: -len(item[0]),
    )
    output_dir = str(tmp_path.resolve()).encode()

    def rebase(request: Request) -> Request:
        for real, placeholder in bases:
            if request.uri.startswith(real):
                request.uri = placeholder + request.uri[len(real) :]
                break
        return request

    # A failed simulation's feedback names this run's output directory.
    def mask_output_dir(request: Request) -> Request:
        if isinstance(request.body, bytes):
            request.body = request.body.replace(output_dir, b"<output_dir>")
        return request

    hosts = {host for real, _ in bases if (host := urlsplit(real).hostname)}
    placeholder_host = urlsplit(CASSETTE_ORIGIN).hostname or ""

    def mask_hosts(response: dict[str, Any]) -> dict[str, Any]:
        body = response["body"]["string"]
        if isinstance(body, bytes):
            for host in hosts:
                body = body.replace(host.encode(), placeholder_host.encode())
            response["body"]["string"] = body
        return response

    return {
        **vcr_config,
        "match_on": ["method", "scheme", "host", "port", "path", "query", "json_body"],
        "before_record_request": [
            rebase,
            vcr_config["before_record_request"],
            mask_output_dir,
        ],
        "before_record_response": [vcr_config["before_record_response"], mask_hosts],
    }


@pytest.fixture
def recorded_env(monkeypatch: pytest.MonkeyPatch, record_mode: str) -> None:
    for name, value in RECORDED_SETTINGS.items():
        monkeypatch.setenv(name, value)
    # Read once at import, so the module attributes are what the code uses.
    monkeypatch.setattr(agent_share, "AGENT_LANGUAGE", "English")
    urls = (
        SERVICES
        if record_mode == "none"
        else {name: os.environ.get(name, "") for name in SERVICES}
    )
    for name, url in urls.items():
        monkeypatch.setenv(name, url)
    monkeypatch.setattr(observe, "IMAGE_TO_3D_URL", urls["IMAGE_TO_3D_URL"])
    monkeypatch.setattr(observe, "SEGMENT_URL", urls["SEGMENT_URL"])
    monkeypatch.setattr(
        observe,
        "IMAGE_TO_3D_API_KEY",
        os.environ.get("IMAGE_TO_3D_API_KEY", "")
        if record_mode != "none"
        else "test-key",
    )
    if record_mode == "none":
        monkeypatch.setattr(mesh_massing, "POLL_INTERVAL_S", 0)


@pytest.mark.vcr
@pytest.mark.skipif(shutil.which("energyplus") is None, reason="EnergyPlus not on PATH")
@pytest.mark.usefixtures("pinned_llm_env", "recorded_env")
def test_office_from_brief_and_photo_runs_to_a_simulation(
    tmp_path: Path, vcr: Cassette, record_mode: str
):
    reference = ReferenceSearch.connect(ReferenceSettings(), EPW)
    context = SimContext(epw_path=EPW, output_dir=tmp_path, reference=reference)
    config: RunnableConfig = {
        "configurable": {"thread_id": "e2e"},
        "recursion_limit": 200,
    }
    reviews: list[dict[str, Any]] = []
    nodes: list[str] = []

    def review(payload: dict[str, Any]) -> InterruptDecision:
        reviews.append(payload)
        if len(reviews) > MAX_REVIEWS:
            pytest.fail(f"still not approved after {MAX_REVIEWS} reviews: {payload}")
        return auto_approval(payload)

    state = run_session(
        build_graph(),
        AgentState(user_input=BRIEF, image_paths=[str(PHOTO)]),
        context,
        config,
        on_interrupt=review,
        on_event=lambda node, _: nodes.append(node),
    )

    assert state["photo_reading"] is not None
    assert sum(s.multiplier for s in state["intake_output"].storeys) == 5
    assert reviews[-1]["errors"] == []
    assert state["validation_errors"] == []
    assert nodes[-1] == "simulate"
    assert state["messages"][-1].content.startswith("[simulate]")
    idf = state["config_state"].idf
    for kind in (
        Zone,
        BuildingSurfaceDetailed,
        FenestrationSurfaceDetailed,
        People,
        Lights,
        ElectricEquipment,
        HVACTemplateThermostat,
        HVACTemplateZoneIdealLoadsAirSystem,
    ):
        assert idf.all_of_type(kind), f"no {kind.__name__} in the model"
    last_run = max(tmp_path.glob("run_*"), key=lambda p: p.stat().st_mtime)
    assert "** Severe" not in (last_run / "eplusout.err").read_text()
    paths = [r.path for r in vcr.requests]
    assert any(p.endswith("/embeddings") for p in paths)
    assert any("/points/" in p for p in paths)
    assert any(p.endswith("/segment") for p in paths)
    assert any(p.endswith("/submit") for p in paths)
    if record_mode == "none":
        assert vcr.all_played
