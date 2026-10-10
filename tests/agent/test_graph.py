"""Retry passes of the whole graph, with fake LLM phases that count their runs.

Zones and surfaces are built by the real code; intake's LLM call is replaced
by a function returning a fixed output or patch.
"""

from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest
from idfpy import IDF
from idfpy.models.constructions import Construction, Material
from idfpy.models.schedules import ScheduleCompact, ScheduleTypeLimits
from idfpy.models.thermal_zones import BuildingSurfaceDetailed, Zone
from langchain_core.runnables import RunnableConfig

from src.agent import graph as graph_module
from src.agent.nodes import intake as intake_module
from src.agent.nodes._share import skipped
from src.agent.phases import Phase
from src.agent.state import AgentState, IntakeOutput, IntakePatch, SimContext
from tests.agent.intake_data import intake, layout, zone_spec

type Build = Callable[[IDF, int], None]


def _material(idf: IDF, _: int) -> None:
    idf.add(
        Material(
            name="Brick",
            roughness="Rough",
            thickness=0.1,
            conductivity=0.9,
            density=1900.0,
            specific_heat=800.0,
        )
    )


def _schedule(idf: IDF, run: int) -> None:
    # The first run forgets the type limits its schedule names.
    if run > 1:
        idf.add(ScheduleTypeLimits(name="Fraction"))
    idf.add(
        ScheduleCompact(name="Occupancy", schedule_type_limits_name="Fraction", data=[])
    )


def _construction(idf: IDF, _: int) -> None:
    for name in ("Wall", "Slab"):
        idf.add(Construction(name=name, outside_layer="Brick"))


def _nothing(idf: IDF, run: int) -> None:
    pass


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> Counter[Phase]:
    """Replace the LLM phases with fakes; returns how often each one ran."""
    counts: Counter[Phase] = Counter()

    def fake(phase: Phase, build: Build) -> Callable[[AgentState], dict[str, Any]]:
        def node(state: AgentState) -> dict[str, Any]:
            if skipped(state, phase):
                return {}
            counts[phase] += 1
            local = state.config_state.model_copy(deep=True)
            build(local.idf, counts[phase])
            return {"config_state": local}

        return node

    builds: dict[str, tuple[Phase, Build]] = {
        "material_agent": ("material", _material),
        "schedule_agent": ("schedule", _schedule),
        "construction_agent": ("construction", _construction),
        "fenestration_agent": ("fenestration", _nothing),
        "hvac_agent": ("hvac", _nothing),
        "people_agent": ("people", _nothing),
        "lights_agent": ("lights", _nothing),
        "equipment_agent": ("equipment", _nothing),
    }
    for attribute, (phase, build) in builds.items():
        monkeypatch.setattr(graph_module, attribute, fake(phase, build))
    return counts


def _intake_llm(
    monkeypatch: pytest.MonkeyPatch, first: IntakeOutput, patch: IntakePatch | None
) -> None:
    def structured(llm: object, schema: type, messages: list, check: Callable) -> tuple:
        reply = first if schema is IntakeOutput else patch
        assert reply is not None, "unexpected intake revision"
        return reply, check(reply)

    monkeypatch.setattr(intake_module, "create_llm", lambda: None)
    monkeypatch.setattr(intake_module, "structured", structured)


def _run(tmp_path) -> dict[str, Any]:
    graph = graph_module.build_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    graph.invoke(
        AgentState(user_input="brief"),
        config,
        context=SimContext(epw_path=tmp_path / "w.epw"),
    )
    snapshot = graph.get_state(config)
    # A clean model stops at the human review in validate.
    assert snapshot.next == ("validate",)
    return snapshot.values


SPECS = {
    "material_specs": "brick",
    "schedule_specs": "occupancy",
    "construction_specs": "walls",
}
BOX = [(0, 0), (5, 0), (5, 5), (0, 5)]


def test_a_phase_problem_reruns_only_that_phase_and_its_dependants(
    monkeypatch, runs, tmp_path
):
    _intake_llm(monkeypatch, intake(zones=[zone_spec("A", BOX)], **SPECS), None)

    state = _run(tmp_path)

    # The schedule problem stopped the first pass after the foundations, so
    # the later phases ran once, in the second pass; material kept its run.
    assert runs == {"material": 1, "schedule": 2, "construction": 1} | dict.fromkeys(
        ("fenestration", "hvac", "people", "lights", "equipment"), 1
    )
    idf = state["config_state"].idf
    assert list(idf.all_of_type(ScheduleCompact)) == ["Occupancy"]
    assert len(idf.all_of_type(BuildingSurfaceDetailed)) == 6
    assert state["validation_errors"] == []


def test_a_zone_problem_revises_intake_and_keeps_unaffected_phases(
    monkeypatch, runs, tmp_path
):
    overlapping = [
        zone_spec("A", BOX),
        zone_spec("B", [(4, 0), (9, 0), (9, 5), (4, 5)]),
    ]
    fixed = [zone_spec("A", BOX), zone_spec("B", [(5, 0), (10, 0), (10, 5), (5, 5)])]
    patch = IntakePatch.model_validate({"reason": "move B east", **layout(*fixed)})
    _intake_llm(monkeypatch, intake(zones=overlapping, **SPECS), patch)
    # Start from a schedule that is right the first time.
    runs["schedule"] = 1

    state = _run(tmp_path)

    # Only the geometry changed: the zones keep their names, so the loads
    # naming them are kept too; surfaces and openings are rebuilt.
    assert runs["material"] == 1
    assert runs["construction"] == 1
    assert runs["schedule"] == 2  # one real run after the preset count
    assert runs["hvac"] == 1
    assert runs["fenestration"] == 2
    assert state["global_retries"] == 1
    idf = state["config_state"].idf
    assert set(idf.all_of_type(Zone)) == {"A", "B"}
    paired = [
        s
        for s in idf.all_of_type(BuildingSurfaceDetailed).values()
        if s.outside_boundary_condition == "Surface"
    ]
    assert len(paired) == 2
