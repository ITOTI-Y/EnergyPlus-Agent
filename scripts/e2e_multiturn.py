"""End-to-end multi-turn test: verify the agent can (1) build from scratch
and (2) revise an existing model incrementally (not rebuild from zero).

Reproduces the UI's run_agent revision-detection logic in a headless
script so we can inspect object counts / names between turns:

  Turn 1 (first run):  AgentState(user_input=...)      -> intake -> ... -> simulate
  Turn 2 (revision):    AgentState(config_state=loaded_from_idf, is_revision=True)
                        -> revise -> ... -> simulate

For each turn we snapshot the IDF object inventory (Zone / Material /
Construction / Surface / Fenestration / Schedule / HVAC / People /
Lights counts + a few sample names) and compare turn-1 vs turn-2 to
confirm the revision preserved the model identity and only applied the
requested change.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from src.agent import AgentState, SimContext, build_graph
from src.agent.runner import run_session
from src.mcp.state import ConfigState

# ── Test prompts ─────────────────────────────────────────────────────────────

TURN1_PROMPT = """Design a 5-zone single-floor office in Shenzhen.
Footprint 20m x 12m x 3.5m.
Exterior walls: 200mm reinforced concrete + 80mm rockwool + 15mm gypsum.
Roof: 200mm concrete + 100mm XPS.
Floor: 200mm concrete slab on ground.
Windows: double glazing U=1.8 SHGC=0.4 covering 30% of south facade.
Zones: OpenOffice (12x8m), MeetingRoom (6x4m), Corridor (8x2m), ServerRoom (4x4m), Lobby (6x4m).
Occupancy: 12 people in office (8am-6pm weekdays, 120 W/person), 6 in meeting room, none in corridor/server.
Lighting: 10 W/m^2 in office/meeting/lobby (8am-6pm), 5 W/m^2 corridor.
HVAC: ideal loads, office/meeting/lobby heating 20C / cooling 24C occupied, setback 15C/28C.
Server room: 24/7 cooling to 22C."""

TURN2_PROMPT = """Reduce the south-facade window-to-wall ratio from 30% to 20%.
Also increase the office lighting power density from 10 W/m^2 to 12 W/m^2.
Leave all other objects (zones, materials, constructions, HVAC) unchanged."""

EPW = Path("data/weather/Shenzhen.epw")
OUT = Path("output/e2e_multiturn")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _ok_interrupt(payload: dict) -> dict:
    """Always approve so the pipeline runs to simulate."""
    errors = payload.get("errors", [])
    if errors:
        # Surface but still approve to see whether simulate tolerates it.
        print(f"  [validate] {len(errors)} errors (approving anyway):")
        for e in errors[:5]:
            print(f"    - {e}")
    return {"approved": True}


def _object_inventory(idf_path: Path) -> dict:
    """Return {ObjectType: count} and a few sample names from an IDF.

    NOTE: must NOT call BaseSchema.set_idf() here — that resets the global
    idfpy IDD registry and causes the subsequent load_idf() to index into
    an empty type table (returning 0 objects). The ConfigState ctor
    initializes the IDD exactly once per process.
    """
    from src.mcp.state import _idf_values

    cs = ConfigState()
    cs.load_idf(idf_path)
    idf = cs.idf

    def _count(*types: str) -> tuple[int, list[str]]:
        objs = _idf_values(idf, *types)
        names: list[str] = []
        for o in objs:
            n = getattr(o, "name", None) or getattr(o, "Name", None)
            if n:
                names.append(str(n))
        return len(objs), names[:6]

    inv: dict[str, dict] = {}
    for key, types in [
        ("zones", ("Zone",)),
        (
            "materials",
            (
                "Material",
                "Material:NoMass",
                "Material:AirGap",
                "WindowMaterial:SimpleGlazingSystem",
            ),
        ),
        ("constructions", ("Construction",)),
        ("surfaces", ("BuildingSurface:Detailed",)),
        ("fenestrations", ("FenestrationSurface:Detailed",)),
        ("schedules", ("Schedule:Compact", "ScheduleCompact")),
        ("thermostats", ("HVACTemplate:Thermostat",)),
        ("ideal_loads", ("HVACTemplate:Zone:IdealLoadsAirSystem",)),
        ("people", ("People",)),
        ("lights", ("Lights",)),
    ]:
        c, names = _count(*types)
        inv[key] = {"count": c, "sample_names": names}
    return inv


def _find_idf(d: Path) -> Path | None:
    if not d.exists():
        return None
    cands = sorted(d.glob("*.idf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _find_results(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return [p for p in d.glob("*") if p.suffix in {".csv", ".htm", ".eso"}]


def _polygon_area(vertices: list[tuple[float, float, float]]) -> float:
    """Return the area of a planar 3D polygon."""
    cross_x = cross_y = cross_z = 0.0
    for current, following in zip(vertices, vertices[1:] + vertices[:1], strict=True):
        x1, y1, z1 = current
        x2, y2, z2 = following
        cross_x += y1 * z2 - z1 * y2
        cross_y += z1 * x2 - x1 * z2
        cross_z += x1 * y2 - y1 * x2
    return 0.5 * math.sqrt(cross_x**2 + cross_y**2 + cross_z**2)


def _revision_properties(idf_path: Path) -> dict:
    """Snapshot revision-sensitive values and objects that must be preserved."""
    from src.mcp.state import _idf_values

    cs = ConfigState()
    cs.load_idf(idf_path)
    idf = cs.idf

    def _records(*object_types: str) -> list[dict]:
        records = [
            {
                "type": type(obj).__name__,
                "name": str(getattr(obj, "name", "")),
                "fields": obj.model_dump(mode="json"),
            }
            for obj in _idf_values(idf, *object_types)
        ]
        return sorted(records, key=lambda record: (record["type"], record["name"]))

    surfaces = _idf_values(idf, "BuildingSurface:Detailed")
    south_walls = {
        surface.name: surface
        for surface in surfaces
        if "south" in surface.name.casefold()
        and str(surface.surface_type).casefold() == "wall"
        and str(surface.outside_boundary_condition).casefold() == "outdoors"
    }
    south_wall_area = sum(
        _polygon_area(
            [
                (
                    float(vertex.vertex_x_coordinate),
                    float(vertex.vertex_y_coordinate),
                    float(vertex.vertex_z_coordinate),
                )
                for vertex in surface.vertices
            ]
        )
        for surface in south_walls.values()
    )

    south_window_area = 0.0
    for fenestration in _idf_values(idf, "FenestrationSurface:Detailed"):
        if fenestration.building_surface_name not in south_walls:
            continue
        vertex_count = int(fenestration.number_of_vertices)
        vertices = [
            (
                float(getattr(fenestration, f"vertex_{index}_x_coordinate")),
                float(getattr(fenestration, f"vertex_{index}_y_coordinate")),
                float(getattr(fenestration, f"vertex_{index}_z_coordinate")),
            )
            for index in range(1, vertex_count + 1)
        ]
        south_window_area += _polygon_area(vertices)

    office_lights = {}
    for light in _idf_values(idf, "Lights"):
        zone_name = str(light.zone_or_zonelist_or_space_or_spacelist_name)
        if "office" not in f"{light.name} {zone_name}".casefold():
            continue
        office_lights[light.name] = {
            "zone_name": zone_name,
            "watts_per_floor_area": light.watts_per_floor_area,
        }

    return {
        "south_wwr": (
            south_window_area / south_wall_area if south_wall_area > 0.0 else None
        ),
        "office_lights": office_lights,
        "preserved_objects": {
            "zones": _records("Zone"),
            "materials": _records(
                "Material",
                "Material:NoMass",
                "Material:AirGap",
                "WindowMaterial:SimpleGlazingSystem",
            ),
            "constructions": _records("Construction"),
            "thermostats": _records("HVACTemplate:Thermostat"),
            "ideal_loads": _records("HVACTemplate:Zone:IdealLoadsAirSystem"),
        },
    }


def _assert_revision_outcome(turn1: dict, turn2: dict) -> None:
    """Assert requested changes and unchanged object groups."""
    failures: list[str] = []

    for key in ("zones", "materials", "constructions", "thermostats", "ideal_loads"):
        if turn1["preserved_objects"][key] != turn2["preserved_objects"][key]:
            failures.append(f"{key} changed during revision")

    office_lights1 = turn1["office_lights"]
    office_lights2 = turn2["office_lights"]
    if not office_lights1 or set(office_lights1) != set(office_lights2):
        failures.append("office Lights objects were not preserved")
    else:
        for name in office_lights1:
            lpd1 = office_lights1[name]["watts_per_floor_area"]
            lpd2 = office_lights2[name]["watts_per_floor_area"]
            if lpd1 is None or not math.isclose(float(lpd1), 10.0, abs_tol=0.01):
                failures.append(f"{name} turn-1 LPD is {lpd1!r}, expected 10 W/m^2")
            if lpd2 is None or not math.isclose(float(lpd2), 12.0, abs_tol=0.01):
                failures.append(f"{name} turn-2 LPD is {lpd2!r}, expected 12 W/m^2")

    for label, actual, expected in (
        ("turn-1 south WWR", turn1["south_wwr"], 0.30),
        ("turn-2 south WWR", turn2["south_wwr"], 0.20),
    ):
        if actual is None or not math.isclose(float(actual), expected, abs_tol=0.01):
            failures.append(f"{label} is {actual!r}, expected {expected:.0%}")

    if failures:
        raise AssertionError("; ".join(failures))


# ── Turns ────────────────────────────────────────────────────────────────────


def run_turn(
    graph,
    prompt: str,
    thread_id: str,
    seed_idf: Path | None,
) -> tuple[dict, Path | None]:
    """Run one agent turn. If seed_idf given, load it as config_state
    and flag is_revision (revision turn). Returns (final_state, idf_path)."""
    if seed_idf and seed_idf.exists():
        cs = ConfigState()
        cs.load_idf(seed_idf)
        # Carry the seed IDF as text in a declared ConfigState field so it
        # survives LangGraph's START-boundary input coercion (which strips
        # the PrivateAttr _idf). merge_config_state rebuilds _idf from it
        # on every channel write; revise_node also recovers defensively.
        cs.seed_idf_text = seed_idf.read_text(encoding="utf-8")
        initial = AgentState(
            user_input=prompt,
            config_state=cs,
            is_revision=True,
        )
        mode = "REVISION"
    else:
        initial = AgentState(user_input=prompt)
        mode = "FIRST-RUN"

    context = SimContext(epw_path=EPW, output_dir=OUT)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'=' * 70}")
    print(f"TURN [{mode}]  thread={thread_id}")
    print(f"{'=' * 70}")
    print(f"Prompt (first 200 chars): {prompt[:200]}...")
    t0 = time.time()
    state = run_session(
        graph,
        initial,
        context,
        config,
        on_interrupt=_ok_interrupt,
    )
    dt = time.time() - t0
    print(f"\nTurn finished in {dt:.1f}s")

    idf_path = _find_idf(OUT)
    return dict(state), idf_path


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Clean any prior idf so turn-1 is a genuine first run
    for p in OUT.glob("*.idf"):
        p.unlink()

    graph = build_graph()

    # ─── Turn 1: build from scratch ──────────────────────────────────────────
    state1, idf1 = run_turn(graph, TURN1_PROMPT, "e2e_t1", seed_idf=None)
    if not idf1:
        print("FAIL: turn 1 produced no IDF")
        return 1
    print(f"\nTurn 1 IDF: {idf1}")
    inv1 = _object_inventory(idf1)
    revision1 = _revision_properties(idf1)
    print("Turn 1 inventory:")
    for k, v in inv1.items():
        print(f"  {k:14s} count={v['count']:3d}  samples={v['sample_names']}")
    results1 = _find_results(OUT)
    print(f"Turn 1 result files: {[p.name for p in results1]}")

    # Snapshot the final messages
    msgs1 = [m.content for m in state1.get("messages", []) if hasattr(m, "content")]
    print(f"\nTurn 1 final messages ({len(msgs1)}):")
    for m in msgs1[-4:]:
        print(f"  - {str(m)[:160]}")

    # ─── Turn 2: revise the existing model ───────────────────────────────────
    _state2, idf2 = run_turn(graph, TURN2_PROMPT, "e2e_t2", seed_idf=idf1)
    if not idf2:
        print("FAIL: turn 2 produced no IDF")
        return 1
    print(f"\nTurn 2 IDF: {idf2}")
    inv2 = _object_inventory(idf2)
    print("Turn 2 inventory:")
    revision2 = _revision_properties(idf2)
    for k, v in inv2.items():
        print(f"  {k:14s} count={v['count']:3d}  samples={v['sample_names']}")

    # ─── Compare turn-1 vs turn-2 ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("TURN-1 vs TURN-2 DELTA")
    print(f"{'=' * 70}")
    deltas: dict[str, tuple[int, int]] = {}
    for k in inv1:
        c1, c2 = inv1[k]["count"], inv2[k]["count"]
        deltas[k] = (c1, c2)
        marker = "  " if c1 == c2 else "!!"
        print(f"  {marker} {k:14s} {c1:3d} -> {c2:3d}  (delta {c2 - c1:+d})")

    assertion_error: str | None = None
    try:
        _assert_revision_outcome(revision1, revision2)
    except AssertionError as exc:
        assertion_error = str(exc)
    # ─── Save artifact for inspection ────────────────────────────────────────
    artifact = {
        "turn1": {
            "idf": str(idf1),
            "inventory": inv1,
            "revision_properties": revision1,
        },
        "turn2": {
            "idf": str(idf2),
            "inventory": inv2,
            "revision_properties": revision2,
        },
        "deltas": deltas,
        "assertions": {"passed": assertion_error is None, "error": assertion_error},
    }
    report_path = OUT / "e2e_report.json"
    report_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nReport saved: {report_path}")
    if assertion_error is not None:
        print(f"FAIL: {assertion_error}")
        return 1
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
