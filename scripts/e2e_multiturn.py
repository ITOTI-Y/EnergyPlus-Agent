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
import sys
import time
from collections import Counter
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
        ("materials", (
            "Material", "Material:NoMass", "Material:AirGap",
            "WindowMaterial:SimpleGlazingSystem",
        )),
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

    print(f"\n{'='*70}")
    print(f"TURN [{mode}]  thread={thread_id}")
    print(f"{'='*70}")
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
    state2, idf2 = run_turn(graph, TURN2_PROMPT, "e2e_t2", seed_idf=idf1)
    if not idf2:
        print("FAIL: turn 2 produced no IDF")
        return 1
    print(f"\nTurn 2 IDF: {idf2}")
    inv2 = _object_inventory(idf2)
    print("Turn 2 inventory:")
    for k, v in inv2.items():
        print(f"  {k:14s} count={v['count']:3d}  samples={v['sample_names']}")

    # ─── Compare turn-1 vs turn-2 ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TURN-1 vs TURN-2 DELTA")
    print(f"{'='*70}")
    deltas: dict[str, tuple[int, int]] = {}
    for k in inv1:
        c1, c2 = inv1[k]["count"], inv2[k]["count"]
        deltas[k] = (c1, c2)
        marker = "  " if c1 == c2 else "!!"
        print(f"  {marker} {k:14s} {c1:3d} -> {c2:3d}  (delta {c2-c1:+d})")

    # ─── Save artifact for inspection ────────────────────────────────────────
    artifact = {
        "turn1": {"idf": str(idf1), "inventory": inv1},
        "turn2": {"idf": str(idf2), "inventory": inv2},
        "deltas": deltas,
    }
    (OUT / "e2e_report.json").write_text(json.dumps(artifact, indent=2))
    print(f"\nReport saved: {OUT / 'e2e_report.json'}")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
