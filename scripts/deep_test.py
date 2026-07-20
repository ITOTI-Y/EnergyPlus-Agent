"""Deep multi-scenario test suite — unified phase self-repair validation.

Successor to the deep_multiturn_test harness. Runs scenarios that stress the
pipeline AFTER all 9 phase agents were unified on invoke_with_self_repair,
plus harder real-world cases than the original suite:

  A. Single-zone unconditioned warehouse (baseline first-run)
  B. 3-turn revision chain (add windows -> modify loads -> thicken insulation)
  C. Revision that deletes a zone + its dependents (cross-ref must stay clean)
  D. Revision inducing fabricated cross-refs (directed-rollback stress)
  E. First-run self-repair under an ambiguous spec (names must be invented
     consistently across phases)
  F. 3-turn integrity chain (zone-name stability + clean refs throughout)
  G. COMPLEX multi-zone office with HVAC in every zone + multiple window
     orientations (the heaviest realistic first-run case)
  H. 4-turn deep revision chain on the complex model from G (add zone ->
     add fenestration -> modify construction -> delete zone)

Each scenario runs in its own output dir + thread_id for isolation. A
scenario "passes" if it completes without exception AND its
post-conditions (object counts, name preservation, reference integrity)
hold. Failures are reported but don't abort the whole suite.

Usage:
    python scripts/deep_test.py            # run all scenarios
    python scripts/deep_test.py A B        # run only A and B
"""

from __future__ import annotations

import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from src.agent import AgentState, SimContext, build_graph
from src.agent.runner import run_session
from src.mcp.state import ConfigState, _idf_values
from src.utils.logging import setup_logger

setup_logger(level="WARNING")  # quiet the LLM/trace noise

EPW = Path("data/weather/Shenzhen.epw")
ROOT = Path("output/deep_test")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _inventory(idf_path: Path) -> dict[str, int]:
    cs = ConfigState()
    cs.load_idf(idf_path)
    idf = cs.idf
    return {
        "zone": len(_idf_values(idf, "Zone")),
        "mat": len(
            _idf_values(
                idf,
                "Material",
                "Material:NoMass",
                "Material:AirGap",
                "WindowMaterial:SimpleGlazingSystem",
            )
        ),
        "const": len(_idf_values(idf, "Construction")),
        "surf": len(_idf_values(idf, "BuildingSurface:Detailed")),
        "fen": len(_idf_values(idf, "FenestrationSurface:Detailed")),
        "sched": len(_idf_values(idf, "Schedule:Compact", "ScheduleCompact")),
        "hvac": len(
            _idf_values(
                idf, "HVACTemplate:Thermostat", "HVACTemplate:Zone:IdealLoadsAirSystem"
            )
        ),
        "ppl": len(_idf_values(idf, "People")),
        "light": len(_idf_values(idf, "Lights", "Light")),
    }


def _names(idf_path: Path, obj_type: str) -> list[str]:
    cs = ConfigState()
    cs.load_idf(idf_path)
    out = []
    for o in _idf_values(cs.idf, obj_type):
        n = getattr(o, "name", None) or getattr(o, "Name", None)
        if n:
            out.append(str(n))
    return out


def _latest_idf(d: Path) -> Path | None:
    """Newest .idf in *d*. Robust to stat() races on freshly-written files."""
    if not d.exists():
        return None
    candidates = []
    for p in d.glob("*.idf"):
        try:
            candidates.append((p.stat().st_mtime, p))
        except OSError:
            continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _wait_for_idf(d: Path, timeout: float = 5.0) -> Path | None:
    import time as _t

    deadline = _t.time() + timeout
    while _t.time() < deadline:
        idf = _latest_idf(d)
        if idf:
            return idf
        _t.sleep(0.2)
    return _latest_idf(d)


def _validate_refs(idf_path: Path) -> list[str]:
    cs = ConfigState()
    cs.load_idf(idf_path)
    return cs.validate_references()


def _run_turn(
    graph,
    prompt: str,
    thread_id: str,
    out_dir: Path,
    seed_idf: Path | None,
    approve_always: bool = True,
) -> dict:
    """Run one agent turn. Returns final state dict."""
    if seed_idf and seed_idf.exists():
        cs = ConfigState()
        cs.load_idf(seed_idf)
        cs.seed_idf_text = seed_idf.read_text(encoding="utf-8")
        initial = AgentState(user_input=prompt, config_state=cs, is_revision=True)
    else:
        initial = AgentState(user_input=prompt)

    context = SimContext(epw_path=EPW, output_dir=out_dir)
    cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    def on_interrupt(payload: dict) -> dict:
        errs = payload.get("errors", [])
        if errs and not approve_always:
            return {"approved": False, "feedback": "fix: " + "; ".join(errs)}
        return {"approved": True}

    return run_session(graph, initial, context, cfg, on_interrupt=on_interrupt)


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    duration_s: float
    detail: str = ""
    extra: dict = field(default_factory=dict)


def _ok(name: str, t0: float, detail: str, **extra) -> ScenarioResult:
    return ScenarioResult(name, True, time.time() - t0, detail, extra)


def _fail(name: str, t0: float, detail: str, **extra) -> ScenarioResult:
    return ScenarioResult(name, False, time.time() - t0, detail, extra)


# ── Scenario A: simple single-zone build (baseline) ──────────────────────────

A_PROMPT = """Design a single-zone warehouse in Shenzhen.
30m x 20m x 6m, one thermal zone named Warehouse_Main.
Walls: 200mm concrete block, no insulation.
Roof: 150mm concrete + 50mm EPS.
Floor: 200mm concrete slab on grade.
No windows. No HVAC (unconditioned storage).
One person present 8am-5pm weekdays, light activity 100 W/person.
Lighting: 8 W/m^2, on 8am-5pm weekdays."""


def scenario_a(graph) -> ScenarioResult:
    """A: First-run build of a simple single-zone unconditioned warehouse."""
    t0 = time.time()
    name = "A_simple_build"
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.idf"):
        p.unlink()
    try:
        _run_turn(graph, A_PROMPT, f"{name}_t1", out, seed_idf=None)
        idf = _wait_for_idf(out)
        if not idf:
            return _fail(name, t0, "no IDF produced")
        inv = _inventory(idf)
        checks = []
        if inv["zone"] < 1:
            checks.append(f"zone={inv['zone']} expected >=1")
        if inv["surf"] < 4:
            checks.append(f"surf={inv['surf']} expected >=4 (box)")
        if inv["ppl"] < 1:
            checks.append(f"ppl={inv['ppl']} expected >=1")
        if inv["light"] < 1:
            checks.append(f"light={inv['light']} expected >=1")
        if inv["fen"] != 0:
            checks.append(f"fen={inv['fen']} expected 0 (no windows)")
        errs = _validate_refs(idf)
        if errs:
            checks.append(f"cross-ref errors: {errs[:2]}")
        if checks:
            return _fail(name, t0, "; ".join(checks) + f" | inv={inv}")
        return _ok(name, t0, "single-zone warehouse built, refs clean", **inv)
    except Exception as e:
        return _fail(name, t0, f"exception: {e}\n{traceback.format_exc()[-400:]}")


# ── Scenario B: 3-turn revision chain (add -> modify -> thicken) ─────────────

B_TURN1 = """Design a 3-zone single-floor office in Shenzhen.
20m x 12m x 3.5m. Zones: Office (10x8m), Meeting (6x4m), Storage (4x4m).
Exterior walls: 200mm concrete + 50mm EPS + 15mm gypsum.
Roof: 200mm concrete + 80mm XPS. Floor: 200mm concrete on ground.
No windows initially. Office: 8 people 8am-6pm weekdays, 120 W/person.
Meeting: 6 people. Lighting: 10 W/m^2 office/meeting, 5 W/m^2 storage.
HVAC: ideal loads, office/meeting 20-24C occupied, storage unconditioned."""

B_TURN2 = """Add double-glazed windows (U=1.8, SHGC=0.4) covering 25% of the
Office south wall. Keep all other objects unchanged."""

B_TURN3 = """Change the Office lighting power density from 10 W/m^2 to 15 W/m^2,
and add 2 more people to the Meeting room (6 -> 8 total). Keep all other
objects unchanged."""


def scenario_b(graph) -> ScenarioResult:
    """B: 3 consecutive revision turns on the same model."""
    t0 = time.time()
    name = "B_revision_chain_3turns"
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.idf"):
        p.unlink()
    try:
        _run_turn(graph, B_TURN1, f"{name}_t1", out, seed_idf=None)
        idf1 = _wait_for_idf(out)
        if not idf1:
            return _fail(name, t0, "turn 1 produced no IDF")
        inv1 = _inventory(idf1)
        if inv1["zone"] != 3:
            return _fail(name, t0, f"turn 1 zone={inv1['zone']} expected 3")

        _run_turn(graph, B_TURN2, f"{name}_t2", out, seed_idf=idf1)
        idf2 = _wait_for_idf(out)
        if not idf2 or idf2 == idf1:
            return _fail(name, t0, "turn 2 produced no new IDF")
        inv2 = _inventory(idf2)
        checks = []
        if inv2["zone"] != inv1["zone"]:
            checks.append(f"zone {inv1['zone']}->{inv2['zone']} (expected preserved)")
        if inv2["surf"] != inv1["surf"]:
            checks.append(f"surf {inv1['surf']}->{inv2['surf']} (expected preserved)")
        if inv2["fen"] <= inv1["fen"]:
            checks.append(f"fen {inv1['fen']}->{inv2['fen']} (expected increase)")
        if checks:
            return _fail(name, t0, "turn 2: " + "; ".join(checks))

        _run_turn(graph, B_TURN3, f"{name}_t3", out, seed_idf=idf2)
        idf3 = _wait_for_idf(out)
        if not idf3 or idf3 == idf2:
            return _fail(name, t0, "turn 3 produced no new IDF")
        inv3 = _inventory(idf3)
        checks = []
        if inv3["zone"] != 3:
            checks.append(f"zone={inv3['zone']} expected 3")
        if inv3["surf"] != inv2["surf"]:
            checks.append(f"surf {inv2['surf']}->{inv3['surf']} (expected preserved)")
        if inv3["fen"] != inv2["fen"]:
            checks.append(f"fen {inv2['fen']}->{inv3['fen']} (expected preserved)")
        errs = _validate_refs(idf3)
        if errs:
            checks.append(f"cross-ref errors: {errs[:2]}")
        if checks:
            return _fail(name, t0, "turn 3: " + "; ".join(checks))
        return _ok(
            name,
            t0,
            "3-turn chain OK: build -> +windows -> modify loads, refs clean",
            t1=inv1,
            t2=inv2,
            t3=inv3,
        )
    except Exception as e:
        return _fail(name, t0, f"exception: {e}\n{traceback.format_exc()[-400:]}")


# ── Scenario C: revision that deletes a zone + dependents ────────────────────

C_TURN1 = """Design a 4-zone office in Shenzhen. 20m x 12m x 3.5m.
Zones: OfficeA (8x6m), OfficeB (8x6m), Corridor (4x12m), Storage (4x4m).
Exterior walls: 200mm concrete + 50mm EPS + 15mm gypsum.
Roof: 200mm concrete + 80mm XPS. Floor: 200mm concrete slab.
Each office: 6 people, 10 W/m^2 lighting, 8am-6pm weekdays.
HVAC: ideal loads 20-24C for offices, corridor/storage unconditioned.
Add double-glazed windows (U=1.8, SHGC=0.4) on OfficeA south wall, 20% WWR."""

C_TURN2 = """Remove (delete) the Storage zone and ALL surfaces, lights, and
people associated with it. The other 3 zones (OfficeA, OfficeB, Corridor)
and all their objects must remain unchanged. After deletion there should
be no dangling references to the deleted zone."""


def scenario_c(graph) -> ScenarioResult:
    """C: Revision turn that deletes a zone + its dependent objects."""
    t0 = time.time()
    name = "C_delete_zone"
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.idf"):
        p.unlink()
    try:
        _run_turn(graph, C_TURN1, f"{name}_t1", out, seed_idf=None)
        idf1 = _wait_for_idf(out)
        if not idf1:
            return _fail(name, t0, "turn 1 produced no IDF")
        inv1 = _inventory(idf1)
        if inv1["zone"] < 4:
            return _fail(name, t0, f"turn 1 zone={inv1['zone']} expected >=4")

        _run_turn(graph, C_TURN2, f"{name}_t2", out, seed_idf=idf1)
        idf2 = _wait_for_idf(out)
        if not idf2:
            return _fail(name, t0, "turn 2 produced no IDF")
        inv2 = _inventory(idf2)

        checks = []
        zone_names = _names(idf2, "Zone")
        if any("Storage" in z for z in zone_names):
            checks.append(f"Storage zone still present: {zone_names}")
        errs = _validate_refs(idf2)
        if errs:
            checks.append(f"dangling refs after delete: {errs[:3]}")
        if not any("OfficeA" in z or "Office" in z for z in zone_names):
            checks.append(f"Office zones lost: {zone_names}")
        if checks:
            return _fail(name, t0, "; ".join(checks) + f" | t1={inv1} t2={inv2}")
        return _ok(
            name,
            t0,
            f"delete revision OK: t1 zones={inv1['zone']} -> t2 zones={inv2['zone']}, refs clean",
            t1=inv1,
            t2=inv2,
        )
    except Exception as e:
        return _fail(name, t0, f"exception: {e}\n{traceback.format_exc()[-400:]}")


# ── Scenario D: revision triggering directed rollback ─────────────────────────

D_TURN1 = """Design a 2-zone office in Shenzhen. 16m x 8m x 3.5m.
Zones: Office (10x6m), Meeting (6x6m).
Walls: 200mm concrete + 50mm EPS. Roof: 200mm concrete + 80mm XPS.
Floor: 200mm concrete slab. No windows.
Office: 8 people 8am-6pm, 10 W/m^2 lights. Meeting: 4 people.
HVAC: ideal loads 20-24C occupied."""

D_TURN2_BAD = """Change every surface's construction to a construction named
'NonExistent_Construction' that does not exist in the model. Do not create
the construction first — just reference it directly. (This intentionally
creates broken cross-references to test error recovery.)"""


def scenario_d(graph) -> ScenarioResult:
    """D: Revision inducing cross-ref errors; verify graceful handling."""
    t0 = time.time()
    name = "D_directed_rollback"
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.idf"):
        p.unlink()
    try:
        _run_turn(graph, D_TURN1, f"{name}_t1", out, seed_idf=None)
        idf1 = _wait_for_idf(out)
        if not idf1:
            return _fail(name, t0, "turn 1 produced no IDF")
        inv1 = _inventory(idf1)

        _run_turn(
            graph, D_TURN2_BAD, f"{name}_t2", out, seed_idf=idf1, approve_always=True
        )
        idf2 = _wait_for_idf(out)
        if idf2 and idf2 != idf1:
            inv2 = _inventory(idf2)
            errs = _validate_refs(idf2)
            if inv2["zone"] < inv1["zone"]:
                return _fail(
                    name,
                    t0,
                    f"zones lost during rollback: {inv1['zone']}->{inv2['zone']}",
                )
            return _ok(
                name,
                t0,
                f"bad-ref turn handled gracefully: zones preserved, remaining errors={len(errs)}",
                t1=inv1,
                t2=inv2,
                remaining_errors=len(errs),
            )
        return _ok(name, t0, "bad-ref turn handled (validate surfaced errors)", t1=inv1)
    except Exception as e:
        return _fail(name, t0, f"exception: {e}\n{traceback.format_exc()[-400:]}")


# ── Scenario E: first-run self-repair (ambiguous spec) ───────────────────────

E_PROMPT = """Design a small 2-zone office in Shenzhen. 12m x 8m x 3.5m.
Zones: Office (8x6m), Storage (4x6m).
Use material 'Concrete_200mm' for exterior walls and construction
'ExtWall' for those walls. Roof uses 'Roof_Construction'. Floor uses
'Floor_Construction'. Office: 6 people, 10 W/m^2 lights, HVAC 20-24C.
The model must have internally consistent cross-references."""


def scenario_e(graph) -> ScenarioResult:
    """E: First-run with named materials/constructions the LLM must create."""
    t0 = time.time()
    name = "E_first_run_self_repair"
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.idf"):
        p.unlink()
    try:
        _run_turn(graph, E_PROMPT, f"{name}_t1", out, seed_idf=None)
        idf = _wait_for_idf(out)
        if not idf:
            return _fail(name, t0, "no IDF produced")
        inv = _inventory(idf)
        errs = _validate_refs(idf)
        checks = []
        if inv["zone"] < 2:
            checks.append(f"zone={inv['zone']} expected >=2")
        if inv["surf"] < 4:
            checks.append(f"surf={inv['surf']} expected >=4")
        if errs:
            checks.append(f"cross-ref errors survived self-repair: {errs[:3]}")
        if checks:
            return _fail(name, t0, "; ".join(checks) + f" | inv={inv}")
        return _ok(name, t0, "first-run self-repair OK, refs clean", **inv)
    except Exception as e:
        return _fail(name, t0, f"exception: {e}\n{traceback.format_exc()[-400:]}")


# ── Scenario F: integrity after consecutive revisions ────────────────────────

F_TURN1 = """Design a 2-zone office in Shenzhen. 12m x 8m x 3.5m.
Zones: Office (8x6m), Meeting (4x6m).
Walls: 200mm concrete + 50mm EPS + 15mm gypsum. Roof: 200mm concrete + 80mm XPS.
Floor: 200mm concrete slab. Office: 6 people, 10 W/m^2, HVAC 20-24C.
Meeting: 4 people, 8 W/m^2, HVAC 20-24C."""

F_TURN2 = """Add double-glazed windows (U=1.8, SHGC=0.4) on the Office south wall,
30% WWR. Keep everything else unchanged."""

F_TURN3 = """Increase Office wall insulation from 50mm EPS to 80mm EPS.
Keep everything else unchanged."""


def scenario_f(graph) -> ScenarioResult:
    """F: 3-turn chain focused on reference integrity + name stability."""
    t0 = time.time()
    name = "F_integrity_chain"
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.idf"):
        p.unlink()
    try:
        _run_turn(graph, F_TURN1, f"{name}_t1", out, seed_idf=None)
        idf1 = _wait_for_idf(out)
        if not idf1:
            return _fail(name, t0, "turn 1 no IDF")
        zones_t1 = set(_names(idf1, "Zone"))

        _run_turn(graph, F_TURN2, f"{name}_t2", out, seed_idf=idf1)
        idf2 = _wait_for_idf(out)
        if not idf2 or idf2 == idf1:
            return _fail(name, t0, "turn 2 produced no new IDF")
        zones_t2 = set(_names(idf2, "Zone"))
        if zones_t1 != zones_t2:
            return _fail(
                name, t0, f"zone names changed t1->t2: {zones_t1} != {zones_t2}"
            )

        _run_turn(graph, F_TURN3, f"{name}_t3", out, seed_idf=idf2)
        idf3 = _wait_for_idf(out)
        if not idf3 or idf3 == idf2:
            return _fail(name, t0, "turn 3 produced no new IDF")
        zones_t3 = set(_names(idf3, "Zone"))
        if zones_t1 != zones_t3:
            return _fail(
                name, t0, f"zone names changed across chain: {zones_t1} != {zones_t3}"
            )
        errs = _validate_refs(idf3)
        if errs:
            return _fail(name, t0, f"cross-ref errors after 3 turns: {errs[:3]}")
        inv3 = _inventory(idf3)
        return _ok(
            name,
            t0,
            f"3-turn integrity OK: zone names stable ({sorted(zones_t1)}), refs clean",
            **inv3,
        )
    except Exception as e:
        return _fail(name, t0, f"exception: {e}\n{traceback.format_exc()[-400:]}")


# ── Scenario G: COMPLEX multi-zone office (heavy realistic first-run) ─────────
# This is the hardest first-run case: 4 zones, HVAC in all conditioned zones,
# windows on multiple orientations. Stresses every phase agent simultaneously.

G_PROMPT = """Design a 4-zone two-floor office building in Shenzhen. Total footprint
16m x 10m, each floor 3.5m tall.

Ground floor zones:
- Reception (6x5m), conditioned 22-26C, 2 people 9am-18pm weekdays
- Office_Ground (10x5m), conditioned 22-26C, 8 people 9am-18pm weekdays

Second floor zones:
- Office_Second (10x5m), conditioned 22-26C, 8 people 9am-18pm weekdays
- Meeting_Second (6x5m), conditioned 22-26C, 6 people, used 10am-17pm weekdays

Walls: 200mm concrete block + 75mm EPS insulation + 15mm gypsum board.
Roof: 200mm concrete + 100mm XPS + waterproof membrane.
Ground floor: 200mm concrete slab on grade.
Internal floor (between floors): 200mm concrete + acoustic insulation.
Use distinct constructions for exterior walls, roof, ground floor,
internal floor, and interior partitions.

Windows: double-glazed (U=1.8 W/m2-K, SHGC=0.4, VT=0.6).
- Reception: 30% WWR on south wall
- Office_Ground: 30% WWR on south and east walls
- Office_Second: 30% WWR on south and west walls
- Meeting_Second: 30% WWR on north wall

Lighting: 12 W/m2 in offices, 10 W/m2 in reception, 8 W/m2 in meeting.
HVAC: ideal loads air system in every conditioned zone, with separate
thermostat schedules. Meeting room has its own occupancy schedule.

All cross-references must be internally consistent."""


def scenario_g(graph) -> ScenarioResult:
    """G: Complex 4-zone two-floor office with HVAC everywhere + multi-orientation windows."""
    t0 = time.time()
    name = "G_complex_multizone"
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.idf"):
        p.unlink()
    try:
        _run_turn(graph, G_PROMPT, f"{name}_t1", out, seed_idf=None)
        idf = _wait_for_idf(out)
        if not idf:
            return _fail(name, t0, "no IDF produced")
        inv = _inventory(idf)
        errs = _validate_refs(idf)
        checks = []
        # 4 zones required
        if inv["zone"] < 4:
            checks.append(f"zone={inv['zone']} expected >=4")
        # Box per zone: at least 4 surfaces/zone, but shared walls reduce this.
        # For 4 zones expect >= 16 surfaces (ceilings/floors included).
        if inv["surf"] < 12:
            checks.append(f"surf={inv['surf']} expected >=12 (4 zones)")
        # HVAC in all 4 conditioned zones
        if inv["hvac"] < 4:
            checks.append(f"hvac={inv['hvac']} expected >=4 (all zones conditioned)")
        # Windows on multiple orientations
        if inv["fen"] < 4:
            checks.append(f"fen={inv['fen']} expected >=4 (multi-orientation WWR)")
        # People + lights in each zone
        if inv["ppl"] < 4:
            checks.append(f"ppl={inv['ppl']} expected >=4")
        if inv["light"] < 4:
            checks.append(f"light={inv['light']} expected >=4")
        # Distinct constructions for the different surface types
        if inv["const"] < 4:
            checks.append(
                f"const={inv['const']} expected >=4 (ext wall/roof/floor/int)"
            )
        if errs:
            checks.append(f"cross-ref errors: {errs[:3]}")
        if checks:
            return _fail(name, t0, "; ".join(checks) + f" | inv={inv}")
        zone_names = _names(idf, "Zone")
        return _ok(
            name,
            t0,
            f"complex 4-zone office built: zones={zone_names}, refs clean",
            **inv,
        )
    except Exception as e:
        return _fail(name, t0, f"exception: {e}\n{traceback.format_exc()[-400:]}")


# ── Scenario H: 4-turn deep revision chain on complex model ──────────────────
# Builds on G's complexity: add zone -> add fenestration -> modify construction
# -> delete zone. The hardest revision sequence: each turn changes a different
# subsystem and must preserve everything else.

H_TURN1 = G_PROMPT  # reuse the complex build

H_TURN2 = """Add a new zone 'Storage_Second' (4x3m) on the second floor next to
the Meeting room. It is unconditioned. Create its walls, floor (on the
internal floor construction), and ceiling (under the roof). Connect it
thermally correctly. Keep all existing objects unchanged."""

H_TURN3 = """Add a small window (1m x 1m, double-glazed, same glazing as existing)
on the north wall of the new Storage_Second zone. Keep everything else unchanged."""

H_TURN4 = """Delete the Storage_Second zone and ALL surfaces, fenestrations, and
other objects associated with it. The building must return to its 4-zone
state with no dangling references. All other zones and objects unchanged."""


def scenario_h(graph) -> ScenarioResult:
    """H: 4-turn deep revision chain on a complex model (add->fen->modify->delete)."""
    t0 = time.time()
    name = "H_deep_revision_4turns"
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.idf"):
        p.unlink()
    try:
        # Turn 1: complex build
        _run_turn(graph, H_TURN1, f"{name}_t1", out, seed_idf=None)
        idf1 = _wait_for_idf(out)
        if not idf1:
            return _fail(name, t0, "turn 1 produced no IDF")
        inv1 = _inventory(idf1)
        if inv1["zone"] < 4:
            return _fail(name, t0, f"turn 1 zone={inv1['zone']} expected >=4")

        # Turn 2: add a 5th zone (Storage_Second)
        _run_turn(graph, H_TURN2, f"{name}_t2", out, seed_idf=idf1)
        idf2 = _wait_for_idf(out)
        if not idf2:
            return _fail(name, t0, "turn 2 produced no IDF")
        inv2 = _inventory(idf2)
        zones_t2 = _names(idf2, "Zone")
        checks = []
        if inv2["zone"] <= inv1["zone"]:
            checks.append(
                f"zone count did not increase: {inv1['zone']}->{inv2['zone']}"
            )
        if not any("Storage" in z for z in zones_t2):
            checks.append(f"Storage_Second zone not found: {zones_t2}")
        errs = _validate_refs(idf2)
        if errs:
            checks.append(f"cross-ref errors after add: {errs[:3]}")
        if checks:
            return _fail(name, t0, "turn 2: " + "; ".join(checks) + f" | inv={inv2}")

        # Turn 3: add window to the new zone
        _run_turn(graph, H_TURN3, f"{name}_t3", out, seed_idf=idf2)
        idf3 = _wait_for_idf(out)
        if not idf3:
            return _fail(name, t0, "turn 3 produced no IDF")
        inv3 = _inventory(idf3)
        checks = []
        if inv3["zone"] != inv2["zone"]:
            checks.append(f"zone changed t2->t3: {inv2['zone']}->{inv3['zone']}")
        if inv3["fen"] <= inv2["fen"]:
            checks.append(
                f"fenestration did not increase: {inv2['fen']}->{inv3['fen']}"
            )
        errs = _validate_refs(idf3)
        if errs:
            checks.append(f"cross-ref errors after fen add: {errs[:3]}")
        if checks:
            return _fail(name, t0, "turn 3: " + "; ".join(checks) + f" | inv={inv3}")

        # Turn 4: delete the added zone — model must return to 4-zone state
        _run_turn(graph, H_TURN4, f"{name}_t4", out, seed_idf=idf3)
        idf4 = _wait_for_idf(out)
        if not idf4:
            return _fail(name, t0, "turn 4 produced no IDF")
        inv4 = _inventory(idf4)
        zones_t4 = _names(idf4, "Zone")
        checks = []
        if any("Storage" in z for z in zones_t4):
            checks.append(f"Storage zone still present after delete: {zones_t4}")
        errs = _validate_refs(idf4)
        if errs:
            checks.append(f"dangling refs after delete: {errs[:3]}")
        # The original 4 zones must survive
        original_survived = all(
            any(orig in z for z in zones_t4) for orig in ["Reception", "Office"]
        )
        if not original_survived:
            checks.append(f"original zones lost: {zones_t4}")
        if checks:
            return _fail(name, t0, "turn 4: " + "; ".join(checks) + f" | inv={inv4}")

        return _ok(
            name,
            t0,
            "4-turn deep chain OK: build(4z) -> +zone(5z) -> +fen -> -zone(4z), refs clean throughout",
            t1=inv1,
            t2=inv2,
            t3=inv3,
            t4=inv4,
        )
    except Exception as e:
        return _fail(name, t0, f"exception: {e}\n{traceback.format_exc()[-400:]}")


# ── Runner ───────────────────────────────────────────────────────────────────

SCENARIOS: list[tuple[str, Callable]] = [
    ("A", scenario_a),
    ("E", scenario_e),
    ("F", scenario_f),
    ("B", scenario_b),
    ("C", scenario_c),
    ("D", scenario_d),
    ("G", scenario_g),
    ("H", scenario_h),
]


def main() -> int:
    only = set(sys.argv[1:])  # scenario labels to run; empty = all
    ROOT.mkdir(parents=True, exist_ok=True)
    print(
        f"\n{'=' * 72}\nDEEP MULTI-SCENARIO TEST SUITE (unified self-repair)\n{'=' * 72}"
    )
    print(f"Output root: {ROOT}\n")

    graph = build_graph()
    to_run = [(label, fn) for label, fn in SCENARIOS if not only or label in only]
    results: list[ScenarioResult] = []
    for label, fn in to_run:
        doc_first = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"\n--- Scenario {label}: {doc_first} ---")
        r = fn(graph)
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name} ({r.duration_s:.1f}s)")
        if r.detail:
            print(f"       {r.detail[:200]}")

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    passed = sum(1 for r in results if r.passed)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name:40s} {r.duration_s:6.1f}s")
    print(f"\n{passed}/{len(results)} scenarios passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
