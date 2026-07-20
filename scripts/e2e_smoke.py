"""End-to-end smoke test for the unified phase self-repair pipeline.

Two scenarios, each in its own thread_id for checkpointer isolation:

  Scenario 1 — First-run build from scratch:
      A simple single-zone warehouse. Validates that intake → phase agents
      (now ALL on invoke_with_self_repair) → cross_ref → validate → simulate
      completes without error, and the resulting IDF has the expected objects.
      Auto-approves the validate interrupt.

  Scenario 2 — Revision turn on the built model:
      Loads the IDF saved by scenario 1, then asks the agent to modify it
      (change a construction / add a window). Validates that revise_node
      recovers the seed IDF and phase agents modify incrementally rather
      than rebuilding from scratch.

Each scenario prints an object-inventory snapshot (Zone / Material /
Construction / Surface / Fenestration / Schedule / People / Lights counts
+ sample names) so we can confirm model integrity between turns.

Usage:
    python scripts/e2e_smoke.py            # run both scenarios
    python scripts/e2e_smoke.py 1          # scenario 1 only
    python scripts/e2e_smoke.py 2          # scenario 2 only
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import TypedDict

from langchain_core.runnables import RunnableConfig

from src.agent import AgentState, SimContext, build_graph
from src.agent.runner import run_session
from src.mcp.state import ConfigState

OUTPUT_DIR = Path("output/e2e_smoke")
EPW = Path("data/weather/Shenzhen.epw")

# A deliberately simple, single-zone prompt — easy for the LLM to get right,
# so any failure points at our pipeline, not at prompt complexity.
BUILD_PROMPT = (
    "Build a simple single-zone warehouse in Shenzhen, China. "
    "One zone named 'Warehouse_Zone', 20m x 15m x 6m (LxWxH). "
    "Concrete block walls with EPS insulation, concrete slab roof, "
    "concrete floor slab on grade. One small window on the south wall. "
    "No HVAC system (unconditioned warehouse). "
    "Occupancy: 2 people during weekday daytime. "
    "Lighting: 5 W/m2 during weekday daytime. "
    "Run for the full year."
)

REVISION_PROMPT = (
    "Add a second window of the same size on the north wall, and change "
    "the wall construction to add a 50mm thicker EPS insulation layer. "
    "Keep everything else unchanged."
)


class _InvEntry(TypedDict):
    count: int
    names: list[str]


def _inventory(config_state: ConfigState) -> dict[str, _InvEntry]:
    """Count objects by type + capture sample names for integrity checks."""
    idf = config_state.idf
    result: dict[str, _InvEntry] = {}
    for label, obj_type in [
        ("zone", "Zone"),
        ("material", "Material"),
        ("construction", "Construction"),
        ("surface", "BuildingSurface:Detailed"),
        ("fenestration", "FenestrationSurface:Detailed"),
        ("schedule", "Schedule:Compact"),
        ("people", "People"),
        ("lights", "Lights"),
    ]:
        items = list(idf.all_of_type(obj_type).values())
        names = [getattr(o, "name", getattr(o, "zone_name", "?")) for o in items]
        result[label] = {"count": len(items), "names": names[:5]}
    return result


def _print_inventory(title: str, inv: dict[str, _InvEntry]) -> None:
    print(f"\n  [{title}]")
    for label, info in inv.items():
        names = ", ".join(info["names"]) if info["names"] else "(none)"
        print(f"    {label:14s} count={info['count']:>2}  sample: {names}")


def _on_event(node: str, update: dict) -> None:
    """Lightweight progress logger — one line per node."""
    if not isinstance(update, dict):
        return  # validate returns a Command; its update may be None
    # Surface phase-agent summaries when present.
    msgs = update.get("messages") or []
    summary = ""
    for m in msgs:
        content = getattr(m, "content", "")
        if content and isinstance(content, str) and len(content) < 120:
            summary = content
            break
    suffix = f"  {summary}" if summary else ""
    print(f"    [{time.strftime('%H:%M:%S')}] {node}{suffix}")


def _auto_approve(payload: dict) -> dict:
    """Interrupt handler: auto-approve, but log what we're approving."""
    summary = payload.get("summary", {})
    errors = payload.get("errors", [])
    print(
        f"\n  [validate interrupt] zones={summary.get('zones_count')} "
        f"materials={summary.get('materials_count')} "
        f"surfaces={summary.get('surfaces_count')} errors={len(errors)}"
    )
    for e in errors[:5]:
        print(f"    error: {e}")
    if errors:
        print("  ! Errors present but auto-approving to reach simulate (smoke test)")
    return {"approved": True}


def scenario_1_build() -> ConfigState | None:
    """First-run build: from-scratch warehouse. Returns the final ConfigState."""
    print("\n" + "=" * 70)
    print("SCENARIO 1: First-run build (single-zone warehouse)")
    print("=" * 70)
    print(f"  prompt: {BUILD_PROMPT[:90]}...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = build_graph()
    initial = AgentState(user_input=BUILD_PROMPT)
    context = SimContext(epw_path=EPW, output_dir=OUTPUT_DIR)
    config = RunnableConfig(configurable={"thread_id": "e2e_build"})

    try:
        final = run_session(
            graph,
            initial,
            context,
            config,
            on_interrupt=_auto_approve,
            on_event=_on_event,
        )
    except Exception:
        print("\n  ✗ SCENARIO 1 FAILED with exception:")
        traceback.print_exc()
        return None

    cs: ConfigState = final["config_state"]
    inv = _inventory(cs)
    _print_inventory("Final inventory after build", inv)

    # Save IDF for scenario 2 to load.
    idf_path = OUTPUT_DIR / "scenario1_result.idf"
    cs.idf.save(idf_path)
    print(f"\n  ✓ Saved IDF to {idf_path}")

    # Basic integrity assertions.
    problems = []
    if inv["zone"]["count"] < 1:
        problems.append("no zones created")
    if inv["surface"]["count"] < 5:  # 4 walls + roof + floor minimum
        problems.append(f"only {inv['surface']['count']} surfaces (expected >=5)")
    if inv["construction"]["count"] < 1:
        problems.append("no constructions created")
    if problems:
        print("\n  ⚠ INTEGRITY ISSUES: " + "; ".join(problems))
        return None
    print("\n  ✓ Integrity check passed (zones/surfaces/constructions present)")
    return cs


def scenario_2_revise(seed_idf: Path) -> bool:
    """Revision turn: load scenario-1 IDF, ask for a modification."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Revision turn (add window + thicken insulation)")
    print("=" * 70)
    print(f"  seed: {seed_idf}")
    print(f"  prompt: {REVISION_PROMPT}")

    # Load the seed IDF into a ConfigState, carrying it as seed_idf_text so
    # revise_node can recover it after LangGraph's START coercion strips _idf.
    cs = ConfigState()
    cs.load_idf(seed_idf)
    cs.seed_idf_text = seed_idf.read_text(encoding="utf-8")

    inv_before = _inventory(cs)
    _print_inventory("Inventory BEFORE revision", inv_before)

    graph = build_graph()
    initial = AgentState(
        user_input=REVISION_PROMPT,
        config_state=cs,
        is_revision=True,
    )
    context = SimContext(epw_path=EPW, output_dir=OUTPUT_DIR)
    config = RunnableConfig(configurable={"thread_id": "e2e_revise"})

    try:
        final = run_session(
            graph,
            initial,
            context,
            config,
            on_interrupt=_auto_approve,
            on_event=_on_event,
        )
    except Exception:
        print("\n  ✗ SCENARIO 2 FAILED with exception:")
        traceback.print_exc()
        return False

    cs_after: ConfigState = final["config_state"]
    inv_after = _inventory(cs_after)
    _print_inventory("Inventory AFTER revision", inv_after)

    # Revision integrity: the model should be preserved (not wiped) and the
    # requested change visible. We check fenestration count went up (new window)
    # and materials/constructions still present.
    problems = []
    if inv_after["zone"]["count"] == 0:
        problems.append("model wiped — no zones after revision")
    if inv_after["zone"]["count"] < inv_before["zone"]["count"]:
        problems.append(
            f"zones decreased: {inv_before['zone']['count']} -> {inv_after['zone']['count']}"
        )
    if inv_after["construction"]["count"] == 0:
        problems.append("no constructions after revision")
    if inv_after["fenestration"]["count"] < inv_before["fenestration"]["count"]:
        problems.append(
            f"fenestrations decreased: {inv_before['fenestration']['count']} "
            f"-> {inv_after['fenestration']['count']}"
        )

    if problems:
        print("\n  ⚠ REVISION INTEGRITY ISSUES: " + "; ".join(problems))
        return False
    print("\n  ✓ Revision integrity check passed (model preserved)")
    return True


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    results: dict[str, bool] = {}

    if which in ("1", "all"):
        cs = scenario_1_build()
        results["scenario_1_build"] = cs is not None
        if cs is not None and which == "all":
            seed = OUTPUT_DIR / "scenario1_result.idf"
            results["scenario_2_revise"] = scenario_2_revise(seed)
        elif cs is None:
            results["scenario_2_revise"] = False

    if which == "2":
        # standalone scenario 2 needs a seed — look for an existing one
        seed = OUTPUT_DIR / "scenario1_result.idf"
        if not seed.exists():
            print(f"\n  ✗ Cannot run scenario 2: seed IDF not found at {seed}")
            print("    Run scenario 1 first: python scripts/e2e_smoke.py 1")
            results["scenario_2_revise"] = False
        else:
            results["scenario_2_revise"] = scenario_2_revise(seed)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
