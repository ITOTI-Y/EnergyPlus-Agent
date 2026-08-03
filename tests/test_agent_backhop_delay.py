"""Tests for the delayed back-hop behaviour in invoke_with_self_repair.

The back-hop is deferred until the self-repair budget is exhausted so the
phase LLM gets a chance to self-heal (e.g. by switching to an existing
upstream name). Two invariants are asserted:

1. A genuine, persistent upstream gap (the LLM never recovers) eventually
   surfaces a hop_request after MAX_SELF_REPAIR_ROUNDS.
2. A gap that the LLM HEALS (it repairs the IDF so validate_references no
   longer reports the missing upstream object) does NOT produce a
   hop_request — even though a stale ``missing_ref`` ToolMessage may still
   be present in the accumulated message history.

Invariant (2) is the regression case: an earlier version of
detect_upstream_gap scanned the message history, so a missing_ref reported
on round 0 stayed visible on round N and triggered a false back-hop after
the LLM had already fixed the reference. The fix re-derives the gap from
the LIVE IDF via detect_upstream_gap_from_state.
"""

from idfpy.models.constructions import Construction, Material
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    BuildingSurfaceDetailedVerticesItem,
    Zone,
)
from langchain_core.messages import AIMessage, ToolMessage

from src.agent.nodes._share import MAX_SELF_REPAIR_ROUNDS, invoke_with_self_repair
from src.mcp.state import ConfigState


def _missing_zone_tool_result(name: str = "Missing_Zone") -> dict:
    """A ToolMessage payload reporting a missing upstream Zone."""
    import json

    return {
        "messages": [
            ToolMessage(
                content=json.dumps(
                    {
                        "success": False,
                        "message": f"Zone '{name}' not found.",
                        "data": {"missing_ref": "Zone", "missing_name": name},
                    }
                ),
                tool_call_id="missing-zone",
            )
        ]
    }


_VERTS = [
    BuildingSurfaceDetailedVerticesItem(
        vertex_x_coordinate=0.0, vertex_y_coordinate=0.0, vertex_z_coordinate=0.0
    )
] * 3


class _PersistentGapAgent:
    """Every invoke re-emits a missing_ref tool error AND keeps the IDF
    dangling (surface references a zone that is never created). The gap is
    genuinely persistent — both the tool signal and validate_references()
    agree it never recovers."""

    def __init__(self):
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        return _missing_zone_tool_result()


class _SelfHealingAgent:
    """On the first invoke, emit a missing_ref tool error (the LLM tried a
    wrong zone name). On the second invoke, 'succeed' — and critically,
    REPAIR the IDF by switching the surface's zone_name to an existing
    zone. This models the LLM calling list_zones and adopting a valid name.

    Note the stale ToolMessage from round 0 remains in the history passed
    in (state.messages), exactly as add_messages would accumulate it. The
    gap detector must NOT be fooled by it.
    """

    def __init__(self, local: ConfigState):
        self.calls = 0
        self.local = local

    def invoke(self, state):
        self.calls += 1
        if self.calls == 1:
            return _missing_zone_tool_result()
        # Round 2: the LLM "heals" by switching to an existing zone name.
        # Seed that existing zone (idempotent — invoke may be called again
        # on later repair rounds) and rewrite the surface's reference so
        # validate_references() no longer reports a dangling zone.
        existing = {
            getattr(z, "name", "") for z in self.local.idf.all_of_type("Zone").values()
        }
        if "Real_Zone" not in existing:
            self.local.idf.add(Zone(name="Real_Zone"))
        for surf in self.local.idf.all_of_type("BuildingSurface:Detailed").values():
            surf.zone_name = "Real_Zone"
        return {"messages": [AIMessage(content="switched to existing zone name")]}


def _surface_with_dangling_zone(local: ConfigState, zone_name: str = "Missing_Zone"):
    """Seed a surface whose zone_name does not yet exist. The construction
    is made real so the ONLY gap is the missing zone — keeping the
    assertion (hop_request.target == 'zone') unambiguous."""
    local.idf.add(
        Material(
            name="M1",
            roughness="rough",
            thickness=0.1,
            conductivity=1.0,
            density=1000.0,
            specific_heat=1000.0,
        )
    )
    local.idf.add(Construction(name="C1", outside_layer="M1"))
    local.idf.add(
        BuildingSurfaceDetailed(
            name="S1",
            surface_type="Wall",
            construction_name="C1",
            zone_name=zone_name,
            outside_boundary_condition="Outdoors",
            sun_exposure="SunExposed",
            wind_exposure="WindExposed",
            vertices=_VERTS,
        )
    )


def test_persistent_gap_surfaces_backhop_after_repair_budget():
    """A gap that never heals must still produce a hop_request once the
    repair budget is exhausted."""
    local = ConfigState()
    _surface_with_dangling_zone(local)
    agent = _PersistentGapAgent()

    result = invoke_with_self_repair(
        agent,
        local,
        "Create surfaces for Missing_Zone",
        phase="surface",
    )

    # MAX_SELF_REPAIR_ROUNDS + 1 invokes, then hop_request.
    assert agent.calls == MAX_SELF_REPAIR_ROUNDS + 1
    assert result["hop_request"]["target"] == "zone"
    assert result["hop_request"]["missing_name"] == "Missing_Zone"


def test_self_healed_gap_does_not_backhop():
    """If the LLM repairs the IDF so validate_references() no longer sees
    the missing upstream object, no hop_request must be issued — even
    though the stale missing_ref ToolMessage is still in the history."""
    local = ConfigState()
    _surface_with_dangling_zone(local)
    agent = _SelfHealingAgent(local)

    result = invoke_with_self_repair(
        agent,
        local,
        "Create surfaces for Missing_Zone",
        phase="surface",
    )

    assert agent.calls == 2  # one failed, one healed
    assert "hop_request" not in result
