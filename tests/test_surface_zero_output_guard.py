"""Tests for the surface phase's 0-output guard (P4).

When the surface phase has produced ZERO BuildingSurface:Detailed objects
and still has self-repair budget, invoke_with_self_repair forces another
repair round with a "you've built nothing yet" nudge instead of issuing a
hop_request. This prevents surface from back-hopping on the first missing
zone/construction when it could simply build every surface whose upstream
already exists.
"""

from idfpy.models import (
    BuildingSurfaceDetailed,
    BuildingSurfaceDetailedVerticesItem,
    Construction,
    Material,
    Zone,
)
from langchain_core.messages import AIMessage

from src.agent.nodes._share import MAX_SELF_REPAIR_ROUNDS, invoke_with_self_repair
from src.mcp.state import ConfigState


class _NeverBuildsAgent:
    """An agent that never creates any surface (simulates an LLM that keeps
    requesting back-hop for a missing ref instead of building what it can).
    Used to verify the guard keeps retrying until the budget is exhausted,
    and that the final hop_request is for the genuine gap (not '0 surfaces').
    """

    def __init__(self):
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        return {"messages": [AIMessage(content="did nothing")]}


class _BuildsOnSecondCallAgent:
    """An agent that builds nothing on round 0, then creates a real surface
    on round 1. The 0-output guard must give it that second chance rather
    than terminating / back-hopping immediately."""

    def __init__(self, local: ConfigState):
        self.calls = 0
        self.local = local

    def invoke(self, state):
        self.calls += 1
        if self.calls == 1:
            return {"messages": [AIMessage(content="did nothing yet")]}
        # Round 2: actually build a complete, valid surface.

        self.local.idf.add(
            Material(
                name="M1",
                roughness="rough",
                thickness=0.1,
                conductivity=1.0,
                density=1000.0,
                specific_heat=1000.0,
            )
        )
        self.local.idf.add(Construction(name="C1", outside_layer="M1"))
        self.local.idf.add(Zone(name="Z1"))
        verts = [
            BuildingSurfaceDetailedVerticesItem(
                vertex_x_coordinate=0.0,
                vertex_y_coordinate=0.0,
                vertex_z_coordinate=0.0,
            )
        ] * 3
        self.local.idf.add(
            BuildingSurfaceDetailed(
                name="S1",
                surface_type="Wall",
                construction_name="C1",
                zone_name="Z1",
                outside_boundary_condition="Outdoors",
                sun_exposure="SunExposed",
                wind_exposure="WindExposed",
                vertices=verts,
            )
        )
        return {"messages": [AIMessage(content="built the surface")]}


def test_zero_surface_output_keeps_repairing_until_budget_exhausted():
    """With zero surfaces and no other gap, the guard drives all
    MAX_SELF_REPAIR_ROUNDS+1 invokes. No hop_request is issued because
    there is no actual upstream gap (the IDF is just empty)."""
    local = ConfigState()
    agent = _NeverBuildsAgent()

    result = invoke_with_self_repair(
        agent,
        local,
        "Create surfaces for the zones",
        phase="surface",
    )

    assert agent.calls == MAX_SELF_REPAIR_ROUNDS + 1
    # No upstream gap existed (no dangling refs), so no back-hop.
    assert "hop_request" not in result


def test_zero_surface_output_gets_second_chance_to_build():
    """The guard must not terminate on round 0 with zero surfaces — it
    gives the agent another round, and if the agent then builds a valid
    surface, the phase completes cleanly (no hop_request)."""
    local = ConfigState()
    agent = _BuildsOnSecondCallAgent(local)

    result = invoke_with_self_repair(
        agent,
        local,
        "Create surfaces for the zones",
        phase="surface",
    )

    assert agent.calls == 2
    assert "hop_request" not in result
    surfaces = local.idf.all_of_type("BuildingSurface:Detailed")
    assert len(surfaces) == 1


def test_non_surface_phase_is_unaffected_by_zero_output_guard():
    """The guard is surface-specific: a construction phase that produces
    zero output should terminate on round 0 (no scoped errors, no gap),
    NOT be forced into extra repair rounds."""
    local = ConfigState()
    agent = _NeverBuildsAgent()

    result = invoke_with_self_repair(
        agent,
        local,
        "Create constructions",
        phase="construction",
    )

    # No errors, no gap, and the guard does not apply to construction ->
    # terminates after a single invoke.
    assert agent.calls == 1
    assert "hop_request" not in result


def test_zero_surface_feedback_does_not_claim_cross_ref_failure():
    """Regression: when surface_empty is the ONLY signal (no scoped errors,
    no gap), the repair feedback must NOT claim 'cross-reference validation
    failed' or 'missing upstream reference' — that would mislead the LLM.
    It should describe the real condition (zero output)."""
    from src.agent.nodes._share import _build_repair_feedback

    body = _build_repair_feedback(scoped=[], gap=None, surface_empty=True)

    assert (
        "zero output" in body
        or "produced zero" in body.lower()
        or "not created ANY BuildingSurface" in body
    )
    # Must NOT falsely claim a cross-ref or upstream failure.
    assert "Cross-reference validation failed" not in body
    assert "missing upstream reference" not in body


def test_scoped_error_feedback_still_claims_cross_ref_failure():
    """Sanity: when scoped errors ARE present, the feedback must still
    report them as cross-reference failures (the normal path)."""
    from src.agent.nodes._share import _build_repair_feedback

    body = _build_repair_feedback(
        scoped=["Surface 'S1' references zone 'Z' which does not exist."],
        gap=None,
        surface_empty=False,
    )

    assert "Cross-reference validation failed" in body
    assert "Surface 'S1' references zone 'Z'" in body
