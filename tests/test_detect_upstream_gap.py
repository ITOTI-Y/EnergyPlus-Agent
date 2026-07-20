"""Unit tests for the IDF-grounded upstream-gap detector.

detect_upstream_gap_from_state re-derives the gap from the LIVE
validate_references() result rather than scanning the message history.
This avoids the stale-gap false back-hop where a missing_ref ToolMessage
reported on an earlier round keeps triggering a back-hop even after the
LLM successfully healed the reference.
"""

from idfpy.models import (
    BuildingSurfaceDetailed,
    BuildingSurfaceDetailedVerticesItem,
    Construction,
    Material,
    Zone,
)

from src.agent.nodes._share import detect_upstream_gap_from_state
from src.mcp.state import ConfigState

_VERTS = [
    BuildingSurfaceDetailedVerticesItem(
        vertex_x_coordinate=0.0, vertex_y_coordinate=0.0, vertex_z_coordinate=0.0
    )
] * 3


def _surface(state: ConfigState, name="S1", zone_name="Z1", construction_name="C1"):
    state.idf.add(
        BuildingSurfaceDetailed(
            name=name,
            surface_type="Wall",
            construction_name=construction_name,
            zone_name=zone_name,
            outside_boundary_condition="Outdoors",
            sun_exposure="SunExposed",
            wind_exposure="WindExposed",
            vertices=_VERTS,
        )
    )


def _material(state: ConfigState, name="M1"):
    state.idf.add(
        Material(
            name=name,
            roughness="rough",
            thickness=0.1,
            conductivity=1.0,
            density=1000.0,
            specific_heat=1000.0,
        )
    )


def _construction(state: ConfigState, name="C1", outside_layer="M1"):
    state.idf.add(Construction(name=name, outside_layer=outside_layer))


def test_no_gap_when_model_is_clean():
    state = ConfigState()
    _material(state)
    _construction(state)
    state.idf.add(Zone(name="Z1"))
    _surface(state)
    assert detect_upstream_gap_from_state(state, "surface") is None


def test_reports_missing_zone_for_surface_phase():
    state = ConfigState()
    _material(state)
    _construction(state)
    _surface(state, zone_name="Ghost_Zone")  # zone absent
    gap = detect_upstream_gap_from_state(state, "surface")
    assert gap == {
        "target": "zone",
        "missing_ref": "Zone",
        "missing_name": "Ghost_Zone",
    }


def test_gap_vanishes_after_zone_is_added():
    """The regression case: once the missing object is created, the gap
    must NOT be reported — even if a stale missing_ref ToolMessage is
    floating around (the detector never looks at messages)."""
    state = ConfigState()
    _material(state)
    _construction(state)
    _surface(state, zone_name="Will_Appear")
    assert detect_upstream_gap_from_state(state, "surface") is not None
    state.idf.add(Zone(name="Will_Appear"))
    assert detect_upstream_gap_from_state(state, "surface") is None


def test_reports_missing_material_for_construction_phase():
    state = ConfigState()
    _construction(state, outside_layer="Ghost_Material")  # material absent
    gap = detect_upstream_gap_from_state(state, "construction")
    assert gap == {
        "target": "material",
        "missing_ref": "Material",
        "missing_name": "Ghost_Material",
    }


def test_returns_none_when_no_errors():
    assert detect_upstream_gap_from_state(ConfigState(), "surface") is None
